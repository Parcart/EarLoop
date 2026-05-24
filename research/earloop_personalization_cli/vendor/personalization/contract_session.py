from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .contract_mapper import InterpretableContractMapper8D
from .contract_metrics import mapped_curve_metrics, mapped_distance_to_target, mapped_pair_metrics
from .contract_pair_generator import ContractPairConfig, ContractPairGenerator
from .contract_space import contract_distance
from .contract_feedback import (
    apply_feedback_to_model,
    apply_feedback_to_state,
    synthetic_feedback_decision,
)
from .preference_model import LogisticDistancePreferenceModel
from .preference_update import update_state_from_choice
from .state import PreferenceState, init_preference_state
from .synthetic_dataset import row_to_synthetic_user
from .synthetic_user import SyntheticUser

ContractStrategy = Literal[
    "semantic_contract_v6",
    "phase_mixed_contract_v6",
    "direct_contract_v6",
]


@dataclass
class ContractSessionConfig:
    """End-to-end personalization settings in z-contract space."""

    strategy: ContractStrategy = "phase_mixed_contract_v6"
    experiment_label: str | None = None
    n_steps: int = 25
    seed: int | None = None
    init_std: float = 0.95
    std_decay: float = 0.95
    min_std: float = 0.10
    clip_value: float = 2.0

    # Preference-state update learning rates.
    semantic_lr: float = 0.23
    direct_lr: float = 0.35
    anchor_selected_lr: float = 0.00

    # Online preference model.
    model_lr: float = 0.06
    model_temperature: float = 0.75
    model_l2: float = 0.003

    # Phase routing for phase_mixed_contract_v6.
    # v6.1 keeps the semantic backbone longer and uses direct refinement sparsely.
    # The session does not hard-stop at the marker; the marker only enables late refinement.
    phase_warmup_steps: int = 3
    phase_semantic_until_step: int = 16
    phase_axis_every: int = 5
    phase_candidate_pool_every: int = 7

    # Post-marker routing. v6.2 keeps semantic as the default while the session is
    # still improving. Direct/axis/candidate probes are used only after a local
    # plateau, and then with explicit probabilities.
    post_marker_probe_window: int = 3
    post_marker_probe_min_gain_z: float = 0.025
    post_marker_probe_min_gain_db_rms: float = 0.070
    post_marker_semantic_probability: float = 0.70
    post_marker_direct_probability: float = 0.20
    post_marker_axis_probability: float = 0.07
    post_marker_candidate_pool_probability: float = 0.03

    # Readiness / soft-stop marker. The full session can continue after marker.
    # The marker means: "we could reasonably stop here, but can continue if the user wants".
    min_ready_step: int = 15
    ready_mean_std_threshold: float = 0.62
    ready_update_norm_threshold: float = 0.28
    ready_distance_z_threshold: float = 0.95
    ready_distance_db_rms_threshold: float = 1.65
    ready_distance_step_window: int = 3
    ready_min_gain_z: float = 0.045
    ready_min_gain_db_rms: float = 0.10

    # Post-marker behavior.
    direct_mode: Literal["blend", "pm", "trust"] = "trust"
    post_marker_use_direct: bool = True

    # dB-aware pair safety.
    pair_config: ContractPairConfig | None = None

    # Optional simulated in-session directional feedback.
    # Feedback is triggered when the virtual user considers both A/B options poor
    # or misaligned, but only stochastically. It updates the session state and
    # online Preference Model; the learned/contract mapper is not retrained.
    enable_feedback: bool = False
    feedback_policy: Literal["none", "random", "cosine", "hybrid"] = "hybrid"
    feedback_min_step: int = 4
    feedback_cooldown_steps: int = 2
    feedback_strength: float = 1.0
    feedback_max_probability: float = 0.34
    feedback_base_probability: float = 0.015
    feedback_random_probability: float = 0.12
    feedback_bad_distance_z: float = 0.95
    feedback_cosine_threshold: float = 0.42
    feedback_ambiguity_margin: float = 0.18
    feedback_misalignment_weight: float = 0.35
    feedback_model_lr: float = 0.35

    # Optional phase-aware feedback scaling. Early feedback is steering/rescue,
    # middle feedback is normal correction, and late/ready feedback is audible
    # fine-tuning rather than an almost inaudible micro-step.
    feedback_phase_aware: bool = False
    feedback_early_until_step: int = 6
    feedback_late_from_step: int = 17
    feedback_early_strength_multiplier: float = 1.35
    feedback_mid_strength_multiplier: float = 1.00
    feedback_late_strength_multiplier: float = 0.80
    feedback_early_std_decay: float = 0.82
    feedback_mid_std_decay: float = 0.88
    feedback_late_std_decay: float = 0.93
    feedback_early_model_lr_multiplier: float = 1.15
    feedback_mid_model_lr_multiplier: float = 1.00
    feedback_late_model_lr_multiplier: float = 0.80
    feedback_min_audible_strength: float = 0.55


@dataclass
class ContractSessionResult:
    strategy_name: str
    target_mode: str
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[dict]
    distances_z: np.ndarray
    distances_db_rms: np.ndarray
    ready_step: int | None
    used_steps: int
    final_status: str
    final_distance_z: float
    final_distance_db_rms: float


def _is_ready(
    step: int,
    cfg: ContractSessionConfig,
    state: PreferenceState,
    update_norm: float,
    distances_z: list[float],
    distances_db_rms: list[float],
) -> bool:
    """Soft-stop marker heuristic for contract-space sessions.

    The marker is intentionally not a hard stop. It is used to answer:
    \"Would it be reasonable to stop here if the user is satisfied?\"
    """
    if step < int(cfg.min_ready_step):
        return False

    mean_std = float(np.mean(state.z_std))
    current_z = float(distances_z[-1]) if distances_z else np.inf
    current_db = float(distances_db_rms[-1]) if distances_db_rms else np.inf

    mean_std_ok = mean_std <= float(cfg.ready_mean_std_threshold)
    update_ok = float(update_norm) <= float(cfg.ready_update_norm_threshold)
    distance_ok = (
        current_z <= float(cfg.ready_distance_z_threshold)
        or current_db <= float(cfg.ready_distance_db_rms_threshold)
    )

    plateau_ok = False
    window = int(cfg.ready_distance_step_window)
    if len(distances_z) > window:
        gain_z = float(distances_z[-window - 1] - distances_z[-1])
        gain_db = float(distances_db_rms[-window - 1] - distances_db_rms[-1])
        plateau_ok = (
            gain_z <= float(cfg.ready_min_gain_z)
            and gain_db <= float(cfg.ready_min_gain_db_rms)
        )

    return bool(mean_std_ok and update_ok and (distance_ok or plateau_ok))


def _selected_lr(pair_meta: dict, choice: str, cfg: ContractSessionConfig, ready_active: bool) -> float:
    source_group = str(pair_meta.get("source_group", ""))
    direct_role = pair_meta.get("selected_direct_role")
    if source_group.startswith("direct"):
        if direct_role == choice:
            return float(cfg.direct_lr)
        return float(cfg.anchor_selected_lr)
    if ready_active:
        return float(cfg.semantic_lr) * 0.70
    return float(cfg.semantic_lr)


def _recent_improvement(
    distances_z: list[float],
    distances_db_rms: list[float],
    window: int,
) -> tuple[float, float]:
    """Return improvement over the last `window` completed steps."""
    window = int(window)
    if window <= 0 or len(distances_z) <= window or len(distances_db_rms) <= window:
        return np.inf, np.inf
    gain_z = float(distances_z[-window - 1] - distances_z[-1])
    gain_db = float(distances_db_rms[-window - 1] - distances_db_rms[-1])
    return gain_z, gain_db


def _weighted_late_probe_source(
    cfg: ContractSessionConfig,
    rng: np.random.Generator,
) -> Literal["semantic", "candidate_pool", "axis", "direct"]:
    """Sample a late-stage probe source with semantic kept as the dominant default."""
    weights = np.asarray([
        float(cfg.post_marker_semantic_probability),
        float(cfg.post_marker_direct_probability),
        float(cfg.post_marker_axis_probability),
        float(cfg.post_marker_candidate_pool_probability),
    ], dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 1e-12:
        return "semantic"
    weights = weights / float(weights.sum())
    # numpy may return a plain Python str here, which has no .item().
    # Cast explicitly so this works across NumPy versions.
    return str(rng.choice(
        np.asarray(["semantic", "direct", "axis", "candidate_pool"], dtype=object),
        p=weights,
    ))


def _select_pair_source(
    step: int,
    cfg: ContractSessionConfig,
    ready_step: int | None,
    distances_z: list[float] | None = None,
    distances_db_rms: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> Literal["semantic", "candidate_pool", "axis", "direct"]:
    """Choose the next question source for the configured v6.2 strategy.

    v6.1 already restored the semantic backbone, but late direct/axis probes still
    appeared while the semantic trajectory was improving. v6.2 is more conservative
    about switching away from semantic: after the marker it first checks whether the
    recent distance is still decreasing. If yes, it keeps asking semantic questions;
    if no, it uses a small weighted mixture of direct/axis/candidate probes.
    """
    ready_active = ready_step is not None
    distances_z = [] if distances_z is None else distances_z
    distances_db_rms = [] if distances_db_rms is None else distances_db_rms
    rng = np.random.default_rng(0) if rng is None else rng

    if cfg.strategy == "semantic_contract_v6":
        return "semantic"

    if cfg.strategy == "direct_contract_v6":
        return "direct" if step > int(cfg.phase_warmup_steps) else "semantic"

    if cfg.strategy == "phase_mixed_contract_v6":
        if step <= int(cfg.phase_warmup_steps):
            return "candidate_pool"

        # Keep semantic backbone long enough to avoid the conservative plateau.
        if step <= int(cfg.phase_semantic_until_step):
            return "semantic"

        if ready_active and cfg.post_marker_use_direct:
            gain_z, gain_db = _recent_improvement(
                distances_z=distances_z,
                distances_db_rms=distances_db_rms,
                window=int(cfg.post_marker_probe_window),
            )

            # If semantic refinement is still making measurable progress, do not
            # replace it with direct/axis probes yet.
            still_improving = (
                gain_z >= float(cfg.post_marker_probe_min_gain_z)
                or gain_db >= float(cfg.post_marker_probe_min_gain_db_rms)
            )
            if still_improving:
                return "semantic"

            return _weighted_late_probe_source(cfg, rng)

        # Before marker: semantic by default, with occasional focused probes.
        if int(cfg.phase_axis_every) > 0 and step % int(cfg.phase_axis_every) == 0:
            return "axis"
        if int(cfg.phase_candidate_pool_every) > 0 and step % int(cfg.phase_candidate_pool_every) == 0:
            return "candidate_pool"
        return "semantic"

    return "semantic"


def _feedback_phase_params(
    cfg: ContractSessionConfig,
    step: int,
    ready_active: bool,
) -> tuple[float, float, float, str]:
    """Return strength/std/model multipliers for stage-aware feedback.

    The values are deliberately not too small late in the session: late feedback
    should remain audible for real users, but it should avoid throwing away the
    refined state.
    """
    if not bool(cfg.feedback_phase_aware):
        return 1.0, 0.88, 1.0, "constant"

    if int(step) <= int(cfg.feedback_early_until_step):
        return (
            float(cfg.feedback_early_strength_multiplier),
            float(cfg.feedback_early_std_decay),
            float(cfg.feedback_early_model_lr_multiplier),
            "early",
        )

    if ready_active or int(step) >= int(cfg.feedback_late_from_step):
        return (
            max(float(cfg.feedback_late_strength_multiplier), float(cfg.feedback_min_audible_strength)),
            float(cfg.feedback_late_std_decay),
            float(cfg.feedback_late_model_lr_multiplier),
            "late",
        )

    return (
        float(cfg.feedback_mid_strength_multiplier),
        float(cfg.feedback_mid_std_decay),
        float(cfg.feedback_mid_model_lr_multiplier),
        "mid",
    )


def run_contract_personalization_session(
    synthetic_user: SyntheticUser,
    target_mode: str = "unknown",
    config: ContractSessionConfig | None = None,
    mapper=None,
) -> ContractSessionResult:
    cfg = ContractSessionConfig() if config is None else config
    rng = np.random.default_rng(cfg.seed)
    mapper = InterpretableContractMapper8D() if mapper is None else mapper
    pair_cfg = cfg.pair_config or ContractPairConfig(clip_value=cfg.clip_value)
    pair_generator = ContractPairGenerator(config=pair_cfg, mapper=mapper, rng=rng)

    state = init_preference_state(dim=len(synthetic_user.z_target), init_std=cfg.init_std)
    model = LogisticDistancePreferenceModel(
        dim=len(synthetic_user.z_target),
        lr=cfg.model_lr,
        temperature=cfg.model_temperature,
        l2=cfg.model_l2,
        clip_value=cfg.clip_value,
    )

    records: list[dict] = []
    distances_z: list[float] = []
    distances_db_rms: list[float] = []
    ready_step: int | None = None
    last_feedback_step: int | None = None
    pending_feedback_recovery: dict | None = None
    feedback_count = 0
    final_status = "completed_full_budget"

    for step in range(1, int(cfg.n_steps) + 1):
        state_before = state.copy()
        ready_active = ready_step is not None

        pair_source = _select_pair_source(
            step=step,
            cfg=cfg,
            ready_step=ready_step,
            distances_z=distances_z,
            distances_db_rms=distances_db_rms,
            rng=rng,
        )

        if pending_feedback_recovery is not None:
            pair_source = "feedback_recovery"

        if pair_source == "feedback_recovery":
            z_a, z_b, direction, pair_meta = pair_generator.feedback_recovery(
                state=state,
                feedback_label=str(pending_feedback_recovery.get("feedback_label", "too_bright")),
                strength=float(pending_feedback_recovery.get("feedback_strength", cfg.feedback_strength)),
            )
            pending_feedback_recovery = None
        elif pair_source == "direct":
            z_a, z_b, direction, pair_meta = pair_generator.direct_refinement(
                state=state,
                model=model,
                mode=cfg.direct_mode,
            )
        elif pair_source == "candidate_pool":
            z_a, z_b, direction, pair_meta = pair_generator.candidate_pool(state)
        elif pair_source == "axis":
            z_a, z_b, direction, pair_meta = pair_generator.axis_refinement(state)
        else:
            z_a, z_b, direction, pair_meta = pair_generator.semantic_active(state)

        pair_meta = dict(pair_meta)
        pair_meta["strategy"] = cfg.experiment_label or cfg.strategy
        pair_meta["step"] = int(step)
        pair_meta["ready_active"] = bool(ready_active)
        pair_meta["ready_step"] = ready_step

        p_before = model.predict_proba_a(z_a, z_b)
        pred_before = "A" if p_before >= 0.5 else "B"
        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)

        cooldown_active = (
            last_feedback_step is not None
            and (int(step) - int(last_feedback_step)) <= int(cfg.feedback_cooldown_steps)
        )
        (
            feedback_strength_multiplier,
            feedback_std_decay,
            feedback_model_lr_multiplier,
            feedback_phase,
        ) = _feedback_phase_params(cfg, step=step, ready_active=ready_active)
        effective_feedback_strength = float(cfg.feedback_strength) * float(feedback_strength_multiplier)
        effective_feedback_model_lr = float(cfg.feedback_model_lr) * float(feedback_model_lr_multiplier)

        feedback_decision = synthetic_feedback_decision(
            z_a=z_a,
            z_b=z_b,
            state=state,
            z_target=synthetic_user.z_target,
            feature_importance=synthetic_user.feature_importance,
            rng=rng,
            step=step,
            policy=cfg.feedback_policy if cfg.enable_feedback else "none",
            min_step=cfg.feedback_min_step,
            cooldown_active=cooldown_active,
            max_probability=cfg.feedback_max_probability,
            base_probability=cfg.feedback_base_probability,
            random_probability=cfg.feedback_random_probability,
            bad_distance_z=cfg.feedback_bad_distance_z,
            cosine_threshold=cfg.feedback_cosine_threshold,
            ambiguity_margin=cfg.feedback_ambiguity_margin,
            misalignment_weight=cfg.feedback_misalignment_weight,
            strength=effective_feedback_strength,
        )

        if feedback_decision.use_feedback and feedback_decision.label is not None:
            observed_action = "feedback"
            feedback_count += 1
            feedback_label = str(feedback_decision.label)
            feedback_strength = float(feedback_decision.strength)
            model_record = {
                "loss_before": np.nan,
                "loss_after": np.nan,
                "feedback_model_update": True,
            }
            state = apply_feedback_to_state(
                state=state,
                label=feedback_label,
                strength=feedback_strength,
                clip_value=cfg.clip_value,
                std_decay=feedback_std_decay,
                min_std=cfg.min_std,
            )
            feedback_model_record = apply_feedback_to_model(
                model=model,
                state_before=state_before,
                state_after=state,
                feedback_model_lr=effective_feedback_model_lr,
            )
            last_feedback_step = int(step)
            pending_feedback_recovery = {
                "feedback_label": feedback_label,
                "feedback_strength": feedback_strength,
            }
            lr = 0.0
        else:
            observed_action = "choice"
            feedback_label = None
            feedback_strength = 0.0
            feedback_model_record = {"model_delta_norm": 0.0}
            model_record = model.update(z_a, z_b, choice)
            lr = _selected_lr(pair_meta, choice, cfg, ready_active=ready_active)
            state = update_state_from_choice(
                state=state,
                z_a=z_a,
                z_b=z_b,
                choice=choice,
                lr=lr,
                std_decay=cfg.std_decay,
                min_std=cfg.min_std,
                clip_value=cfg.clip_value,
                pair_meta=pair_meta,
            )

        update_norm = float(np.linalg.norm(state.z_mean - state_before.z_mean))

        target_metrics = mapped_distance_to_target(state.z_mean, synthetic_user.z_target, mapper)
        pair_metrics = mapped_pair_metrics(z_a, z_b, mapper)
        state_curve_metrics = mapped_curve_metrics(state.z_mean, mapper)
        target_curve_metrics = mapped_curve_metrics(synthetic_user.z_target, mapper)

        dist_z = float(target_metrics["distance_to_target_z"])
        dist_db = float(target_metrics["distance_to_target_db_rms"])
        distances_z.append(dist_z)
        distances_db_rms.append(dist_db)

        if ready_step is None and _is_ready(
            step=step,
            cfg=cfg,
            state=state,
            update_norm=update_norm,
            distances_z=distances_z,
            distances_db_rms=distances_db_rms,
        ):
            ready_step = int(step)
            pair_meta["ready_marker_set"] = True
        else:
            pair_meta["ready_marker_set"] = False

        record = {
            "step": int(step),
            "strategy": cfg.experiment_label or cfg.strategy,
            "pair_source": pair_source,
            "pair_source_group": pair_meta.get("source_group"),
            "contract_mode": pair_meta.get("contract_mode"),
            "ready_marker_set": bool(pair_meta.get("ready_marker_set", False)),
            "choice": choice,
            "selected_role": choice if observed_action == "choice" else "FEEDBACK",
            "observed_action": observed_action,
            "feedback_used": bool(observed_action == "feedback"),
            "feedback_label": feedback_label,
            "feedback_strength": float(feedback_strength),
            "feedback_phase": feedback_phase,
            "feedback_strength_multiplier": float(feedback_strength_multiplier),
            "feedback_std_decay": float(feedback_std_decay),
            "feedback_model_lr_effective": float(effective_feedback_model_lr),
            "feedback_probability": float(feedback_decision.probability),
            "feedback_best_label_cosine": float(feedback_decision.best_label_cosine),
            "feedback_badness_score": float(feedback_decision.badness_score),
            "feedback_ambiguity_score": float(feedback_decision.ambiguity_score),
            "feedback_misalignment_score": float(feedback_decision.misalignment_score),
            "feedback_best_candidate_distance_z": float(feedback_decision.best_candidate_distance_z),
            "feedback_state_distance_z": float(feedback_decision.state_distance_z),
            "feedback_reason": feedback_decision.reason,
            "feedback_count_so_far": int(feedback_count),
            "feedback_model_delta_norm": float(feedback_model_record.get("model_delta_norm", 0.0)),
            "applied_lr": float(lr),
            "u_a": float(u_a),
            "u_b": float(u_b),
            "utility_margin": float(abs(u_a - u_b)),
            "p_before": float(p_before),
            "pred_before": pred_before,
            "correct_before": bool(pred_before == choice),
            "loss_before": float(model_record["loss_before"]),
            "loss_after": float(model_record["loss_after"]),
            "z_mean_norm": float(np.linalg.norm(state.z_mean)),
            "z_model_norm": float(np.linalg.norm(model.z_pref)),
            "z_target_norm": float(np.linalg.norm(synthetic_user.z_target)),
            "mean_z_std": float(np.mean(state.z_std)),
            "update_norm": update_norm,
            "ready_step": ready_step,
            "ready_active": bool(ready_active),
            "distance_to_target_z": dist_z,
            "distance_to_target_db_rms": dist_db,
            "distance_to_target_db_mae": float(target_metrics["distance_to_target_db_mae"]),
            "distance_to_target_db_max_abs": float(target_metrics["distance_to_target_db_max_abs"]),
            "state_mapped_max_abs_db": float(state_curve_metrics["mapped_max_abs_db"]),
            "state_mapped_mean_abs_db": float(state_curve_metrics["mapped_mean_abs_db"]),
            "target_mapped_max_abs_db": float(target_curve_metrics["mapped_max_abs_db"]),
            "pair_distance_z": float(pair_metrics["pair_distance_z"]),
            "pair_distance_db_rms": float(pair_metrics["pair_distance_db_rms"]),
            "pair_max_abs_db": float(pair_metrics["pair_max_abs_db"]),
            "safety_ok": bool(pair_meta.get("safety_ok", True)),
            "safety_shrink": float(pair_meta.get("safety_shrink", 1.0)),
            "z_a": z_a.copy(),
            "z_b": z_b.copy(),
            "z_mean": state.z_mean.copy(),
            "z_target": synthetic_user.z_target.copy(),
        }
        records.append(record)

    if ready_step is not None:
        final_status = "completed_after_ready_marker"

    return ContractSessionResult(
        strategy_name=cfg.experiment_label or cfg.strategy,
        target_mode=target_mode,
        final_state=state,
        final_model=model,
        records=records,
        distances_z=np.asarray(distances_z, dtype=np.float64),
        distances_db_rms=np.asarray(distances_db_rms, dtype=np.float64),
        ready_step=ready_step,
        used_steps=int(cfg.n_steps),
        final_status=final_status,
        final_distance_z=float(distances_z[-1]),
        final_distance_db_rms=float(distances_db_rms[-1]),
    )


def run_contract_comparison_on_dataset(
    dataset: pd.DataFrame,
    strategies: Iterable[ContractStrategy] = ("semantic_contract_v6", "phase_mixed_contract_v6"),
    n_steps: int = 25,
    seed: int = 42,
    mapper=None,
    max_users: int | None = None,
    pair_config: ContractPairConfig | None = None,
    session_kwargs: dict | None = None,
) -> list[ContractSessionResult]:
    results: list[ContractSessionResult] = []
    session_kwargs = {} if session_kwargs is None else dict(session_kwargs)
    df = dataset.copy()
    if max_users is not None and len(df) > int(max_users):
        df = df.sample(n=int(max_users), random_state=seed).reset_index(drop=True)

    for strategy in strategies:
        for row_idx, row in df.iterrows():
            user_seed = int(seed + row_idx * 1009 + abs(hash(strategy)) % 997)
            user = row_to_synthetic_user(row, seed=user_seed)
            cfg = ContractSessionConfig(
                strategy=strategy,
                n_steps=n_steps,
                seed=user_seed,
                pair_config=pair_config,
                **session_kwargs,
            )
            result = run_contract_personalization_session(
                synthetic_user=user,
                target_mode=str(row.get("target_mode", "unknown")),
                config=cfg,
                mapper=mapper,
            )
            results.append(result)
    return results


def contract_records_dataframe(results: list[ContractSessionResult]) -> pd.DataFrame:
    rows = []
    for result_index, result in enumerate(results):
        for rec in result.records:
            row = {
                "result_index": int(result_index),
                "strategy": result.strategy_name,
                "target_mode": result.target_mode,
            }
            for key, value in rec.items():
                if isinstance(value, np.ndarray):
                    continue
                row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def contract_summary_dataframe(results: list[ContractSessionResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        final_record = result.records[-1] if result.records else {}
        if result.ready_step is not None and len(result.records) >= int(result.ready_step):
            ready_record = result.records[int(result.ready_step) - 1]
            distance_z_at_ready = float(ready_record.get("distance_to_target_z", np.nan))
            distance_db_at_ready = float(ready_record.get("distance_to_target_db_rms", np.nan))
        else:
            distance_z_at_ready = np.nan
            distance_db_at_ready = np.nan

        rows.append({
            "strategy": result.strategy_name,
            "target_mode": result.target_mode,
            "used_steps": result.used_steps,
            "ready_step": result.ready_step,
            "steps_after_ready": np.nan if result.ready_step is None else int(result.used_steps - result.ready_step),
            "final_status": result.final_status,
            "distance_z_at_ready": distance_z_at_ready,
            "distance_db_at_ready": distance_db_at_ready,
            "extra_gain_z_after_ready": np.nan if np.isnan(distance_z_at_ready) else float(distance_z_at_ready - result.final_distance_z),
            "extra_gain_db_after_ready": np.nan if np.isnan(distance_db_at_ready) else float(distance_db_at_ready - result.final_distance_db_rms),
            "final_distance_z": result.final_distance_z,
            "final_distance_db_rms": result.final_distance_db_rms,
            "final_mapped_max_abs_db": float(final_record.get("state_mapped_max_abs_db", np.nan)),
            "final_mapped_mean_abs_db": float(final_record.get("state_mapped_mean_abs_db", np.nan)),
            "final_mean_z_std": float(final_record.get("mean_z_std", np.nan)),
            "feedback_count": int(sum(1 for rec in result.records if bool(rec.get("feedback_used", False)))),
            "feedback_rate": float(np.mean([bool(rec.get("feedback_used", False)) for rec in result.records])) if result.records else 0.0,
        })
    return pd.DataFrame(rows)
