from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import distance_to_target
from .pair_generator import PairGenerator
from .preference_model import LogisticDistancePreferenceModel
from .preference_update import update_state_from_choice
from .state import PreferenceState, init_preference_state
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser
from .safe_anchor_challenger import (
    V5_STRATEGY_NAME,
    _direction_lock_status,
    _model_blend_direction,
    _update_intensity_score,
)
from .safe_anchor_challenger_v51 import (
    V51_STRATEGY_NAME,
    make_conservative_anchor_challenger_pair,
    _update_state_conservative,
    _summary_row as _v51_summary_row,
)
from .safe_anchor_challenger_v52 import (
    V52_STRATEGY_NAME,
    V52Config,
    V52_DISPLAY_NAMES,
    run_semantic_backbone_confirmation_session,
    run_v52_comparison_on_dataset,
    summarize_v52_sessions,
    source_usage_table_v52,
)


V53_STRATEGY_NAME = "semantic_backbone_confirmation_gate_v53"

V53_DISPLAY_NAMES = dict(V52_DISPLAY_NAMES)
V53_DISPLAY_NAMES.update({
    V53_STRATEGY_NAME: "Semantic backbone + confirmation gate v5.3",
})


@dataclass
class V53Config(V52Config):
    """V5.3: semantic backbone with confirmation gate instead of hard stop.

    Difference from V5.2:
    - V5.2 starts a confirmation block and then stops after the block.
    - V5.3 treats confirmation as a gate:
        * if anchor is confirmed and state is stable, stop;
        * if challenger wins, apply a tiny zone update and return to semantic backbone;
        * if uncertainty remains, continue semantic backbone.
    """

    min_confirmation_step: int = 16
    ready_threshold: float = 0.90
    mean_std_ready_threshold: float = 0.46
    update_norm_ready_threshold: float = 0.22

    max_confirmation_questions: int = 2
    required_anchor_confirmations: int = 2
    confirmation_cooldown_steps: int = 3
    max_confirmation_blocks: int = 2
    freeze_after_confirmed: bool = True

    # Keep confirmation as a local test. Anchor does not push the state.
    anchor_selected_lr: float = 0.0
    challenger_selected_lr: float = 0.055
    ready_anchor_lr: float = 0.0
    ready_challenger_lr: float = 0.035

    challenger_scales: tuple[float, ...] = (0.16, 0.26, 0.38)
    max_anchor_challenger_distance: float = 0.78
    min_audible_distance: float = 0.22


@dataclass
class V53SessionResult:
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[dict]
    distances: np.ndarray
    intensity_score: float
    direction_lock_step: int | None
    ready_step: int | None
    confirmation_start_step: int | None
    used_steps: int
    final_status: str
    confirmation_blocks: int
    anchor_confirmations: int
    challenger_corrections: int


def _semantic_update(
    state: PreferenceState,
    z_a: np.ndarray,
    z_b: np.ndarray,
    choice: str,
    lr: float,
    std_decay: float,
    min_std: float,
    clip_value: float | None,
    pair_meta: dict,
) -> PreferenceState:
    return update_state_from_choice(
        state=state,
        z_a=z_a,
        z_b=z_b,
        choice=choice,
        lr=lr,
        std_decay=std_decay,
        min_std=min_std,
        clip_value=clip_value,
        pair_meta=pair_meta,
    )


def _is_ready(
    *,
    step: int,
    cfg: V53Config,
    agreement_after: float,
    mean_std: float,
    update_norm: float,
) -> bool:
    return (
        step >= cfg.min_confirmation_step
        and agreement_after >= cfg.ready_threshold
        and mean_std <= cfg.mean_std_ready_threshold
        and update_norm <= cfg.update_norm_ready_threshold
    )


def run_semantic_backbone_confirmation_gate_session(
    synthetic_user: SyntheticUser,
    n_steps: int = 25,
    step_scale: float = 0.6,
    heuristic_lr: float = 0.25,
    model_lr: float = 0.06,
    model_temperature: float = 0.75,
    model_l2: float = 0.003,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    seed: int | None = None,
    config: V53Config | None = None,
) -> V53SessionResult:
    """Run V5.3 semantic backbone + confirmation gate.

    The backbone is always semantic_active_v21. Confirmation questions are only
    entered after readiness is detected. Confirmation does not automatically
    stop the session: it either validates the current profile or sends the loop
    back to the semantic backbone.
    """
    cfg = V53Config(base_step_scale=step_scale, clip_value=clip_value) if config is None else config
    rng = np.random.default_rng(seed)
    state = init_preference_state(dim=len(synthetic_user.z_target), init_std=init_std)
    pair_generator = PairGenerator(step_scale=step_scale, clip_value=clip_value, rng=rng)
    model = LogisticDistancePreferenceModel(
        dim=len(synthetic_user.z_target),
        lr=model_lr,
        temperature=model_temperature,
        l2=model_l2,
        clip_value=clip_value,
        feature_weight=np.ones(len(synthetic_user.z_target), dtype=np.float64),
    )

    records: list[dict] = []
    distances: list[float] = []
    intensity_score = 0.0
    direction_lock_step: int | None = None
    ready_step: int | None = None
    confirmation_start_step: int | None = None
    used_steps = int(n_steps)

    lock_hits = 0
    finalized = False
    confirmation_active = False
    confirmation_count = 0
    confirmation_blocks = 0
    anchor_confirmations = 0
    challenger_corrections = 0
    cooldown_until_step = 0
    final_status = "completed_without_stop"

    for step in range(1, int(n_steps) + 1):
        if finalized:
            dist = distance_to_target(state.z_mean, synthetic_user.z_target)
            distances.append(float(dist))
            records.append({
                "step": int(step),
                "phase": "stopped_after_confirmed_gate",
                "choice": "stop",
                "selected_role": "stopped",
                "applied_lr": 0.0,
                "u_a": np.nan,
                "u_b": np.nan,
                "utility_margin": np.nan,
                "distance_to_target": float(dist),
                "p_before": np.nan,
                "pred_before": None,
                "correct_before": False,
                "loss_before": np.nan,
                "loss_after": np.nan,
                "agreement_before": np.nan,
                "agreement_after": np.nan,
                "intensity_score": float(intensity_score),
                "mean_z_std": float(np.mean(state.z_std)),
                "update_norm": 0.0,
                "z_mean_norm": float(np.linalg.norm(state.z_mean)),
                "z_model_norm": float(np.linalg.norm(model.z_pref)),
                "z_blend_norm": np.nan,
                "pair_source": "stopped",
                "pair_source_group": "stopped",
                "sub_strategy": "stopped_after_confirmed_gate",
                "pair_distance": 0.0,
                "audibility_score": np.nan,
                "acceptability_score": np.nan,
                "anchor_acceptability_score": np.nan,
                "model_uncertainty": np.nan,
                "midrange_disturbance_penalty": np.nan,
                "safety_penalty": np.nan,
                "ready_step": ready_step,
                "confirmation_count": confirmation_count,
                "confirmation_blocks": confirmation_blocks,
                "confirmation_gate_decision": "already_stopped",
            })
            continue

        state_before = state.copy()
        locked, agreement, lock_hits = _direction_lock_status(state, model, step, cfg, lock_hits)
        if locked and direction_lock_step is None:
            direction_lock_step = int(step)

        # Activate a confirmation block only when not on cooldown and not too many blocks were already tried.
        if (
            not confirmation_active
            and step >= cooldown_until_step
            and confirmation_blocks < cfg.max_confirmation_blocks
        ):
            # Use current agreement proxy before the question. The stronger readiness
            # gate is checked after each update below.
            mean_std_pre = float(np.mean(state.z_std))
            if ready_step is not None or (step >= cfg.min_confirmation_step and agreement >= cfg.ready_threshold and mean_std_pre <= cfg.mean_std_ready_threshold):
                confirmation_active = True
                confirmation_count = 0
                anchor_confirmations = 0
                confirmation_blocks += 1
                if ready_step is None:
                    ready_step = int(step)
                if confirmation_start_step is None:
                    confirmation_start_step = int(step)

        if confirmation_active:
            phase = "confirmation_gate"
            z_a, z_b, direction, pair_meta = make_conservative_anchor_challenger_pair(
                state=state,
                model=model,
                rng=rng,
                config=cfg,
                intensity_score=intensity_score,
            )
            pair_meta = dict(pair_meta)
            pair_meta["strategy"] = V53_STRATEGY_NAME
            pair_meta["sub_strategy"] = "confirmation_gate"
            update_mode = "confirmation"
        else:
            phase = "semantic_active_backbone"
            z_a, z_b, direction, pair_meta = pair_generator.semantic_active_v21(state)
            pair_meta = dict(pair_meta)
            pair_meta["strategy"] = V53_STRATEGY_NAME
            pair_meta["sub_strategy"] = "semantic_active_backbone"
            pair_meta["source"] = "semantic_active_v21"
            pair_meta["source_group"] = "semantic"
            pair_meta["agreement"] = float(agreement)
            pair_meta["intensity_score"] = float(intensity_score)
            update_mode = "semantic"

        p_before = model.predict_proba_a(z_a, z_b)
        pred_before = "A" if p_before >= 0.5 else "B"
        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)
        model_record = model.update(z_a, z_b, choice)
        intensity_score = _update_intensity_score(intensity_score, state_before, z_a, z_b, choice)

        confirmation_gate_decision = "none"
        if update_mode == "confirmation":
            state, applied_lr, selected_role = _update_state_conservative(
                state=state,
                z_a=z_a,
                z_b=z_b,
                choice=choice,
                pair_meta=pair_meta,
                config=cfg,
                std_decay=std_decay,
                min_std=min_std,
                ready_active=True,
            )
            confirmation_count += 1
            if selected_role == "anchor_selected":
                anchor_confirmations += 1
                confirmation_gate_decision = "anchor_confirmed"
            elif selected_role == "challenger_selected":
                challenger_corrections += 1
                confirmation_gate_decision = "challenger_correction_return_to_backbone"
                confirmation_active = False
                confirmation_count = 0
                anchor_confirmations = 0
                cooldown_until_step = int(step + cfg.confirmation_cooldown_steps)
        else:
            applied_lr = heuristic_lr
            selected_role = "standard_semantic"
            state = _semantic_update(
                state=state,
                z_a=z_a,
                z_b=z_b,
                choice=choice,
                lr=heuristic_lr,
                std_decay=std_decay,
                min_std=min_std,
                clip_value=clip_value,
                pair_meta=pair_meta,
            )

        dist = distance_to_target(state.z_mean, synthetic_user.z_target)
        distances.append(float(dist))

        _, model_blend, agreement_after = _model_blend_direction(state, model, clip_value)
        update_norm = float(np.linalg.norm(state.z_mean - state_before.z_mean))
        mean_std = float(np.mean(state.z_std))

        if ready_step is None and _is_ready(
            step=step,
            cfg=cfg,
            agreement_after=agreement_after,
            mean_std=mean_std,
            update_norm=update_norm,
        ):
            ready_step = int(step)

        if update_mode == "confirmation" and confirmation_active:
            enough_anchor_confirmation = anchor_confirmations >= cfg.required_anchor_confirmations
            confirmation_budget_done = confirmation_count >= cfg.max_confirmation_questions
            stable_after_confirmation = _is_ready(
                step=step,
                cfg=cfg,
                agreement_after=agreement_after,
                mean_std=mean_std,
                update_norm=update_norm,
            )
            if enough_anchor_confirmation and stable_after_confirmation:
                confirmation_gate_decision = "stop_confirmed_anchor"
                final_status = "confirmed_anchor_stop"
                if cfg.freeze_after_confirmed:
                    finalized = True
                    used_steps = int(step)
            elif confirmation_budget_done:
                confirmation_gate_decision = "confirmation_budget_done_continue_backbone"
                confirmation_active = False
                confirmation_count = 0
                anchor_confirmations = 0
                cooldown_until_step = int(step + cfg.confirmation_cooldown_steps)

        records.append({
            "step": int(step),
            "phase": phase,
            "choice": choice,
            "selected_role": selected_role,
            "applied_lr": float(applied_lr),
            "u_a": float(u_a),
            "u_b": float(u_b),
            "utility_margin": float(abs(u_a - u_b)),
            "distance_to_target": float(dist),
            "p_before": float(p_before),
            "pred_before": pred_before,
            "correct_before": bool(pred_before == choice),
            "loss_before": float(model_record["loss_before"]),
            "loss_after": float(model_record["loss_after"]),
            "agreement_before": float(agreement),
            "agreement_after": float(agreement_after),
            "intensity_score": float(intensity_score),
            "mean_z_std": mean_std,
            "update_norm": update_norm,
            "z_mean_norm": float(np.linalg.norm(state.z_mean)),
            "z_model_norm": float(np.linalg.norm(model.z_pref)),
            "z_blend_norm": float(np.linalg.norm(model_blend)),
            "pair_source": pair_meta.get("source"),
            "pair_source_group": pair_meta.get("source_group"),
            "sub_strategy": pair_meta.get("sub_strategy"),
            "pair_distance": float(pair_meta.get("pair_distance", np.linalg.norm(z_a - z_b))),
            "audibility_score": float(pair_meta.get("audibility_score", np.nan)),
            "acceptability_score": float(pair_meta.get("acceptability_score", np.nan)),
            "anchor_acceptability_score": float(pair_meta.get("anchor_acceptability_score", np.nan)),
            "model_uncertainty": float(pair_meta.get("model_uncertainty", np.nan)),
            "midrange_disturbance_penalty": float(pair_meta.get("midrange_disturbance_penalty", np.nan)),
            "safety_penalty": float(pair_meta.get("safety_penalty", np.nan)),
            "ready_step": ready_step,
            "confirmation_count": confirmation_count,
            "confirmation_blocks": confirmation_blocks,
            "confirmation_gate_decision": confirmation_gate_decision,
        })

    if not finalized:
        if confirmation_blocks > 0:
            final_status = "confirmation_tried_completed_full_session"
        else:
            final_status = "completed_without_confirmation"

    return V53SessionResult(
        final_state=state,
        final_model=model,
        records=records,
        distances=np.asarray(distances, dtype=np.float64),
        intensity_score=float(intensity_score),
        direction_lock_step=direction_lock_step,
        ready_step=ready_step,
        confirmation_start_step=confirmation_start_step,
        used_steps=int(used_steps),
        final_status=final_status,
        confirmation_blocks=int(confirmation_blocks),
        anchor_confirmations=int(anchor_confirmations),
        challenger_corrections=int(challenger_corrections),
    )


def _summary_row(user_id: int, target_mode: str, strategy: str, result, z_target: np.ndarray) -> dict:
    d = np.concatenate([[float(np.linalg.norm(z_target))], result.distances])
    records_df = pd.DataFrame(result.records) if hasattr(result, "records") else pd.DataFrame()
    confirmation_rows = records_df[records_df.get("phase", pd.Series(dtype=str)).astype(str).eq("confirmation_gate")] if not records_df.empty else pd.DataFrame()
    anchor_rate = np.nan
    challenger_rate = np.nan
    if not confirmation_rows.empty and "selected_role" in confirmation_rows:
        anchor_rate = float((confirmation_rows["selected_role"] == "anchor_selected").mean())
        challenger_rate = float((confirmation_rows["selected_role"] == "challenger_selected").mean())
    return {
        "user_id": int(user_id),
        "target_mode": target_mode,
        "strategy": strategy,
        "n_steps": int(len(result.distances)),
        "used_steps": int(getattr(result, "used_steps", len(result.distances))),
        "initial_distance": float(d[0]),
        "final_distance": float(d[-1]),
        "best_distance": float(np.min(d)),
        "mean_distance": float(np.mean(d)),
        "improvement_abs": float(d[0] - d[-1]),
        "improvement_pct": float(100.0 * (d[0] - d[-1]) / (d[0] + 1e-8)),
        "direction_lock_step": getattr(result, "direction_lock_step", np.nan),
        "ready_step": getattr(result, "ready_step", np.nan),
        "confirmation_start_step": getattr(result, "confirmation_start_step", np.nan),
        "final_status": getattr(result, "final_status", "baseline"),
        "final_intensity_score": getattr(result, "intensity_score", np.nan),
        "confirmation_questions": int(len(confirmation_rows)) if not confirmation_rows.empty else 0,
        "confirmation_blocks": int(getattr(result, "confirmation_blocks", 0)),
        "anchor_selected_rate": anchor_rate,
        "challenger_selected_rate": challenger_rate,
        "challenger_corrections": int(getattr(result, "challenger_corrections", 0)),
        "mean_pair_distance": float(records_df["pair_distance"].mean()) if "pair_distance" in records_df else np.nan,
        "mean_audibility_score": float(records_df["audibility_score"].mean()) if "audibility_score" in records_df else np.nan,
        "mean_acceptability_score": float(records_df["acceptability_score"].mean()) if "acceptability_score" in records_df else np.nan,
        "mean_midrange_penalty": float(records_df["midrange_disturbance_penalty"].mean()) if "midrange_disturbance_penalty" in records_df else np.nan,
        "mean_applied_lr": float(records_df["applied_lr"].mean()) if "applied_lr" in records_df else np.nan,
        "stop_rate": float(getattr(result, "final_status", "") == "confirmed_anchor_stop"),
    }


def run_v53_comparison_on_dataset(
    dataset: pd.DataFrame,
    baseline_strategies: Iterable[str] = ("semantic_active_v21", "candidate_pool_active"),
    include_v5: bool = True,
    include_v51: bool = True,
    include_v52: bool = True,
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_seed_base: int = 40_000,
    user_seed_base: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    # Reuse V5.2 comparison for all older strategies, then append V5.3.
    sessions_base, steps_base, curves_base = run_v52_comparison_on_dataset(
        dataset=dataset,
        baseline_strategies=baseline_strategies,
        include_v5=include_v5,
        include_v51=include_v51,
        n_steps=n_steps,
        step_scale=step_scale,
        lr=lr,
        init_std=init_std,
        std_decay=std_decay,
        min_std=min_std,
        clip_value=clip_value,
        pair_seed_base=pair_seed_base,
        user_seed_base=user_seed_base,
    )

    if not include_v52:
        sessions_base = sessions_base[sessions_base["strategy"] != V52_STRATEGY_NAME].copy()
        steps_base = steps_base[steps_base["strategy"] != V52_STRATEGY_NAME].copy() if not steps_base.empty else steps_base
        for mode in list(curves_base.keys()):
            curves_base[mode].pop(V52_STRATEGY_NAME, None)

    rows: list[dict] = []
    step_rows: list[dict] = []

    # Convert curves to mutable lists.
    curve_store: dict[str, dict[str, list[np.ndarray]]] = {}
    for mode, by_strategy in curves_base.items():
        curve_store[mode] = {strategy: [arr for arr in values] for strategy, values in by_strategy.items()}
        curve_store[mode].setdefault(V53_STRATEGY_NAME, [])

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        curve_store.setdefault(target_mode, {}).setdefault(V53_STRATEGY_NAME, [])

        user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
        result_v53 = run_semantic_backbone_confirmation_gate_session(
            synthetic_user=user,
            n_steps=n_steps,
            step_scale=step_scale,
            heuristic_lr=lr,
            init_std=init_std,
            std_decay=std_decay,
            min_std=min_std,
            clip_value=clip_value,
            seed=pair_seed_base + user_id,
        )
        d = np.concatenate([[float(np.linalg.norm(z_target))], result_v53.distances])
        curve_store[target_mode][V53_STRATEGY_NAME].append(d)
        rows.append(_summary_row(user_id, target_mode, V53_STRATEGY_NAME, result_v53, z_target))
        for rec in result_v53.records:
            item = dict(rec)
            item.update({"user_id": user_id, "target_mode": target_mode, "strategy": V53_STRATEGY_NAME})
            step_rows.append(item)

    sessions = pd.concat([sessions_base, pd.DataFrame(rows)], axis=0, ignore_index=True)
    steps = pd.concat([steps_base, pd.DataFrame(step_rows)], axis=0, ignore_index=True) if not steps_base.empty else pd.DataFrame(step_rows)
    curves = {
        target_mode: {strategy: np.asarray(items, dtype=np.float64) for strategy, items in by_strategy.items()}
        for target_mode, by_strategy in curve_store.items()
    }
    return sessions, steps, curves


def summarize_v53_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    return (
        sessions
        .groupby(["target_mode", "strategy"])
        .agg(
            users=("user_id", "count"),
            mean_used_steps=("used_steps", "mean"),
            mean_initial_distance=("initial_distance", "mean"),
            mean_final_distance=("final_distance", "mean"),
            std_final_distance=("final_distance", "std"),
            mean_best_distance=("best_distance", "mean"),
            mean_mean_distance=("mean_distance", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            mean_direction_lock_step=("direction_lock_step", "mean"),
            mean_ready_step=("ready_step", "mean"),
            ready_rate=("ready_step", lambda s: float(pd.notna(s).mean())),
            mean_confirmation_start_step=("confirmation_start_step", "mean"),
            mean_confirmation_questions=("confirmation_questions", "mean"),
            mean_confirmation_blocks=("confirmation_blocks", "mean") if "confirmation_blocks" in sessions.columns else ("confirmation_questions", "mean"),
            mean_anchor_selected_rate=("anchor_selected_rate", "mean"),
            mean_challenger_selected_rate=("challenger_selected_rate", "mean") if "challenger_selected_rate" in sessions.columns else ("anchor_selected_rate", "mean"),
            mean_challenger_corrections=("challenger_corrections", "mean") if "challenger_corrections" in sessions.columns else ("confirmation_questions", "mean"),
            stop_rate=("stop_rate", "mean") if "stop_rate" in sessions.columns else ("used_steps", lambda s: 0.0),
            mean_pair_distance=("mean_pair_distance", "mean"),
            mean_audibility_score=("mean_audibility_score", "mean"),
            mean_acceptability_score=("mean_acceptability_score", "mean"),
            mean_midrange_penalty=("mean_midrange_penalty", "mean"),
            mean_applied_lr=("mean_applied_lr", "mean"),
        )
        .reset_index()
        .sort_values(["target_mode", "mean_final_distance"])
    )


def source_usage_table_v53(steps: pd.DataFrame) -> pd.DataFrame:
    return source_usage_table_v52(steps)


def save_v53_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    source_usage: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "article_v53_confirmation_gate",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    steps.to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    win_rates.to_csv(output_dir / f"{prefix}_win_rates.csv", index=False)
    source_usage.to_csv(output_dir / f"{prefix}_source_usage.csv", index=False)
