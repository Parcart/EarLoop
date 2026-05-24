from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .batch_eval import distances_with_initial
from .control_basis import CONTROL_BASIS_6D_TO_8D, CONTROL_NAMES_6D
from .loop import run_personalization_session_v0
from .metrics import distance_to_target
from .pair_generator import PairGenerator
from .preference_model import LogisticDistancePreferenceModel, cosine_similarity
from .preference_update import update_state_from_choice
from .state import PreferenceState, clip_vector, init_preference_state
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser
from .safe_anchor_challenger import (
    V5_STRATEGY_NAME,
    ZONE_DIRECTIONS,
    ZONE_GROUP_BY_DIRECTION,
    make_safe_calibration_pair,
    run_safe_anchor_challenger_session,
    _audibility_score,
    _candidate_acceptability,
    _direction_lock_status,
    _intensity_multiplier,
    _model_blend_direction,
    _randomize_pair_order,
    _unit,
    _update_intensity_score,
)


V51_STRATEGY_NAME = "safe_anchor_challenger_v51"

V51_DISPLAY_NAMES = {
    "semantic_active_v21": "Semantic active v3",
    "candidate_pool_active": "Candidate pool active",
    V5_STRATEGY_NAME: "Safe anchor-challenger v5",
    V51_STRATEGY_NAME: "Conservative anchor-challenger v5.1",
}

MID_FEATURE_INDICES = [2, 3, 4]  # lowmid, warmth, presence


@dataclass
class V51Config:
    """Conservative V5.1 configuration.

    The key difference from V5 is that anchor-challenger becomes a confirmation
    / refinement layer, not a full-speed exploration engine.
    """

    calibration_steps: int = 3
    min_anchor_step: int = 14
    direction_lock_threshold: float = 0.88
    ready_threshold: float = 0.90
    base_step_scale: float = 0.6
    calibration_scale: float = 0.55
    challenger_scales: tuple[float, ...] = (0.20, 0.35, 0.50, 0.65)
    max_anchor_challenger_distance: float = 1.10
    min_audible_distance: float = 0.28
    clip_value: float | None = 2.0

    # Update policy after direction lock.
    anchor_selected_lr: float = 0.015
    challenger_selected_lr: float = 0.085
    ready_anchor_lr: float = 0.0
    ready_challenger_lr: float = 0.035

    # Score weights for safe challenger selection.
    model_uncertainty_weight: float = 0.25
    zone_uncertainty_weight: float = 0.45
    audibility_weight: float = 0.35
    acceptability_weight: float = 0.35
    anchor_acceptability_weight: float = 0.10
    challenger_alignment_weight: float = 0.20
    safety_penalty_weight: float = 1.30
    midrange_penalty_weight: float = 0.70
    distance_penalty_weight: float = 0.90
    repetition_penalty_weight: float = 0.30

    direction_locked_patience: int = 2
    max_refinement_questions_after_ready: int = 3


@dataclass
class V51SessionResult:
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[dict]
    distances: np.ndarray
    intensity_score: float
    direction_lock_step: int | None
    ready_step: int | None
    final_status: str


def _recent_sources(state: PreferenceState, lookback: int = 3) -> list[str | None]:
    out: list[str | None] = []
    for item in state.history[-int(lookback):]:
        meta = item.get("pair_meta") or {}
        out.append(meta.get("source") or meta.get("control_name"))
    return out


def _conservative_anchor_point(
    state: PreferenceState,
    model: LogisticDistancePreferenceModel,
    clip_value: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return anchor, calibrated model vector, blend and agreement.

    The anchor is deliberately close to the current online state. It represents
    "the current best profile" rather than "one more step forward".
    """
    model_calibrated, blend, agreement = _model_blend_direction(state, model, clip_value)
    # Very small pull toward the calibrated blend keeps the anchor near the
    # current best state and avoids the V5 overshoot failure mode.
    anchor = clip_vector(0.85 * state.z_mean + 0.15 * blend, clip_value)
    return anchor, model_calibrated, blend, float(agreement)


def make_conservative_anchor_challenger_pair(
    state: PreferenceState,
    model: LogisticDistancePreferenceModel,
    rng: np.random.Generator,
    config: V51Config,
    intensity_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build a conservative anchor-challenger pair.

    Anchor is the current best profile. Challenger is a local, zone-aware
    alternative. This pair should diagnose one local preference without pushing
    the state aggressively after direction lock.
    """
    anchor, model_calibrated, blend, agreement = _conservative_anchor_point(
        state=state,
        model=model,
        clip_value=config.clip_value,
    )

    consensus_delta = blend - state.z_mean
    if np.linalg.norm(consensus_delta) < 1e-6:
        consensus_delta = model_calibrated if np.linalg.norm(model_calibrated) > 1e-6 else state.z_mean
    if np.linalg.norm(consensus_delta) < 1e-6:
        consensus_delta = np.ones_like(state.z_mean)
    consensus_direction = _unit(consensus_delta)

    # If the user repeatedly chooses softer variants, shrink challenger strength.
    intensity_mult = _intensity_multiplier(intensity_score)
    if intensity_score < 0:
        intensity_mult *= 0.82
    if intensity_score <= -3:
        intensity_mult *= 0.75

    zone_dirs = {
        f"zone_{name}": (direction, ZONE_GROUP_BY_DIRECTION.get(name, "zone"))
        for name, direction in ZONE_DIRECTIONS.items()
    }
    # A small consensus challenger is allowed, but no direct countercheck after lock.
    model_dirs = {
        "model_direction_local": (consensus_direction, "model_direction"),
    }
    candidate_directions: dict[str, tuple[np.ndarray, str]] = {}
    candidate_directions.update(zone_dirs)
    candidate_directions.update(model_dirs)

    z_std = np.asarray(state.z_std, dtype=np.float64)
    recent = _recent_sources(state)
    anchor_acceptability, anchor_safety, anchor_mid = _candidate_acceptability(
        anchor, state.z_mean, "model_direction", config.clip_value
    )

    candidates: list[dict] = []
    for source, (direction_raw, group) in candidate_directions.items():
        base_direction = _unit(direction_raw)
        for sign in (+1.0, -1.0):
            direction = sign * base_direction
            alignment = float(np.dot(direction, consensus_direction))

            # In refinement, avoid strongly adversarial challengers. We want a
            # local alternative, not a new global hypothesis that derails state.
            if alignment < -0.35 and source != "model_direction_local":
                continue
            if source == "model_direction_local" and sign < 0:
                continue

            # Prefer either local alternatives near orthogonal zones or mild
            # agreement with consensus. Avoid identical forward pushes.
            alignment_bonus = float(1.0 - min(abs(alignment), 1.0))
            if 0.10 <= alignment <= 0.70:
                alignment_bonus += 0.10

            uncertainty_score = float(np.sum(np.abs(direction) * z_std) / (np.sum(np.abs(direction)) + 1e-8))
            for scale_mult in config.challenger_scales:
                scale = float(config.base_step_scale) * float(scale_mult) * float(intensity_mult)
                challenger = clip_vector(anchor + scale * direction, config.clip_value)
                pair_distance = float(np.linalg.norm(anchor - challenger))

                acceptability, safety_penalty, mid_penalty = _candidate_acceptability(
                    challenger, anchor, group, config.clip_value
                )
                audibility = _audibility_score(pair_distance, config.min_audible_distance, ideal_distance=0.70)
                if pair_distance > config.max_anchor_challenger_distance:
                    distance_penalty = (pair_distance - config.max_anchor_challenger_distance) ** 2
                else:
                    distance_penalty = 0.0

                p_anchor = model.predict_proba_a(anchor, challenger)
                model_uncertainty = float(1.0 - 2.0 * abs(p_anchor - 0.5))
                repeat_penalty = 1.0 if source in recent else 0.0

                score = (
                    config.model_uncertainty_weight * model_uncertainty
                    + config.zone_uncertainty_weight * uncertainty_score
                    + config.audibility_weight * audibility
                    + config.challenger_alignment_weight * alignment_bonus
                    + config.acceptability_weight * acceptability
                    + config.anchor_acceptability_weight * anchor_acceptability
                    - config.safety_penalty_weight * (safety_penalty + anchor_safety)
                    - config.midrange_penalty_weight * (mid_penalty + 0.25 * anchor_mid)
                    - config.distance_penalty_weight * distance_penalty
                    - config.repetition_penalty_weight * repeat_penalty
                )

                candidates.append({
                    "source": source,
                    "source_group": group,
                    "direction": direction.copy(),
                    "challenger": challenger.copy(),
                    "scale": float(scale),
                    "scale_multiplier": float(scale_mult),
                    "score": float(score),
                    "agreement": float(agreement),
                    "alignment_to_consensus": float(alignment),
                    "alignment_bonus": float(alignment_bonus),
                    "uncertainty_score": float(uncertainty_score),
                    "audibility_score": float(audibility),
                    "pair_distance": float(pair_distance),
                    "acceptability_score": float(acceptability),
                    "anchor_acceptability_score": float(anchor_acceptability),
                    "model_uncertainty": float(model_uncertainty),
                    "p_anchor_preferred": float(p_anchor),
                    "safety_penalty": float(safety_penalty + anchor_safety),
                    "midrange_disturbance_penalty": float(mid_penalty + 0.25 * anchor_mid),
                    "distance_penalty": float(distance_penalty),
                    "repetition_penalty": float(repeat_penalty),
                })

    if not candidates:
        # Robust fallback: tiny local consensus question.
        challenger = clip_vector(anchor + 0.25 * config.base_step_scale * consensus_direction, config.clip_value)
        best = {
            "source": "model_direction_local_fallback",
            "source_group": "model_direction",
            "direction": consensus_direction.copy(),
            "challenger": challenger.copy(),
            "scale": float(0.25 * config.base_step_scale),
            "scale_multiplier": 0.25,
            "score": 0.0,
            "agreement": float(agreement),
            "alignment_to_consensus": 1.0,
            "alignment_bonus": 0.0,
            "uncertainty_score": 0.0,
            "audibility_score": _audibility_score(float(np.linalg.norm(anchor - challenger)), config.min_audible_distance),
            "pair_distance": float(np.linalg.norm(anchor - challenger)),
            "acceptability_score": 1.0,
            "anchor_acceptability_score": float(anchor_acceptability),
            "model_uncertainty": 0.0,
            "p_anchor_preferred": 0.5,
            "safety_penalty": 0.0,
            "midrange_disturbance_penalty": 0.0,
            "distance_penalty": 0.0,
            "repetition_penalty": 0.0,
        }
    else:
        best = max(candidates, key=lambda item: item["score"])
        challenger = best["challenger"]

    direction_hint = anchor - challenger
    z_a, z_b, signed_direction, order = _randomize_pair_order(anchor, challenger, direction_hint, rng)
    anchor_label = "A" if order == "candidate_1_as_A" else "B"
    challenger_label = "B" if anchor_label == "A" else "A"

    meta = {
        "strategy": V51_STRATEGY_NAME,
        "sub_strategy": "anchor_challenger_conservative",
        "source": best["source"],
        "source_group": best["source_group"],
        "control_name": best["source"],
        "control_direction": signed_direction.copy(),
        "anchor_challenger_role": "current_anchor_vs_safe_challenger",
        "anchor": anchor.copy(),
        "challenger": challenger.copy(),
        "anchor_label": anchor_label,
        "challenger_label": challenger_label,
        "agreement": best["agreement"],
        "scale": best["scale"],
        "scale_multiplier": best["scale_multiplier"],
        "intensity_score": float(intensity_score),
        "intensity_multiplier": float(intensity_mult),
        "score": best["score"],
        "alignment_to_consensus": best["alignment_to_consensus"],
        "uncertainty_score": best["uncertainty_score"],
        "audibility_score": best["audibility_score"],
        "pair_distance": best["pair_distance"],
        "acceptability_score": best["acceptability_score"],
        "anchor_acceptability_score": best["anchor_acceptability_score"],
        "model_uncertainty": best["model_uncertainty"],
        "p_anchor_preferred": best["p_anchor_preferred"],
        "safety_penalty": best["safety_penalty"],
        "midrange_disturbance_penalty": best["midrange_disturbance_penalty"],
        "distance_penalty": best["distance_penalty"],
        "repetition_penalty": best["repetition_penalty"],
        "order": order,
    }
    return z_a, z_b, signed_direction, meta


def _update_state_conservative(
    state: PreferenceState,
    z_a: np.ndarray,
    z_b: np.ndarray,
    choice: str,
    pair_meta: dict,
    config: V51Config,
    std_decay: float,
    min_std: float,
    ready_active: bool,
) -> tuple[PreferenceState, float, str]:
    """Anchor-aware update for conservative refinement."""
    anchor_label = pair_meta.get("anchor_label")
    challenger_label = pair_meta.get("challenger_label")
    if anchor_label is None or challenger_label is None:
        lr = config.challenger_selected_lr
        role = "unknown"
    elif choice == anchor_label:
        lr = config.ready_anchor_lr if ready_active else config.anchor_selected_lr
        role = "anchor_selected"
    else:
        lr = config.ready_challenger_lr if ready_active else config.challenger_selected_lr
        role = "challenger_selected"

    state = update_state_from_choice(
        state=state,
        z_a=z_a,
        z_b=z_b,
        choice=choice,
        lr=lr,
        std_decay=std_decay,
        min_std=min_std,
        clip_value=config.clip_value,
        pair_meta=pair_meta,
    )
    return state, float(lr), role


def run_conservative_anchor_challenger_session(
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
    config: V51Config | None = None,
) -> V51SessionResult:
    """Run V5.1 conservative anchor-challenger session."""
    cfg = V51Config(base_step_scale=step_scale, clip_value=clip_value) if config is None else config
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
    lock_hits = 0
    refinement_questions_after_ready = 0

    for step in range(1, int(n_steps) + 1):
        state_before = state.copy()
        locked, agreement, lock_hits = _direction_lock_status(state, model, step, cfg, lock_hits)
        if locked and direction_lock_step is None:
            direction_lock_step = int(step)

        ready_active = ready_step is not None
        if step <= cfg.calibration_steps:
            phase = "safe_calibration"
            z_a, z_b, direction, pair_meta = make_safe_calibration_pair(state, step, rng, cfg)
            update_mode = "standard_calibration"
        elif locked:
            phase = "anchor_challenger_conservative"
            if ready_active:
                refinement_questions_after_ready += 1
                phase = "ready_confirmation" if refinement_questions_after_ready > cfg.max_refinement_questions_after_ready else phase
            z_a, z_b, direction, pair_meta = make_conservative_anchor_challenger_pair(
                state=state,
                model=model,
                rng=rng,
                config=cfg,
                intensity_score=intensity_score,
            )
            if phase == "ready_confirmation":
                pair_meta = dict(pair_meta)
                pair_meta["sub_strategy"] = "ready_confirmation"
            update_mode = "conservative_anchor_challenger"
        else:
            phase = "semantic_active_warmup"
            z_a, z_b, direction, pair_meta = pair_generator.semantic_active_v21(state)
            pair_meta = dict(pair_meta)
            pair_meta["sub_strategy"] = "semantic_active_warmup"
            pair_meta["source"] = "semantic_active_v21"
            pair_meta["source_group"] = "semantic"
            pair_meta["agreement"] = float(agreement)
            pair_meta["intensity_score"] = float(intensity_score)
            update_mode = "standard_semantic"

        p_before = model.predict_proba_a(z_a, z_b)
        pred_before = "A" if p_before >= 0.5 else "B"
        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)
        model_record = model.update(z_a, z_b, choice)
        intensity_score = _update_intensity_score(intensity_score, state_before, z_a, z_b, choice)

        if update_mode == "conservative_anchor_challenger":
            state, applied_lr, selected_role = _update_state_conservative(
                state=state,
                z_a=z_a,
                z_b=z_b,
                choice=choice,
                pair_meta=pair_meta,
                config=cfg,
                std_decay=std_decay,
                min_std=min_std,
                ready_active=ready_active,
            )
        else:
            applied_lr = heuristic_lr
            selected_role = "standard"
            state = update_state_from_choice(
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
        if ready_step is None and step >= cfg.min_anchor_step:
            if agreement_after >= cfg.ready_threshold and mean_std <= 0.35 and update_norm <= 0.22:
                ready_step = int(step)

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
        })

    final_status = "ready_to_finalize" if ready_step is not None else "completed"
    return V51SessionResult(
        final_state=state,
        final_model=model,
        records=records,
        distances=np.asarray(distances, dtype=np.float64),
        intensity_score=float(intensity_score),
        direction_lock_step=direction_lock_step,
        ready_step=ready_step,
        final_status=final_status,
    )


def _summary_row(user_id: int, target_mode: str, strategy: str, result, z_target: np.ndarray) -> dict:
    d = np.concatenate([[float(np.linalg.norm(z_target))], result.distances])
    records_df = pd.DataFrame(result.records) if hasattr(result, "records") else pd.DataFrame()
    return {
        "user_id": int(user_id),
        "target_mode": target_mode,
        "strategy": strategy,
        "n_steps": int(len(result.distances)),
        "initial_distance": float(d[0]),
        "final_distance": float(d[-1]),
        "best_distance": float(np.min(d)),
        "mean_distance": float(np.mean(d)),
        "improvement_abs": float(d[0] - d[-1]),
        "improvement_pct": float(100.0 * (d[0] - d[-1]) / (d[0] + 1e-8)),
        "direction_lock_step": getattr(result, "direction_lock_step", np.nan),
        "ready_step": getattr(result, "ready_step", np.nan),
        "final_status": getattr(result, "final_status", "baseline"),
        "final_intensity_score": getattr(result, "intensity_score", np.nan),
        "mean_pair_distance": float(records_df["pair_distance"].mean()) if "pair_distance" in records_df else np.nan,
        "mean_audibility_score": float(records_df["audibility_score"].mean()) if "audibility_score" in records_df else np.nan,
        "mean_acceptability_score": float(records_df["acceptability_score"].mean()) if "acceptability_score" in records_df else np.nan,
        "mean_midrange_penalty": float(records_df["midrange_disturbance_penalty"].mean()) if "midrange_disturbance_penalty" in records_df else np.nan,
        "mean_applied_lr": float(records_df["applied_lr"].mean()) if "applied_lr" in records_df else np.nan,
    }


def run_v51_comparison_on_dataset(
    dataset: pd.DataFrame,
    baseline_strategies: Iterable[str] = ("semantic_active_v21", "candidate_pool_active"),
    include_v5: bool = True,
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_seed_base: int = 20_000,
    user_seed_base: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """Run V5.1 and selected baselines on a fixed dataset."""
    rows: list[dict] = []
    step_rows: list[dict] = []
    strategies = list(baseline_strategies)
    if include_v5:
        strategies.append(V5_STRATEGY_NAME)
    strategies.append(V51_STRATEGY_NAME)
    curve_store: dict[str, dict[str, list[np.ndarray]]] = {}

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        curve_store.setdefault(target_mode, {strategy: [] for strategy in strategies})

        for strategy in baseline_strategies:
            user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
            result = run_personalization_session_v0(
                synthetic_user=user,
                n_steps=n_steps,
                step_scale=step_scale,
                lr=lr,
                init_std=init_std,
                std_decay=std_decay,
                min_std=min_std,
                clip_value=clip_value,
                pair_strategy=strategy,
                seed=pair_seed_base + user_id,
            )
            d = distances_with_initial(result, z_target)
            curve_store[target_mode][strategy].append(d)
            rows.append({
                "user_id": user_id,
                "target_mode": target_mode,
                "strategy": strategy,
                "n_steps": int(len(result.distances)),
                "initial_distance": float(d[0]),
                "final_distance": float(d[-1]),
                "best_distance": float(np.min(d)),
                "mean_distance": float(np.mean(d)),
                "improvement_abs": float(d[0] - d[-1]),
                "improvement_pct": float(100.0 * (d[0] - d[-1]) / (d[0] + 1e-8)),
                "direction_lock_step": np.nan,
                "ready_step": np.nan,
                "final_status": "baseline",
                "final_intensity_score": np.nan,
                "mean_pair_distance": np.nan,
                "mean_audibility_score": np.nan,
                "mean_acceptability_score": np.nan,
                "mean_midrange_penalty": np.nan,
                "mean_applied_lr": np.nan,
            })

        if include_v5:
            user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
            result_v5 = run_safe_anchor_challenger_session(
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
            d = np.concatenate([[float(np.linalg.norm(z_target))], result_v5.distances])
            curve_store[target_mode][V5_STRATEGY_NAME].append(d)
            rows.append(_summary_row(user_id, target_mode, V5_STRATEGY_NAME, result_v5, z_target))
            for rec in result_v5.records:
                item = dict(rec)
                item.update({"user_id": user_id, "target_mode": target_mode, "strategy": V5_STRATEGY_NAME})
                step_rows.append(item)

        user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
        result_v51 = run_conservative_anchor_challenger_session(
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
        d = np.concatenate([[float(np.linalg.norm(z_target))], result_v51.distances])
        curve_store[target_mode][V51_STRATEGY_NAME].append(d)
        rows.append(_summary_row(user_id, target_mode, V51_STRATEGY_NAME, result_v51, z_target))
        for rec in result_v51.records:
            item = dict(rec)
            item.update({"user_id": user_id, "target_mode": target_mode, "strategy": V51_STRATEGY_NAME})
            step_rows.append(item)

    sessions = pd.DataFrame(rows)
    steps = pd.DataFrame(step_rows)
    curves = {
        target_mode: {strategy: np.asarray(items, dtype=np.float64) for strategy, items in by_strategy.items()}
        for target_mode, by_strategy in curve_store.items()
    }
    return sessions, steps, curves


def summarize_v51_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    return (
        sessions
        .groupby(["target_mode", "strategy"])
        .agg(
            users=("user_id", "count"),
            mean_initial_distance=("initial_distance", "mean"),
            mean_final_distance=("final_distance", "mean"),
            std_final_distance=("final_distance", "std"),
            mean_best_distance=("best_distance", "mean"),
            mean_mean_distance=("mean_distance", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            mean_direction_lock_step=("direction_lock_step", "mean"),
            mean_ready_step=("ready_step", "mean"),
            ready_rate=("ready_step", lambda s: float(pd.notna(s).mean())),
            mean_pair_distance=("mean_pair_distance", "mean"),
            mean_audibility_score=("mean_audibility_score", "mean"),
            mean_acceptability_score=("mean_acceptability_score", "mean"),
            mean_midrange_penalty=("mean_midrange_penalty", "mean"),
            mean_applied_lr=("mean_applied_lr", "mean"),
        )
        .reset_index()
        .sort_values(["target_mode", "mean_final_distance"])
    )


def source_usage_table_v51(steps: pd.DataFrame) -> pd.DataFrame:
    if steps.empty:
        return pd.DataFrame()
    return (
        steps.groupby(["target_mode", "strategy", "phase", "pair_source_group"])
        .size()
        .reset_index(name="count")
        .assign(share=lambda df: df["count"] / df.groupby(["target_mode", "strategy", "phase"])["count"].transform("sum"))
        .sort_values(["target_mode", "strategy", "phase", "share"], ascending=[True, True, True, False])
    )


def save_v51_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    source_usage: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "article_v51_conservative_anchor_challenger",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    steps.to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    win_rates.to_csv(output_dir / f"{prefix}_win_rates.csv", index=False)
    source_usage.to_csv(output_dir / f"{prefix}_source_usage.csv", index=False)
