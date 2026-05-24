from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .batch_eval import distances_with_initial, summarize_by_strategy, win_rates_vs_baseline
from .control_basis import CONTROL_BASIS_6D_TO_8D, CONTROL_NAMES_6D
from .loop import run_personalization_session_v0
from .metrics import distance_to_target
from .pair_generator import PairGenerator, normalize
from .preference_model import LogisticDistancePreferenceModel, cosine_similarity
from .preference_model_calibration import calibrate_by_heuristic_norm
from .preference_update import update_state_from_choice
from .state import FEATURE_NAMES_8D, PreferenceState, clip_vector, init_preference_state
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser


V5_STRATEGY_NAME = "safe_anchor_challenger_v5"

V5_DISPLAY_NAMES = {
    "semantic_active_v21": "Semantic active v3",
    "candidate_pool_active": "Candidate pool active",
    "phase_aware_controller": "Phase-aware controller",
    V5_STRATEGY_NAME: "Safe anchor-challenger v5",
}

# Local, zone-level refinement directions. These are intentionally more precise
# than broad semantic controls: they are meant for a later session phase, after
# the global direction has become reasonably stable.
ZONE_DIRECTIONS = {
    "sub_bass_depth": np.array([1.00, 0.25, 0.00, 0.00, -0.05, 0.00, 0.00, -0.05], dtype=np.float64),
    "bass_punch": np.array([0.20, 1.00, -0.12, -0.05, -0.05, 0.05, 0.00, 0.00], dtype=np.float64),
    "clean_bass_local": np.array([0.75, 0.95, -0.30, -0.25, -0.05, 0.10, 0.05, -0.05], dtype=np.float64),
    "warm_bass": np.array([0.35, 0.75, 0.35, 0.35, -0.10, -0.10, -0.05, -0.15], dtype=np.float64),
    "warmth_body_local": np.array([0.00, 0.10, 0.70, 0.85, -0.15, -0.10, 0.00, -0.15], dtype=np.float64),
    "vocal_presence": np.array([0.00, 0.00, -0.10, 0.00, 0.85, 0.55, 0.05, 0.00], dtype=np.float64),
    "clarity_without_harshness": np.array([-0.05, -0.05, -0.10, -0.10, 0.15, 0.85, 0.35, -0.15], dtype=np.float64),
    "air_detail": np.array([-0.10, -0.15, 0.00, -0.10, 0.05, 0.25, 0.90, 0.45], dtype=np.float64),
    "brightness_smoothing": np.array([0.05, 0.05, 0.00, 0.10, -0.20, -0.20, 0.10, -0.90], dtype=np.float64),
}

ZONE_GROUP_BY_DIRECTION = {
    "sub_bass_depth": "low_end",
    "bass_punch": "low_end",
    "clean_bass_local": "low_end",
    "warm_bass": "low_end_body",
    "warmth_body_local": "body_mid",
    "vocal_presence": "presence",
    "clarity_without_harshness": "detail_high",
    "air_detail": "detail_high",
    "brightness_smoothing": "detail_high",
}

MID_FEATURE_INDICES = [2, 3, 4]  # lowmid, warmth, presence


@dataclass
class V5Config:
    """Configuration for Safe Calibration + Anchor-Challenger Pair Generator."""

    calibration_steps: int = 3
    min_anchor_step: int = 14
    direction_lock_threshold: float = 0.88
    ready_threshold: float = 0.90
    base_step_scale: float = 0.6
    calibration_scale: float = 0.55
    anchor_scale: float = 0.40
    challenger_scales: tuple[float, ...] = (0.35, 0.50, 0.70, 0.90)
    max_anchor_challenger_distance: float = 1.45
    min_audible_distance: float = 0.35
    clip_value: float | None = 2.0
    midrange_penalty_weight: float = 0.55
    safety_penalty_weight: float = 1.20
    repetition_penalty_weight: float = 0.25
    model_uncertainty_weight: float = 0.35
    zone_uncertainty_weight: float = 0.45
    audibility_weight: float = 0.25
    challenger_alignment_weight: float = 0.20
    direction_locked_patience: int = 2


@dataclass
class V5SessionResult:
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[dict]
    distances: np.ndarray
    intensity_score: float
    direction_lock_step: int | None
    ready_step: int | None
    final_status: str


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def _unit(v: np.ndarray) -> np.ndarray:
    return normalize(np.asarray(v, dtype=np.float64))


def _model_blend_direction(state: PreferenceState, model: LogisticDistancePreferenceModel, clip_value: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return calibrated model vector, blend vector and agreement cosine."""
    heuristic = state.z_mean.copy()
    model_calibrated = calibrate_by_heuristic_norm(model.z_pref, heuristic, clip_value=clip_value)
    blend = clip_vector(0.70 * heuristic + 0.30 * model_calibrated, clip_value)
    agreement = cosine_similarity(heuristic, model_calibrated)
    return model_calibrated, blend, float(agreement)


def _intensity_multiplier(intensity_score: float) -> float:
    """Map online intensity estimate into scale multiplier."""
    if intensity_score >= 5:
        return 1.45
    if intensity_score >= 3:
        return 1.25
    if intensity_score <= -3:
        return 0.80
    if intensity_score <= -1:
        return 0.90
    return 1.00


def _randomize_pair_order(
    candidate_1: np.ndarray,
    candidate_2: np.ndarray,
    direction_hint: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return z_a, z_b and signed direction after random A/B order shuffle."""
    if rng.random() < 0.5:
        return candidate_1, candidate_2, _unit(direction_hint), "candidate_1_as_A"
    return candidate_2, candidate_1, -_unit(direction_hint), "candidate_2_as_A"


def _overflow_penalty(z: np.ndarray, clip_value: float | None) -> float:
    if clip_value is None:
        return 0.0
    overflow = np.maximum(np.abs(np.asarray(z, dtype=np.float64)) - float(clip_value), 0.0)
    return float(np.mean(overflow * overflow))


def _midrange_disturbance(candidate: np.ndarray, center: np.ndarray) -> float:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return float(np.mean(np.abs(delta[MID_FEATURE_INDICES])))


def _candidate_acceptability(
    candidate: np.ndarray,
    center: np.ndarray,
    source_group: str,
    clip_value: float | None,
) -> tuple[float, float, float]:
    """
    Return acceptability score and component penalties.

    Acceptability does not mean small or weak. It means compact-vector safety and
    musical plausibility. Midrange changes are penalized only when the question is
    not explicitly about body/mids/presence.
    """
    safety_penalty = _overflow_penalty(candidate, clip_value)
    mid_dist = _midrange_disturbance(candidate, center)
    if source_group in {"body_mid", "presence", "low_end_body"}:
        mid_penalty = 0.25 * mid_dist
    else:
        mid_penalty = mid_dist
    acceptability = float(np.exp(-1.6 * safety_penalty - 1.0 * mid_penalty))
    return acceptability, safety_penalty, mid_penalty


def _audibility_score(distance: float, min_distance: float, ideal_distance: float = 0.95) -> float:
    """High score when a pair is clearly audible but not absurdly large."""
    if distance < min_distance:
        return float(distance / (min_distance + 1e-8)) * 0.5
    # Smoothly prefer distances around ideal_distance, but keep strong contrasts allowed.
    return float(np.exp(-0.5 * ((distance - ideal_distance) / 0.70) ** 2))


def make_safe_calibration_pair(
    state: PreferenceState,
    step_index: int,
    rng: np.random.Generator,
    config: V5Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    First 2-3 curated questions. They are intentionally broad and audible, but
    still music-plausible. Their role is to initialize coarse direction and
    intensity before the system starts using adaptive anchor-challenger logic.
    """
    semantic_by_name = {name: CONTROL_BASIS_6D_TO_8D[i] for i, name in enumerate(CONTROL_NAMES_6D)}

    if step_index == 1:
        source_name = "calibration_clean_bass_vs_balanced"
        c1 = state.z_mean + config.calibration_scale * semantic_by_name["clean_bass"]
        c2 = state.z_mean
        direction = c1 - c2
        group = "low_end"
    elif step_index == 2:
        source_name = "calibration_air_detail_vs_soft_dark"
        c1 = state.z_mean + config.calibration_scale * semantic_by_name["air_brightness"]
        c2 = state.z_mean - 0.75 * config.calibration_scale * semantic_by_name["air_brightness"]
        direction = c1 - c2
        group = "detail_high"
    else:
        source_name = "calibration_warm_body_vs_clean_neutral"
        c1 = state.z_mean + config.calibration_scale * semantic_by_name["warmth_body"]
        c2 = state.z_mean + 0.35 * config.calibration_scale * semantic_by_name["clean_bass"]
        direction = c1 - c2
        group = "body_mid"

    c1 = clip_vector(c1, config.clip_value)
    c2 = clip_vector(c2, config.clip_value)
    z_a, z_b, signed_direction, order = _randomize_pair_order(c1, c2, direction, rng)
    distance = float(np.linalg.norm(z_a - z_b))

    meta = {
        "strategy": V5_STRATEGY_NAME,
        "sub_strategy": "safe_calibration",
        "source": source_name,
        "source_group": group,
        "control_name": source_name,
        "control_direction": signed_direction.copy(),
        "anchor_challenger_role": "calibration",
        "pair_distance": distance,
        "audibility_score": _audibility_score(distance, config.min_audible_distance),
        "acceptability_score": 1.0,
        "order": order,
    }
    return z_a, z_b, signed_direction, meta


def make_anchor_challenger_pair(
    state: PreferenceState,
    model: LogisticDistancePreferenceModel,
    rng: np.random.Generator,
    config: V5Config,
    intensity_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build an anchor-challenger A/B pair.

    Anchor: a conservative step toward the current consensus direction.
    Challenger: a safe but informative alternative from semantic/zone directions.
    """
    model_calibrated, blend, agreement = _model_blend_direction(state, model, config.clip_value)

    # Main consensus direction. If blend is too close to current state, fall back to z_mean/model direction.
    direct_delta = blend - state.z_mean
    if np.linalg.norm(direct_delta) < 1e-6:
        direct_delta = blend if np.linalg.norm(blend) > 1e-6 else model_calibrated
    if np.linalg.norm(direct_delta) < 1e-6:
        direct_delta = np.ones_like(state.z_mean)
    direct_direction = _unit(direct_delta)

    intensity_mult = _intensity_multiplier(intensity_score)
    anchor_scale = float(config.anchor_scale) * intensity_mult
    anchor = clip_vector(state.z_mean + anchor_scale * direct_direction, config.clip_value)

    semantic_dirs = {
        f"semantic_{name}": (CONTROL_BASIS_6D_TO_8D[i], "semantic")
        for i, name in enumerate(CONTROL_NAMES_6D)
    }
    zone_dirs = {
        f"zone_{name}": (direction, ZONE_GROUP_BY_DIRECTION.get(name, "zone"))
        for name, direction in ZONE_DIRECTIONS.items()
    }
    model_dirs = {
        "model_direction_forward": (direct_direction, "model_direction"),
        "model_direction_countercheck": (-direct_direction, "model_direction"),
    }

    candidate_directions: dict[str, tuple[np.ndarray, str]] = {}
    candidate_directions.update(semantic_dirs)
    candidate_directions.update(zone_dirs)
    candidate_directions.update(model_dirs)

    recent_sources = []
    for item in state.history[-3:]:
        meta = item.get("pair_meta") or {}
        recent_sources.append(meta.get("source") or meta.get("control_name"))

    candidates: list[dict] = []
    z_std = np.asarray(state.z_std, dtype=np.float64)
    for source, (direction_raw, group) in candidate_directions.items():
        base_direction = _unit(direction_raw)
        for sign in (+1.0, -1.0):
            direction = sign * base_direction
            alignment = float(np.dot(direction, direct_direction))
            # Avoid challengers that are just identical to the anchor direction.
            if source == "model_direction_forward":
                alignment_bonus = 0.15
            else:
                alignment_bonus = 1.0 - abs(alignment)

            uncertainty_score = float(np.sum(np.abs(direction) * z_std) / (np.sum(np.abs(direction)) + 1e-8))
            for scale_mult in config.challenger_scales:
                scale = float(config.base_step_scale) * float(scale_mult) * intensity_mult
                challenger = clip_vector(state.z_mean + scale * direction, config.clip_value)
                pair_distance = float(np.linalg.norm(anchor - challenger))

                acceptability, safety_penalty, mid_penalty = _candidate_acceptability(
                    challenger, state.z_mean, group, config.clip_value
                )
                anchor_acceptability, anchor_safety, anchor_mid = _candidate_acceptability(
                    anchor, state.z_mean, "model_direction", config.clip_value
                )
                audibility = _audibility_score(pair_distance, config.min_audible_distance)
                if pair_distance > config.max_anchor_challenger_distance:
                    distance_penalty = (pair_distance - config.max_anchor_challenger_distance) ** 2
                else:
                    distance_penalty = 0.0

                p_anchor = model.predict_proba_a(anchor, challenger)
                model_uncertainty = float(1.0 - 2.0 * abs(p_anchor - 0.5))

                repeat_penalty = 1.0 if source in recent_sources else 0.0

                # A strong challenger is allowed if it remains acceptable and audible.
                score = (
                    config.model_uncertainty_weight * model_uncertainty
                    + config.zone_uncertainty_weight * uncertainty_score
                    + config.audibility_weight * audibility
                    + config.challenger_alignment_weight * alignment_bonus
                    + 0.25 * acceptability
                    + 0.15 * anchor_acceptability
                    - config.safety_penalty_weight * (safety_penalty + anchor_safety)
                    - config.midrange_penalty_weight * (mid_penalty + 0.35 * anchor_mid)
                    - config.repetition_penalty_weight * repeat_penalty
                    - 0.65 * distance_penalty
                )

                candidates.append({
                    "source": source,
                    "source_group": group,
                    "direction": direction.copy(),
                    "challenger": challenger.copy(),
                    "scale": scale,
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
                    "midrange_disturbance_penalty": float(mid_penalty + 0.35 * anchor_mid),
                    "distance_penalty": float(distance_penalty),
                    "repetition_penalty": float(repeat_penalty),
                })

    best = max(candidates, key=lambda item: item["score"])
    challenger = best["challenger"]
    direction_hint = anchor - challenger
    z_a, z_b, signed_direction, order = _randomize_pair_order(anchor, challenger, direction_hint, rng)

    meta = {
        "strategy": V5_STRATEGY_NAME,
        "sub_strategy": "anchor_challenger",
        "source": best["source"],
        "source_group": best["source_group"],
        "control_name": best["source"],
        "control_direction": signed_direction.copy(),
        "anchor_challenger_role": "anchor_vs_challenger",
        "anchor": anchor.copy(),
        "challenger": challenger.copy(),
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


def _update_intensity_score(
    intensity_score: float,
    state_before: PreferenceState,
    z_a: np.ndarray,
    z_b: np.ndarray,
    choice: str,
) -> float:
    """Update online intensity estimate from whether the user chose the stronger candidate."""
    preferred = z_a if choice == "A" else z_b
    rejected = z_b if choice == "A" else z_a
    pref_strength = np.linalg.norm(preferred - state_before.z_mean)
    rej_strength = np.linalg.norm(rejected - state_before.z_mean)
    if pref_strength > rej_strength + 0.05:
        intensity_score += 1.0
    elif pref_strength < rej_strength - 0.05:
        intensity_score -= 1.0
    return float(np.clip(intensity_score, -8.0, 8.0))


def _direction_lock_status(
    state: PreferenceState,
    model: LogisticDistancePreferenceModel,
    step: int,
    config: V5Config,
    lock_hits: int,
) -> tuple[bool, float, int]:
    _, _, agreement = _model_blend_direction(state, model, config.clip_value)
    if step >= int(config.min_anchor_step) and agreement >= float(config.direction_lock_threshold):
        lock_hits += 1
    else:
        lock_hits = 0
    return lock_hits >= int(config.direction_locked_patience), float(agreement), int(lock_hits)


def run_safe_anchor_challenger_session(
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
    config: V5Config | None = None,
) -> V5SessionResult:
    """Run Safe Calibration + Anchor-Challenger personalization session."""
    cfg = V5Config(base_step_scale=step_scale, clip_value=clip_value) if config is None else config
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

    for step in range(1, int(n_steps) + 1):
        state_before = state.copy()
        locked, agreement, lock_hits = _direction_lock_status(state, model, step, cfg, lock_hits)
        if locked and direction_lock_step is None:
            direction_lock_step = int(step)

        if step <= cfg.calibration_steps:
            phase = "safe_calibration"
            z_a, z_b, direction, pair_meta = make_safe_calibration_pair(state, step, rng, cfg)
        elif locked:
            phase = "anchor_challenger"
            z_a, z_b, direction, pair_meta = make_anchor_challenger_pair(
                state=state,
                model=model,
                rng=rng,
                config=cfg,
                intensity_score=intensity_score,
            )
        else:
            phase = "semantic_active_warmup"
            z_a, z_b, direction, pair_meta = pair_generator.semantic_active_v21(state)
            pair_meta = dict(pair_meta)
            pair_meta["sub_strategy"] = "semantic_active_warmup"
            pair_meta["source"] = "semantic_active_v21"
            pair_meta["source_group"] = "semantic"
            pair_meta["agreement"] = float(agreement)
            pair_meta["intensity_score"] = float(intensity_score)

        p_before = model.predict_proba_a(z_a, z_b)
        pred_before = "A" if p_before >= 0.5 else "B"
        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)
        model_record = model.update(z_a, z_b, choice)
        intensity_score = _update_intensity_score(intensity_score, state_before, z_a, z_b, choice)

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
            if agreement_after >= cfg.ready_threshold and mean_std <= 0.35 and update_norm <= 0.30:
                ready_step = int(step)

        records.append({
            "step": int(step),
            "phase": phase,
            "choice": choice,
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
            "model_uncertainty": float(pair_meta.get("model_uncertainty", np.nan)),
            "midrange_disturbance_penalty": float(pair_meta.get("midrange_disturbance_penalty", np.nan)),
            "safety_penalty": float(pair_meta.get("safety_penalty", np.nan)),
        })

    final_status = "ready_to_finalize" if ready_step is not None else "completed"
    return V5SessionResult(
        final_state=state,
        final_model=model,
        records=records,
        distances=np.asarray(distances, dtype=np.float64),
        intensity_score=float(intensity_score),
        direction_lock_step=direction_lock_step,
        ready_step=ready_step,
        final_status=final_status,
    )


def _summary_row_from_v5_result(user_id: int, target_mode: str, result: V5SessionResult, z_target: np.ndarray) -> dict:
    d = np.concatenate([[float(np.linalg.norm(z_target))], result.distances])
    records_df = pd.DataFrame(result.records)
    return {
        "user_id": int(user_id),
        "target_mode": target_mode,
        "strategy": V5_STRATEGY_NAME,
        "n_steps": int(len(result.distances)),
        "initial_distance": float(d[0]),
        "final_distance": float(d[-1]),
        "best_distance": float(np.min(d)),
        "mean_distance": float(np.mean(d)),
        "improvement_abs": float(d[0] - d[-1]),
        "improvement_pct": float(100.0 * (d[0] - d[-1]) / (d[0] + 1e-8)),
        "direction_lock_step": result.direction_lock_step,
        "ready_step": result.ready_step,
        "final_status": result.final_status,
        "final_intensity_score": float(result.intensity_score),
        "mean_pair_distance": float(records_df["pair_distance"].mean()) if not records_df.empty else np.nan,
        "mean_audibility_score": float(records_df["audibility_score"].mean()) if "audibility_score" in records_df else np.nan,
        "mean_acceptability_score": float(records_df["acceptability_score"].mean()) if "acceptability_score" in records_df else np.nan,
        "mean_midrange_penalty": float(records_df["midrange_disturbance_penalty"].mean()) if "midrange_disturbance_penalty" in records_df else np.nan,
    }


def run_v5_comparison_on_dataset(
    dataset: pd.DataFrame,
    baseline_strategies: Iterable[str] = ("semantic_active_v21", "candidate_pool_active"),
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
    """Run V5 and selected baselines on a fixed dataset."""
    rows: list[dict] = []
    step_rows: list[dict] = []
    curve_store: dict[str, dict[str, list[np.ndarray]]] = {}
    strategies = list(baseline_strategies) + [V5_STRATEGY_NAME]

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
            })

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
        rows.append(_summary_row_from_v5_result(user_id, target_mode, result_v5, z_target))
        for rec in result_v5.records:
            item = dict(rec)
            item.update({"user_id": user_id, "target_mode": target_mode, "strategy": V5_STRATEGY_NAME})
            step_rows.append(item)

    sessions = pd.DataFrame(rows)
    steps = pd.DataFrame(step_rows)
    curves = {
        target_mode: {
            strategy: np.asarray(items, dtype=np.float64)
            for strategy, items in by_strategy.items()
        }
        for target_mode, by_strategy in curve_store.items()
    }
    return sessions, steps, curves


def summarize_v5_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate V5 comparison sessions by target_mode and strategy."""
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
        )
        .reset_index()
        .sort_values(["target_mode", "mean_final_distance"])
    )


def source_usage_table(steps: pd.DataFrame) -> pd.DataFrame:
    if steps.empty:
        return pd.DataFrame()
    return (
        steps.groupby(["target_mode", "phase", "pair_source_group"])
        .size()
        .reset_index(name="count")
        .assign(share=lambda df: df["count"] / df.groupby(["target_mode", "phase"])["count"].transform("sum"))
        .sort_values(["target_mode", "phase", "share"], ascending=[True, True, False])
    )


def save_v5_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    source_usage: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "article_v5_safe_anchor_challenger",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    steps.to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    win_rates.to_csv(output_dir / f"{prefix}_win_rates.csv", index=False)
    source_usage.to_csv(output_dir / f"{prefix}_source_usage.csv", index=False)
