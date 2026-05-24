from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .batch_eval import win_rates_vs_baseline
from .metrics import distance_to_target
from .pair_generator import PairGenerator
from .preference_model import LogisticDistancePreferenceModel
from .preference_update import update_state_from_choice
from .safe_anchor_challenger import (
    _audibility_score,
    _candidate_acceptability,
    _direction_lock_status,
    _intensity_multiplier,
    _model_blend_direction,
    _randomize_pair_order,
    _unit,
    _update_intensity_score,
)
from .soft_stop_monitor_v54 import (
    V54_STRATEGY_NAME,
    V54Config,
    _is_ready_candidate,
    _source_group_from_meta,
    _summary_row,
)
from .phase_aware_full_budget_v55 import (
    V55_DISPLAY_NAMES,
    V55_MIXED_STRATEGY_NAME,
    V55_ZONE_STRATEGY_NAME,
    phase_step_budget_table,
    post_marker_source_usage_table_v55,
    run_v55_comparison_on_dataset,
    save_v55_outputs,
    source_usage_table_v55,
    summarize_v55_sessions,
)
from .state import PreferenceState, clip_vector, init_preference_state
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser


V56_BLEND_STRATEGY_NAME = "direct_blend_refinement_v56"
V56_PM_STRATEGY_NAME = "direct_pm_refinement_v56"
V56_TRUST_STRATEGY_NAME = "trust_region_direct_refinement_v56"

V56_DISPLAY_NAMES = dict(V55_DISPLAY_NAMES)
V56_DISPLAY_NAMES.update({
    V56_BLEND_STRATEGY_NAME: "Direct to blend v5.6",
    V56_PM_STRATEGY_NAME: "Direct to PM v5.6",
    V56_TRUST_STRATEGY_NAME: "Trust-region direct v5.6",
})

DirectMode = Literal["blend", "pm", "trust"]


def _cosine_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


@dataclass
class V56Config(V54Config):
    """V5.6: direct taste-vector refinement after the soft-stop marker.

    The session still uses the full 25-step budget. Before the soft-stop marker
    the generator is Semantic active v3. After the marker it can ask direct
    refinement questions of the form:

        anchor = current best profile
        challenger = small controlled step toward estimated taste vector

    Estimated taste vector variants:
    - blend: calibrated blend of heuristic state and online Preference Model;
    - pm: calibrated Preference Model vector;
    - trust: blend vector with zone/trust masks to avoid unnecessary midrange
      disturbance and overshoot.
    """

    mode: DirectMode = "trust"
    min_direct_step: float = 0.24
    max_direct_step: float = 0.62
    direct_step_fraction: float = 0.45
    direct_selected_lr: float = 0.85
    anchor_selected_lr: float = 0.0
    ready_direct_selected_lr: float = 0.55
    min_delta_norm_for_direct: float = 0.06
    min_audible_distance: float = 0.24
    ideal_direct_distance: float = 0.55
    trust_mid_weight: float = 0.42
    trust_presence_weight: float = 0.62
    trust_high_weight: float = 0.88
    direct_fallback_to_semantic: bool = False


@dataclass
class V56SessionResult:
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[dict]
    distances: np.ndarray
    intensity_score: float
    direction_lock_step: int | None
    ready_step: int | None
    recommended_stop_step: int | None
    final_status: str


def _trust_mask(dim: int, cfg: V56Config) -> np.ndarray:
    # 8D: sub_bass, bass, lowmid, warmth, presence, clarity, air, brightness.
    if dim != 8:
        return np.ones(dim, dtype=np.float64)
    return np.asarray([
        1.00,
        1.00,
        cfg.trust_mid_weight,
        cfg.trust_mid_weight,
        cfg.trust_presence_weight,
        cfg.trust_high_weight,
        cfg.trust_high_weight,
        cfg.trust_high_weight,
    ], dtype=np.float64)


def _target_estimate_from_mode(
    state: PreferenceState,
    model: LogisticDistancePreferenceModel,
    mode: DirectMode,
    clip_value: float | None,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    model_calibrated, blend, agreement = _model_blend_direction(state, model, clip_value)
    if mode == "pm":
        return model_calibrated, model_calibrated, float(agreement), "pm_calibrated"
    if mode in {"blend", "trust"}:
        return blend, model_calibrated, float(agreement), "blend_calibrated"
    raise ValueError(f"Unknown V5.6 direct mode: {mode}")


def make_direct_taste_vector_pair(
    state: PreferenceState,
    model: LogisticDistancePreferenceModel,
    rng: np.random.Generator,
    config: V56Config,
    intensity_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build current-profile vs direct-step pair.

    The direct candidate is a bounded, audible step toward the current taste
    vector estimate. If the estimate is too close to the current state, the
    caller may fall back to semantic exploration.
    """
    model_calibrated, blend_estimate, agreement = _model_blend_direction(state, model, config.clip_value)
    if config.mode == "pm":
        target_est = model_calibrated
        target_source = "pm_calibrated"
    elif config.mode in {"blend", "trust"}:
        target_est = blend_estimate
        target_source = "blend_calibrated"
    else:
        raise ValueError(f"Unknown V5.6 direct mode: {config.mode}")

    z_current = np.asarray(state.z_mean, dtype=np.float64)
    delta = np.asarray(target_est, dtype=np.float64) - z_current
    delta_pm = np.asarray(model_calibrated, dtype=np.float64) - z_current
    delta_blend = np.asarray(blend_estimate, dtype=np.float64) - z_current
    raw_delta_norm = float(np.linalg.norm(delta))
    delta_pm_norm = float(np.linalg.norm(delta_pm))
    delta_blend_norm = float(np.linalg.norm(delta_blend))
    target_model_cosine = _cosine_safe(np.asarray(target_est, dtype=np.float64), np.asarray(model_calibrated, dtype=np.float64))
    target_blend_cosine = _cosine_safe(np.asarray(target_est, dtype=np.float64), np.asarray(blend_estimate, dtype=np.float64))
    model_blend_cosine = _cosine_safe(np.asarray(model_calibrated, dtype=np.float64), np.asarray(blend_estimate, dtype=np.float64))
    delta_pm_blend_cosine = _cosine_safe(delta_pm, delta_blend)

    if config.mode == "trust":
        mask = _trust_mask(len(delta), config)
        # Allow dimensions that are still uncertain to move a bit more.
        std = np.asarray(state.z_std, dtype=np.float64)
        std_boost = 0.75 + 0.25 * (std / (float(np.max(std)) + 1e-8))
        direction_raw = delta * mask * std_boost
        source_group = "direct_trust_region"
        trust_mask_mean = float(np.mean(mask))
        trust_mask_min = float(np.min(mask))
        trust_mask_max = float(np.max(mask))
    elif config.mode == "pm":
        mask = np.ones(len(delta), dtype=np.float64)
        direction_raw = delta
        source_group = "direct_pm"
        trust_mask_mean = 1.0
        trust_mask_min = 1.0
        trust_mask_max = 1.0
    else:
        mask = np.ones(len(delta), dtype=np.float64)
        direction_raw = delta
        source_group = "direct_blend"
        trust_mask_mean = 1.0
        trust_mask_min = 1.0
        trust_mask_max = 1.0

    direction_norm = float(np.linalg.norm(direction_raw))
    direction_cosine_to_raw_delta = _cosine_safe(direction_raw, delta)
    direction_cosine_to_pm_delta = _cosine_safe(direction_raw, delta_pm)
    direction_cosine_to_blend_delta = _cosine_safe(direction_raw, delta_blend)
    direction_effective_dims = float((np.sum(np.abs(direction_raw)) ** 2) / (np.sum(direction_raw ** 2) + 1e-12))
    if direction_norm < config.min_delta_norm_for_direct:
        # Keep metadata explicit; the session runner will decide whether to use it.
        direction_raw = delta if raw_delta_norm > 1e-8 else np.ones_like(state.z_mean)
        direction_norm = float(np.linalg.norm(direction_raw))

    direction = _unit(direction_raw)
    intensity_mult = _intensity_multiplier(float(intensity_score))
    step_size = float(config.direct_step_fraction) * max(raw_delta_norm, config.min_direct_step)
    step_size *= float(intensity_mult)
    step_size = float(np.clip(step_size, config.min_direct_step, config.max_direct_step))

    anchor = z_current.copy()
    direct_candidate = clip_vector(anchor + step_size * direction, config.clip_value)
    pair_distance = float(np.linalg.norm(direct_candidate - anchor))
    if pair_distance < config.min_audible_distance:
        direct_candidate = clip_vector(anchor + config.min_audible_distance * direction, config.clip_value)
        pair_distance = float(np.linalg.norm(direct_candidate - anchor))

    acceptability, safety_penalty, mid_penalty = _candidate_acceptability(
        direct_candidate,
        anchor,
        source_group,
        config.clip_value,
    )
    audibility = _audibility_score(pair_distance, config.min_audible_distance, ideal_distance=config.ideal_direct_distance)

    z_a, z_b, signed_direction, order = _randomize_pair_order(direct_candidate, anchor, direction, rng)
    direct_label = "A" if order == "candidate_1_as_A" else "B"

    meta = {
        "strategy": V56_TRUST_STRATEGY_NAME if config.mode == "trust" else (V56_PM_STRATEGY_NAME if config.mode == "pm" else V56_BLEND_STRATEGY_NAME),
        "source": f"direct_taste_vector_{config.mode}",
        "source_group": source_group,
        "phase_source": "direct_taste_vector",
        "sub_strategy": f"direct_{config.mode}_post_marker",
        "target_source": target_source,
        "direct_label": direct_label,
        "direct_candidate": direct_candidate.copy(),
        "anchor_candidate": anchor.copy(),
        "model_calibrated": model_calibrated.copy(),
        "target_estimate": target_est.copy(),
        "control_direction": signed_direction.copy(),
        "agreement": float(agreement),
        "target_model_cosine": float(target_model_cosine),
        "target_blend_cosine": float(target_blend_cosine),
        "model_blend_cosine": float(model_blend_cosine),
        "delta_pm_blend_cosine": float(delta_pm_blend_cosine),
        "direction_cosine_to_raw_delta": float(direction_cosine_to_raw_delta),
        "direction_cosine_to_pm_delta": float(direction_cosine_to_pm_delta),
        "direction_cosine_to_blend_delta": float(direction_cosine_to_blend_delta),
        "delta_pm_norm": delta_pm_norm,
        "delta_blend_norm": delta_blend_norm,
        "trust_mask_mean": trust_mask_mean,
        "trust_mask_min": trust_mask_min,
        "trust_mask_max": trust_mask_max,
        "direction_effective_dims": direction_effective_dims,
        "raw_delta_norm": raw_delta_norm,
        "direction_norm": direction_norm,
        "scale": step_size,
        "pair_distance": pair_distance,
        "audibility_score": float(audibility),
        "acceptability_score": float(acceptability),
        "midrange_disturbance_penalty": float(mid_penalty),
        "safety_penalty": float(safety_penalty),
        "fallback_recommended": bool(raw_delta_norm < config.min_delta_norm_for_direct),
    }
    return z_a, z_b, signed_direction, meta


def _update_state_direct_refinement(
    state: PreferenceState,
    z_a: np.ndarray,
    z_b: np.ndarray,
    choice: str,
    pair_meta: dict,
    config: V56Config,
    std_decay: float,
    min_std: float,
    ready_active: bool = False,
) -> tuple[PreferenceState, float, str]:
    """Update state for anchor-vs-direct refinement.

    If the direct candidate wins, move toward that point. If the anchor wins,
    keep the mean almost unchanged and mainly reduce uncertainty along the tested
    direction. This avoids the common failure mode where rejecting a challenger
    pushes the profile in the opposite direction.
    """
    if choice not in {"A", "B"}:
        raise ValueError("choice must be 'A' or 'B'")
    direct_label = str(pair_meta.get("direct_label", ""))
    direct_selected = choice == direct_label
    direct_candidate = np.asarray(pair_meta.get("direct_candidate"), dtype=np.float64)
    anchor_candidate = np.asarray(pair_meta.get("anchor_candidate"), dtype=np.float64)

    if direct_selected:
        applied_lr = float(config.ready_direct_selected_lr if ready_active else config.direct_selected_lr)
        selected_role = "direct_candidate_selected"
        state.z_mean = clip_vector(
            state.z_mean + applied_lr * (direct_candidate - state.z_mean),
            config.clip_value,
        )
    else:
        applied_lr = float(config.anchor_selected_lr)
        selected_role = "anchor_selected"
        if applied_lr > 0:
            state.z_mean = clip_vector(
                state.z_mean + applied_lr * (anchor_candidate - state.z_mean),
                config.clip_value,
            )

    direction = np.abs(np.asarray(pair_meta.get("control_direction", np.zeros_like(state.z_mean)), dtype=np.float64))
    if direction.max() > 0:
        information = direction / (direction.max() + 1e-8)
        state.z_std = state.z_std * 0.995
        state.z_std = state.z_std * (1.0 - 0.22 * information)
        state.z_std = np.maximum(state.z_std, float(min_std))
    else:
        state.z_std = np.maximum(state.z_std * float(std_decay), float(min_std))

    state.step += 1
    state.history.append({
        "type": "ab_choice",
        "step": state.step,
        "choice": choice,
        "z_a": np.asarray(z_a, dtype=np.float64).copy(),
        "z_b": np.asarray(z_b, dtype=np.float64).copy(),
        "pair_meta": pair_meta,
        "selected_role": selected_role,
        "applied_lr": applied_lr,
        "z_mean_after": state.z_mean.copy(),
        "z_std_after": state.z_std.copy(),
    })
    return state, applied_lr, selected_role


def run_direct_taste_vector_session(
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
    mode: DirectMode = "trust",
    config: V56Config | None = None,
) -> V56SessionResult:
    cfg = V56Config(base_step_scale=step_scale, clip_value=clip_value, mode=mode) if config is None else config
    cfg.mode = mode
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

    strategy_name = V56_TRUST_STRATEGY_NAME if mode == "trust" else (V56_PM_STRATEGY_NAME if mode == "pm" else V56_BLEND_STRATEGY_NAME)
    records: list[dict] = []
    distances: list[float] = []
    intensity_score = 0.0
    direction_lock_step: int | None = None
    ready_step: int | None = None
    recommended_stop_step: int | None = None
    lock_hits = 0
    ready_hits = 0

    for step in range(1, int(n_steps) + 1):
        state_before = state.copy()
        locked, agreement_before, lock_hits = _direction_lock_status(state, model, step, cfg, lock_hits)
        if locked and direction_lock_step is None:
            direction_lock_step = int(step)

        after_marker = recommended_stop_step is not None and step > recommended_stop_step
        source_choice = "direct_taste_vector" if after_marker else "semantic_active"

        if source_choice == "direct_taste_vector":
            z_a, z_b, direction, pair_meta = make_direct_taste_vector_pair(
                state=state,
                model=model,
                rng=rng,
                config=cfg,
                intensity_score=intensity_score,
            )
            # If direct vector has essentially no remaining signal, use semantic as a safe fallback.
            if pair_meta.get("fallback_recommended") and cfg.direct_fallback_to_semantic:
                z_a, z_b, direction, pair_meta = pair_generator.semantic_active_v21(state)
                pair_meta = dict(pair_meta)
                pair_meta["strategy"] = strategy_name
                pair_meta["sub_strategy"] = "semantic_fallback_after_marker"
                pair_meta["source"] = "semantic_active_v21"
                pair_meta["source_group"] = "semantic"
                pair_meta["phase_source"] = "semantic_fallback"
            else:
                pair_meta = dict(pair_meta)
                pair_meta["strategy"] = strategy_name
        else:
            z_a, z_b, direction, pair_meta = pair_generator.semantic_active_v21(state)
            pair_meta = dict(pair_meta)
            pair_meta["strategy"] = strategy_name
            pair_meta["sub_strategy"] = "semantic_active_backbone"
            pair_meta["source"] = "semantic_active_v21"
            pair_meta["source_group"] = "semantic"
            pair_meta["phase_source"] = "semantic_active"

        pair_meta["agreement_before"] = float(agreement_before)
        pair_meta["intensity_score"] = float(intensity_score)
        pair_meta["recommended_stop_step"] = recommended_stop_step

        p_before = model.predict_proba_a(z_a, z_b)
        pred_before = "A" if p_before >= 0.5 else "B"
        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)
        model_record = model.update(z_a, z_b, choice)
        intensity_score = _update_intensity_score(intensity_score, state_before, z_a, z_b, choice)

        ready_active = recommended_stop_step is not None
        if str(pair_meta.get("phase_source")) == "direct_taste_vector":
            state, applied_lr, selected_role = _update_state_direct_refinement(
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
            applied_lr = float(heuristic_lr)
            selected_role = "semantic_active_selected"
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

        _, z_blend, agreement_after = _model_blend_direction(state, model, clip_value)
        update_norm = float(np.linalg.norm(state.z_mean - state_before.z_mean))
        mean_std = float(np.mean(state.z_std))
        ready_candidate = _is_ready_candidate(
            step=step,
            cfg=cfg,
            agreement_after=agreement_after,
            mean_std=mean_std,
            update_norm=update_norm,
        )
        if ready_candidate:
            ready_hits += 1
        else:
            ready_hits = 0
        if ready_step is None and ready_candidate:
            ready_step = int(step)
        if recommended_stop_step is None and ready_hits >= int(cfg.ready_patience):
            recommended_stop_step = int(step)

        if recommended_stop_step is None:
            phase = "before_soft_stop_marker"
            soft_stop_marker = "not_ready"
            post_marker_index = 0
        elif step == recommended_stop_step:
            phase = "soft_stop_marker"
            soft_stop_marker = "recommended_stop"
            post_marker_index = 0
        else:
            phase = f"after_soft_stop_{pair_meta.get('phase_source', source_choice)}"
            soft_stop_marker = "continued_after_recommendation"
            post_marker_index = int(step - recommended_stop_step)

        pair_distance = float(np.linalg.norm(z_a - z_b))
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
            "agreement_before": float(agreement_before),
            "agreement_after": float(agreement_after),
            "intensity_score": float(intensity_score),
            "mean_z_std": mean_std,
            "update_norm": update_norm,
            "z_mean_norm": float(np.linalg.norm(state.z_mean)),
            "z_model_norm": float(np.linalg.norm(model.z_pref)),
            "z_blend_norm": float(np.linalg.norm(z_blend)),
            "pair_source": pair_meta.get("source"),
            "pair_source_group": _source_group_from_meta(pair_meta),
            "phase_source": pair_meta.get("phase_source", source_choice),
            "sub_strategy": pair_meta.get("sub_strategy"),
            "control_name": pair_meta.get("control_name"),
            "target_source": pair_meta.get("target_source"),
            "direct_label": pair_meta.get("direct_label"),
            "raw_delta_norm": float(pair_meta.get("raw_delta_norm", np.nan)),
            "delta_pm_norm": float(pair_meta.get("delta_pm_norm", np.nan)),
            "delta_blend_norm": float(pair_meta.get("delta_blend_norm", np.nan)),
            "target_model_cosine": float(pair_meta.get("target_model_cosine", np.nan)),
            "target_blend_cosine": float(pair_meta.get("target_blend_cosine", np.nan)),
            "model_blend_cosine": float(pair_meta.get("model_blend_cosine", np.nan)),
            "delta_pm_blend_cosine": float(pair_meta.get("delta_pm_blend_cosine", np.nan)),
            "direction_cosine_to_raw_delta": float(pair_meta.get("direction_cosine_to_raw_delta", np.nan)),
            "direction_cosine_to_pm_delta": float(pair_meta.get("direction_cosine_to_pm_delta", np.nan)),
            "direction_cosine_to_blend_delta": float(pair_meta.get("direction_cosine_to_blend_delta", np.nan)),
            "trust_mask_mean": float(pair_meta.get("trust_mask_mean", np.nan)),
            "trust_mask_min": float(pair_meta.get("trust_mask_min", np.nan)),
            "trust_mask_max": float(pair_meta.get("trust_mask_max", np.nan)),
            "direction_effective_dims": float(pair_meta.get("direction_effective_dims", np.nan)),
            "scale": float(pair_meta.get("scale", np.nan)),
            "scale_multiplier": float(pair_meta.get("scale_multiplier", np.nan)),
            "pair_distance": pair_distance,
            "audibility_score": float(pair_meta.get("audibility_score", min(pair_distance / (step_scale + 1e-8), 1.0))),
            "acceptability_score": float(pair_meta.get("acceptability_score", 1.0)),
            "midrange_disturbance_penalty": float(pair_meta.get("midrange_disturbance_penalty", 0.0)),
            "safety_penalty": float(pair_meta.get("safety_penalty", np.nan)),
            "direction_lock_step": direction_lock_step,
            "ready_step": ready_step,
            "recommended_stop_step": recommended_stop_step,
            "ready_candidate": bool(ready_candidate),
            "ready_hits": int(ready_hits),
            "soft_stop_marker": soft_stop_marker,
            "post_marker_index": int(post_marker_index),
        })

    final_status = "soft_stop_recommended_and_continued" if recommended_stop_step is not None else "no_soft_stop_recommendation"
    return V56SessionResult(
        final_state=state,
        final_model=model,
        records=records,
        distances=np.asarray(distances, dtype=np.float64),
        intensity_score=float(intensity_score),
        direction_lock_step=direction_lock_step,
        ready_step=ready_step,
        recommended_stop_step=recommended_stop_step,
        final_status=final_status,
    )


def _summary_row_v56(user_id: int, target_mode: str, strategy: str, result: V56SessionResult, z_target: np.ndarray) -> dict:
    row = _summary_row(user_id, target_mode, strategy, result, z_target)
    stop_step = row.get("recommended_stop_step", np.nan)
    n_steps = int(row.get("n_steps", len(result.distances)))
    if pd.notna(stop_step):
        row["steps_before_recommendation"] = float(stop_step)
        row["steps_after_recommendation"] = float(max(n_steps - int(stop_step), 0))
    else:
        row["steps_before_recommendation"] = float(n_steps)
        row["steps_after_recommendation"] = 0.0
    return row


def run_v56_comparison_on_dataset(
    dataset: pd.DataFrame,
    include_v55: bool = True,
    include_blend: bool = True,
    include_pm: bool = True,
    include_trust: bool = True,
    baseline_strategies: Iterable[str] = ("semantic_active_v21", "candidate_pool_active"),
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_seed_base: int = 60_000,
    user_seed_base: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    if include_v55:
        sessions_base, steps_base, curves_base = run_v55_comparison_on_dataset(
            dataset=dataset,
            baseline_strategies=baseline_strategies,
            include_previous_v5=False,
            include_v54=True,
            include_zone=True,
            include_mixed=True,
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
    else:
        sessions_base = pd.DataFrame()
        steps_base = pd.DataFrame()
        curves_base: dict[str, dict[str, list[np.ndarray]]] = {}

    curve_store: dict[str, dict[str, list[np.ndarray]]] = {}
    for mode_name, by_strategy in curves_base.items():
        curve_store[mode_name] = {strategy: [arr for arr in values] for strategy, values in by_strategy.items()}

    modes_to_run: list[DirectMode] = []
    if include_blend:
        modes_to_run.append("blend")
    if include_pm:
        modes_to_run.append("pm")
    if include_trust:
        modes_to_run.append("trust")

    rows: list[dict] = []
    step_rows: list[dict] = []
    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        curve_store.setdefault(target_mode, {})
        for direct_mode in modes_to_run:
            strategy_name = V56_TRUST_STRATEGY_NAME if direct_mode == "trust" else (V56_PM_STRATEGY_NAME if direct_mode == "pm" else V56_BLEND_STRATEGY_NAME)
            curve_store[target_mode].setdefault(strategy_name, [])
            user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
            result = run_direct_taste_vector_session(
                synthetic_user=user,
                n_steps=n_steps,
                step_scale=step_scale,
                heuristic_lr=lr,
                init_std=init_std,
                std_decay=std_decay,
                min_std=min_std,
                clip_value=clip_value,
                seed=pair_seed_base + user_id,
                mode=direct_mode,
            )
            d = np.concatenate([[float(np.linalg.norm(z_target))], result.distances])
            curve_store[target_mode][strategy_name].append(d)
            rows.append(_summary_row_v56(user_id, target_mode, strategy_name, result, z_target))
            for rec in result.records:
                item = dict(rec)
                item.update({"user_id": user_id, "target_mode": target_mode, "strategy": strategy_name})
                step_rows.append(item)

    sessions = pd.concat([sessions_base, pd.DataFrame(rows)], axis=0, ignore_index=True) if not sessions_base.empty else pd.DataFrame(rows)
    steps = pd.concat([steps_base, pd.DataFrame(step_rows)], axis=0, ignore_index=True) if not steps_base.empty else pd.DataFrame(step_rows)

    n_steps_col = pd.to_numeric(sessions.get("n_steps", sessions.get("used_steps", n_steps)), errors="coerce").fillna(float(n_steps))
    stop_col = pd.to_numeric(sessions.get("recommended_stop_step", np.nan), errors="coerce")
    has_stop = stop_col.notna()
    sessions["steps_before_recommendation"] = np.where(has_stop, stop_col, n_steps_col).astype(float)
    sessions["steps_after_recommendation"] = np.where(has_stop, np.maximum(n_steps_col - stop_col, 0.0), 0.0).astype(float)

    curves = {
        target_mode: {strategy: np.asarray(items, dtype=np.float64) for strategy, items in by_strategy.items()}
        for target_mode, by_strategy in curve_store.items()
    }
    return sessions, steps, curves


def summarize_v56_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    return summarize_v55_sessions(sessions)


def source_usage_table_v56(steps: pd.DataFrame) -> pd.DataFrame:
    return source_usage_table_v55(steps)


def post_marker_source_usage_table_v56(steps: pd.DataFrame, target_mode: str = "archetype8d") -> pd.DataFrame:
    return post_marker_source_usage_table_v55(steps, target_mode=target_mode)


def phase_step_budget_table_v56(sessions: pd.DataFrame, target_mode: str = "archetype8d") -> pd.DataFrame:
    return phase_step_budget_table(sessions, target_mode=target_mode)




def direct_selection_table_v56(steps: pd.DataFrame, target_mode: str = "archetype8d") -> pd.DataFrame:
    """Share of anchor/direct choices after the soft-stop marker."""
    if steps.empty:
        return pd.DataFrame()
    direct_strategies = [V56_BLEND_STRATEGY_NAME, V56_PM_STRATEGY_NAME, V56_TRUST_STRATEGY_NAME]
    df = steps[(steps["target_mode"].astype(str).eq(target_mode)) & (steps["strategy"].isin(direct_strategies))].copy()
    if df.empty:
        return pd.DataFrame()
    df = df[df.get("soft_stop_marker", "").astype(str).eq("continued_after_recommendation")].copy()
    if df.empty:
        return pd.DataFrame()
    df["selected_role"] = df["selected_role"].fillna("unknown").astype(str)
    counts = df.groupby(["target_mode", "strategy", "selected_role"]).size().rename("count").reset_index()
    totals = counts.groupby(["target_mode", "strategy"])["count"].transform("sum")
    counts["share"] = counts["count"] / totals.replace(0, np.nan)
    return counts


def direct_vector_diagnostics_table_v56(steps: pd.DataFrame, target_mode: str = "archetype8d") -> pd.DataFrame:
    """Aggregate diagnostics for why V5.6 direct variants may collapse to similar behavior."""
    if steps.empty:
        return pd.DataFrame()
    direct_strategies = [V56_BLEND_STRATEGY_NAME, V56_PM_STRATEGY_NAME, V56_TRUST_STRATEGY_NAME]
    df = steps[(steps["target_mode"].astype(str).eq(target_mode)) & (steps["strategy"].isin(direct_strategies))].copy()
    if df.empty:
        return pd.DataFrame()
    df = df[df.get("soft_stop_marker", "").astype(str).eq("continued_after_recommendation")].copy()
    if df.empty:
        return pd.DataFrame()

    df["direct_selected"] = df.get("selected_role", "").astype(str).eq("direct_candidate_selected").astype(float)
    metric_cols = [
        "direct_selected",
        "raw_delta_norm",
        "delta_pm_norm",
        "delta_blend_norm",
        "model_blend_cosine",
        "delta_pm_blend_cosine",
        "target_model_cosine",
        "target_blend_cosine",
        "direction_cosine_to_raw_delta",
        "direction_cosine_to_pm_delta",
        "direction_cosine_to_blend_delta",
        "trust_mask_mean",
        "trust_mask_min",
        "trust_mask_max",
        "direction_effective_dims",
        "scale",
        "pair_distance",
        "audibility_score",
        "acceptability_score",
        "midrange_disturbance_penalty",
        "applied_lr",
    ]
    for col in metric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows: list[dict] = []
    for strategy, g in df.groupby("strategy", sort=False):
        row = {
            "target_mode": target_mode,
            "strategy": strategy,
            "n_post_marker_questions": int(len(g)),
            "direct_selected_rate": float(g["direct_selected"].mean()),
        }
        for col in metric_cols:
            row[f"mean_{col}"] = float(g[col].mean()) if g[col].notna().any() else np.nan
            row[f"std_{col}"] = float(g[col].std(ddof=0)) if g[col].notna().any() else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    return out

def save_v56_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    source_usage: pd.DataFrame,
    step_budget: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "article_v56_direct_taste_vector_refinement",
    post_marker_source_usage: pd.DataFrame | None = None,
) -> None:
    save_v55_outputs(
        sessions=sessions,
        steps=steps,
        summary=summary,
        win_rates=win_rates,
        source_usage=source_usage,
        step_budget=step_budget,
        output_dir=output_dir,
        prefix=prefix,
        post_marker_source_usage=post_marker_source_usage,
    )
