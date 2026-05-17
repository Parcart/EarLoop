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
    _direction_lock_status,
    _model_blend_direction,
    _update_intensity_score,
)
from .safe_anchor_challenger_v51 import V51_STRATEGY_NAME
from .safe_anchor_challenger_v52 import V52_STRATEGY_NAME
from .safe_anchor_challenger_v53 import (
    V53_DISPLAY_NAMES,
    V53_STRATEGY_NAME,
    run_v53_comparison_on_dataset,
    source_usage_table_v53,
)


V54_STRATEGY_NAME = "semantic_active_soft_stop_monitor_v54"

V54_DISPLAY_NAMES = dict(V53_DISPLAY_NAMES)
V54_DISPLAY_NAMES.update({
    V54_STRATEGY_NAME: "Semantic active v3 + soft-stop monitor v5.4",
})


@dataclass
class V54Config:
    """V5.4: soft-stop monitor over Semantic active v3.

    The generator is intentionally NOT changed: Semantic active v3 keeps asking
    questions until n_steps. The controller only marks the step where the product
    could recommend stopping, then measures what happens if the user continues.

    This answers a different question from V5.2/V5.3:
        - not "can confirmation replace optimization?"
        - but "when would the system recommend that the profile is stable, and
          how much quality remains to be gained after that marker?"
    """

    min_anchor_step: int = 12
    direction_lock_threshold: float = 0.88
    direction_locked_patience: int = 2

    min_recommendation_step: int = 14
    ready_threshold: float = 0.90
    mean_std_ready_threshold: float = 0.46
    update_norm_ready_threshold: float = 0.30
    ready_patience: int = 2

    base_step_scale: float = 0.6
    clip_value: float | None = 2.0


@dataclass
class V54SessionResult:
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[dict]
    distances: np.ndarray
    intensity_score: float
    direction_lock_step: int | None
    ready_step: int | None
    recommended_stop_step: int | None
    final_status: str


def _is_ready_candidate(
    *,
    step: int,
    cfg: V54Config,
    agreement_after: float,
    mean_std: float,
    update_norm: float,
) -> bool:
    return (
        step >= int(cfg.min_recommendation_step)
        and agreement_after >= float(cfg.ready_threshold)
        and mean_std <= float(cfg.mean_std_ready_threshold)
        and update_norm <= float(cfg.update_norm_ready_threshold)
    )


def _source_group_from_meta(pair_meta: dict) -> str:
    source_group = pair_meta.get("source_group")
    if source_group is not None:
        return str(source_group)
    source = str(pair_meta.get("source", pair_meta.get("strategy", "semantic")))
    if "semantic" in source:
        return "semantic"
    if "axis" in source:
        return "axis"
    if "random" in source:
        return "random"
    return source


def run_semantic_soft_stop_monitor_session(
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
    config: V54Config | None = None,
) -> V54SessionResult:
    """Run Semantic active v3 for all steps while only marking soft-stop readiness."""
    cfg = V54Config(base_step_scale=step_scale, clip_value=clip_value) if config is None else config
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
    recommended_stop_step: int | None = None
    lock_hits = 0
    ready_hits = 0

    for step in range(1, int(n_steps) + 1):
        state_before = state.copy()
        locked, agreement_before, lock_hits = _direction_lock_status(state, model, step, cfg, lock_hits)
        if locked and direction_lock_step is None:
            direction_lock_step = int(step)

        # The actual Pair Generator remains Semantic active v3 for the whole session.
        z_a, z_b, direction, pair_meta = pair_generator.semantic_active_v21(state)
        pair_meta = dict(pair_meta)
        pair_meta["strategy"] = V54_STRATEGY_NAME
        pair_meta["sub_strategy"] = "semantic_active_with_soft_stop_monitor"
        pair_meta["source"] = "semantic_active_v21"
        pair_meta["source_group"] = "semantic"
        pair_meta["agreement_before"] = float(agreement_before)
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
            phase = "semantic_before_soft_stop"
            soft_stop_marker = "not_ready"
        elif step == recommended_stop_step:
            phase = "semantic_recommended_stop_marker"
            soft_stop_marker = "recommended_stop"
        else:
            phase = "semantic_after_soft_stop_continuation"
            soft_stop_marker = "continued_after_recommendation"

        records.append({
            "step": int(step),
            "phase": phase,
            "choice": choice,
            "selected_role": "semantic_active_question",
            "applied_lr": float(heuristic_lr),
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
            "sub_strategy": pair_meta.get("sub_strategy"),
            "control_name": pair_meta.get("control_name"),
            "scale": float(pair_meta.get("scale", np.nan)),
            "scale_multiplier": float(pair_meta.get("scale_multiplier", np.nan)),
            "pair_distance": float(np.linalg.norm(z_a - z_b)),
            "audibility_score": float(min(np.linalg.norm(z_a - z_b) / (step_scale + 1e-8), 1.0)),
            "acceptability_score": 1.0,
            "midrange_disturbance_penalty": 0.0,
            "safety_penalty": float(pair_meta.get("safety_penalty", np.nan)),
            "direction_lock_step": direction_lock_step,
            "ready_step": ready_step,
            "recommended_stop_step": recommended_stop_step,
            "ready_candidate": bool(ready_candidate),
            "ready_hits": int(ready_hits),
            "soft_stop_marker": soft_stop_marker,
        })

    final_status = "soft_stop_recommended_and_continued" if recommended_stop_step is not None else "no_soft_stop_recommendation"
    return V54SessionResult(
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


def _value_at_step(distances: np.ndarray, step: int | None) -> float:
    if step is None or pd.isna(step):
        return np.nan
    step_i = int(step)
    if step_i <= 0:
        return np.nan
    if step_i > len(distances):
        return np.nan
    return float(distances[step_i - 1])


def _summary_row(user_id: int, target_mode: str, strategy: str, result, z_target: np.ndarray) -> dict:
    d = np.concatenate([[float(np.linalg.norm(z_target))], result.distances])
    recommended_stop_step = getattr(result, "recommended_stop_step", np.nan)
    dist_at_stop = _value_at_step(result.distances, recommended_stop_step)
    final_distance = float(d[-1])
    initial_distance = float(d[0])
    extra_gain_after_stop = np.nan
    quality_loss_if_stop = np.nan
    retained_quality_at_stop = np.nan
    steps_saved_if_stop = np.nan
    if not np.isnan(dist_at_stop):
        extra_gain_after_stop = float(dist_at_stop - final_distance)
        quality_loss_if_stop = float(dist_at_stop - final_distance)
        full_gain = initial_distance - final_distance
        stop_gain = initial_distance - dist_at_stop
        retained_quality_at_stop = float(100.0 * stop_gain / (full_gain + 1e-8))
        steps_saved_if_stop = float(len(result.distances) - int(recommended_stop_step))

    records_df = pd.DataFrame(result.records) if hasattr(result, "records") else pd.DataFrame()
    continued_rows = records_df[records_df.get("phase", pd.Series(dtype=str)).astype(str).eq("semantic_after_soft_stop_continuation")] if not records_df.empty else pd.DataFrame()
    return {
        "user_id": int(user_id),
        "target_mode": target_mode,
        "strategy": strategy,
        "n_steps": int(len(result.distances)),
        "used_steps": int(len(result.distances)),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "best_distance": float(np.min(d)),
        "mean_distance": float(np.mean(d)),
        "improvement_abs": float(initial_distance - final_distance),
        "improvement_pct": float(100.0 * (initial_distance - final_distance) / (initial_distance + 1e-8)),
        "direction_lock_step": getattr(result, "direction_lock_step", np.nan),
        "ready_step": getattr(result, "ready_step", np.nan),
        "recommended_stop_step": recommended_stop_step,
        "recommended_stop_rate": float(pd.notna(recommended_stop_step)),
        "distance_at_recommended_stop": dist_at_stop,
        "extra_gain_after_stop": extra_gain_after_stop,
        "quality_loss_if_stop": quality_loss_if_stop,
        "retained_quality_at_stop_pct": retained_quality_at_stop,
        "steps_saved_if_stop": steps_saved_if_stop,
        "final_status": getattr(result, "final_status", "baseline"),
        "continued_steps_after_stop": int(len(continued_rows)) if not continued_rows.empty else 0,
        "final_intensity_score": getattr(result, "intensity_score", np.nan),
        "mean_pair_distance": float(records_df["pair_distance"].mean()) if "pair_distance" in records_df else np.nan,
        "mean_audibility_score": float(records_df["audibility_score"].mean()) if "audibility_score" in records_df else np.nan,
        "mean_acceptability_score": float(records_df["acceptability_score"].mean()) if "acceptability_score" in records_df else np.nan,
        "mean_midrange_penalty": float(records_df["midrange_disturbance_penalty"].mean()) if "midrange_disturbance_penalty" in records_df else np.nan,
        "mean_applied_lr": float(records_df["applied_lr"].mean()) if "applied_lr" in records_df else np.nan,
        "stop_rate": 0.0,
    }


def run_v54_comparison_on_dataset(
    dataset: pd.DataFrame,
    baseline_strategies: Iterable[str] = ("semantic_active_v21", "candidate_pool_active"),
    include_v51: bool = True,
    include_v52: bool = True,
    include_v53: bool = True,
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
    sessions_base, steps_base, curves_base = run_v53_comparison_on_dataset(
        dataset=dataset,
        baseline_strategies=baseline_strategies,
        include_v5=False,
        include_v51=include_v51,
        include_v52=include_v52,
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

    if not include_v53:
        sessions_base = sessions_base[sessions_base["strategy"] != V53_STRATEGY_NAME].copy()
        steps_base = steps_base[steps_base["strategy"] != V53_STRATEGY_NAME].copy() if not steps_base.empty else steps_base
        for mode in list(curves_base.keys()):
            curves_base[mode].pop(V53_STRATEGY_NAME, None)

    rows: list[dict] = []
    step_rows: list[dict] = []

    curve_store: dict[str, dict[str, list[np.ndarray]]] = {}
    for mode, by_strategy in curves_base.items():
        curve_store[mode] = {strategy: [arr for arr in values] for strategy, values in by_strategy.items()}
        curve_store[mode].setdefault(V54_STRATEGY_NAME, [])

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        curve_store.setdefault(target_mode, {}).setdefault(V54_STRATEGY_NAME, [])

        user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
        result_v54 = run_semantic_soft_stop_monitor_session(
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
        d = np.concatenate([[float(np.linalg.norm(z_target))], result_v54.distances])
        curve_store[target_mode][V54_STRATEGY_NAME].append(d)
        rows.append(_summary_row(user_id, target_mode, V54_STRATEGY_NAME, result_v54, z_target))
        for rec in result_v54.records:
            item = dict(rec)
            item.update({"user_id": user_id, "target_mode": target_mode, "strategy": V54_STRATEGY_NAME})
            step_rows.append(item)

    sessions = pd.concat([sessions_base, pd.DataFrame(rows)], axis=0, ignore_index=True)
    steps = pd.concat([steps_base, pd.DataFrame(step_rows)], axis=0, ignore_index=True) if not steps_base.empty else pd.DataFrame(step_rows)
    curves = {
        target_mode: {strategy: np.asarray(items, dtype=np.float64) for strategy, items in by_strategy.items()}
        for target_mode, by_strategy in curve_store.items()
    }
    return sessions, steps, curves


def summarize_v54_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
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
            mean_recommended_stop_step=("recommended_stop_step", "mean") if "recommended_stop_step" in sessions.columns else ("used_steps", "mean"),
            recommended_stop_rate=("recommended_stop_rate", "mean") if "recommended_stop_rate" in sessions.columns else ("used_steps", lambda s: 0.0),
            mean_distance_at_recommended_stop=("distance_at_recommended_stop", "mean") if "distance_at_recommended_stop" in sessions.columns else ("final_distance", "mean"),
            mean_extra_gain_after_stop=("extra_gain_after_stop", "mean") if "extra_gain_after_stop" in sessions.columns else ("final_distance", lambda s: np.nan),
            mean_quality_loss_if_stop=("quality_loss_if_stop", "mean") if "quality_loss_if_stop" in sessions.columns else ("final_distance", lambda s: np.nan),
            mean_retained_quality_at_stop_pct=("retained_quality_at_stop_pct", "mean") if "retained_quality_at_stop_pct" in sessions.columns else ("final_distance", lambda s: np.nan),
            mean_steps_saved_if_stop=("steps_saved_if_stop", "mean") if "steps_saved_if_stop" in sessions.columns else ("used_steps", lambda s: np.nan),
            mean_continued_steps_after_stop=("continued_steps_after_stop", "mean") if "continued_steps_after_stop" in sessions.columns else ("used_steps", lambda s: np.nan),
            mean_pair_distance=("mean_pair_distance", "mean"),
            mean_audibility_score=("mean_audibility_score", "mean"),
            mean_acceptability_score=("mean_acceptability_score", "mean"),
            mean_midrange_penalty=("mean_midrange_penalty", "mean"),
            mean_applied_lr=("mean_applied_lr", "mean"),
        )
        .reset_index()
        .sort_values(["target_mode", "mean_final_distance"])
    )


def source_usage_table_v54(steps: pd.DataFrame) -> pd.DataFrame:
    return source_usage_table_v53(steps)


def save_v54_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    source_usage: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "article_v54_soft_stop_monitor",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    steps.to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    win_rates.to_csv(output_dir / f"{prefix}_win_rates.csv", index=False)
    source_usage.to_csv(output_dir / f"{prefix}_source_usage.csv", index=False)
