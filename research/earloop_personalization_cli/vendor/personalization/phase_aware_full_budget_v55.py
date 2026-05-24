from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .batch_eval import win_rates_vs_baseline
from .loop import run_personalization_session_v0
from .metrics import distance_to_target
from .pair_generator import PairGenerator
from .preference_model import LogisticDistancePreferenceModel
from .preference_update import update_state_from_choice
from .safe_anchor_challenger import _direction_lock_status, _model_blend_direction, _update_intensity_score
from .safe_anchor_challenger_v51 import V51Config, make_conservative_anchor_challenger_pair, _update_state_conservative
from .soft_stop_monitor_v54 import (
    V54_DISPLAY_NAMES,
    V54_STRATEGY_NAME,
    V54Config,
    _is_ready_candidate,
    _source_group_from_meta,
    _summary_row,
    run_v54_comparison_on_dataset,
    source_usage_table_v54,
)
from .state import PreferenceState, init_preference_state
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser


V55_ZONE_STRATEGY_NAME = "phase_zone_refinement_full_budget_v55"
V55_MIXED_STRATEGY_NAME = "phase_mixed_refinement_full_budget_v55"

V55_DISPLAY_NAMES = dict(V54_DISPLAY_NAMES)
# Make the marker-only baseline explicit in plots. It is the same Semantic active v3
# Pair Generator, but with a soft-stop monitor attached.
V55_DISPLAY_NAMES[V54_STRATEGY_NAME] = "Semantic active v3 + soft-stop marker v5.4"
V55_DISPLAY_NAMES.update({
    V55_ZONE_STRATEGY_NAME: "Phase zone refinement v5.5",
    V55_MIXED_STRATEGY_NAME: "Phase mixed refinement v5.5",
})


PhaseMode = Literal["zone", "mixed"]


@dataclass
class V55Config(V54Config):
    """V5.5: full-budget phase-aware generation.

    Unlike V5.2/V5.3, this strategy never hard-stops before n_steps.
    It uses the same soft-stop marker as V5.4, but if the user continues, the
    generator may change its question type after the marker.

    Modes:
    - zone: after the recommended marker, use conservative zone-level
      anchor/challenger refinement for the remaining steps;
    - mixed: after the marker, mostly keep Semantic active v3, but inject local
      conservative anchor/challenger questions periodically.

    This separates two questions:
    1. When would we recommend stopping?
    2. If the user continues, which pair generator should we use after that?
    """

    mode: PhaseMode = "mixed"
    mixed_anchor_every: int = 3
    min_post_marker_semantic_steps: int = 1

    # Conservative refinement update config.
    anchor_selected_lr: float = 0.015
    challenger_selected_lr: float = 0.085
    ready_anchor_lr: float = 0.0
    ready_challenger_lr: float = 0.035


@dataclass
class V55SessionResult:
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[dict]
    distances: np.ndarray
    intensity_score: float
    direction_lock_step: int | None
    ready_step: int | None
    recommended_stop_step: int | None
    final_status: str


def _v51_config_from_v55(cfg: V55Config) -> V51Config:
    """Use V5.1 conservative settings for local refinement steps."""
    return V51Config(
        calibration_steps=0,
        min_anchor_step=cfg.min_anchor_step,
        direction_lock_threshold=cfg.direction_lock_threshold,
        ready_threshold=cfg.ready_threshold,
        base_step_scale=cfg.base_step_scale,
        clip_value=cfg.clip_value,
        anchor_selected_lr=cfg.anchor_selected_lr,
        challenger_selected_lr=cfg.challenger_selected_lr,
        ready_anchor_lr=cfg.ready_anchor_lr,
        ready_challenger_lr=cfg.ready_challenger_lr,
    )


def _choose_post_marker_source(
    mode: PhaseMode,
    post_marker_index: int,
    cfg: V55Config,
) -> str:
    if mode == "zone":
        return "zone_refinement"
    if mode == "mixed":
        # Keep at least one semantic question immediately after the marker.
        if post_marker_index <= int(cfg.min_post_marker_semantic_steps):
            return "semantic_active"
        every = max(int(cfg.mixed_anchor_every), 2)
        if post_marker_index % every == 0:
            return "zone_refinement"
        return "semantic_active"
    raise ValueError(f"Unknown phase-aware mode: {mode}")


def run_phase_aware_full_budget_session(
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
    mode: PhaseMode = "mixed",
    config: V55Config | None = None,
) -> V55SessionResult:
    """Run a full 25-step phase-aware session without hard stopping."""
    cfg = V55Config(base_step_scale=step_scale, clip_value=clip_value, mode=mode) if config is None else config
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
    v51_cfg = _v51_config_from_v55(cfg)

    strategy_name = V55_MIXED_STRATEGY_NAME if mode == "mixed" else V55_ZONE_STRATEGY_NAME

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
        post_marker_index = int(step - recommended_stop_step) if after_marker and recommended_stop_step is not None else 0
        source_choice = _choose_post_marker_source(mode, post_marker_index, cfg) if after_marker else "semantic_active"

        if source_choice == "zone_refinement":
            z_a, z_b, direction, pair_meta = make_conservative_anchor_challenger_pair(
                state=state,
                model=model,
                rng=rng,
                config=v51_cfg,
                intensity_score=intensity_score,
            )
            pair_meta = dict(pair_meta)
            pair_meta["strategy"] = strategy_name
            pair_meta["sub_strategy"] = "post_stop_zone_refinement"
            pair_meta["phase_source"] = "zone_refinement"
            pair_meta["source_group"] = pair_meta.get("source_group", "zone_refinement")
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
        if source_choice == "zone_refinement":
            state, applied_lr, selected_role = _update_state_conservative(
                state=state,
                z_a=z_a,
                z_b=z_b,
                choice=choice,
                pair_meta=pair_meta,
                config=v51_cfg,
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
        elif step == recommended_stop_step:
            phase = "soft_stop_marker"
            soft_stop_marker = "recommended_stop"
        else:
            phase = f"after_soft_stop_{source_choice}"
            soft_stop_marker = "continued_after_recommendation"

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
    return V55SessionResult(
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


def _summary_row_v55(user_id: int, target_mode: str, strategy: str, result: V55SessionResult, z_target: np.ndarray) -> dict:
    row = _summary_row(user_id, target_mode, strategy, result, z_target)
    # Add explicit before/after split for the requested chart.
    stop_step = row.get("recommended_stop_step", np.nan)
    n_steps = int(row.get("n_steps", len(result.distances)))
    if pd.notna(stop_step):
        row["steps_before_recommendation"] = float(stop_step)
        row["steps_after_recommendation"] = float(max(n_steps - int(stop_step), 0))
    else:
        row["steps_before_recommendation"] = float(n_steps)
        row["steps_after_recommendation"] = 0.0
    return row


def run_v55_comparison_on_dataset(
    dataset: pd.DataFrame,
    baseline_strategies: Iterable[str] = ("semantic_active_v21", "candidate_pool_active"),
    include_previous_v5: bool = True,
    include_v54: bool = True,
    include_zone: bool = True,
    include_mixed: bool = True,
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_seed_base: int = 50_000,
    user_seed_base: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    if include_v54:
        sessions_base, steps_base, curves_base = run_v54_comparison_on_dataset(
            dataset=dataset,
            baseline_strategies=baseline_strategies,
            include_v51=include_previous_v5,
            include_v52=include_previous_v5,
            include_v53=include_previous_v5,
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
        rows_base: list[dict] = []
        steps_base = pd.DataFrame()
        curves_base: dict[str, dict[str, list[np.ndarray]]] = {}
        for _, row in dataset.iterrows():
            user_id = int(row["user_id"])
            target_mode = str(row["target_mode"])
            z_target = row_to_target(row)
            curves_base.setdefault(target_mode, {})
            for strategy in baseline_strategies:
                user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
                res = run_personalization_session_v0(
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
                d = np.concatenate([[float(np.linalg.norm(z_target))], res.distances])
                curves_base[target_mode].setdefault(strategy, []).append(d)
                rows_base.append(_summary_row(user_id, target_mode, strategy, res, z_target))
        sessions_base = pd.DataFrame(rows_base)

    curve_store: dict[str, dict[str, list[np.ndarray]]] = {}
    for mode, by_strategy in curves_base.items():
        curve_store[mode] = {strategy: [arr for arr in values] for strategy, values in by_strategy.items()}

    rows: list[dict] = []
    step_rows: list[dict] = []
    modes_to_run: list[PhaseMode] = []
    if include_zone:
        modes_to_run.append("zone")
    if include_mixed:
        modes_to_run.append("mixed")

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        curve_store.setdefault(target_mode, {})
        for mode in modes_to_run:
            strategy_name = V55_MIXED_STRATEGY_NAME if mode == "mixed" else V55_ZONE_STRATEGY_NAME
            curve_store[target_mode].setdefault(strategy_name, [])
            user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
            result = run_phase_aware_full_budget_session(
                synthetic_user=user,
                n_steps=n_steps,
                step_scale=step_scale,
                heuristic_lr=lr,
                init_std=init_std,
                std_decay=std_decay,
                min_std=min_std,
                clip_value=clip_value,
                seed=pair_seed_base + user_id,
                mode=mode,
            )
            d = np.concatenate([[float(np.linalg.norm(z_target))], result.distances])
            curve_store[target_mode][strategy_name].append(d)
            rows.append(_summary_row_v55(user_id, target_mode, strategy_name, result, z_target))
            for rec in result.records:
                item = dict(rec)
                item.update({"user_id": user_id, "target_mode": target_mode, "strategy": strategy_name})
                step_rows.append(item)

    sessions = pd.concat([sessions_base, pd.DataFrame(rows)], axis=0, ignore_index=True)
    steps = pd.concat([steps_base, pd.DataFrame(step_rows)], axis=0, ignore_index=True) if not steps_base.empty else pd.DataFrame(step_rows)

    # Make all rows compatible with the before/after chart.
    # Earlier versions filled missing values with n_steps, which accidentally hid
    # the soft-stop marker for the V5.4 marker-only baseline. Here we recompute
    # the split from recommended_stop_step for every strategy that has one.
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


def summarize_v55_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    summary = (
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
            mean_recommended_stop_step=("recommended_stop_step", "mean") if "recommended_stop_step" in sessions.columns else ("used_steps", lambda s: np.nan),
            recommended_stop_rate=("recommended_stop_rate", "mean") if "recommended_stop_rate" in sessions.columns else ("used_steps", lambda s: 0.0),
            mean_distance_at_recommended_stop=("distance_at_recommended_stop", "mean") if "distance_at_recommended_stop" in sessions.columns else ("final_distance", lambda s: np.nan),
            mean_extra_gain_after_stop=("extra_gain_after_stop", "mean") if "extra_gain_after_stop" in sessions.columns else ("final_distance", lambda s: np.nan),
            mean_retained_quality_at_stop_pct=("retained_quality_at_stop_pct", "mean") if "retained_quality_at_stop_pct" in sessions.columns else ("final_distance", lambda s: np.nan),
            mean_steps_saved_if_stop=("steps_saved_if_stop", "mean") if "steps_saved_if_stop" in sessions.columns else ("used_steps", lambda s: np.nan),
            mean_steps_before_recommendation=("steps_before_recommendation", "mean") if "steps_before_recommendation" in sessions.columns else ("used_steps", "mean"),
            mean_steps_after_recommendation=("steps_after_recommendation", "mean") if "steps_after_recommendation" in sessions.columns else ("used_steps", lambda s: 0.0),
            mean_pair_distance=("mean_pair_distance", "mean"),
            mean_audibility_score=("mean_audibility_score", "mean"),
            mean_acceptability_score=("mean_acceptability_score", "mean"),
            mean_midrange_penalty=("mean_midrange_penalty", "mean"),
            mean_applied_lr=("mean_applied_lr", "mean"),
        )
        .reset_index()
    )

    # Article-safe retained-quality metric. The old per-user ratio can explode
    # when a denominator is tiny. This aggregate version is computed from mean
    # distances and is stable/readable for charts.
    denom = summary["mean_initial_distance"] - summary["mean_final_distance"]
    numer = summary["mean_initial_distance"] - summary["mean_distance_at_recommended_stop"]
    summary["aggregate_retained_quality_at_stop_pct"] = np.where(
        np.isfinite(denom) & (np.abs(denom) > 1e-8),
        100.0 * numer / denom,
        np.nan,
    )
    summary["aggregate_retained_quality_at_stop_pct"] = summary["aggregate_retained_quality_at_stop_pct"].clip(lower=0.0, upper=100.0)
    # Keep the legacy column article-safe too: avoid misleading values like 269%.
    summary["mean_retained_quality_at_stop_pct"] = summary["aggregate_retained_quality_at_stop_pct"]

    return summary.sort_values(["target_mode", "mean_final_distance"])

def source_usage_table_v55(steps: pd.DataFrame) -> pd.DataFrame:
    """Whole-session source usage normalized per strategy.

    Older V5.x helpers normalize source shares inside each phase. That is useful
    for phase diagnostics, but misleading for the article chart "за всю
    сессию": summing per-phase shares can produce bars above 1.0. Here the
    denominator is all A/B questions of a strategy inside a target mode, so each
    stacked bar sums to 1.0.
    """
    if steps.empty:
        return pd.DataFrame(columns=["target_mode", "strategy", "pair_source_group", "count", "share"])
    df = steps.copy()
    if "pair_source_group" not in df.columns:
        df["pair_source_group"] = "unknown"
    df["pair_source_group"] = df["pair_source_group"].fillna("unknown").astype(str)
    counts = (
        df.groupby(["target_mode", "strategy", "pair_source_group"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(["target_mode", "strategy"])["count"].transform("sum")
    counts["share"] = counts["count"] / totals.replace(0, np.nan)
    return counts.sort_values(["target_mode", "strategy", "share"], ascending=[True, True, False])


def post_marker_source_usage_table_v55(steps: pd.DataFrame, target_mode: str = "archetype8d") -> pd.DataFrame:
    """Source usage after the soft-stop marker only.

    This table is useful for checking what happens if the user decides to
    continue after the product would already recommend saving the profile.
    """
    if steps.empty:
        return pd.DataFrame()
    df = steps[steps["target_mode"].astype(str).eq(target_mode)].copy()
    if "soft_stop_marker" in df.columns:
        df = df[df["soft_stop_marker"].astype(str).eq("continued_after_recommendation")]
    elif "recommended_stop_step" in df.columns:
        step = pd.to_numeric(df.get("step"), errors="coerce")
        stop = pd.to_numeric(df.get("recommended_stop_step"), errors="coerce")
        df = df[step > stop]
    else:
        df = df.iloc[0:0]
    if df.empty:
        return pd.DataFrame(columns=["target_mode", "strategy", "pair_source_group", "count", "share"])
    if "pair_source_group" not in df.columns:
        df["pair_source_group"] = "unknown"
    counts = (
        df.groupby(["target_mode", "strategy", "pair_source_group"])
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(["target_mode", "strategy"])["count"].transform("sum")
    counts["share"] = counts["count"] / totals.replace(0, np.nan)
    return counts


def phase_step_budget_table(sessions: pd.DataFrame, target_mode: str = "archetype8d") -> pd.DataFrame:
    df = sessions[sessions["target_mode"].astype(str).eq(target_mode)].copy()
    if df.empty:
        return pd.DataFrame()
    if "steps_before_recommendation" not in df.columns:
        df["steps_before_recommendation"] = df["n_steps"].astype(float)
    if "steps_after_recommendation" not in df.columns:
        df["steps_after_recommendation"] = 0.0
    return (
        df.groupby("strategy")
        .agg(
            users=("user_id", "count"),
            mean_total_steps=("used_steps", "mean"),
            mean_steps_before_recommendation=("steps_before_recommendation", "mean"),
            mean_steps_after_recommendation=("steps_after_recommendation", "mean"),
            mean_recommended_stop_step=("recommended_stop_step", "mean"),
            recommendation_rate=("recommended_stop_rate", "mean") if "recommended_stop_rate" in df.columns else ("used_steps", lambda s: 0.0),
        )
        .reset_index()
    )


def save_v55_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    source_usage: pd.DataFrame,
    step_budget: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "article_v55_phase_aware_full_budget",
    post_marker_source_usage: pd.DataFrame | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    steps.to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    win_rates.to_csv(output_dir / f"{prefix}_win_rates.csv", index=False)
    source_usage.to_csv(output_dir / f"{prefix}_source_usage.csv", index=False)
    step_budget.to_csv(output_dir / f"{prefix}_step_budget.csv", index=False)
    if post_marker_source_usage is not None:
        post_marker_source_usage.to_csv(output_dir / f"{prefix}_post_marker_source_usage.csv", index=False)
