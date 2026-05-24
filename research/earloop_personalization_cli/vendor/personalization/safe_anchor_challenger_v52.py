from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .batch_eval import distances_with_initial
from .loop import run_personalization_session_v0
from .metrics import distance_to_target
from .pair_generator import PairGenerator
from .preference_model import LogisticDistancePreferenceModel
from .preference_update import update_state_from_choice
from .state import PreferenceState, init_preference_state
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser
from .safe_anchor_challenger import (
    V5_STRATEGY_NAME,
    run_safe_anchor_challenger_session,
    _direction_lock_status,
    _model_blend_direction,
    _update_intensity_score,
)
from .safe_anchor_challenger_v51 import (
    V51_STRATEGY_NAME,
    V51Config,
    run_conservative_anchor_challenger_session,
    make_conservative_anchor_challenger_pair,
    _update_state_conservative,
    _summary_row as _v51_summary_row,
)


V52_STRATEGY_NAME = "semantic_backbone_confirmation_v52"

V52_DISPLAY_NAMES = {
    "semantic_active_v21": "Semantic active v3",
    "candidate_pool_active": "Candidate pool active",
    V5_STRATEGY_NAME: "Safe anchor-challenger v5",
    V51_STRATEGY_NAME: "Conservative anchor-challenger v5.1",
    V52_STRATEGY_NAME: "Semantic backbone + confirmation v5.2",
}


@dataclass
class V52Config(V51Config):
    """V5.2 configuration: semantic-active backbone + short confirmation block.

    V5.2 does not replace the main Pair Generator with anchor-challenger.
    Instead, `semantic_active_v21` remains the backbone. Anchor-challenger is
    used only after the controller thinks the profile is stable enough, and only
    for a small number of confirmation/refinement questions.
    """

    # No safe calibration by default: keep the strongest known generator as the
    # backbone from step 1. This makes the ablation easy to interpret.
    use_safe_calibration: bool = False
    calibration_steps: int = 0

    min_confirmation_step: int = 14
    ready_threshold: float = 0.88
    mean_std_ready_threshold: float = 0.48
    update_norm_ready_threshold: float = 0.25
    max_confirmation_questions: int = 2
    freeze_after_confirmation: bool = True

    # Confirmation should be a test/refinement, not a new optimizer.
    anchor_selected_lr: float = 0.0
    challenger_selected_lr: float = 0.06
    ready_anchor_lr: float = 0.0
    ready_challenger_lr: float = 0.04

    challenger_scales: tuple[float, ...] = (0.18, 0.30, 0.45)
    max_anchor_challenger_distance: float = 0.85
    min_audible_distance: float = 0.24


@dataclass
class V52SessionResult:
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


def _standard_update(
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


def run_semantic_backbone_confirmation_session(
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
    config: V52Config | None = None,
) -> V52SessionResult:
    """Run V5.2 semantic backbone + short anchor-challenger confirmation.

    The backbone is `semantic_active_v21`. Confirmation questions are used only
    after readiness is detected. After the confirmation budget is exhausted the
    session is treated as stopped; the remaining distance curve is padded with
    the same value for fair plotting against fixed-25-step baselines.
    """
    cfg = V52Config(base_step_scale=step_scale, clip_value=clip_value) if config is None else config
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
    lock_hits = 0
    confirmation_count = 0
    finalized = False
    used_steps = int(n_steps)

    for step in range(1, int(n_steps) + 1):
        if finalized:
            # Product behavior: session would already stop here. We pad the
            # curve to compare against fixed-length experiments.
            dist = distance_to_target(state.z_mean, synthetic_user.z_target)
            distances.append(float(dist))
            records.append({
                "step": int(step),
                "phase": "stopped_after_confirmation",
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
                "sub_strategy": "stopped_after_confirmation",
                "pair_distance": 0.0,
                "audibility_score": np.nan,
                "acceptability_score": np.nan,
                "anchor_acceptability_score": np.nan,
                "model_uncertainty": np.nan,
                "midrange_disturbance_penalty": np.nan,
                "safety_penalty": np.nan,
                "ready_step": ready_step,
                "confirmation_count": confirmation_count,
            })
            continue

        state_before = state.copy()
        locked, agreement, lock_hits = _direction_lock_status(state, model, step, cfg, lock_hits)
        if locked and direction_lock_step is None:
            direction_lock_step = int(step)

        in_confirmation = ready_step is not None and confirmation_count < cfg.max_confirmation_questions
        if in_confirmation:
            phase = "anchor_challenger_confirmation"
            if confirmation_start_step is None:
                confirmation_start_step = int(step)
            z_a, z_b, direction, pair_meta = make_conservative_anchor_challenger_pair(
                state=state,
                model=model,
                rng=rng,
                config=cfg,
                intensity_score=intensity_score,
            )
            pair_meta = dict(pair_meta)
            pair_meta["strategy"] = V52_STRATEGY_NAME
            pair_meta["sub_strategy"] = "anchor_challenger_confirmation"
            update_mode = "confirmation"
        else:
            phase = "semantic_active_backbone"
            z_a, z_b, direction, pair_meta = pair_generator.semantic_active_v21(state)
            pair_meta = dict(pair_meta)
            pair_meta["strategy"] = V52_STRATEGY_NAME
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
        else:
            applied_lr = heuristic_lr
            selected_role = "standard_semantic"
            state = _standard_update(
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

        if ready_step is None and step >= cfg.min_confirmation_step:
            # Readiness is deliberately less aggressive than V5.1 direction lock:
            # it must see agreement and local stability, but it does not end the
            # session immediately; it starts a small confirmation block.
            if agreement_after >= cfg.ready_threshold and mean_std <= cfg.mean_std_ready_threshold and update_norm <= cfg.update_norm_ready_threshold:
                ready_step = int(step)

        if cfg.freeze_after_confirmation and ready_step is not None and confirmation_count >= cfg.max_confirmation_questions:
            finalized = True
            used_steps = int(step)

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
        })

    final_status = "confirmed_and_stopped" if confirmation_count >= cfg.max_confirmation_questions else (
        "ready_no_confirmation_budget" if ready_step is not None else "completed_without_ready"
    )
    return V52SessionResult(
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
    )


def _summary_row(user_id: int, target_mode: str, strategy: str, result, z_target: np.ndarray) -> dict:
    d = np.concatenate([[float(np.linalg.norm(z_target))], result.distances])
    records_df = pd.DataFrame(result.records) if hasattr(result, "records") else pd.DataFrame()
    confirmation_rows = records_df[records_df.get("phase", pd.Series(dtype=str)).astype(str).eq("anchor_challenger_confirmation")] if not records_df.empty else pd.DataFrame()
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
        "anchor_selected_rate": float((confirmation_rows.get("selected_role") == "anchor_selected").mean()) if not confirmation_rows.empty and "selected_role" in confirmation_rows else np.nan,
        "mean_pair_distance": float(records_df["pair_distance"].mean()) if "pair_distance" in records_df else np.nan,
        "mean_audibility_score": float(records_df["audibility_score"].mean()) if "audibility_score" in records_df else np.nan,
        "mean_acceptability_score": float(records_df["acceptability_score"].mean()) if "acceptability_score" in records_df else np.nan,
        "mean_midrange_penalty": float(records_df["midrange_disturbance_penalty"].mean()) if "midrange_disturbance_penalty" in records_df else np.nan,
        "mean_applied_lr": float(records_df["applied_lr"].mean()) if "applied_lr" in records_df else np.nan,
    }


def run_v52_comparison_on_dataset(
    dataset: pd.DataFrame,
    baseline_strategies: Iterable[str] = ("semantic_active_v21", "candidate_pool_active"),
    include_v5: bool = True,
    include_v51: bool = True,
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_seed_base: int = 30_000,
    user_seed_base: int = 10_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    rows: list[dict] = []
    step_rows: list[dict] = []
    strategies = list(baseline_strategies)
    if include_v5:
        strategies.append(V5_STRATEGY_NAME)
    if include_v51:
        strategies.append(V51_STRATEGY_NAME)
    strategies.append(V52_STRATEGY_NAME)
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
                "used_steps": int(len(result.distances)),
                "initial_distance": float(d[0]),
                "final_distance": float(d[-1]),
                "best_distance": float(np.min(d)),
                "mean_distance": float(np.mean(d)),
                "improvement_abs": float(d[0] - d[-1]),
                "improvement_pct": float(100.0 * (d[0] - d[-1]) / (d[0] + 1e-8)),
                "direction_lock_step": np.nan,
                "ready_step": np.nan,
                "confirmation_start_step": np.nan,
                "final_status": "baseline",
                "final_intensity_score": np.nan,
                "confirmation_questions": 0,
                "anchor_selected_rate": np.nan,
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
            rows.append(_v51_summary_row(user_id, target_mode, V5_STRATEGY_NAME, result_v5, z_target))
            rows[-1]["used_steps"] = int(len(result_v5.distances))
            for rec in result_v5.records:
                item = dict(rec)
                item.update({"user_id": user_id, "target_mode": target_mode, "strategy": V5_STRATEGY_NAME})
                step_rows.append(item)

        if include_v51:
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
            rows.append(_v51_summary_row(user_id, target_mode, V51_STRATEGY_NAME, result_v51, z_target))
            rows[-1]["used_steps"] = int(len(result_v51.distances))
            for rec in result_v51.records:
                item = dict(rec)
                item.update({"user_id": user_id, "target_mode": target_mode, "strategy": V51_STRATEGY_NAME})
                step_rows.append(item)

        user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
        result_v52 = run_semantic_backbone_confirmation_session(
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
        d = np.concatenate([[float(np.linalg.norm(z_target))], result_v52.distances])
        curve_store[target_mode][V52_STRATEGY_NAME].append(d)
        rows.append(_summary_row(user_id, target_mode, V52_STRATEGY_NAME, result_v52, z_target))
        for rec in result_v52.records:
            item = dict(rec)
            item.update({"user_id": user_id, "target_mode": target_mode, "strategy": V52_STRATEGY_NAME})
            step_rows.append(item)

    sessions = pd.DataFrame(rows)
    steps = pd.DataFrame(step_rows)
    curves = {
        target_mode: {strategy: np.asarray(items, dtype=np.float64) for strategy, items in by_strategy.items()}
        for target_mode, by_strategy in curve_store.items()
    }
    return sessions, steps, curves


def summarize_v52_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
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
            mean_anchor_selected_rate=("anchor_selected_rate", "mean"),
            mean_pair_distance=("mean_pair_distance", "mean"),
            mean_audibility_score=("mean_audibility_score", "mean"),
            mean_acceptability_score=("mean_acceptability_score", "mean"),
            mean_midrange_penalty=("mean_midrange_penalty", "mean"),
            mean_applied_lr=("mean_applied_lr", "mean"),
        )
        .reset_index()
        .sort_values(["target_mode", "mean_final_distance"])
    )


def source_usage_table_v52(steps: pd.DataFrame) -> pd.DataFrame:
    if steps.empty:
        return pd.DataFrame()
    return (
        steps.groupby(["target_mode", "strategy", "phase", "pair_source_group"])
        .size()
        .reset_index(name="count")
        .assign(share=lambda df: df["count"] / df.groupby(["target_mode", "strategy", "phase"])["count"].transform("sum"))
        .sort_values(["target_mode", "strategy", "phase", "share"], ascending=[True, True, True, False])
    )


def save_v52_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    source_usage: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "article_v52_semantic_backbone_confirmation",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    steps.to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    win_rates.to_csv(output_dir / f"{prefix}_win_rates.csv", index=False)
    source_usage.to_csv(output_dir / f"{prefix}_source_usage.csv", index=False)
