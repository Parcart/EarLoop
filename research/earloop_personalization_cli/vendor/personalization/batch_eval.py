from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .loop import PairStrategy, run_personalization_session_v0
from .synthetic_dataset import row_to_synthetic_user, row_to_target


DEFAULT_STRATEGIES: list[PairStrategy] = [
    "random",
    "uncertainty_axis",
    "semantic_control",
    "semantic_control_v21",
    "semantic_active_v21",
    "candidate_pool_active",
    "adaptive_router_v32",
    "hybrid",
    "hybrid_v21",
    "hybrid_active_v21",
]


def distances_with_initial(result, z_target: np.ndarray) -> np.ndarray:
    """Prepend true distance at step 0 to result.distances."""
    initial_distance = float(np.linalg.norm(np.zeros_like(z_target) - z_target))
    return np.concatenate([[initial_distance], np.asarray(result.distances, dtype=np.float64)])


def make_summary_row(
    user_id: int,
    target_mode: str,
    strategy: str,
    result,
    z_target: np.ndarray,
) -> dict:
    d = distances_with_initial(result, z_target)
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
    }


def run_batch_on_dataset(
    dataset: pd.DataFrame,
    strategies: Iterable[PairStrategy] = DEFAULT_STRATEGIES,
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_seed_base: int = 20_000,
    user_seed_base: int = 10_000,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """
    Run multiple Pair Generator strategies on a fixed synthetic user dataset.

    Returns:
        sessions: long-form DataFrame with one row per user/strategy.
        curves: nested dict curves[target_mode][strategy] -> array [n_users, n_steps+1].
    """
    rows: list[dict] = []
    curve_store: dict[str, dict[str, list[np.ndarray]]] = {}
    strategies = list(strategies)

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)

        curve_store.setdefault(target_mode, {strategy: [] for strategy in strategies})

        for strategy in strategies:
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
            rows.append(make_summary_row(user_id, target_mode, strategy, result, z_target))

    sessions = pd.DataFrame(rows)
    curves: dict[str, dict[str, np.ndarray]] = {}
    for target_mode, by_strategy in curve_store.items():
        curves[target_mode] = {
            strategy: np.asarray(items, dtype=np.float64)
            for strategy, items in by_strategy.items()
        }
    return sessions, curves


def summarize_by_strategy(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sessions by target_mode and strategy."""
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
            std_improvement_pct=("improvement_pct", "std"),
        )
        .reset_index()
        .sort_values(["target_mode", "mean_final_distance"])
    )


def win_rates_vs_baseline(
    sessions: pd.DataFrame,
    baseline: str = "random",
) -> pd.DataFrame:
    """Compute per-target-mode win rates against a baseline strategy."""
    rows: list[dict] = []
    for target_mode, group in sessions.groupby("target_mode"):
        pivot_final = group.pivot(index="user_id", columns="strategy", values="final_distance")
        pivot_best = group.pivot(index="user_id", columns="strategy", values="best_distance")
        if baseline not in pivot_final.columns:
            continue
        for strategy in pivot_final.columns:
            if strategy == baseline:
                continue
            rows.append({
                "target_mode": target_mode,
                "strategy": strategy,
                "baseline": baseline,
                "win_rate_final_distance": float((pivot_final[strategy] < pivot_final[baseline]).mean()),
                "win_rate_best_distance": float((pivot_best[strategy] < pivot_best[baseline]).mean()),
            })
    return pd.DataFrame(rows)


def save_batch_outputs(
    sessions: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    output_dir: str | Path,
    prefix: str,
) -> None:
    """Save batch result tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    strategy_summary.to_csv(output_dir / f"{prefix}_strategy_summary.csv", index=False)
    win_rates.to_csv(output_dir / f"{prefix}_win_rates.csv", index=False)
