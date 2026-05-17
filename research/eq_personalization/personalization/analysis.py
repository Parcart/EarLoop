from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_STRATEGY_ORDER = [
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

STRATEGY_DISPLAY_NAMES_RU = {
    "random": "Random direction",
    "uncertainty_axis": "Uncertainty axis",
    "semantic_control": "Semantic 4D",
    "semantic_control_v21": "Semantic 6D v2.1",
    "semantic_active_v21": "Semantic active v3",
    "candidate_pool_active": "Candidate pool active",
    "adaptive_router_v32": "Adaptive router v3.2",
    "hybrid": "Hybrid 4D",
    "hybrid_v21": "Hybrid 6D v2.1",
    "hybrid_active_v21": "Hybrid active v3",
}

GROUP_DISPLAY_NAMES_RU = {
    "random8d": "Random 8D",
    "semantic4d": "Semantic 4D",
    "semantic6d": "Semantic 6D",
    "archetype8d": "Archetype 8D",
    "mild": "Mild",
    "moderate": "Moderate",
    "strong": "Strong",
    "extreme": "Extreme",
}


def display_strategy_name(strategy: str) -> str:
    return STRATEGY_DISPLAY_NAMES_RU.get(strategy, strategy)


def display_group_name(value) -> str:
    return GROUP_DISPLAY_NAMES_RU.get(str(value), str(value))


def merge_sessions_with_user_metadata(
    sessions: pd.DataFrame,
    dataset: pd.DataFrame,
    metadata_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Attach synthetic-user metadata to per-session strategy results.

    Args:
        sessions: long-form results with user_id, target_mode, strategy, distances.
        dataset: synthetic user dataset with target/user metadata.
        metadata_cols: optional list of metadata columns to copy from dataset.

    Returns:
        DataFrame with one row per user/strategy and metadata columns attached.
    """
    if metadata_cols is None:
        metadata_cols = [
            "main_archetype",
            "secondary_archetype",
            "is_extreme_archetype",
            "intensity_label",
            "intensity_value",
        ]

    keep_cols = ["user_id", "target_mode"] + [
        col for col in metadata_cols if col in dataset.columns
    ]

    meta = dataset[keep_cols].drop_duplicates(["user_id", "target_mode"])
    return sessions.merge(meta, on=["user_id", "target_mode"], how="left")


def summarize_by_group(
    sessions: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate pair-generator results by user group and strategy.

    Lower distance metrics are better. Higher improvement_pct is better.
    """
    return (
        sessions
        .groupby(group_cols + ["strategy"], dropna=False)
        .agg(
            users=("user_id", "nunique"),
            mean_initial_distance=("initial_distance", "mean"),
            mean_final_distance=("final_distance", "mean"),
            std_final_distance=("final_distance", "std"),
            mean_best_distance=("best_distance", "mean"),
            mean_mean_distance=("mean_distance", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            std_improvement_pct=("improvement_pct", "std"),
        )
        .reset_index()
        .sort_values(group_cols + ["mean_final_distance"])
    )


def winners_by_group(
    group_summary: pd.DataFrame,
    group_cols: list[str],
    metric: str = "mean_final_distance",
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Return the best strategy for each group by a selected summary metric."""
    summary = group_summary.copy()
    summary = summary.sort_values(metric, ascending=lower_is_better)
    winners = summary.groupby(group_cols, dropna=False).head(1).copy()
    winners = winners[group_cols + ["strategy", "users", metric]]
    winners = winners.rename(
        columns={
            "strategy": "winner_strategy",
            metric: f"winner_{metric}",
        }
    )
    return winners.reset_index(drop=True)


def pivot_group_metric(
    group_summary: pd.DataFrame,
    group_col: str,
    metric: str = "mean_final_distance",
    strategy_order: list[str] | None = None,
) -> pd.DataFrame:
    """Create a group x strategy table for a selected metric."""
    strategy_order = DEFAULT_STRATEGY_ORDER if strategy_order is None else strategy_order
    table = group_summary.pivot(index=group_col, columns="strategy", values=metric)
    cols = [strategy for strategy in strategy_order if strategy in table.columns]
    return table[cols]


def win_rates_vs_baseline_by_group(
    sessions: pd.DataFrame,
    group_cols: list[str],
    baseline: str = "random",
    metric: str = "final_distance",
    strategy_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute strategy win-rate against a baseline inside each user group.

    For distance metrics, smaller values are treated as wins.
    """
    strategy_order = DEFAULT_STRATEGY_ORDER if strategy_order is None else strategy_order

    pivot = (
        sessions
        .pivot_table(
            index=group_cols + ["user_id"],
            columns="strategy",
            values=metric,
            aggfunc="mean",
        )
        .reset_index()
    )

    rows: list[dict] = []
    for group_values, group_df in pivot.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        if baseline not in group_df.columns:
            continue

        baseline_values = group_df[baseline]
        for strategy in strategy_order:
            if strategy == baseline or strategy not in group_df.columns:
                continue
            strategy_values = group_df[strategy]
            row = {col: val for col, val in zip(group_cols, group_values)}
            row.update({
                "strategy": strategy,
                "baseline": baseline,
                f"win_rate_{metric}": float((strategy_values < baseline_values).mean()),
                "n_users": int(len(group_df)),
            })
            rows.append(row)

    return pd.DataFrame(rows)


def plot_group_metric_bars(
    group_summary: pd.DataFrame,
    group_col: str,
    metric: str = "mean_final_distance",
    strategy_order: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (11, 5),
):
    """Plot grouped bars for a selected metric by group and strategy."""
    table = pivot_group_metric(
        group_summary=group_summary,
        group_col=group_col,
        metric=metric,
        strategy_order=strategy_order,
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    table.plot(kind="bar", ax=ax)

    ax.set_title(title or f"{metric} по группам: {group_col}")
    ax.set_xlabel("")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", color="gray")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [display_strategy_name(label) for label in labels],
        title="Стратегия",
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
    )
    ax.tick_params(axis="x", labelrotation=30, colors="black")
    ax.tick_params(axis="y", colors="black")

    for label in ax.get_xticklabels():
        label.set_ha("right")

    for spine in ax.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    return fig, ax


def plot_win_rate_bars(
    win_rates: pd.DataFrame,
    group_col: str,
    metric_col: str = "win_rate_final_distance",
    strategy_order: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (11, 5),
):
    """Plot win-rate against baseline by group."""
    strategy_order = DEFAULT_STRATEGY_ORDER if strategy_order is None else strategy_order

    table = win_rates.pivot(index=group_col, columns="strategy", values=metric_col)
    cols = [strategy for strategy in strategy_order if strategy in table.columns]
    table = table[cols]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    table.plot(kind="bar", ax=ax)

    ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_title(title or f"Win-rate по группам: {group_col}")
    ax.set_xlabel("")
    ax.set_ylabel("Доля побед над random")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", color="gray")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        [display_strategy_name(label) for label in labels],
        title="Стратегия",
        frameon=True,
        facecolor="white",
        edgecolor="0.75",
    )
    ax.tick_params(axis="x", labelrotation=30, colors="black")
    ax.tick_params(axis="y", colors="black")

    for label in ax.get_xticklabels():
        label.set_ha("right")

    for spine in ax.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    return fig, ax


def compare_two_strategies_by_group(
    group_summary: pd.DataFrame,
    group_col: str,
    old_strategy: str = "semantic_control",
    new_strategy: str = "semantic_control_v21",
    metric: str = "mean_final_distance",
) -> pd.DataFrame:
    """
    Compare two strategies inside every group using an aggregated metric.

    For distance metrics lower values are better, so positive improvement_abs means
    new_strategy improved over old_strategy.
    """
    table = group_summary.pivot(index=group_col, columns="strategy", values=metric)
    if old_strategy not in table.columns or new_strategy not in table.columns:
        missing = [s for s in [old_strategy, new_strategy] if s not in table.columns]
        raise ValueError(f"Missing strategy columns for comparison: {missing}")

    out = pd.DataFrame({
        group_col: table.index,
        f"{old_strategy}_{metric}": table[old_strategy].values,
        f"{new_strategy}_{metric}": table[new_strategy].values,
    })
    out["improvement_abs"] = out[f"{old_strategy}_{metric}"] - out[f"{new_strategy}_{metric}"]
    out["improvement_pct"] = 100.0 * out["improvement_abs"] / out[f"{old_strategy}_{metric}"]
    out["winner"] = np.where(out["improvement_abs"] > 0, new_strategy, old_strategy)
    return out.sort_values("improvement_abs", ascending=False).reset_index(drop=True)


def win_rate_between_strategies_by_group(
    sessions: pd.DataFrame,
    group_cols: list[str],
    strategy_a: str = "semantic_control_v21",
    strategy_b: str = "semantic_control",
    metric: str = "final_distance",
) -> pd.DataFrame:
    """
    Compute how often strategy_a beats strategy_b inside each group.

    For distance metrics, lower values are treated as wins.
    """
    pivot = (
        sessions
        .pivot_table(
            index=group_cols + ["user_id"],
            columns="strategy",
            values=metric,
            aggfunc="mean",
        )
        .reset_index()
    )
    if strategy_a not in pivot.columns or strategy_b not in pivot.columns:
        missing = [s for s in [strategy_a, strategy_b] if s not in pivot.columns]
        raise ValueError(f"Missing strategy columns for comparison: {missing}")

    rows = []
    for group_values, group_df in pivot.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {col: val for col, val in zip(group_cols, group_values)}
        row.update({
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            f"win_rate_{metric}": float((group_df[strategy_a] < group_df[strategy_b]).mean()),
            "n_users": int(len(group_df)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def plot_strategy_improvement_bars(
    comparison_df: pd.DataFrame,
    group_col: str,
    value_col: str = "improvement_pct",
    title: str | None = None,
    figsize: tuple[float, float] = (12, 5),
):
    """Plot v2.1 improvement over v2 by group."""
    df = comparison_df.copy().sort_values(value_col, ascending=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.bar(df[group_col].astype(str), df[value_col])
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_title(title or "Улучшение semantic basis v2.1 относительно v2")
    ax.set_xlabel("")
    ax.set_ylabel("Улучшение, %")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", color="gray")
    ax.tick_params(axis="x", labelrotation=30, colors="black")
    ax.tick_params(axis="y", colors="black")

    for label in ax.get_xticklabels():
        label.set_ha("right")
    for spine in ax.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    return fig, ax
