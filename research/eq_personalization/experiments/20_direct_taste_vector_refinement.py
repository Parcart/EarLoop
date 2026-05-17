from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from personalization.batch_eval import win_rates_vs_baseline
from personalization.direct_taste_vector_v56 import (
    V56_BLEND_STRATEGY_NAME,
    V56_DISPLAY_NAMES,
    V56_PM_STRATEGY_NAME,
    V56_TRUST_STRATEGY_NAME,
    phase_step_budget_table_v56,
    post_marker_source_usage_table_v56,
    direct_selection_table_v56,
    direct_vector_diagnostics_table_v56,
    run_v56_comparison_on_dataset,
    save_v56_outputs,
    source_usage_table_v56,
    summarize_v56_sessions,
)
from personalization.phase_aware_full_budget_v55 import V55_MIXED_STRATEGY_NAME, V55_ZONE_STRATEGY_NAME
from personalization.plotting import use_article_style
from personalization.soft_stop_monitor_v54 import V54_STRATEGY_NAME


TARGET_MODE_DISPLAY = {
    "random8d": "Random 8D",
    "semantic4d": "Semantic 4D",
    "semantic6d": "Semantic 6D",
    "archetype8d": "Archetype 8D",
}

ARTICLE_STRATEGIES = [
    "semantic_active_v21",
    V54_STRATEGY_NAME,
    V55_MIXED_STRATEGY_NAME,
    V56_BLEND_STRATEGY_NAME,
    V56_PM_STRATEGY_NAME,
    V56_TRUST_STRATEGY_NAME,
    "candidate_pool_active",
]

DIRECT_STRATEGIES = [V56_BLEND_STRATEGY_NAME, V56_PM_STRATEGY_NAME, V56_TRUST_STRATEGY_NAME]
POST_MARKER_STRATEGIES = [V54_STRATEGY_NAME, V55_MIXED_STRATEGY_NAME, *DIRECT_STRATEGIES]


def sample_dataset(dataset: pd.DataFrame, sample_per_mode: int | None) -> pd.DataFrame:
    if sample_per_mode is None or int(sample_per_mode) <= 0:
        return dataset.copy().reset_index(drop=True)
    parts = []
    for _, group in dataset.groupby("target_mode"):
        parts.append(group.sample(n=min(int(sample_per_mode), len(group)), random_state=42))
    return pd.concat(parts, axis=0).reset_index(drop=True)


def _strategy_label(strategy: str) -> str:
    return V56_DISPLAY_NAMES.get(strategy, strategy)


def plot_final_distance_by_target(summary: pd.DataFrame, fig_dir: Path, prefix: str) -> None:
    df = summary[summary["strategy"].isin(ARTICLE_STRATEGIES)].copy()
    df["target_label"] = df["target_mode"].map(TARGET_MODE_DISPLAY).fillna(df["target_mode"])
    df["strategy_label"] = df["strategy"].map(_strategy_label)
    order_targets = ["Random 8D", "Semantic 4D", "Semantic 6D", "Archetype 8D"]
    order_strategies = [_strategy_label(s) for s in ARTICLE_STRATEGIES]
    pivot = df.pivot(index="target_label", columns="strategy_label", values="mean_final_distance")
    pivot = pivot.reindex(order_targets).dropna(how="all")
    pivot = pivot[[c for c in order_strategies if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(13, 5.4), facecolor="white")
    pivot.plot(kind="bar", ax=ax, width=0.84)
    ax.set_title("V5.6: direct taste-vector refinement по режимам target", fontsize=15, fontweight="bold")
    ax.set_xlabel("Режим генерации target")
    ax.set_ylabel("Mean final distance в weighted 8D")
    ax.grid(True, axis="y", alpha=0.30, linestyle="--")
    ax.legend(title="Pair Generator", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_mean_final_distance_by_target.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_archetype(curves: dict, summary: pd.DataFrame, fig_dir: Path, prefix: str) -> None:
    mode = "archetype8d"
    if mode not in curves:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.5), facecolor="white")
    for strategy in ARTICLE_STRATEGIES:
        arr = curves.get(mode, {}).get(strategy)
        if arr is None or len(arr) == 0:
            continue
        mean_curve = np.nanmean(arr, axis=0)
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, marker="o", linewidth=2, label=_strategy_label(strategy))

    marker_df = summary[(summary["target_mode"].astype(str).eq(mode)) & (summary["strategy"].astype(str).eq(V54_STRATEGY_NAME))]
    if not marker_df.empty and "mean_recommended_stop_step" in marker_df.columns:
        marker_step = float(marker_df["mean_recommended_stop_step"].iloc[0])
        if np.isfinite(marker_step):
            ax.axvline(marker_step, linestyle="--", linewidth=2, alpha=0.75, label=f"soft-stop marker ≈ {marker_step:.1f}")
            ax.axvspan(marker_step, 25, alpha=0.06)

    ax.set_title("Средняя сходимость V5.6: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Шаг A/B-сессии")
    ax.set_ylabel("Distance to target")
    ax.grid(True, alpha=0.30, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_convergence_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stop_vs_final_distance(summary: pd.DataFrame, fig_dir: Path, prefix: str) -> None:
    df = summary[(summary["target_mode"].astype(str).eq("archetype8d")) & (summary["strategy"].isin(POST_MARKER_STRATEGIES))].copy()
    if df.empty:
        return
    order = [s for s in POST_MARKER_STRATEGIES if s in set(df["strategy"])]
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)
    df = df.sort_values("strategy")
    labels = [_strategy_label(str(s)) for s in df["strategy"]]
    at_stop = df["mean_distance_at_recommended_stop"].to_numpy(dtype=float)
    final = df["mean_final_distance"].to_numpy(dtype=float)
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.4), facecolor="white")
    ax.bar(x - width / 2, at_stop, width=width, label="На soft-stop marker")
    ax.bar(x + width / 2, final, width=width, label="Финал после 25 шагов")
    ax.set_title("Что даёт direct refinement после soft-stop marker: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Стратегия")
    ax.set_ylabel("Mean distance to target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.30, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
    for i, (a, f) in enumerate(zip(at_stop, final)):
        if np.isfinite(a):
            ax.text(i - width / 2, a + 0.012, f"{a:.3f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(f):
            ax.text(i + width / 2, f + 0.012, f"{f:.3f}", ha="center", va="bottom", fontsize=9)
        if np.isfinite(a) and np.isfinite(f):
            ax.text(i, max(a, f) + 0.055, f"gain {a-f:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_stop_vs_final_distance_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_post_marker_source_usage(post_usage: pd.DataFrame, fig_dir: Path, prefix: str) -> None:
    df = post_usage[post_usage["strategy"].isin(POST_MARKER_STRATEGIES)].copy() if not post_usage.empty else pd.DataFrame()
    if df.empty:
        return
    df["strategy_label"] = df["strategy"].map(_strategy_label)
    order = [_strategy_label(s) for s in POST_MARKER_STRATEGIES]
    pivot = df.pivot_table(index="strategy_label", columns="pair_source_group", values="share", aggfunc="sum").fillna(0.0)
    pivot = pivot.reindex([x for x in order if x in pivot.index])

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor="white")
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.72)
    ax.set_title("Какие вопросы задаются после soft-stop marker: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Стратегия")
    ax.set_ylabel("Доля post-marker вопросов")
    ax.grid(True, axis="y", alpha=0.30, linestyle="--")
    ax.legend(title="Источник / зона", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_post_marker_source_usage_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def direct_selection_table(steps: pd.DataFrame, target_mode: str = "archetype8d") -> pd.DataFrame:
    if steps.empty:
        return pd.DataFrame()
    df = steps[(steps["target_mode"].astype(str).eq(target_mode)) & (steps["strategy"].isin(DIRECT_STRATEGIES))].copy()
    if df.empty:
        return pd.DataFrame()
    df = df[df.get("soft_stop_marker", "").astype(str).eq("continued_after_recommendation")]
    if df.empty:
        return pd.DataFrame()
    df["selected_role"] = df["selected_role"].fillna("unknown").astype(str)
    counts = df.groupby(["target_mode", "strategy", "selected_role"]).size().rename("count").reset_index()
    totals = counts.groupby(["target_mode", "strategy"])["count"].transform("sum")
    counts["share"] = counts["count"] / totals.replace(0, np.nan)
    return counts


def plot_direct_selection(selection: pd.DataFrame, fig_dir: Path, prefix: str) -> None:
    if selection.empty:
        return
    df = selection.copy()
    df["strategy_label"] = df["strategy"].map(_strategy_label)
    order = [_strategy_label(s) for s in DIRECT_STRATEGIES]
    pivot = df.pivot_table(index="strategy_label", columns="selected_role", values="share", aggfunc="sum").fillna(0.0)
    pivot = pivot.reindex([x for x in order if x in pivot.index])
    fig, ax = plt.subplots(figsize=(10.5, 5.0), facecolor="white")
    pivot.plot(kind="bar", stacked=True, ax=ax, width=0.72)
    ax.set_title("Выбор anchor/direct после marker: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Стратегия")
    ax.set_ylabel("Доля post-marker вопросов")
    ax.grid(True, axis="y", alpha=0.30, linestyle="--")
    ax.legend(title="Что выбрал synthetic user", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_direct_selection_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)




def plot_direct_vector_cosines(diagnostics: pd.DataFrame, fig_dir: Path, prefix: str) -> None:
    if diagnostics.empty:
        return
    df = diagnostics.copy()
    df["strategy_label"] = df["strategy"].map(_strategy_label)
    order = [_strategy_label(s) for s in DIRECT_STRATEGIES]
    df = df.set_index("strategy_label").reindex([x for x in order if x in set(df["strategy_label"])]).reset_index()
    metrics = {
        "PM vs blend delta": "mean_delta_pm_blend_cosine",
        "direction vs blend delta": "mean_direction_cosine_to_blend_delta",
        "direction vs PM delta": "mean_direction_cosine_to_pm_delta",
    }
    plot_df = df[["strategy_label", *metrics.values()]].copy()
    plot_df = plot_df.set_index("strategy_label").rename(columns={v: k for k, v in metrics.items()})

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="white")
    plot_df.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Диагностика V5.6: совпадают ли direct-направления", fontsize=15, fontweight="bold")
    ax.set_xlabel("Стратегия")
    ax.set_ylabel("Mean cosine")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, axis="y", alpha=0.30, linestyle="--")
    ax.legend(title="Косинус", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_direct_vector_cosines_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_direct_vector_norms(diagnostics: pd.DataFrame, fig_dir: Path, prefix: str) -> None:
    if diagnostics.empty:
        return
    df = diagnostics.copy()
    df["strategy_label"] = df["strategy"].map(_strategy_label)
    order = [_strategy_label(s) for s in DIRECT_STRATEGIES]
    df = df.set_index("strategy_label").reindex([x for x in order if x in set(df["strategy_label"])]).reset_index()
    metrics = {
        "||PM - z||": "mean_delta_pm_norm",
        "||blend - z||": "mean_delta_blend_norm",
        "pair distance": "mean_pair_distance",
        "direct selected rate": "direct_selected_rate",
    }
    plot_df = df[["strategy_label", *metrics.values()]].copy()
    plot_df = plot_df.set_index("strategy_label").rename(columns={v: k for k, v in metrics.items()})

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="white")
    plot_df.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Диагностика V5.6: масштаб direct-кандидата", fontsize=15, fontweight="bold")
    ax.set_xlabel("Стратегия")
    ax.set_ylabel("Среднее значение")
    ax.grid(True, axis="y", alpha=0.30, linestyle="--")
    ax.legend(title="Метрика", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_direct_vector_norms_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_step_budget(step_budget: pd.DataFrame, fig_dir: Path, prefix: str) -> None:
    df = step_budget[step_budget["strategy"].isin(POST_MARKER_STRATEGIES)].copy()
    if df.empty:
        return
    order = [s for s in POST_MARKER_STRATEGIES if s in set(df["strategy"])]
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)
    df = df.sort_values("strategy")
    labels = [_strategy_label(str(s)) for s in df["strategy"]]
    before = df["mean_steps_before_recommendation"].to_numpy(dtype=float)
    after = df["mean_steps_after_recommendation"].to_numpy(dtype=float)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 5.2), facecolor="white")
    ax.bar(x, before, label="До soft-stop marker")
    ax.bar(x, after, bottom=before, label="После marker, пользователь продолжил")
    ax.set_title("Среднее число шагов до/после soft-stop marker: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Стратегия")
    ax.set_ylabel("Количество A/B-шагов")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 26.5)
    ax.grid(True, axis="y", alpha=0.30, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i, b / 2, f"{b:.1f}", ha="center", va="center", fontsize=9)
        if a > 0.2:
            ax.text(i, b + a / 2, f"{a:.1f}", ha="center", va="center", fontsize=9)
        ax.text(i, b + a + 0.30, f"{b+a:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_steps_before_after_soft_stop_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="outputs/datasets/synthetic_users_v21.csv")
    parser.add_argument("--sample-per-mode", type=int, default=50, help="0 means full dataset")
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--prefix", type=str, default="article_v56_direct_taste_vector_refinement")
    args = parser.parse_args()

    use_article_style()
    dataset_path = PROJECT_ROOT / args.dataset
    dataset = pd.read_csv(dataset_path)
    dataset_eval = sample_dataset(dataset, None if args.sample_per_mode == 0 else args.sample_per_mode)

    sessions, steps, curves = run_v56_comparison_on_dataset(
        dataset=dataset_eval,
        n_steps=args.n_steps,
    )
    summary = summarize_v56_sessions(sessions)
    win_rates = win_rates_vs_baseline(sessions, baseline="semantic_active_v21")
    source_usage = source_usage_table_v56(steps)
    post_marker_source_usage = post_marker_source_usage_table_v56(steps, target_mode="archetype8d")
    step_budget = phase_step_budget_table_v56(sessions, target_mode="archetype8d")
    direct_selection = direct_selection_table_v56(steps, target_mode="archetype8d")
    direct_diagnostics = direct_vector_diagnostics_table_v56(steps, target_mode="archetype8d")

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    save_v56_outputs(
        sessions, steps, summary, win_rates, source_usage, step_budget, metrics_dir,
        prefix=args.prefix, post_marker_source_usage=post_marker_source_usage,
    )
    step_budget.to_csv(table_dir / f"{args.prefix}_step_budget_archetype8d.csv", index=False)
    post_marker_source_usage.to_csv(table_dir / f"{args.prefix}_post_marker_source_usage_archetype8d.csv", index=False)
    direct_selection.to_csv(table_dir / f"{args.prefix}_direct_selection_archetype8d.csv", index=False)
    direct_diagnostics.to_csv(table_dir / f"{args.prefix}_direct_vector_diagnostics_archetype8d.csv", index=False)

    plot_final_distance_by_target(summary, fig_dir, args.prefix)
    plot_convergence_archetype(curves, summary, fig_dir, args.prefix)
    plot_phase_step_budget(step_budget, fig_dir, args.prefix)
    plot_stop_vs_final_distance(summary, fig_dir, args.prefix)
    plot_post_marker_source_usage(post_marker_source_usage, fig_dir, args.prefix)
    plot_direct_selection(direct_selection, fig_dir, args.prefix)
    plot_direct_vector_cosines(direct_diagnostics, fig_dir, args.prefix)
    plot_direct_vector_norms(direct_diagnostics, fig_dir, args.prefix)

    print("Saved metrics to", metrics_dir)
    print("Saved figures to", fig_dir)
    print("\nTarget-mode summary:")
    print(summary[summary["strategy"].isin(ARTICLE_STRATEGIES)].to_string(index=False))
    print("\nArchetype 8D direct selection:")
    print(direct_selection.to_string(index=False))
    print("\nArchetype 8D direct vector diagnostics:")
    print(direct_diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
