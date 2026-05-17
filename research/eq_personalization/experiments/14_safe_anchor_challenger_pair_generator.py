from __future__ import annotations

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from personalization.batch_eval import win_rates_vs_baseline
from personalization.safe_anchor_challenger import (
    V5_DISPLAY_NAMES,
    V5_STRATEGY_NAME,
    run_v5_comparison_on_dataset,
    save_v5_outputs,
    source_usage_table,
    summarize_v5_sessions,
)

DATASET_PATH = ROOT / "outputs" / "datasets" / "synthetic_users_v21.csv"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"
METRICS_DIR = ROOT / "outputs" / "metrics"

TARGET_MODE_DISPLAY_NAMES = {
    "random8d": "Random 8D",
    "semantic4d": "Semantic 4D",
    "semantic6d": "Semantic 6D",
    "archetype8d": "Archetype 8D",
}

TARGET_MODE_ORDER = ["random8d", "semantic4d", "semantic6d", "archetype8d"]
STRATEGY_ORDER = ["semantic_active_v21", "candidate_pool_active", V5_STRATEGY_NAME]


def _display_strategy(strategy: str) -> str:
    return V5_DISPLAY_NAMES.get(strategy, strategy)


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_mode_display"] = out["target_mode"].map(TARGET_MODE_DISPLAY_NAMES).fillna(out["target_mode"])
    out["strategy_display"] = out["strategy"].map(V5_DISPLAY_NAMES).fillna(out["strategy"])
    return out


def plot_mean_final_distance(summary: pd.DataFrame) -> None:
    df = add_display_columns(summary)
    df["target_mode"] = pd.Categorical(df["target_mode"], TARGET_MODE_ORDER, ordered=True)
    df["strategy"] = pd.Categorical(df["strategy"], STRATEGY_ORDER, ordered=True)
    df = df.sort_values(["target_mode", "strategy"])
    pivot = df.pivot(index="target_mode_display", columns="strategy_display", values="mean_final_distance")
    pivot = pivot[[V5_DISPLAY_NAMES[s] for s in STRATEGY_ORDER if V5_DISPLAY_NAMES[s] in pivot.columns]]

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Mean final distance: V5 Safe anchor-challenger", fontsize=15, fontweight="bold")
    ax.set_xlabel("Target mode")
    ax.set_ylabel("Mean final distance in weighted 8D")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Pair Generator", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=0)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v5_mean_final_distance_by_target.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_archetype(curves: dict[str, dict[str, np.ndarray]]) -> None:
    mode = "archetype8d"
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    for strategy in STRATEGY_ORDER:
        arr = curves.get(mode, {}).get(strategy)
        if arr is None or len(arr) == 0:
            continue
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, marker="o", linewidth=2, label=_display_strategy(strategy))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.12)

    ax.set_title("Средняя сходимость A/B-сессии: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Шаг A/B-сессии")
    ax.set_ylabel("Расстояние до скрытого target")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v5_convergence_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_v5_pair_source_usage(source_usage: pd.DataFrame) -> None:
    df = source_usage[source_usage["target_mode"] == "archetype8d"].copy()
    if df.empty:
        return
    pivot = df.pivot_table(index="phase", columns="pair_source_group", values="share", aggfunc="sum").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Источники A/B-вопросов в V5: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Фаза сессии")
    ax.set_ylabel("Доля вопросов")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Источник", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v5_pair_source_usage_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_v5_pair_quality(steps: pd.DataFrame) -> None:
    df = steps[steps["target_mode"] == "archetype8d"].copy()
    if df.empty:
        return
    metrics = ["pair_distance", "audibility_score", "acceptability_score", "midrange_disturbance_penalty"]
    grouped = df.groupby("phase")[metrics].mean().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    x = np.arange(len(grouped))
    width = 0.20
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, grouped[metric], width=width, label=metric)
    ax.set_title("Диагностика качества A/B-пар V5: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Фаза сессии")
    ax.set_ylabel("Среднее значение")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["phase"], rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v5_pair_quality_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _sample_dataset(dataset: pd.DataFrame, sample_per_mode: int | None, seed: int = 42) -> pd.DataFrame:
    if sample_per_mode is None or int(sample_per_mode) <= 0:
        return dataset.copy()
    return (
        dataset
        .groupby("target_mode", group_keys=False)
        .apply(lambda g: g.sample(n=min(int(sample_per_mode), len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V5 Safe anchor-challenger Pair Generator experiment.")
    parser.add_argument("--sample-per-mode", type=int, default=50, help="Users per target_mode. Use 0 for full dataset.")
    parser.add_argument("--n-steps", type=int, default=25)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    dataset = pd.read_csv(DATASET_PATH)
    dataset = _sample_dataset(dataset, sample_per_mode=args.sample_per_mode)
    print(f"Dataset rows used: {len(dataset)}")

    sessions, steps, curves = run_v5_comparison_on_dataset(
        dataset,
        baseline_strategies=("semantic_active_v21", "candidate_pool_active"),
        n_steps=int(args.n_steps),
        step_scale=0.6,
        lr=0.25,
        clip_value=2.0,
    )
    summary = summarize_v5_sessions(sessions)
    win_rates = win_rates_vs_baseline(sessions, baseline="semantic_active_v21")
    source_usage = source_usage_table(steps)

    save_v5_outputs(sessions, steps, summary, win_rates, source_usage, METRICS_DIR, prefix="article_v5_safe_anchor_challenger")
    summary.to_csv(TABLE_DIR / "article_v5_safe_anchor_challenger_summary.csv", index=False)
    win_rates.to_csv(TABLE_DIR / "article_v5_safe_anchor_challenger_win_rates.csv", index=False)
    source_usage.to_csv(TABLE_DIR / "article_v5_safe_anchor_challenger_source_usage.csv", index=False)

    plot_mean_final_distance(summary)
    plot_convergence_archetype(curves)
    plot_v5_pair_source_usage(source_usage)
    plot_v5_pair_quality(steps)

    print("Saved V5 results to:", METRICS_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
