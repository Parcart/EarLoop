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
from personalization.safe_anchor_challenger_v51 import V51_STRATEGY_NAME
from personalization.safe_anchor_challenger_v52 import V52_STRATEGY_NAME
from personalization.safe_anchor_challenger_v53 import V53_STRATEGY_NAME
from personalization.soft_stop_monitor_v54 import (
    V54_DISPLAY_NAMES,
    V54_STRATEGY_NAME,
    run_v54_comparison_on_dataset,
    save_v54_outputs,
    source_usage_table_v54,
    summarize_v54_sessions,
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
STRATEGY_ORDER = [
    "semantic_active_v21",
    "candidate_pool_active",
    V51_STRATEGY_NAME,
    V52_STRATEGY_NAME,
    V53_STRATEGY_NAME,
    V54_STRATEGY_NAME,
]
SOFT_STOP_COMPARISON_ORDER = ["semantic_active_v21", V52_STRATEGY_NAME, V53_STRATEGY_NAME, V54_STRATEGY_NAME]


def _display_strategy(strategy: str) -> str:
    return V54_DISPLAY_NAMES.get(strategy, strategy)


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_mode_display"] = out["target_mode"].map(TARGET_MODE_DISPLAY_NAMES).fillna(out["target_mode"])
    out["strategy_display"] = out["strategy"].map(V54_DISPLAY_NAMES).fillna(out["strategy"])
    return out


def _sample_dataset(dataset: pd.DataFrame, sample_per_mode: int | None, seed: int = 42) -> pd.DataFrame:
    if sample_per_mode is None or int(sample_per_mode) <= 0:
        return dataset.copy()
    sampled_parts = []
    for _, group in dataset.groupby("target_mode"):
        sampled_parts.append(group.sample(n=min(int(sample_per_mode), len(group)), random_state=seed))
    return pd.concat(sampled_parts, axis=0).reset_index(drop=True)


def plot_mean_final_distance(summary: pd.DataFrame) -> None:
    df = add_display_columns(summary)
    df = df[df["strategy"].isin(STRATEGY_ORDER)].copy()
    df["target_mode"] = pd.Categorical(df["target_mode"], TARGET_MODE_ORDER, ordered=True)
    df["strategy"] = pd.Categorical(df["strategy"], STRATEGY_ORDER, ordered=True)
    df = df.sort_values(["target_mode", "strategy"])
    pivot = df.pivot(index="target_mode_display", columns="strategy_display", values="mean_final_distance")
    cols = [V54_DISPLAY_NAMES[s] for s in STRATEGY_ORDER if V54_DISPLAY_NAMES.get(s) in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Mean final distance: V5.4 soft-stop monitor", fontsize=15, fontweight="bold")
    ax.set_xlabel("Target mode")
    ax.set_ylabel("Mean final distance in weighted 8D")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Pair Generator", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=0)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v54_mean_final_distance_by_target.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_archetype(curves: dict[str, dict[str, np.ndarray]], summary: pd.DataFrame) -> None:
    mode = "archetype8d"
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    for strategy in ["semantic_active_v21", V52_STRATEGY_NAME, V53_STRATEGY_NAME, V54_STRATEGY_NAME]:
        arr = curves.get(mode, {}).get(strategy)
        if arr is None or len(arr) == 0:
            continue
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, marker="o", linewidth=2, label=_display_strategy(strategy))
        ax.fill_between(steps, mean - std, mean + std, alpha=0.10)

    archetype_summary = summary[(summary["target_mode"] == "archetype8d") & (summary["strategy"] == V54_STRATEGY_NAME)]
    if not archetype_summary.empty:
        stop_step = float(archetype_summary["mean_recommended_stop_step"].iloc[0])
        if np.isfinite(stop_step):
            ax.axvline(stop_step, color="black", linestyle="--", linewidth=1.5, alpha=0.75)
            ax.text(stop_step + 0.25, ax.get_ylim()[1] * 0.92, f"soft-stop ≈ {stop_step:.1f}", fontsize=10)

    ax.set_title("Средняя сходимость A/B-сессии: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Шаг A/B-сессии")
    ax.set_ylabel("Расстояние до скрытого target")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v54_convergence_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_archetype_final_distance(summary: pd.DataFrame) -> None:
    df = add_display_columns(summary)
    df = df[(df["target_mode"] == "archetype8d") & (df["strategy"].isin(STRATEGY_ORDER))].copy()
    df["strategy"] = pd.Categorical(df["strategy"], STRATEGY_ORDER, ordered=True)
    df = df.sort_values("strategy")
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    bars = ax.bar(df["strategy_display"], df["mean_final_distance"])
    ax.set_title("V5.4 soft-stop monitor на Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Pair Generator")
    ax.set_ylabel("Mean final distance in weighted 8D")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v54_archetype8d_final_distance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_soft_stop_tradeoff(summary: pd.DataFrame) -> None:
    df = add_display_columns(summary)
    df = df[(df["target_mode"] == "archetype8d") & (df["strategy"].isin(SOFT_STOP_COMPARISON_ORDER))].copy()
    df["strategy"] = pd.Categorical(df["strategy"], SOFT_STOP_COMPARISON_ORDER, ordered=True)
    df = df.sort_values("strategy")

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    bars = ax.bar(df["strategy_display"], df["mean_used_steps"])
    ax.set_title("Использованные A/B-шаги: hard stop vs soft-stop marker", fontsize=15, fontweight="bold")
    ax.set_xlabel("Стратегия")
    ax.set_ylabel("A/B-шаги")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.bar_label(bars, fmt="%.1f", padding=3)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v54_used_steps_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    v54 = summary[(summary["target_mode"] == "archetype8d") & (summary["strategy"] == V54_STRATEGY_NAME)]
    if not v54.empty:
        metrics = [
            ("mean_recommended_stop_step", "Recommended stop step"),
            ("mean_distance_at_recommended_stop", "Distance at soft stop"),
            ("mean_final_distance", "Distance at 25 steps"),
            ("mean_extra_gain_after_stop", "Extra gain after stop"),
            ("mean_steps_saved_if_stop", "Steps saved if stop"),
            ("mean_retained_quality_at_stop_pct", "Quality retained, %"),
        ]
        values = [float(v54[m].iloc[0]) for m, _ in metrics]
        labels = [label for _, label in metrics]
        fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
        bars = ax.bar(labels, values)
        ax.set_title("Soft-stop диагностика V5.4: Archetype 8D", fontsize=15, fontweight="bold")
        ax.set_ylabel("Значение")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.bar_label(bars, fmt="%.2f", padding=3)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "article_v54_soft_stop_diagnostics_archetype8d.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_v54_pair_quality(steps: pd.DataFrame) -> None:
    df = steps[(steps["target_mode"] == "archetype8d") & (steps["strategy"] == V54_STRATEGY_NAME)].copy()
    if df.empty:
        return
    metrics = ["pair_distance", "audibility_score", "acceptability_score", "midrange_disturbance_penalty", "applied_lr"]
    grouped = df.groupby("phase")[metrics].mean().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    x = np.arange(len(grouped))
    width = 0.16
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 2) * width, grouped[metric], width=width, label=metric)
    ax.set_title("Диагностика V5.4 semantic вопросов: Archetype 8D", fontsize=15, fontweight="bold")
    ax.set_xlabel("Фаза сессии")
    ax.set_ylabel("Среднее значение")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["phase"], rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "article_v54_pair_quality_archetype8d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V5.4 soft-stop monitor experiment.")
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

    sessions, steps, curves = run_v54_comparison_on_dataset(
        dataset,
        baseline_strategies=("semantic_active_v21", "candidate_pool_active"),
        include_v51=True,
        include_v52=True,
        include_v53=True,
        n_steps=int(args.n_steps),
        step_scale=0.6,
        lr=0.25,
        clip_value=2.0,
    )
    summary = summarize_v54_sessions(sessions)
    win_rates = win_rates_vs_baseline(sessions, baseline="semantic_active_v21")
    source_usage = source_usage_table_v54(steps)

    save_v54_outputs(sessions, steps, summary, win_rates, source_usage, METRICS_DIR)
    summary.to_csv(TABLE_DIR / "article_v54_soft_stop_monitor_summary.csv", index=False)
    win_rates.to_csv(TABLE_DIR / "article_v54_soft_stop_monitor_win_rates.csv", index=False)
    source_usage.to_csv(TABLE_DIR / "article_v54_soft_stop_monitor_source_usage.csv", index=False)

    plot_mean_final_distance(summary)
    plot_convergence_archetype(curves, summary)
    plot_archetype_final_distance(summary)
    plot_soft_stop_tradeoff(summary)
    plot_v54_pair_quality(steps)

    print("Saved V5.4 results to:", METRICS_DIR)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
