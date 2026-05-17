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

from personalization.phase_controller import (  # noqa: E402
    PHASE_DISPLAY_NAMES,
    PhaseControllerConfig,
    run_phase_controller_batch,
    summarize_phase_results,
)

STRATEGY_DISPLAY = {
    "semantic_active_v3_fixed": "Semantic active v3 fixed",
    "phase_aware_v1": "Phase-aware controller",
}

TARGET_MODE_DISPLAY = {
    "random8d": "Random 8D",
    "semantic4d": "Semantic 4D",
    "semantic6d": "Semantic 6D",
    "archetype8d": "Archetype 8D",
}


def ensure_dirs() -> None:
    (PROJECT_ROOT / "outputs" / "metrics").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "strategy" in out.columns:
        out["strategy_display"] = out["strategy"].map(STRATEGY_DISPLAY).fillna(out["strategy"])
    if "target_mode" in out.columns:
        out["target_mode_display"] = out["target_mode"].map(TARGET_MODE_DISPLAY).fillna(out["target_mode"])
    return out


def plot_convergence_archetype(steps: pd.DataFrame, out_path: Path | None = None) -> None:
    df = steps[steps["target_mode"] == "archetype8d"].copy()
    if df.empty:
        return
    df = add_display_columns(df)
    curve = (
        df.groupby(["strategy_display", "step"])["distance_to_target"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    for name, group in curve.groupby("strategy_display"):
        ax.plot(group["step"], group["distance_to_target"], marker="o", linewidth=2.2, label=name)
    ax.set_title("Сходимость A/B-сессии с phase-aware controller: Archetype 8D", fontsize=16, fontweight="bold")
    ax.set_xlabel("Шаг A/B-сессии")
    ax.set_ylabel("Среднее расстояние до скрытого target")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(title="Стратегия", frameon=True, facecolor="white", edgecolor="0.75")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_distance_archetype(sessions: pd.DataFrame, out_path: Path | None = None) -> None:
    df = sessions[sessions["target_mode"] == "archetype8d"].copy()
    if df.empty:
        return
    df = add_display_columns(df)
    summary = (
        df.groupby("strategy_display")["selected_blend_final_distance"]
        .mean()
        .reset_index()
        .sort_values("selected_blend_final_distance")
    )
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    bars = ax.bar(summary["strategy_display"], summary["selected_blend_final_distance"], alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_title("Final distance: fixed vs phase-aware loop — Archetype 8D", fontsize=16, fontweight="bold")
    ax.set_xlabel("Стратегия")
    ax.set_ylabel("Mean final distance в weighted 8D")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_speed(sessions: pd.DataFrame, out_path: Path | None = None) -> None:
    df = sessions[(sessions["target_mode"] == "archetype8d") & (sessions["strategy"] == "phase_aware_v1")].copy()
    if df.empty:
        return
    metrics = [
        ("direction_lock_step", "Direction locked"),
        ("ready_step", "Ready to finalize"),
        ("synthetic_threshold_step", "Distance threshold"),
    ]
    rows = []
    for col, label in metrics:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(values):
            rows.append({"metric": label, "mean_step": values.mean(), "rate": len(values) / len(df)})
        else:
            rows.append({"metric": label, "mean_step": np.nan, "rate": 0.0})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 5), facecolor="white")
    bars = ax.bar(table["metric"], table["mean_step"], alpha=0.9)
    for bar, rate in zip(bars, table["rate"]):
        h = bar.get_height()
        if np.isfinite(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f} шаг\n{rate:.0%} сессий", ha="center", va="bottom", fontsize=10)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 0.5, "нет", ha="center", va="bottom", fontsize=10)
    ax.set_title("Скорость сходимости: Archetype 8D", fontsize=16, fontweight="bold")
    ax.set_xlabel("Маркер")
    ax.set_ylabel("Средний шаг срабатывания")
    ax.set_ylim(0, 26)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_status_distribution(sessions: pd.DataFrame, out_path: Path | None = None) -> None:
    df = sessions[sessions["strategy"] == "phase_aware_v1"].copy()
    if df.empty:
        return
    df = add_display_columns(df)
    status_counts = (
        df.groupby(["target_mode_display", "final_status"])["user_id"]
        .nunique()
        .reset_index(name="users")
    )
    status_counts["pct"] = status_counts.groupby("target_mode_display")["users"].transform(lambda x: 100.0 * x / x.sum())
    table = status_counts.pivot(index="target_mode_display", columns="final_status", values="pct").fillna(0.0)
    order = [TARGET_MODE_DISPLAY[x] for x in ["random8d", "semantic4d", "semantic6d", "archetype8d"] if TARGET_MODE_DISPLAY[x] in table.index]
    table = table.loc[order]
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    bottom = np.zeros(len(table))
    for status in table.columns:
        values = table[status].values
        ax.bar(table.index, values, bottom=bottom, label=PHASE_DISPLAY_NAMES.get(status, status), alpha=0.9)
        bottom += values
    ax.set_title("Статус A/B-сессии по режимам target", fontsize=16, fontweight="bold")
    ax.set_xlabel("Режим генерации target")
    ax.set_ylabel("Доля сессий, %")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(title="Статус", frameon=True, facecolor="white", edgecolor="0.75")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_phase_usage(steps: pd.DataFrame, out_path: Path | None = None) -> None:
    df = steps[(steps["target_mode"] == "archetype8d") & (steps["strategy"] == "phase_aware_v1")].copy()
    if df.empty:
        return
    counts = df.groupby("phase").size().reset_index(name="count")
    counts["pct"] = 100.0 * counts["count"] / counts["count"].sum()
    counts["phase_display"] = counts["phase"].map(PHASE_DISPLAY_NAMES).fillna(counts["phase"])
    counts = counts.sort_values("pct", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    bars = ax.bar(counts["phase_display"], counts["pct"], alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_title("Использование фаз: Phase-aware controller на Archetype 8D", fontsize=16, fontweight="bold")
    ax.set_xlabel("Фаза")
    ax.set_ylabel("Доля A/B-вопросов, %")
    ax.set_ylim(0, max(100, counts["pct"].max() * 1.15))
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="outputs/datasets/synthetic_users_v21.csv")
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--prefix", type=str, default="article_phase_controller")
    parser.add_argument("--sample-per-mode", type=int, default=50, help="Use N users per target_mode. Set <=0 for full dataset.")
    args = parser.parse_args()

    ensure_dirs()
    dataset_path = PROJECT_ROOT / args.dataset if not str(args.dataset).startswith("/") else Path(args.dataset)
    dataset = pd.read_csv(dataset_path)
    if int(args.sample_per_mode) > 0:
        dataset = (
            dataset
            .groupby("target_mode", group_keys=False)
            .head(int(args.sample_per_mode))
            .reset_index(drop=True)
        )

    config = PhaseControllerConfig()
    sessions, steps, statuses = run_phase_controller_batch(
        dataset=dataset,
        n_steps=args.n_steps,
        config=config,
    )
    summary = summarize_phase_results(sessions)

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    tables_dir = PROJECT_ROOT / "outputs" / "tables"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"

    sessions.to_csv(metrics_dir / f"{args.prefix}_sessions.csv", index=False)
    steps.to_csv(metrics_dir / f"{args.prefix}_steps.csv", index=False)
    statuses.to_csv(metrics_dir / f"{args.prefix}_statuses.csv", index=False)
    summary.to_csv(metrics_dir / f"{args.prefix}_summary.csv", index=False)
    summary.to_csv(tables_dir / f"{args.prefix}_summary.csv", index=False)

    plot_convergence_archetype(steps, figures_dir / f"{args.prefix}_convergence_archetype8d.png")
    plot_final_distance_archetype(sessions, figures_dir / f"{args.prefix}_final_distance_archetype8d.png")
    plot_convergence_speed(sessions, figures_dir / f"{args.prefix}_convergence_speed_archetype8d.png")
    plot_status_distribution(sessions, figures_dir / f"{args.prefix}_status_distribution.png")
    plot_phase_usage(steps, figures_dir / f"{args.prefix}_phase_usage_archetype8d.png")

    print("Saved:")
    print(metrics_dir / f"{args.prefix}_summary.csv")
    print(figures_dir / f"{args.prefix}_convergence_archetype8d.png")


if __name__ == "__main__":
    main()
