from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from personalization.batch_eval import (
    DEFAULT_STRATEGIES,
    run_batch_on_dataset,
    save_batch_outputs,
    summarize_by_strategy,
    win_rates_vs_baseline,
)
from personalization.plotting import save_figure, use_article_style
from personalization.synthetic_dataset import load_synthetic_users_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Pair Generator batch test using fixed synthetic users dataset.")
    parser.add_argument("--dataset", type=str, default="outputs/datasets/synthetic_users_v2.csv")
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--step-scale", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--output-metrics-dir", type=str, default="outputs/metrics")
    parser.add_argument("--output-figures-dir", type=str, default="outputs/figures")
    parser.add_argument("--prefix", type=str, default="v2_dataset_batch")
    return parser.parse_args()


def plot_mean_convergence(curves, target_mode: str, output_path: Path | None = None) -> None:
    use_article_style()
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for strategy, arr in curves[target_mode].items():
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, marker="o", linewidth=2, label=strategy)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.12)
    ax.set_title(f"Средняя сходимость: {target_mode}")
    ax.set_xlabel("Шаг A/B-сессии")
    ax.set_ylabel("Расстояние до скрытого target")
    ax.grid(True, alpha=0.3, linestyle="--", color="gray")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75", framealpha=0.95)
    plt.tight_layout()
    if output_path is not None:
        save_figure(fig, str(output_path))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_synthetic_users_dataset(args.dataset)
    sessions, curves = run_batch_on_dataset(
        dataset=dataset,
        strategies=DEFAULT_STRATEGIES,
        n_steps=args.n_steps,
        step_scale=args.step_scale,
        lr=args.lr,
    )
    strategy_summary = summarize_by_strategy(sessions)
    win_rates = win_rates_vs_baseline(sessions, baseline="random")

    save_batch_outputs(
        sessions=sessions,
        strategy_summary=strategy_summary,
        win_rates=win_rates,
        output_dir=args.output_metrics_dir,
        prefix=args.prefix,
    )

    figures_dir = Path(args.output_figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for target_mode in sorted(curves.keys()):
        plot_mean_convergence(
            curves=curves,
            target_mode=target_mode,
            output_path=figures_dir / f"{args.prefix}_{target_mode}_mean_convergence.png",
        )

    print("Strategy summary:")
    print(strategy_summary.to_string(index=False))
    print("\nWin rates vs random:")
    print(win_rates.to_string(index=False))


if __name__ == "__main__":
    main()
