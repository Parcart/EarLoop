from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from personalization.control_basis import CONTROL_BASIS_4D_TO_8D
from personalization.loop import run_personalization_session_v0
from personalization.synthetic_user import SyntheticUser
from personalization.plotting import use_article_style, save_figure


STRATEGIES = [
    "random",
    "uncertainty_axis",
    "semantic_control",
    "hybrid",
]

STRATEGY_LABELS = {
    "random": "v0: random direction",
    "uncertainty_axis": "v1: uncertainty axis",
    "semantic_control": "v2: semantic control",
    "hybrid": "v3: hybrid",
}


def distances_with_initial(result, z_target: np.ndarray) -> np.ndarray:
    """Add the honest distance before the first A/B step."""
    initial_distance = np.linalg.norm(np.zeros_like(z_target) - z_target)
    return np.concatenate([[initial_distance], result.distances])


def make_summary_row(strategy_name: str, result, z_target: np.ndarray, user_id: int) -> dict:
    d = distances_with_initial(result, z_target)
    return {
        "user_id": user_id,
        "strategy": strategy_name,
        "n_steps": len(result.distances),
        "initial_distance": d[0],
        "final_distance": d[-1],
        "best_distance": np.min(d),
        "mean_distance": np.mean(d),
        "improvement_abs": d[0] - d[-1],
        "improvement_pct": 100.0 * (d[0] - d[-1]) / d[0],
    }


def sample_target(
    rng: np.random.Generator,
    dim: int,
    target_scale: float,
    target_mode: str,
) -> np.ndarray:
    """Sample a hidden synthetic target.

    random8d: arbitrary target in the full 8D space.
    semantic4d: target generated from the same semantic controls, then mapped to 8D.
    """
    if target_mode == "random8d":
        z_target = rng.normal(0.0, target_scale, size=dim)
        return np.clip(z_target, -1.5, 1.5)

    if target_mode == "semantic4d":
        coeffs = rng.normal(0.0, target_scale, size=CONTROL_BASIS_4D_TO_8D.shape[0])
        z_target = coeffs @ CONTROL_BASIS_4D_TO_8D
        return np.clip(z_target, -1.5, 1.5)

    raise ValueError(f"Unknown target_mode: {target_mode}")


def run_batch_compare(
    n_users: int = 100,
    dim: int = 8,
    target_scale: float = 0.8,
    noise_std: float = 0.05,
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
    seed: int = 42,
    strategies: list[str] | None = None,
    target_mode: str = "random8d",
):
    """Run all strategies on the same synthetic targets."""
    rng = np.random.default_rng(seed)
    strategies = list(STRATEGIES if strategies is None else strategies)

    rows: list[dict] = []
    curves: dict[str, list[np.ndarray]] = {strategy: [] for strategy in strategies}

    for user_id in range(n_users):
        z_target = sample_target(
            rng=rng,
            dim=dim,
            target_scale=target_scale,
            target_mode=target_mode,
        )

        for strategy_idx, strategy in enumerate(strategies):
            # Separate user object per strategy keeps noisy choices reproducible.
            # Same user_id and strategy_idx seeds avoid cross-run RNG leakage.
            user = SyntheticUser(
                z_target=z_target,
                noise_std=noise_std,
                seed=10_000 + user_id,
            )

            result = run_personalization_session_v0(
                synthetic_user=user,
                n_steps=n_steps,
                step_scale=step_scale,
                lr=lr,
                pair_strategy=strategy,
                seed=20_000 + user_id,
            )

            d = distances_with_initial(result, z_target)
            curves[strategy].append(d)
            rows.append(make_summary_row(strategy, result, z_target, user_id))

    summary = pd.DataFrame(rows)
    curve_arrays = {k: np.asarray(v, dtype=np.float64) for k, v in curves.items()}
    return summary, curve_arrays


def summarize_strategies(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary
        .groupby("strategy")
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
        .sort_values("mean_final_distance")
    )


def compute_win_rates(summary: pd.DataFrame, reference_strategy: str = "random") -> pd.DataFrame:
    rows = []
    pivot_final = summary.pivot(index="user_id", columns="strategy", values="final_distance")
    pivot_best = summary.pivot(index="user_id", columns="strategy", values="best_distance")

    for strategy in pivot_final.columns:
        if strategy == reference_strategy:
            continue
        rows.append({
            "strategy": strategy,
            "reference": reference_strategy,
            "win_rate_final_distance": float((pivot_final[strategy] < pivot_final[reference_strategy]).mean()),
            "win_rate_best_distance": float((pivot_best[strategy] < pivot_best[reference_strategy]).mean()),
        })
    return pd.DataFrame(rows)


def plot_mean_convergence(curves: dict[str, np.ndarray], output_path: Path) -> None:
    use_article_style()
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")

    for strategy, arr in curves.items():
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = np.arange(len(mean))
        label = STRATEGY_LABELS.get(strategy, strategy)
        ax.plot(steps, mean, marker="o", linewidth=2.1, markersize=4, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.10)

    ax.set_title("Средняя сходимость стратегий Pair Generator")
    ax.set_xlabel("Шаг A/B-сессии")
    ax.set_ylabel("Расстояние до скрытого target")
    ax.grid(True, alpha=0.3, linestyle="--", color="gray")
    ax.legend(frameon=True, facecolor="white", edgecolor="0.75", framealpha=0.95)
    plt.tight_layout()
    save_figure(fig, str(output_path), dpi=300)
    plt.close(fig)


def plot_final_distance_boxplot(summary: pd.DataFrame, output_path: Path) -> None:
    use_article_style()
    strategies = list(summary["strategy"].unique())
    data = [summary.loc[summary["strategy"] == s, "final_distance"].values for s in strategies]
    labels = [STRATEGY_LABELS.get(s, s) for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.boxplot(data, tick_labels=labels)
    ax.set_title("Распределение final_distance по synthetic users")
    ax.set_ylabel("Final distance to target")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--", color="gray")
    ax.tick_params(axis="x", labelrotation=10)
    plt.tight_layout()
    save_figure(fig, str(output_path), dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch test Pair Generator strategies.")
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--target-scale", type=float, default=0.8)
    parser.add_argument("--step-scale", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-mode",
        choices=["random8d", "semantic4d"],
        default="random8d",
        help="How synthetic hidden targets are sampled.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    figures_dir = root / "outputs" / "figures"
    metrics_dir = root / "outputs" / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    summary, curves = run_batch_compare(
        n_users=args.n_users,
        target_scale=args.target_scale,
        noise_std=args.noise_std,
        n_steps=args.n_steps,
        step_scale=args.step_scale,
        lr=args.lr,
        seed=args.seed,
        target_mode=args.target_mode,
    )

    strategy_summary = summarize_strategies(summary)
    win_rates = compute_win_rates(summary, reference_strategy="random")

    prefix = f"pair_generator_batch_{args.target_mode}"
    summary_path = metrics_dir / f"{prefix}_sessions.csv"
    strategy_path = metrics_dir / f"{prefix}_strategy_summary.csv"
    win_path = metrics_dir / f"{prefix}_win_rates.csv"
    convergence_path = figures_dir / f"{prefix}_mean_convergence.png"
    boxplot_path = figures_dir / f"{prefix}_final_distance_boxplot.png"

    summary.to_csv(summary_path, index=False)
    strategy_summary.to_csv(strategy_path, index=False)
    win_rates.to_csv(win_path, index=False)

    plot_mean_convergence(curves, convergence_path)
    plot_final_distance_boxplot(summary, boxplot_path)

    print(f"\n=== Target mode: {args.target_mode} ===")
    print("\n=== Strategy summary ===")
    print(strategy_summary.to_string(index=False))

    print("\n=== Win rates vs random ===")
    print(win_rates.to_string(index=False))

    print("\nSaved metrics:")
    print(f"- {summary_path}")
    print(f"- {strategy_path}")
    print(f"- {win_path}")
    print("\nSaved figures:")
    print(f"- {convergence_path}")
    print(f"- {boxplot_path}")


if __name__ == "__main__":
    main()
