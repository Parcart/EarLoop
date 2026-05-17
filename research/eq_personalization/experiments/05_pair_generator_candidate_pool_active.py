from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from personalization.analysis import (
    compare_two_strategies_by_group,
    merge_sessions_with_user_metadata,
    summarize_by_group,
    winners_by_group,
    win_rate_between_strategies_by_group,
    win_rates_vs_baseline_by_group,
)
from personalization.batch_eval import (
    run_batch_on_dataset,
    save_batch_outputs,
    summarize_by_strategy,
    win_rates_vs_baseline,
)
from personalization.synthetic_dataset import (
    dataset_metadata,
    generate_synthetic_users_dataset,
    load_synthetic_users_dataset,
    save_synthetic_users_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate V3.1 candidate-pool active Pair Generator against previous strategies."
    )
    parser.add_argument("--dataset", type=str, default="outputs/datasets/synthetic_users_v21.csv")
    parser.add_argument("--n-per-mode", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="v31_candidate_pool_active")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        dataset = generate_synthetic_users_dataset(
            n_per_mode=args.n_per_mode,
            seed=args.seed,
            noise_std=args.noise_std,
        )
        save_synthetic_users_dataset(
            dataset=dataset,
            dataset_path=dataset_path,
            metadata_path=dataset_path.with_suffix(".metadata.json"),
            metadata=dataset_metadata(
                dataset_name=dataset_path.stem,
                n_per_mode=args.n_per_mode,
                seed=args.seed,
                noise_std=args.noise_std,
                importance_sigma=0.35,
            ),
        )
    else:
        dataset = load_synthetic_users_dataset(dataset_path)

    strategies = [
        "random",
        "uncertainty_axis",
        "semantic_control_v21",
        "semantic_active_v21",
        "candidate_pool_active",
        "hybrid_active_v21",
    ]

    sessions, _curves = run_batch_on_dataset(
        dataset=dataset,
        strategies=strategies,
        n_steps=args.n_steps,
    )

    strategy_summary = summarize_by_strategy(sessions)
    win_rates = win_rates_vs_baseline(sessions, baseline="random")
    save_batch_outputs(
        sessions=sessions,
        strategy_summary=strategy_summary,
        win_rates=win_rates,
        output_dir=metrics_dir,
        prefix=args.prefix,
    )

    sessions_with_meta = merge_sessions_with_user_metadata(sessions, dataset)
    sessions_with_meta.to_csv(metrics_dir / f"{args.prefix}_sessions_with_meta.csv", index=False)

    target_mode_summary = summarize_by_group(sessions_with_meta, ["target_mode"])
    target_mode_winners = winners_by_group(target_mode_summary, ["target_mode"])
    target_mode_win_rates = win_rates_vs_baseline_by_group(sessions_with_meta, ["target_mode"])

    target_mode_summary.to_csv(metrics_dir / f"{args.prefix}_target_mode_group_summary.csv", index=False)
    target_mode_winners.to_csv(metrics_dir / f"{args.prefix}_target_mode_winners.csv", index=False)
    target_mode_win_rates.to_csv(metrics_dir / f"{args.prefix}_target_mode_win_rates.csv", index=False)

    archetype_sessions = sessions_with_meta[sessions_with_meta["target_mode"] == "archetype8d"].copy()
    if len(archetype_sessions) > 0:
        intensity_summary = summarize_by_group(archetype_sessions, ["intensity_label"])
        intensity_win_rates = win_rates_vs_baseline_by_group(archetype_sessions, ["intensity_label"])
        intensity_summary.to_csv(metrics_dir / f"{args.prefix}_intensity_group_summary.csv", index=False)
        intensity_win_rates.to_csv(metrics_dir / f"{args.prefix}_intensity_win_rates.csv", index=False)

        if "main_archetype" in archetype_sessions.columns:
            main_archetype_summary = summarize_by_group(archetype_sessions, ["main_archetype"])
            main_archetype_win_rates = win_rates_vs_baseline_by_group(archetype_sessions, ["main_archetype"])
            main_archetype_summary.to_csv(metrics_dir / f"{args.prefix}_main_archetype_group_summary.csv", index=False)
            main_archetype_win_rates.to_csv(metrics_dir / f"{args.prefix}_main_archetype_win_rates.csv", index=False)

            candidate_vs_v3_by_archetype = compare_two_strategies_by_group(
                main_archetype_summary,
                group_col="main_archetype",
                old_strategy="semantic_active_v21",
                new_strategy="candidate_pool_active",
                metric="mean_final_distance",
            )
            candidate_vs_v3_win_rates = win_rate_between_strategies_by_group(
                archetype_sessions,
                group_cols=["main_archetype"],
                strategy_a="candidate_pool_active",
                strategy_b="semantic_active_v21",
                metric="final_distance",
            )
            candidate_vs_v3_by_archetype.to_csv(metrics_dir / f"{args.prefix}_candidate_vs_v3_by_archetype.csv", index=False)
            candidate_vs_v3_win_rates.to_csv(metrics_dir / f"{args.prefix}_candidate_vs_v3_win_rates_by_archetype.csv", index=False)

    print("Strategy summary:")
    print(strategy_summary)
    print("\nWin rates vs random:")
    print(win_rates)


if __name__ == "__main__":
    main()
