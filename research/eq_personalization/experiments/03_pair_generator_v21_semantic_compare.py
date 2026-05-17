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
from personalization.batch_eval import run_batch_on_dataset, save_batch_outputs, summarize_by_strategy, win_rates_vs_baseline
from personalization.synthetic_dataset import generate_synthetic_users_dataset, load_synthetic_users_dataset, save_synthetic_users_dataset, dataset_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v2 4D semantic controls with v2.1 6D semantic controls.")
    parser.add_argument("--dataset", type=str, default="outputs/datasets/synthetic_users_v21.csv")
    parser.add_argument("--generate-if-missing", action="store_true", default=True)
    parser.add_argument("--n-per-mode", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="v21_semantic_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)

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
        "semantic_control",
        "semantic_control_v21",
        "hybrid",
        "hybrid_v21",
    ]

    sessions, curves = run_batch_on_dataset(
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
        output_dir="outputs/metrics",
        prefix=args.prefix,
    )

    sessions_with_meta = merge_sessions_with_user_metadata(sessions, dataset)
    sessions_with_meta.to_csv(Path("outputs/metrics") / f"{args.prefix}_sessions_with_meta.csv", index=False)

    target_mode_summary = summarize_by_group(sessions_with_meta, ["target_mode"])
    target_mode_winners = winners_by_group(target_mode_summary, ["target_mode"])
    target_mode_win_rates = win_rates_vs_baseline_by_group(sessions_with_meta, ["target_mode"])

    target_mode_summary.to_csv(Path("outputs/metrics") / f"{args.prefix}_target_mode_group_summary.csv", index=False)
    target_mode_winners.to_csv(Path("outputs/metrics") / f"{args.prefix}_target_mode_winners.csv", index=False)
    target_mode_win_rates.to_csv(Path("outputs/metrics") / f"{args.prefix}_target_mode_win_rates.csv", index=False)

    archetype_sessions = sessions_with_meta[sessions_with_meta["target_mode"] == "archetype8d"].copy()
    if len(archetype_sessions) > 0 and "main_archetype" in archetype_sessions.columns:
        main_archetype_summary = summarize_by_group(archetype_sessions, ["main_archetype"])
        v21_vs_v2_by_archetype = compare_two_strategies_by_group(
            main_archetype_summary,
            group_col="main_archetype",
            old_strategy="semantic_control",
            new_strategy="semantic_control_v21",
            metric="mean_final_distance",
        )
        v21_vs_v2_win_rates = win_rate_between_strategies_by_group(
            archetype_sessions,
            group_cols=["main_archetype"],
            strategy_a="semantic_control_v21",
            strategy_b="semantic_control",
            metric="final_distance",
        )
        v21_vs_v2_by_archetype.to_csv(
            Path("outputs/metrics") / f"{args.prefix}_v21_vs_v2_by_archetype.csv",
            index=False,
        )
        v21_vs_v2_win_rates.to_csv(
            Path("outputs/metrics") / f"{args.prefix}_v21_vs_v2_win_rates_by_archetype.csv",
            index=False,
        )

    print("Strategy summary:")
    print(strategy_summary)
    print("\nWin rates vs random:")
    print(win_rates)


if __name__ == "__main__":
    main()
