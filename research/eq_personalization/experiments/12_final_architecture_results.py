from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personalization.final_architecture_results import (  # noqa: E402
    compare_methods_vs_heuristic,
    final_architecture_description_table,
    plot_archetype_final_vector_comparison,
    plot_cosine_by_target,
    plot_final_architecture_diagram,
    plot_final_distance_by_target,
    plot_heldout_accuracy_by_target,
    plot_selected_vs_heuristic_improvement,
    save_final_architecture_tables,
    summarize_final_architecture,
    summarize_method_improvements,
)
from personalization.preference_model_calibration import (  # noqa: E402
    run_preference_model_calibration_batch_v4a2,
)
from personalization.synthetic_dataset import load_synthetic_users_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final architecture results: Semantic active v3 + calibrated Preference Model final vector."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="outputs/datasets/synthetic_users_v21.csv",
        help="Path to fixed synthetic user dataset CSV.",
    )
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--n-heldout-pairs", type=int, default=100)
    parser.add_argument("--step-scale", type=float, default=0.6)
    parser.add_argument("--heuristic-lr", type=float, default=0.25)
    parser.add_argument("--model-lr", type=float, default=0.06)
    parser.add_argument("--model-temperature", type=float, default=0.75)
    parser.add_argument("--model-l2", type=float, default=0.003)
    parser.add_argument(
        "--max-users-per-mode",
        type=int,
        default=None,
        help="Optional small-sample mode for quick smoke tests.",
    )
    parser.add_argument("--output-tables-dir", type=str, default="outputs/tables")
    parser.add_argument("--output-figures-dir", type=str, default="outputs/figures")
    parser.add_argument("--prefix", type=str, default="article_final_architecture")
    return parser.parse_args()


def _sample_dataset(dataset: pd.DataFrame, max_users_per_mode: int | None) -> pd.DataFrame:
    if max_users_per_mode is None:
        return dataset
    parts = []
    for _, group in dataset.groupby("target_mode", sort=False):
        parts.append(group.head(int(max_users_per_mode)))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    dataset_path = ROOT / args.dataset
    dataset = load_synthetic_users_dataset(dataset_path)
    dataset = _sample_dataset(dataset, args.max_users_per_mode)

    sessions, heldout_pairs, train_steps = run_preference_model_calibration_batch_v4a2(
        dataset=dataset,
        n_steps=args.n_steps,
        n_heldout_pairs=args.n_heldout_pairs,
        pair_strategy="semantic_active_v21",
        step_scale=args.step_scale,
        heuristic_lr=args.heuristic_lr,
        model_lr=args.model_lr,
        model_temperature=args.model_temperature,
        model_l2=args.model_l2,
        model_feature_weight="uniform",
    )

    summary = summarize_final_architecture(sessions)
    improvement = compare_methods_vs_heuristic(sessions)
    improvement_summary = summarize_method_improvements(improvement)

    tables_dir = ROOT / args.output_tables_dir
    figures_dir = ROOT / args.output_figures_dir
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_final_architecture_tables(
        sessions=sessions,
        heldout_pairs=heldout_pairs,
        train_steps=train_steps,
        summary=summary,
        improvement=improvement,
        improvement_summary=improvement_summary,
        output_dir=tables_dir,
        prefix=args.prefix,
    )

    plot_final_architecture_diagram(figures_dir / f"{args.prefix}_pipeline.png")
    plot_final_distance_by_target(summary, figures_dir / f"{args.prefix}_mean_final_distance_by_target.png")
    plot_archetype_final_vector_comparison(summary, figures_dir / f"{args.prefix}_archetype8d_final_vector_methods.png")
    plot_selected_vs_heuristic_improvement(improvement_summary, figures_dir / f"{args.prefix}_improvement_vs_heuristic.png")
    plot_heldout_accuracy_by_target(summary, figures_dir / f"{args.prefix}_heldout_accuracy.png")
    plot_cosine_by_target(summary, figures_dir / f"{args.prefix}_cosine_to_target.png")

    pd.set_option("display.max_columns", 120)
    print("Final architecture description:")
    print(final_architecture_description_table())
    print("\nSummary:")
    print(summary)
    print("\nImprovement summary:")
    print(improvement_summary)
    print("\nSaved tables to:", tables_dir)
    print("Saved figures to:", figures_dir)


if __name__ == "__main__":
    main()
