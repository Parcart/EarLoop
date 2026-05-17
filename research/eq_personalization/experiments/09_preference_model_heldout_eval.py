from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personalization.preference_model_heldout import (  # noqa: E402
    run_preference_model_heldout_batch_v4a1,
    save_v4a1_outputs,
    summarize_v4a1_by_target_mode,
    summarize_v4a1_heldout_by_source,
)
from personalization.synthetic_dataset import load_synthetic_users_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V4a.1: held-out evaluation for online Logistic Preference Model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="outputs/datasets/synthetic_users_v21.csv",
        help="Path to fixed synthetic user dataset CSV.",
    )
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--n-heldout-pairs", type=int, default=100)
    parser.add_argument("--pair-strategy", type=str, default="semantic_active_v21")
    parser.add_argument("--model-lr", type=float, default=0.06)
    parser.add_argument("--model-temperature", type=float, default=0.75)
    parser.add_argument("--model-l2", type=float, default=0.003)
    parser.add_argument("--heuristic-lr", type=float, default=0.25)
    parser.add_argument("--step-scale", type=float, default=0.6)
    parser.add_argument("--output-dir", type=str, default="outputs/metrics")
    parser.add_argument("--prefix", type=str, default="notebook_v4a1_preference_model_heldout")
    parser.add_argument(
        "--model-feature-weight",
        type=str,
        default="uniform",
        choices=["uniform", "oracle"],
        help="Use uniform feature weights, or oracle synthetic feature importance as diagnostic upper bound.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = ROOT / args.dataset
    dataset = load_synthetic_users_dataset(dataset_path)

    sessions, train_steps, heldout_pairs = run_preference_model_heldout_batch_v4a1(
        dataset=dataset,
        n_steps=args.n_steps,
        n_heldout_pairs=args.n_heldout_pairs,
        pair_strategy=args.pair_strategy,
        step_scale=args.step_scale,
        heuristic_lr=args.heuristic_lr,
        model_lr=args.model_lr,
        model_temperature=args.model_temperature,
        model_l2=args.model_l2,
        model_feature_weight=args.model_feature_weight,
    )
    summary = summarize_v4a1_by_target_mode(sessions)
    source_summary = summarize_v4a1_heldout_by_source(heldout_pairs)

    output_dir = ROOT / args.output_dir
    save_v4a1_outputs(
        sessions=sessions,
        train_steps=train_steps,
        heldout_pairs=heldout_pairs,
        summary=summary,
        source_summary=source_summary,
        output_dir=output_dir,
        prefix=args.prefix,
    )

    pd.set_option("display.max_columns", 100)
    print("Saved V4a.1 held-out outputs to:", output_dir)
    print(summary)
    print("\nHeld-out by source:")
    print(source_summary)


if __name__ == "__main__":
    main()
