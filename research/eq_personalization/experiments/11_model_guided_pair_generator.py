from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personalization.model_guided_pair_generation import (  # noqa: E402
    build_population_preference_prior,
    run_model_guided_pair_batch_v4b,
    save_v4b_outputs,
    summarize_model_guided_sessions,
)
from personalization.synthetic_dataset import load_synthetic_users_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V4b: model-guided Pair Generator with model-only and hybrid selection."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="outputs/datasets/synthetic_users_v21.csv",
        help="Path to fixed synthetic user dataset CSV.",
    )
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--model-lr", type=float, default=0.06)
    parser.add_argument("--model-temperature", type=float, default=0.75)
    parser.add_argument("--model-l2", type=float, default=0.003)
    parser.add_argument("--heuristic-lr", type=float, default=0.25)
    parser.add_argument("--step-scale", type=float, default=0.6)
    parser.add_argument("--pretrain-n-per-mode", type=int, default=250)
    parser.add_argument("--pretrain-seed", type=int, default=2026)
    parser.add_argument("--max-users-per-mode", type=int, default=0, help="Optional quick-run limit per target mode. 0 means use all users.")
    parser.add_argument("--output-dir", type=str, default="outputs/metrics")
    parser.add_argument("--prefix", type=str, default="notebook_v4b_model_guided_pair_generator")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = ROOT / args.dataset
    dataset = load_synthetic_users_dataset(dataset_path)
    if args.max_users_per_mode and args.max_users_per_mode > 0:
        dataset = (
            dataset
            .groupby("target_mode", group_keys=False)
            .head(int(args.max_users_per_mode))
            .reset_index(drop=True)
        )

    prior = build_population_preference_prior(
        n_per_mode=args.pretrain_n_per_mode,
        seed=args.pretrain_seed,
        use_zero_target_prior=True,
    )

    sessions, steps = run_model_guided_pair_batch_v4b(
        dataset=dataset,
        n_steps=args.n_steps,
        warmup_steps=args.warmup_steps,
        step_scale=args.step_scale,
        heuristic_lr=args.heuristic_lr,
        model_lr=args.model_lr,
        model_temperature=args.model_temperature,
        model_l2=args.model_l2,
        prior=prior,
    )
    summary = summarize_model_guided_sessions(sessions)

    output_dir = ROOT / args.output_dir
    save_v4b_outputs(
        sessions=sessions,
        steps=steps,
        summary=summary,
        output_dir=output_dir,
        prefix=args.prefix,
    )

    pd.set_option("display.max_columns", 100)
    print("Saved V4b model-guided Pair Generator outputs to:", output_dir)
    print("Population prior: users=", prior.n_users, "modes=", prior.modes)
    print(summary)


if __name__ == "__main__":
    main()
