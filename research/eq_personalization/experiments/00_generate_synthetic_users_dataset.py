from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from personalization.synthetic_dataset import (
    TARGET_MODES,
    dataset_metadata,
    generate_synthetic_users_dataset,
    save_synthetic_users_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed synthetic users dataset.")
    parser.add_argument("--n-per-mode", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--importance-sigma", type=float, default=0.35)
    parser.add_argument("--target-max-abs", type=float, default=2.0)
    parser.add_argument("--archetype-extreme-probability", type=float, default=0.30)
    parser.add_argument("--output-dir", type=str, default="outputs/datasets")
    parser.add_argument("--name", type=str, default="synthetic_users_v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dataset_path = output_dir / f"{args.name}.csv"
    metadata_path = output_dir / f"{args.name}_metadata.json"

    dataset = generate_synthetic_users_dataset(
        n_per_mode=args.n_per_mode,
        modes=TARGET_MODES,
        seed=args.seed,
        noise_std=args.noise_std,
        importance_sigma=args.importance_sigma,
        target_max_abs=args.target_max_abs,
        archetype_extreme_probability=args.archetype_extreme_probability,
    )

    metadata = dataset_metadata(
        dataset_name=args.name,
        n_per_mode=args.n_per_mode,
        seed=args.seed,
        noise_std=args.noise_std,
        importance_sigma=args.importance_sigma,
        target_max_abs=args.target_max_abs,
        archetype_extreme_probability=args.archetype_extreme_probability,
    )
    metadata["n_users_total"] = int(len(dataset))

    save_synthetic_users_dataset(
        dataset=dataset,
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    print(f"Saved dataset: {dataset_path}")
    print(f"Saved metadata: {metadata_path}")
    print(dataset.groupby("target_mode").size())
    if "intensity_label" in dataset.columns:
        print("\nIntensity distribution:")
        print(dataset.groupby(["target_mode", "intensity_label"]).size())


if __name__ == "__main__":
    main()
