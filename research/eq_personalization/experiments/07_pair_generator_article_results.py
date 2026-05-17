from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from personalization.article_results import (
    ARTICLE_STRATEGIES,
    ensure_article_dirs,
    run_article_batch,
    save_article_tables,
    make_archetype_improvement_table,
    plot_mean_final_distance_by_target_mode,
    plot_mean_convergence,
    plot_final_distance_boxplot,
    plot_win_rates_vs_random,
    plot_intensity_analysis,
    plot_archetype_v3_improvement,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate article-ready Pair Generator results and figures.")
    parser.add_argument("--dataset", default="outputs/datasets/synthetic_users_v21.csv")
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--step-scale", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--prefix", default="article_pair_generator")
    args = parser.parse_args()

    project_root = Path.cwd()
    dirs = ensure_article_dirs(project_root)

    dataset_path = project_root / args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run 00_generate_synthetic_users_dataset first.")

    dataset = pd.read_csv(dataset_path)
    sessions, curves, strategy_summary, win_rates, sessions_with_meta = run_article_batch(
        dataset=dataset,
        strategies=ARTICLE_STRATEGIES,
        n_steps=args.n_steps,
        step_scale=args.step_scale,
        lr=args.lr,
    )

    sessions.to_csv(dirs["metrics"] / f"{args.prefix}_sessions.csv", index=False)
    sessions_with_meta.to_csv(dirs["metrics"] / f"{args.prefix}_sessions_with_meta.csv", index=False)
    strategy_summary.to_csv(dirs["metrics"] / f"{args.prefix}_strategy_summary.csv", index=False)
    win_rates.to_csv(dirs["metrics"] / f"{args.prefix}_win_rates.csv", index=False)

    table_paths = save_article_tables(
        strategy_summary=strategy_summary,
        win_rates=win_rates,
        sessions_with_meta=sessions_with_meta,
        output_dir=dirs["tables"],
    )
    improvement_table = make_archetype_improvement_table(sessions_with_meta)

    plot_mean_final_distance_by_target_mode(
        strategy_summary,
        save_path=dirs["figures"] / f"{args.prefix}_mean_final_distance_by_target_mode.png",
    )
    plot_mean_convergence(
        curves,
        target_mode="archetype8d",
        save_path=dirs["figures"] / f"{args.prefix}_convergence_archetype8d.png",
    )
    plot_final_distance_boxplot(
        sessions,
        target_mode="archetype8d",
        save_path=dirs["figures"] / f"{args.prefix}_boxplot_archetype8d.png",
    )
    plot_win_rates_vs_random(
        win_rates,
        target_mode="archetype8d",
        save_path=dirs["figures"] / f"{args.prefix}_win_rate_archetype8d.png",
    )
    plot_intensity_analysis(
        sessions_with_meta,
        save_path=dirs["figures"] / f"{args.prefix}_intensity_archetype8d.png",
    )
    plot_archetype_v3_improvement(
        improvement_table,
        save_path=dirs["figures"] / f"{args.prefix}_v3_vs_v21_by_archetype.png",
    )

    print("Saved metrics:", dirs["metrics"])
    print("Saved tables:", dirs["tables"])
    print("Saved figures:", dirs["figures"])
    print("Article tables:")
    for name, path in table_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
