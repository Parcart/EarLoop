"""
Experiment 01: baseline offline personalization loop v0.

Run from the research directory:
    python experiments/01_personalization_loop_v0.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from personalization import FEATURE_NAMES_8D, SyntheticUser, run_personalization_session_v0
from personalization.metrics import session_summary
from personalization.plotting import plot_convergence, plot_final_vs_target, save_figure


FIG_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rng = np.random.default_rng(42)

    z_target = np.array([0.30, 0.70, -0.20, 0.40, 0.10, 0.50, 0.60, -0.30])
    user = SyntheticUser(
        z_target=z_target,
        noise_std=0.05,
        rng=rng,
    )

    result = run_personalization_session_v0(
        synthetic_user=user,
        n_steps=25,
        step_scale=0.6,
        lr=0.25,
        pair_strategy="random",
        seed=123,
    )

    summary = session_summary(result.final_state, z_target, result.distances)
    print("FEATURE_NAMES_8D:", FEATURE_NAMES_8D)
    print("z_target:", np.round(z_target, 3))
    print("z_final: ", np.round(result.final_state.z_mean, 3))
    print("summary:", summary)

    # Save metrics
    pd.DataFrame([summary]).to_csv(METRICS_DIR / "personalization_loop_v0_summary.csv", index=False)

    records = []
    for r in result.records:
        records.append({
            "step": r.step,
            "choice": r.choice,
            "u_a": r.u_a,
            "u_b": r.u_b,
            "distance_to_target": r.distance_to_target,
        })
    pd.DataFrame(records).to_csv(METRICS_DIR / "personalization_loop_v0_steps.csv", index=False)

    # Save figures
    fig, _ = plot_convergence(result.distances)
    save_figure(fig, FIG_DIR / "personalization_loop_v0_convergence.png", dpi=300)

    fig, _ = plot_final_vs_target(result.final_state.z_mean, z_target)
    save_figure(fig, FIG_DIR / "personalization_loop_v0_final_vs_target.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
