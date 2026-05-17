"""Quick smoke test for the personalization package."""

import numpy as np

from personalization import SyntheticUser, run_personalization_session_v0
from personalization.metrics import session_summary


def main() -> None:
    z_target = np.array([0.3, 0.7, -0.2, 0.4, 0.1, 0.5, 0.6, -0.3])
    user = SyntheticUser(z_target=z_target, noise_std=0.05)

    result = run_personalization_session_v0(
        synthetic_user=user,
        n_steps=25,
        step_scale=0.6,
        lr=0.25,
        pair_strategy="random",
        seed=123,
    )

    print("target:", np.round(z_target, 3))
    print("final: ", np.round(result.final_state.z_mean, 3))
    print("summary:", session_summary(result.final_state, z_target, result.distances))


if __name__ == "__main__":
    main()
