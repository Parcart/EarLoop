from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticUser:
    """
    Synthetic user with a hidden target preference vector.

    The user chooses the candidate with higher utility. Utility is higher when
    the candidate is closer to z_target. Optional feature_importance makes some
    axes more important than others.
    """

    z_target: np.ndarray
    feature_importance: np.ndarray | None = None
    noise_std: float = 0.0
    rng: np.random.Generator | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.z_target = np.asarray(self.z_target, dtype=np.float64)
        if self.feature_importance is None:
            self.feature_importance = np.ones_like(self.z_target, dtype=np.float64)
        else:
            self.feature_importance = np.asarray(self.feature_importance, dtype=np.float64)
            if self.feature_importance.shape != self.z_target.shape:
                raise ValueError("feature_importance must have the same shape as z_target")
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

    def utility(self, z_candidate: np.ndarray) -> float:
        """Negative weighted squared distance to the hidden target."""
        z_candidate = np.asarray(z_candidate, dtype=np.float64)
        diff = z_candidate - self.z_target
        value = -float(np.sum(self.feature_importance * diff * diff))
        if self.noise_std > 0:
            value += float(self.rng.normal(0.0, self.noise_std))
        return value

    def choose(self, z_a: np.ndarray, z_b: np.ndarray) -> tuple[str, float, float]:
        """Return choice ('A' or 'B') and both utilities."""
        u_a = self.utility(z_a)
        u_b = self.utility(z_b)
        return ("A" if u_a >= u_b else "B"), u_a, u_b

    def both_bad(self, z_a: np.ndarray, z_b: np.ndarray, threshold: float = -2.0) -> bool:
        """
        Simple heuristic for future Directional Feedback simulation.

        Returns True if both candidates are below a utility threshold.
        """
        return self.utility(z_a) < threshold and self.utility(z_b) < threshold


def make_random_synthetic_user(
    dim: int = 8,
    target_scale: float = 0.8,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> SyntheticUser:
    """Create a random synthetic user target in compact space."""
    rng = np.random.default_rng(seed)
    z_target = rng.normal(0.0, target_scale, size=dim)
    z_target = np.clip(z_target, -1.5, 1.5)
    return SyntheticUser(z_target=z_target, noise_std=noise_std, rng=rng)
