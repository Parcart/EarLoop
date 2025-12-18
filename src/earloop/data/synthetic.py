# src/earloop/data/synthetic.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


ThetaMode = Literal["random", "archetype"]
ChoiceMode = Literal["distance", "linear"]


@dataclass(frozen=True)
class ThetaSpec:
    """Parameterization of an EQ preset theta.

    theta = [bass_gain_db, treble_gain_db, presence_gain_db, tilt_db, dynamics]
    """
    names: tuple[str, ...] = ("bass", "treble", "presence", "tilt", "dynamics")

    low: np.ndarray = field(
        default_factory=lambda: np.array([-12.0, -12.0, -10.0, -6.0, 0.0], dtype=np.float32)
    )
    high: np.ndarray = field(
        default_factory=lambda: np.array([+12.0, +12.0, +10.0, +6.0, 1.0], dtype=np.float32)
    )

    def dim(self) -> int:
        return int(self.low.shape[0])


DEFAULT_SPEC = ThetaSpec()


def _clip_to_spec(theta: np.ndarray, spec: ThetaSpec) -> np.ndarray:
    return np.clip(theta, spec.low, spec.high)


def sample_theta(rng: np.random.Generator, spec: ThetaSpec = DEFAULT_SPEC) -> np.ndarray:
    """Sample a random preset theta uniformly within the allowed ranges."""
    u = rng.random(spec.dim(), dtype=np.float32)
    theta = spec.low + u * (spec.high - spec.low)
    return theta.astype(np.float32)


def sample_theta_true(
    rng: np.random.Generator,
    spec: ThetaSpec = DEFAULT_SPEC,
    mode: ThetaMode = "random",
) -> np.ndarray:
    """Sample a 'true' user preference theta_true.

    mode="random": draws from a centered normal prior and clips to ranges.
    mode="archetype": samples around a few interpretable archetypes.
    """
    if mode == "random":
        # Typical "people are near neutral" prior with occasional stronger tastes.
        mean = np.zeros(spec.dim(), dtype=np.float32)
        std = np.array([5.0, 5.0, 4.0, 3.0, 0.25], dtype=np.float32)
        theta = rng.normal(mean, std).astype(np.float32)

        # dynamics in [0, 1] is better modeled with Beta; blend into theta[4]
        dyn = rng.beta(2.0, 5.0)  # skewed towards lighter compression
        theta[4] = np.float32(dyn)
        return _clip_to_spec(theta, spec)

    if mode == "archetype":
        # Archetypes are rough, but great for interpretability in experiments.
        archetypes = {
            "neutral": np.array([0.0, 0.0, 0.0, 0.0, 0.15], dtype=np.float32),
            "basshead": np.array([10.0, 0.0, -1.0, 1.0, 0.20], dtype=np.float32),
            "bright": np.array([0.0, 10.0, 4.0, -1.0, 0.10], dtype=np.float32),
            "vocal": np.array([1.0, 3.0, 8.0, -0.5, 0.15], dtype=np.float32),
            "warm": np.array([6.0, -4.0, 1.0, 3.0, 0.25], dtype=np.float32),
        }
        keys = list(archetypes.keys())
        key = keys[int(rng.integers(0, len(keys)))]
        base = archetypes[key].copy()

        # Add small personal variation
        noise = rng.normal(0.0, np.array([2.0, 2.0, 1.5, 1.0, 0.08], dtype=np.float32)).astype(np.float32)
        theta = base + noise
        return _clip_to_spec(theta, spec)

    raise ValueError(f"Unknown mode={mode!r}")


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def choose_ab(
    rng: np.random.Generator,
    theta_true: np.ndarray,
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    *,
    mode: ChoiceMode = "distance",
    beta: float = 1.5,
    noise_std: float = 0.30,
) -> int:
    """Return 1 if user chooses A, else 0.

    mode="distance": user prefers preset closer to theta_true (squared L2 distance).
    mode="linear": user utility is linear u(theta)=theta_true·theta (less intuitive, but classic).
    beta: "discriminability" (higher => more consistent choices)
    noise_std: stochasticity in choices (higher => noisier)
    """
    theta_true = theta_true.astype(np.float32)
    theta_a = theta_a.astype(np.float32)
    theta_b = theta_b.astype(np.float32)

    if mode == "distance":
        # Higher is better: negative distance to the ideal.
        ua = -float(np.sum((theta_a - theta_true) ** 2))
        ub = -float(np.sum((theta_b - theta_true) ** 2))
    elif mode == "linear":
        ua = float(np.dot(theta_true, theta_a))
        ub = float(np.dot(theta_true, theta_b))
    else:
        raise ValueError(f"Unknown mode={mode!r}")

    logit = beta * (ua - ub) + float(rng.normal(0.0, noise_std))
    p_choose_a = float(_sigmoid(logit))
    return int(rng.random() < p_choose_a)


def generate_ab_dataset(
    rng: np.random.Generator,
    n_pairs: int,
    *,
    spec: ThetaSpec = DEFAULT_SPEC,
    theta_true: np.ndarray | None = None,
    theta_true_mode: ThetaMode = "random",
    choice_mode: ChoiceMode = "distance",
    beta: float = 1.5,
    noise_std: float = 0.30,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate an A/B preference dataset for a single user.

    Returns:
      df: columns include theta_a_*, theta_b_*, y (1 means A chosen)
      theta_true: the sampled (or provided) ground-truth preference vector
    """
    if theta_true is None:
        theta_true = sample_theta_true(rng, spec=spec, mode=theta_true_mode)

    a = np.stack([sample_theta(rng, spec=spec) for _ in range(n_pairs)], axis=0)
    b = np.stack([sample_theta(rng, spec=spec) for _ in range(n_pairs)], axis=0)

    y = np.zeros((n_pairs,), dtype=np.int64)
    for i in range(n_pairs):
        y[i] = choose_ab(
            rng,
            theta_true=theta_true,
            theta_a=a[i],
            theta_b=b[i],
            mode=choice_mode,
            beta=beta,
            noise_std=noise_std,
        )

    cols = {}
    for j, name in enumerate(spec.names):
        cols[f"a_{name}"] = a[:, j]
        cols[f"b_{name}"] = b[:, j]
    cols["y"] = y

    df = pd.DataFrame(cols)
    return df, theta_true


def make_rng(seed: int = 42) -> np.random.Generator:
    """Convenience RNG helper for reproducibility."""
    return np.random.default_rng(seed)

if __name__ == "__main__":
    rng = make_rng(0)
    df, theta_true = generate_ab_dataset(rng, n_pairs=200, theta_true_mode="archetype")
    print("theta_true:", theta_true)
    print(df.head())
    print("choice rate A:", df["y"].mean())