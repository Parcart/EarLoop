from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


FEATURE_NAMES_8D = [
    "sub_bass",
    "bass",
    "lowmid",
    "warmth",
    "presence",
    "clarity",
    "air",
    "brightness",
]


@dataclass
class PreferenceState:
    """
    Current personalization state in compact feature space.

    z_mean is the current estimate of the user's preferred point.
    z_std is the current uncertainty per feature axis.
    """

    z_mean: np.ndarray
    z_std: np.ndarray
    step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "PreferenceState":
        return PreferenceState(
            z_mean=self.z_mean.copy(),
            z_std=self.z_std.copy(),
            step=int(self.step),
            history=list(self.history),
        )


def init_preference_state(dim: int = 8, init_std: float = 1.0) -> PreferenceState:
    """Create a zero-centered initial preference state."""
    return PreferenceState(
        z_mean=np.zeros(dim, dtype=np.float64),
        z_std=np.ones(dim, dtype=np.float64) * float(init_std),
        step=0,
        history=[],
    )


def clip_vector(z: np.ndarray, clip_value: float | None = 2.0) -> np.ndarray:
    """Clip compact feature vector to a safe finite range."""
    z = np.asarray(z, dtype=np.float64)
    if clip_value is None:
        return z
    return np.clip(z, -float(clip_value), float(clip_value))


def feature_index(feature_name: str, feature_names: list[str] | None = None) -> int:
    """Return feature index by name."""
    names = FEATURE_NAMES_8D if feature_names is None else feature_names
    if feature_name not in names:
        raise ValueError(f"Unknown feature '{feature_name}'. Available: {names}")
    return names.index(feature_name)
