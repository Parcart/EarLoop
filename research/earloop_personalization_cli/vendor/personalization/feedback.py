from __future__ import annotations

import numpy as np

from .state import FEATURE_NAMES_8D, PreferenceState, clip_vector, feature_index


def direction_vector(
    feature_name: str,
    sign: float = 1.0,
    feature_names: list[str] | None = None,
) -> np.ndarray:
    """Create a unit direction vector for a named feature."""
    names = FEATURE_NAMES_8D if feature_names is None else feature_names
    d = np.zeros(len(names), dtype=np.float64)
    d[feature_index(feature_name, names)] = float(sign)
    return d


def apply_directional_feedback(
    state: PreferenceState,
    feature_name: str,
    sign: float = 1.0,
    alpha: float = 0.25,
    clip_value: float | None = 2.0,
    feature_names: list[str] | None = None,
) -> PreferenceState:
    """
    Apply Directional Feedback as a directed shift in compact feature space.

    Example:
        apply_directional_feedback(state, "bass", sign=+1)  # more bass
        apply_directional_feedback(state, "brightness", sign=-1)  # less brightness
    """
    d = direction_vector(feature_name, sign=sign, feature_names=feature_names)
    state.z_mean = clip_vector(state.z_mean + float(alpha) * d, clip_value)
    state.step += 1
    state.history.append({
        "type": "directional_feedback",
        "step": state.step,
        "feature_name": feature_name,
        "sign": float(sign),
        "alpha": float(alpha),
        "direction": d.copy(),
        "z_mean_after": state.z_mean.copy(),
        "z_std_after": state.z_std.copy(),
    })
    return state
