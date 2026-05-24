from __future__ import annotations

import numpy as np
import pandas as pd

from .state import PreferenceState


def distance_to_target(z: np.ndarray, z_target: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(z, dtype=np.float64) - np.asarray(z_target, dtype=np.float64)))


def weighted_distance_to_target(
    z: np.ndarray,
    z_target: np.ndarray,
    feature_importance: np.ndarray | None = None,
) -> float:
    z = np.asarray(z, dtype=np.float64)
    z_target = np.asarray(z_target, dtype=np.float64)
    if feature_importance is None:
        feature_importance = np.ones_like(z)
    feature_importance = np.asarray(feature_importance, dtype=np.float64)
    return float(np.sqrt(np.sum(feature_importance * (z - z_target) ** 2)))


def regret(best_utility: float, chosen_utility: float) -> float:
    return float(best_utility - chosen_utility)


def history_to_dataframe(state: PreferenceState, z_target: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    for item in state.history:
        row = {
            "step": item.get("step"),
            "type": item.get("type"),
            "choice": item.get("choice"),
            "feature_name": item.get("feature_name"),
            "sign": item.get("sign"),
        }
        z_mean_after = item.get("z_mean_after")
        if z_mean_after is not None and z_target is not None:
            row["distance_to_target"] = distance_to_target(z_mean_after, z_target)
        rows.append(row)
    return pd.DataFrame(rows)


def session_summary(final_state: PreferenceState, z_target: np.ndarray, distances: list[float] | np.ndarray) -> dict[str, float]:
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "n_steps": float(final_state.step),
        "initial_distance": float(distances[0]) if len(distances) else np.nan,
        "final_distance": distance_to_target(final_state.z_mean, z_target),
        "best_distance": float(np.min(distances)) if len(distances) else np.nan,
        "mean_distance": float(np.mean(distances)) if len(distances) else np.nan,
    }
