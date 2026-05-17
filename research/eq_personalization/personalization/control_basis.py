from __future__ import annotations

import numpy as np

from .state import FEATURE_NAMES_8D


# -----------------------------------------------------------------------------
# Semantic control basis v2: compact 4D semantic controls
# -----------------------------------------------------------------------------

CONTROL_NAMES_4D = [
    "low_power",
    "warmth_body",
    "presence_clarity",
    "air_brightness",
]

CONTROL_DISPLAY_NAMES_4D_RU = [
    "low_power",
    "warmth_body",
    "presence_clarity",
    "air_brightness",
]

# Rows are semantic controls, columns are weighted 8D features:
# [sub_bass, bass, lowmid, warmth, presence, clarity, air, brightness]
#
# This is the original v2 hypothesis: four broad musical controls that map
# into the interpretable weighted 8D preference space.
CONTROL_BASIS_4D_TO_8D_RAW = np.array([
    # sub_bass, bass, lowmid, warmth, presence, clarity, air, brightness
    [ 0.85,  1.00,  0.20, -0.05,  0.00,  0.00,  0.00, -0.20],  # low_power
    [ 0.00,  0.15,  0.85,  1.00, -0.20, -0.15,  0.00, -0.25],  # warmth_body
    [ 0.00,  0.00, -0.20, -0.10,  0.85,  1.00,  0.20,  0.25],  # presence_clarity
    [-0.15, -0.20,  0.00, -0.15,  0.10,  0.45,  1.00,  0.85],  # air_brightness
], dtype=np.float64)


# -----------------------------------------------------------------------------
# Semantic control basis v2.1: extended 6D semantic controls
# -----------------------------------------------------------------------------

CONTROL_NAMES_6D = [
    "low_power",
    "warmth_body",
    "presence_clarity",
    "air_brightness",
    "club_energy",
    "clean_bass",
]

CONTROL_DISPLAY_NAMES_6D_RU = [
    "low_power",
    "warmth_body",
    "presence_clarity",
    "air_brightness",
    "club_energy",
    "clean_bass",
]

# v2.1 adds two directions motivated by the group analysis:
# - club_energy: bass-oriented profile with moderate upper energy;
# - clean_bass: stronger low-end without excessive low-mid/warmth buildup.
#
# These directions are still hand-crafted experimental hypotheses, not final
# psychoacoustic truths. They are meant to test whether a richer semantic basis
# improves archetype and extreme-user search.
CONTROL_BASIS_6D_TO_8D_RAW = np.array([
    # sub_bass, bass, lowmid, warmth, presence, clarity, air, brightness
    [ 0.85,  1.00,  0.20, -0.05,  0.00,  0.00,  0.00, -0.20],  # low_power
    [ 0.00,  0.15,  0.85,  1.00, -0.20, -0.15,  0.00, -0.25],  # warmth_body
    [ 0.00,  0.00, -0.20, -0.10,  0.85,  1.00,  0.20,  0.25],  # presence_clarity
    [-0.15, -0.20,  0.00, -0.15,  0.10,  0.45,  1.00,  0.85],  # air_brightness
    [ 0.65,  0.90,  0.05,  0.00, -0.10,  0.25,  0.45,  0.25],  # club_energy
    [ 0.80,  1.00, -0.30, -0.35, -0.05,  0.05,  0.10, -0.10],  # clean_bass
], dtype=np.float64)


# Russian display labels for 8D features. English names are kept as model
# identifiers; Russian labels are used in article-ready plots.
FEATURE_DISPLAY_NAMES_8D_RU = [
    "Sub-bass",
    "Bass",
    "Low-mid",
    "Warmth",
    "Presence",
    "Clarity",
    "Air",
    "Brightness",
]


def normalize_rows(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each row of a matrix to unit L2 norm."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + eps)


CONTROL_BASIS_4D_TO_8D = normalize_rows(CONTROL_BASIS_4D_TO_8D_RAW)
CONTROL_BASIS_6D_TO_8D = normalize_rows(CONTROL_BASIS_6D_TO_8D_RAW)

# Backward-compatible aliases used by existing v2 code.
CONTROL_BASIS_4D_TO_8D_LEGACY = CONTROL_BASIS_4D_TO_8D
CONTROL_NAMES = CONTROL_NAMES_4D
CONTROL_BASIS_4D_TO_8D_DEFAULT = CONTROL_BASIS_4D_TO_8D


def get_control_basis(version: str = "4d") -> tuple[list[str], np.ndarray]:
    """Return semantic control names and normalized basis for a selected version."""
    if version in {"4d", "v2", "semantic4d"}:
        return list(CONTROL_NAMES_4D), CONTROL_BASIS_4D_TO_8D.copy()
    if version in {"6d", "v21", "v2.1", "semantic6d"}:
        return list(CONTROL_NAMES_6D), CONTROL_BASIS_6D_TO_8D.copy()
    raise ValueError(f"Unknown semantic basis version: {version}")


def get_control_display_names_ru(version: str = "4d") -> list[str]:
    """Return Russian display names for the selected semantic basis."""
    if version in {"4d", "v2", "semantic4d"}:
        return list(CONTROL_DISPLAY_NAMES_4D_RU)
    if version in {"6d", "v21", "v2.1", "semantic6d"}:
        return list(CONTROL_DISPLAY_NAMES_6D_RU)
    raise ValueError(f"Unknown semantic basis version: {version}")


def control_uncertainty_scores(
    z_std: np.ndarray,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate uncertainty for every semantic control.

    A control is uncertain if it touches 8D axes with high z_std:

        score_k = sum_j |basis[k, j]| * z_std[j]
    """
    z_std = np.asarray(z_std, dtype=np.float64)
    basis = CONTROL_BASIS_4D_TO_8D if basis is None else np.asarray(basis, dtype=np.float64)
    return np.abs(basis) @ z_std


def control_probabilities(
    z_std: np.ndarray,
    basis: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Convert semantic control uncertainty scores into probabilities."""
    scores = control_uncertainty_scores(z_std, basis=basis)
    total = float(scores.sum())
    if total <= eps:
        return np.ones_like(scores) / len(scores)
    return scores / total


def describe_control_basis(version: str = "4d") -> list[dict[str, float | str]]:
    """Return a readable long-form representation of a semantic basis."""
    names, basis = get_control_basis(version)
    rows = []
    for control_name, row in zip(names, basis):
        item: dict[str, float | str] = {"control": control_name}
        for feature_name, value in zip(FEATURE_NAMES_8D, row):
            item[feature_name] = float(value)
        rows.append(item)
    return rows
