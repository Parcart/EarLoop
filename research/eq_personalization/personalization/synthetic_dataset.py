from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .control_basis import CONTROL_BASIS_4D_TO_8D, CONTROL_BASIS_6D_TO_8D, CONTROL_NAMES_4D, CONTROL_NAMES_6D
from .state import FEATURE_NAMES_8D
from .synthetic_user import SyntheticUser


TARGET_MODES = ["random8d", "semantic4d", "semantic6d", "archetype8d"]


# -----------------------------------------------------------------------------
# Realistic / archetype user targets
# -----------------------------------------------------------------------------

USER_ARCHETYPES_8D: dict[str, np.ndarray] = {
    # mild / neutral-ish profiles
    "neutral": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "bass_lover": np.array([0.8, 1.0, 0.3, 0.1, -0.1, -0.2, -0.1, -0.2], dtype=np.float64),
    "warm": np.array([0.2, 0.4, 0.6, 0.8, -0.2, -0.2, -0.1, -0.3], dtype=np.float64),
    "bright": np.array([-0.2, -0.1, -0.1, -0.2, 0.2, 0.5, 0.7, 0.8], dtype=np.float64),
    "v_shape": np.array([0.7, 0.8, -0.2, -0.2, -0.2, 0.2, 0.5, 0.4], dtype=np.float64),
    "mid_forward": np.array([-0.2, -0.1, 0.3, 0.2, 0.8, 0.4, -0.1, -0.1], dtype=np.float64),
    "soft_dark": np.array([0.2, 0.3, 0.4, 0.6, -0.4, -0.5, -0.4, -0.7], dtype=np.float64),
    "detail_focused": np.array([-0.1, -0.1, -0.2, -0.1, 0.3, 0.8, 0.6, 0.5], dtype=np.float64),

    # stronger / mass-market / extreme profiles
    # These are intended to model users who deliberately prefer strong EQ curves,
    # e.g. basshead or V-shape profiles with large low-end or treble emphasis.
    "basshead": np.array([1.2, 1.4, 0.4, 0.2, -0.3, -0.4, -0.3, -0.4], dtype=np.float64),
    "extreme_vshape": np.array([1.0, 1.2, -0.4, -0.4, -0.3, 0.3, 0.8, 0.8], dtype=np.float64),
    "club_curve": np.array([1.1, 1.3, 0.1, 0.0, -0.2, 0.1, 0.4, 0.2], dtype=np.float64),
    "sparkle_lover": np.array([-0.3, -0.2, -0.1, -0.2, 0.3, 0.6, 0.9, 1.0], dtype=np.float64),
}

NORMAL_ARCHETYPES = [
    "neutral",
    "bass_lover",
    "warm",
    "bright",
    "v_shape",
    "mid_forward",
    "soft_dark",
    "detail_focused",
]

EXTREME_ARCHETYPES = [
    "basshead",
    "extreme_vshape",
    "club_curve",
    "sparkle_lover",
]

INTENSITY_RANGES: dict[str, tuple[float, float]] = {
    "mild": (0.45, 0.75),
    "moderate": (0.75, 1.05),
    "strong": (1.05, 1.40),
    "extreme": (1.40, 1.90),
}

INTENSITY_PRIORS: dict[str, float] = {
    "mild": 0.20,
    "moderate": 0.40,
    "strong": 0.25,
    "extreme": 0.15,
}


# -----------------------------------------------------------------------------
# Target generators
# -----------------------------------------------------------------------------


def _clip_target(z: np.ndarray, max_abs: float = 2.0) -> np.ndarray:
    return np.clip(np.asarray(z, dtype=np.float64), -float(max_abs), float(max_abs))


def sample_intensity(rng: np.random.Generator) -> tuple[str, float]:
    """Sample user preference intensity label and numeric multiplier."""
    labels = list(INTENSITY_PRIORS.keys())
    probs = np.array([INTENSITY_PRIORS[label] for label in labels], dtype=np.float64)
    probs = probs / probs.sum()
    label = str(rng.choice(labels, p=probs))
    lo, hi = INTENSITY_RANGES[label]
    return label, float(rng.uniform(lo, hi))


def make_random8d_target(
    rng: np.random.Generator,
    target_scale: float = 0.8,
    max_abs: float = 2.0,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Generate a stress-test target anywhere in full weighted 8D space."""
    z = rng.normal(0.0, target_scale, size=len(FEATURE_NAMES_8D))
    meta: dict[str, float | int | str | bool] = {
        "target_mode": "random8d",
        "main_archetype": "none",
        "secondary_archetype": "none",
        "is_extreme_archetype": False,
        "intensity_label": "random",
        "intensity_value": 1.0,
    }
    return _clip_target(z, max_abs=max_abs), meta


def make_semantic4d_target(
    rng: np.random.Generator,
    control_scale: float = 0.9,
    jitter_std: float = 0.08,
    max_abs: float = 2.0,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """
    Generate a target from 4D semantic controls and map it into weighted 8D.

    This mode tests the hypothesis that user taste lives close to a semantic
    subspace rather than arbitrary 8D coordinates.
    """
    c = rng.normal(0.0, control_scale, size=len(CONTROL_NAMES_4D))
    z = c @ CONTROL_BASIS_4D_TO_8D
    z = z + rng.normal(0.0, jitter_std, size=len(FEATURE_NAMES_8D))
    z = _clip_target(z, max_abs=max_abs)

    meta: dict[str, float | int | str | bool] = {
        "target_mode": "semantic4d",
        "main_archetype": "none",
        "secondary_archetype": "none",
        "is_extreme_archetype": False,
        "intensity_label": "semantic",
        "intensity_value": 1.0,
    }
    for name, value in zip(CONTROL_NAMES_4D, c):
        meta[f"semantic_{name}"] = float(value)
    return z, meta




def make_semantic6d_target(
    rng: np.random.Generator,
    control_scale: float = 0.9,
    jitter_std: float = 0.08,
    max_abs: float = 2.0,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """
    Generate a target from the extended v2.1 6D semantic controls.

    This mode tests whether the improved semantic basis covers users that are
    not well represented by the original 4D semantic controls.
    """
    c = rng.normal(0.0, control_scale, size=len(CONTROL_NAMES_6D))
    z = c @ CONTROL_BASIS_6D_TO_8D
    z = z + rng.normal(0.0, jitter_std, size=len(FEATURE_NAMES_8D))
    z = _clip_target(z, max_abs=max_abs)

    meta: dict[str, float | int | str | bool] = {
        "target_mode": "semantic6d",
        "main_archetype": "none",
        "secondary_archetype": "none",
        "is_extreme_archetype": False,
        "intensity_label": "semantic",
        "intensity_value": 1.0,
    }
    for name, value in zip(CONTROL_NAMES_6D, c):
        meta[f"semantic6d_{name}"] = float(value)
    return z, meta


def make_archetype8d_target(
    rng: np.random.Generator,
    jitter_std: float = 0.08,
    max_abs: float = 2.0,
    extreme_probability: float = 0.30,
    main_weight: float = 0.75,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """
    Generate a realistic synthetic user target from dominant archetypes.

    Compared with a soft Dirichlet mixture over all archetypes, this generator
    produces clearer user types: a dominant archetype, a secondary archetype,
    an intensity multiplier, and small individual jitter.

    This makes it possible to model both moderate listeners and users who like
    strong/extreme EQ curves, such as basshead or strong V-shape profiles.
    """
    is_extreme = bool(rng.random() < float(extreme_probability))

    if is_extreme:
        main_name = str(rng.choice(EXTREME_ARCHETYPES))
        secondary_pool = [x for x in NORMAL_ARCHETYPES + EXTREME_ARCHETYPES if x != main_name]
    else:
        main_name = str(rng.choice(NORMAL_ARCHETYPES))
        secondary_pool = [x for x in NORMAL_ARCHETYPES if x != main_name]

    secondary_name = str(rng.choice(secondary_pool))

    main_vec = USER_ARCHETYPES_8D[main_name]
    secondary_vec = USER_ARCHETYPES_8D[secondary_name]
    secondary_weight = 1.0 - float(main_weight)

    intensity_label, intensity_value = sample_intensity(rng)

    z = float(intensity_value) * (float(main_weight) * main_vec + secondary_weight * secondary_vec)
    z = z + rng.normal(0.0, jitter_std, size=len(FEATURE_NAMES_8D))
    z = _clip_target(z, max_abs=max_abs)

    meta: dict[str, float | int | str | bool] = {
        "target_mode": "archetype8d",
        "main_archetype": main_name,
        "secondary_archetype": secondary_name,
        "is_extreme_archetype": is_extreme,
        "main_weight": float(main_weight),
        "secondary_weight": float(secondary_weight),
        "intensity_label": intensity_label,
        "intensity_value": float(intensity_value),
    }

    # Store one-hot-ish weights for easy later analysis.
    for name in USER_ARCHETYPES_8D:
        if name == main_name:
            meta[f"mix_{name}"] = float(main_weight)
        elif name == secondary_name:
            meta[f"mix_{name}"] = float(secondary_weight)
        else:
            meta[f"mix_{name}"] = 0.0

    return z, meta


# -----------------------------------------------------------------------------
# Synthetic user dataset
# -----------------------------------------------------------------------------


def make_feature_importance(
    rng: np.random.Generator,
    dim: int = 8,
    sigma: float = 0.35,
) -> np.ndarray:
    """
    Generate individual feature importance for synthetic user choices.

    Values are normalized to mean 1.0, so overall utility scale remains stable.
    """
    importance = rng.lognormal(mean=0.0, sigma=sigma, size=dim)
    importance = importance / (importance.mean() + 1e-8)
    return importance.astype(np.float64)


def generate_synthetic_users_dataset(
    n_per_mode: int = 100,
    modes: Iterable[str] = TARGET_MODES,
    seed: int = 42,
    noise_std: float = 0.05,
    importance_sigma: float = 0.35,
    target_max_abs: float = 2.0,
    archetype_jitter_std: float = 0.08,
    archetype_extreme_probability: float = 0.30,
) -> pd.DataFrame:
    """Generate a fixed synthetic user dataset for fair strategy comparisons."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []
    user_id = 0

    for mode in modes:
        for _ in range(int(n_per_mode)):
            if mode == "random8d":
                z_target, meta = make_random8d_target(rng, max_abs=target_max_abs)
            elif mode == "semantic4d":
                z_target, meta = make_semantic4d_target(rng, max_abs=target_max_abs)
            elif mode == "semantic6d":
                z_target, meta = make_semantic6d_target(rng, max_abs=target_max_abs)
            elif mode == "archetype8d":
                z_target, meta = make_archetype8d_target(
                    rng,
                    jitter_std=archetype_jitter_std,
                    max_abs=target_max_abs,
                    extreme_probability=archetype_extreme_probability,
                )
            else:
                raise ValueError(f"Unknown target mode: {mode}")

            importance = make_feature_importance(
                rng,
                dim=len(FEATURE_NAMES_8D),
                sigma=importance_sigma,
            )

            row: dict[str, float | int | str | bool] = {
                "user_id": user_id,
                "target_mode": mode,
                "noise_std": float(noise_std),
            }

            for name, value in zip(FEATURE_NAMES_8D, z_target):
                row[f"z_{name}"] = float(value)

            for name, value in zip(FEATURE_NAMES_8D, importance):
                row[f"importance_{name}"] = float(value)

            row.update(meta)
            rows.append(row)
            user_id += 1

    return pd.DataFrame(rows)


def row_to_target(row: pd.Series) -> np.ndarray:
    """Read weighted 8D target vector from a dataset row."""
    return np.array([row[f"z_{name}"] for name in FEATURE_NAMES_8D], dtype=np.float64)


def row_to_importance(row: pd.Series) -> np.ndarray:
    """Read feature importance vector from a dataset row."""
    return np.array([row[f"importance_{name}"] for name in FEATURE_NAMES_8D], dtype=np.float64)


def row_to_synthetic_user(row: pd.Series, seed: int | None = None) -> SyntheticUser:
    """Create SyntheticUser object from a dataset row."""
    return SyntheticUser(
        z_target=row_to_target(row),
        feature_importance=row_to_importance(row),
        noise_std=float(row.get("noise_std", 0.0)),
        seed=seed,
    )


def save_synthetic_users_dataset(
    dataset: pd.DataFrame,
    dataset_path: str | Path,
    metadata_path: str | Path | None = None,
    metadata: dict | None = None,
) -> None:
    """Save dataset CSV and optional metadata JSON."""
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(dataset_path, index=False)

    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {} if metadata is None else dict(metadata)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


def load_synthetic_users_dataset(dataset_path: str | Path) -> pd.DataFrame:
    """Load a generated synthetic users dataset."""
    return pd.read_csv(dataset_path)


def dataset_metadata(
    dataset_name: str,
    n_per_mode: int,
    seed: int,
    noise_std: float,
    importance_sigma: float,
    target_max_abs: float = 2.0,
    archetype_extreme_probability: float = 0.30,
) -> dict:
    """Build metadata dictionary for a generated dataset."""
    return {
        "dataset_name": dataset_name,
        "description": "Synthetic users for testing Pair Generator strategies in compact weighted 8D space.",
        "feature_names_8d": FEATURE_NAMES_8D,
        "semantic_control_names_4d": CONTROL_NAMES_4D,
        "semantic_control_names_6d": CONTROL_NAMES_6D,
        "target_modes": TARGET_MODES,
        "n_per_mode": int(n_per_mode),
        "seed": int(seed),
        "noise_std": float(noise_std),
        "importance_sigma": float(importance_sigma),
        "target_max_abs": float(target_max_abs),
        "archetype_extreme_probability": float(archetype_extreme_probability),
        "archetype_names": list(USER_ARCHETYPES_8D.keys()),
        "normal_archetypes": NORMAL_ARCHETYPES,
        "extreme_archetypes": EXTREME_ARCHETYPES,
        "intensity_ranges": INTENSITY_RANGES,
        "intensity_priors": INTENSITY_PRIORS,
    }
