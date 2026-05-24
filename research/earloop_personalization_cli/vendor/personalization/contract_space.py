from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .state import FEATURE_NAMES_8D


# z-contract used by the learned/contract mapper.
# These numbers are raw weighted-feature dB per one z-unit.
# Therefore z=2 means the configured extreme value.
CONTRACT_SCALE_RAW_DB_PER_Z_UNIT: dict[str, float] = {
    "sub_bass": 8.0,      # z=2 -> 16 dB
    "bass": 7.0,          # z=2 -> 14 dB
    "lowmid": 3.0,        # z=2 -> 6 dB
    "warmth": 3.0,        # z=2 -> 6 dB
    "presence": 2.5,      # z=2 -> 5 dB
    "clarity": 3.5,       # z=2 -> 7 dB
    "air": 6.0,           # z=2 -> 12 dB
    "brightness": 4.0,    # z=2 -> 8 dB
}

CONTRACT_Z_CLIP = 2.0


@dataclass(frozen=True)
class ContractScale:
    """Explicit dB-contract for the 8D preference space.

    The state vector is no longer percentile-normalized. It is a product/user
    contract:

        z = 0.0  -> neutral
        z = 0.5  -> mild
        z = 1.0  -> noticeable
        z = 1.5  -> strong
        z = 2.0  -> extreme
    """

    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES_8D)
    raw_db_per_z_unit: tuple[float, ...] = tuple(
        CONTRACT_SCALE_RAW_DB_PER_Z_UNIT[name] for name in FEATURE_NAMES_8D
    )
    clip_value: float = CONTRACT_Z_CLIP

    @property
    def raw_db_per_z_unit_array(self) -> np.ndarray:
        return np.asarray(self.raw_db_per_z_unit, dtype=np.float64)

    @property
    def extreme_raw_db(self) -> np.ndarray:
        return self.raw_db_per_z_unit_array * float(self.clip_value)

    def clip(self, z: np.ndarray) -> np.ndarray:
        return clip_contract_z(z, self.clip_value)

    def to_raw_feature_db(self, z_contract: np.ndarray) -> np.ndarray:
        return contract_z_to_raw_feature_db(z_contract, self)

    def from_raw_feature_db(self, z_raw_db: np.ndarray) -> np.ndarray:
        return raw_feature_db_to_contract_z(z_raw_db, self)

    def table(self) -> pd.DataFrame:
        return contract_scale_table(self)


DEFAULT_CONTRACT_SCALE = ContractScale()


# Feature weights for distances in contract-space. Low-level tonal axes are
# slightly more important because small movement there maps to larger dB shifts.
DEFAULT_CONTRACT_DISTANCE_WEIGHTS = np.asarray(
    [1.15, 1.10, 0.90, 0.90, 0.95, 1.00, 1.05, 1.10],
    dtype=np.float64,
)
DEFAULT_CONTRACT_DISTANCE_WEIGHTS = (
    DEFAULT_CONTRACT_DISTANCE_WEIGHTS / DEFAULT_CONTRACT_DISTANCE_WEIGHTS.mean()
)


def contract_scale_table(scale: ContractScale = DEFAULT_CONTRACT_SCALE) -> pd.DataFrame:
    rows = []
    for name, per_unit, extreme in zip(
        scale.feature_names,
        scale.raw_db_per_z_unit_array,
        scale.extreme_raw_db,
    ):
        rows.append({
            "feature": name,
            "z_0": 0.0,
            "raw_db_per_z_unit": float(per_unit),
            "raw_db_at_z_1": float(per_unit),
            "raw_db_at_z_2": float(extreme),
        })
    return pd.DataFrame(rows)


def clip_contract_z(z: np.ndarray, clip_value: float | None = CONTRACT_Z_CLIP) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if clip_value is None:
        return z
    return np.clip(z, -float(clip_value), float(clip_value))


def contract_z_to_raw_feature_db(
    z_contract: np.ndarray,
    scale: ContractScale = DEFAULT_CONTRACT_SCALE,
) -> np.ndarray:
    z = np.asarray(z_contract, dtype=np.float64)
    return z * scale.raw_db_per_z_unit_array


def raw_feature_db_to_contract_z(
    z_raw_db: np.ndarray,
    scale: ContractScale = DEFAULT_CONTRACT_SCALE,
) -> np.ndarray:
    raw = np.asarray(z_raw_db, dtype=np.float64)
    z = raw / (scale.raw_db_per_z_unit_array + 1e-8)
    return scale.clip(z)


def contract_distance(
    a: np.ndarray,
    b: np.ndarray,
    weights: np.ndarray | None = DEFAULT_CONTRACT_DISTANCE_WEIGHTS,
) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    if weights is None:
        return float(np.linalg.norm(diff))
    w = np.asarray(weights, dtype=np.float64)
    return float(np.sqrt(np.mean(w * diff * diff)) * np.sqrt(len(diff)))


def contract_pair_distance(
    z_a: np.ndarray,
    z_b: np.ndarray,
    weights: np.ndarray | None = DEFAULT_CONTRACT_DISTANCE_WEIGHTS,
) -> float:
    return contract_distance(z_a, z_b, weights=weights)


def z_contract_series(z: np.ndarray, name: str = "z_contract") -> pd.Series:
    z = np.asarray(z, dtype=np.float64)
    return pd.Series(z, index=list(FEATURE_NAMES_8D), name=name)


CONTRACT_ARCHETYPES_8D: dict[str, np.ndarray] = {
    "neutral": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "basshead": np.asarray([1.65, 1.45, -0.25, 0.15, -0.20, 0.10, 0.25, -0.35], dtype=np.float64),
    "warm_dark": np.asarray([0.65, 0.80, 0.50, 1.15, -0.55, -0.65, -0.75, -1.10], dtype=np.float64),
    "bright_air": np.asarray([-0.35, -0.25, -0.35, -0.20, 0.55, 1.10, 1.55, 1.35], dtype=np.float64),
    "v_shape": np.asarray([1.15, 0.95, -0.85, -0.45, 0.35, 0.75, 1.00, 0.65], dtype=np.float64),
    "vocal_clear": np.asarray([-0.25, -0.15, -0.30, 0.10, 1.10, 0.95, 0.45, 0.35], dtype=np.float64),
    "soft_warm": np.asarray([0.30, 0.40, 0.35, 0.85, -0.10, -0.15, 0.05, -0.20], dtype=np.float64),
    "lowmid_cut": np.asarray([0.45, 0.30, -1.25, -0.65, 0.35, 0.45, 0.30, 0.25], dtype=np.float64),
}


def archetype_table() -> pd.DataFrame:
    rows = []
    for name, z in CONTRACT_ARCHETYPES_8D.items():
        row = {"archetype": name}
        row.update({feature: float(value) for feature, value in zip(FEATURE_NAMES_8D, z)})
        rows.append(row)
    return pd.DataFrame(rows)


def make_axis_sweep_contract(
    feature: str,
    values: Iterable[float],
    feature_names: Iterable[str] = FEATURE_NAMES_8D,
) -> np.ndarray:
    names = list(feature_names)
    if feature not in names:
        raise ValueError(f"Unknown feature: {feature}")
    idx = names.index(feature)
    rows = []
    for value in values:
        z = np.zeros(len(names), dtype=np.float64)
        z[idx] = float(value)
        rows.append(z)
    return np.stack(rows, axis=0)
