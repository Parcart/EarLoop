"""Interpretable 8D->23-band EQ basis and safety utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .feature_space import (
    FEATURE_NAMES_8D,
    FREQS_23_DEFAULT,
    build_weight_bank_8d,
    extract_8d_scaled_from_curve,
    FeatureScaleNormalizer,
)

MAX_GAIN_DB_DEFAULT = {
    "sub_bass": 16.0,
    "bass": 14.0,
    "lowmid": 6.0,
    "warmth": 6.0,
    "presence": 5.0,
    "clarity": 7.0,
    "air": 12.0,
    "brightness": 8.0,
}


def normalize_peak_signed(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    m = np.max(np.abs(v))
    if m < eps:
        return np.zeros_like(v, dtype=np.float64)
    return v / m


def build_mapper_basis_8d(freqs_hz: Iterable[float], feature_names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Build generative EQ basis templates.

    Differs from extractor weights:
      extractor weights are L1-normalized for measurement;
      mapper templates are peak-normalized so gain_db means real dB peak.
    """
    names = feature_names or FEATURE_NAMES_8D
    weight_bank = build_weight_bank_8d(freqs_hz)
    basis = {}
    for name in names:
        basis[name] = normalize_peak_signed(weight_bank[name])
    return basis


def build_basis_matrix_8d(freqs_hz: Iterable[float], feature_names: list[str] | None = None) -> np.ndarray:
    names = feature_names or FEATURE_NAMES_8D
    basis = build_mapper_basis_8d(freqs_hz, names)
    return np.stack([basis[name] for name in names]).astype(np.float32)


def z_to_gain_db(
    z: np.ndarray,
    feature_names: list[str] | None = None,
    max_gain_db: dict[str, float] | None = None,
    clip_value: float = 2.0,
    slope: float = 1.25,
) -> np.ndarray:
    """Convert z in [-2,2] to per-feature amplitude in dB.

    tanh keeps the response smooth while allowing z=2 to approach the feature's
    maximum gain.
    """
    names = feature_names or FEATURE_NAMES_8D
    max_gain_db = max_gain_db or MAX_GAIN_DB_DEFAULT
    z = np.asarray(z, dtype=np.float64)
    z = np.clip(z, -clip_value, clip_value)
    gains = []
    for i, name in enumerate(names):
        max_gain = float(max_gain_db.get(name, 8.0))
        gains.append(max_gain * np.tanh(slope * z[i] / clip_value))
    return np.asarray(gains, dtype=np.float64)


def smooth_curve(curve: np.ndarray, strength: float = 0.15, passes: int = 1) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64).copy()
    if strength <= 0 or passes <= 0 or len(curve) < 3:
        return curve.astype(np.float32)
    for _ in range(passes):
        old = curve.copy()
        curve[1:-1] = (1.0 - strength) * old[1:-1] + strength * 0.5 * (old[:-2] + old[2:])
    return curve.astype(np.float32)


def apply_safety_layer(
    curve: np.ndarray,
    max_boost_db: float = 16.0,
    max_cut_db: float = -12.0,
    smoothing_strength: float = 0.12,
    smoothing_passes: int = 1,
) -> tuple[np.ndarray, dict[str, float]]:
    raw = np.asarray(curve, dtype=np.float64)
    clipped = np.clip(raw, max_cut_db, max_boost_db)
    smoothed = smooth_curve(clipped, strength=smoothing_strength, passes=smoothing_passes)
    preamp_db = -max(0.0, float(np.max(smoothed)))
    meta = {
        "raw_max_db": float(np.max(raw)),
        "raw_min_db": float(np.min(raw)),
        "safe_max_db": float(np.max(smoothed)),
        "safe_min_db": float(np.min(smoothed)),
        "required_preamp_db": float(preamp_db),
        "clipped_points": float(np.sum((raw > max_boost_db) | (raw < max_cut_db))),
    }
    return smoothed.astype(np.float32), meta


@dataclass
class InterpretableMapper8D:
    freqs_hz: np.ndarray | None = None
    feature_names: list[str] | None = None
    max_gain_db: dict[str, float] | None = None
    clip_value: float = 2.0
    slope: float = 1.25
    safety: bool = True
    max_boost_db: float = 16.0
    max_cut_db: float = -12.0
    smoothing_strength: float = 0.12
    smoothing_passes: int = 1

    def __post_init__(self) -> None:
        self.freqs_hz = np.asarray(self.freqs_hz if self.freqs_hz is not None else FREQS_23_DEFAULT, dtype=np.float64)
        self.feature_names = self.feature_names or FEATURE_NAMES_8D
        self.max_gain_db = self.max_gain_db or MAX_GAIN_DB_DEFAULT.copy()
        self.basis_matrix = build_basis_matrix_8d(self.freqs_hz, self.feature_names).astype(np.float32)

    def map_one(self, z_scaled: np.ndarray, return_meta: bool = False):
        z_scaled = np.asarray(z_scaled, dtype=np.float64)
        gains = z_to_gain_db(
            z_scaled,
            feature_names=self.feature_names,
            max_gain_db=self.max_gain_db,
            clip_value=self.clip_value,
            slope=self.slope,
        )
        raw_curve = gains @ self.basis_matrix
        if self.safety:
            curve, safety_meta = apply_safety_layer(
                raw_curve,
                max_boost_db=self.max_boost_db,
                max_cut_db=self.max_cut_db,
                smoothing_strength=self.smoothing_strength,
                smoothing_passes=self.smoothing_passes,
            )
        else:
            curve = raw_curve.astype(np.float32)
            safety_meta = {}

        if return_meta:
            return curve, {"gains_db": gains.astype(np.float32), **safety_meta}
        return curve

    def map_batch(self, z_scaled: np.ndarray) -> np.ndarray:
        z_scaled = np.asarray(z_scaled, dtype=np.float64)
        return np.stack([self.map_one(z) for z in z_scaled]).astype(np.float32)

    def cycle_reconstruct(self, z_scaled: np.ndarray, normalizer: FeatureScaleNormalizer) -> np.ndarray:
        curves = self.map_batch(z_scaled)
        return np.stack([
            extract_8d_scaled_from_curve(c, self.freqs_hz, normalizer=normalizer)
            for c in curves
        ]).astype(np.float32)


def make_axis_sweep(feature: str, values: Iterable[float], feature_names: list[str] | None = None) -> np.ndarray:
    names = feature_names or FEATURE_NAMES_8D
    idx = names.index(feature)
    rows = []
    for value in values:
        z = np.zeros(len(names), dtype=np.float32)
        z[idx] = float(value)
        rows.append(z)
    return np.stack(rows).astype(np.float32)


def make_archetype_presets() -> dict[str, np.ndarray]:
    """Small preset bank for mapper sanity checks in z scale [-2,2]."""
    names = FEATURE_NAMES_8D
    def v(**kwargs):
        z = np.zeros(len(names), dtype=np.float32)
        for k, val in kwargs.items():
            z[names.index(k)] = float(val)
        return z
    return {
        "neutral": v(),
        "bass_lover": v(sub_bass=1.0, bass=1.2, brightness=-0.2),
        "basshead_extreme": v(sub_bass=2.0, bass=2.0, lowmid=-0.3, brightness=-0.3),
        "club_curve": v(sub_bass=1.5, bass=1.6, clarity=0.4, air=0.6, brightness=0.2),
        "warm_dark": v(lowmid=0.7, warmth=1.3, brightness=-1.0, air=-0.4),
        "bright_air": v(clarity=0.8, air=1.8, brightness=0.8, bass=-0.2),
        "extreme_vshape": v(sub_bass=1.8, bass=1.4, presence=-0.5, clarity=1.0, air=1.4, brightness=0.7),
    }
