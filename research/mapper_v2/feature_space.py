"""Scale-aligned 8D feature space utilities for mapper_v2.

This module ports the weighted 8D extractor from the mapper notebooks and adds
explicit normalizers so extracted EQ features can live in the same scale as
the personalization loop:

    z = 0.0  neutral
    z = 0.5  mild
    z = 1.0  noticeable
    z = 1.5  strong
    z = 2.0  extreme

The key idea is:
    EQ curve -> raw weighted 8D -> contract/statistical scaled 8D in [-2, 2]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable
import ast
import json

import numpy as np
import pandas as pd

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

FREQS_23_DEFAULT = np.asarray(
    [
        20, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
        400, 500, 630, 800, 1000, 2000, 4000, 8000, 10000, 12500, 16000,
    ],
    dtype=np.float64,
)


CONTRACT_EXTREME_DB_DEFAULT = {
    # z=+2 corresponds to approximately this absolute zone value in raw weighted dB.
    # Example: raw bass = +14 dB -> z_bass ~= +2; raw bass = -2 dB -> z_bass ~= -0.29.
    "sub_bass": 16.0,
    "bass": 14.0,
    "lowmid": 6.0,
    "warmth": 6.0,
    "presence": 5.0,
    "clarity": 7.0,
    "air": 12.0,
    "brightness": 8.0,
}


def contract_extreme_db_vector(
    feature_names: list[str] | None = None,
    extreme_db: dict[str, float] | None = None,
) -> np.ndarray:
    """Return per-feature dB values that correspond to |z|=2 in contract scale."""
    names = feature_names or FEATURE_NAMES_8D
    values = extreme_db or CONTRACT_EXTREME_DB_DEFAULT
    return np.asarray([float(values[name]) for name in names], dtype=np.float64)


def parse_array_value(value: Any, dtype=float) -> np.ndarray:
    """Parse arrays stored as real arrays, JSON strings, or Python-list strings."""
    if isinstance(value, np.ndarray):
        return value.astype(dtype)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype)
    if pd.isna(value):
        raise ValueError("Cannot parse NaN as array")
    if isinstance(value, str):
        text = value.strip()
        try:
            return np.asarray(json.loads(text), dtype=dtype)
        except Exception:
            return np.asarray(ast.literal_eval(text), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def gaussian_weights(freqs_hz: Iterable[float], center_hz: float, sigma_oct: float) -> np.ndarray:
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    log_f = np.log2(freqs_hz)
    center = np.log2(float(center_hz))
    x = (log_f - center) / float(sigma_oct)
    w = np.exp(-0.5 * (x ** 2))
    s = w.sum()
    if s > 0:
        w = w / s
    return w.astype(np.float64)


def high_shelf_weights(freqs_hz: Iterable[float], start_hz: float, slope_oct: float) -> np.ndarray:
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    x = (np.log2(freqs_hz) - np.log2(float(start_hz))) / float(slope_oct)
    w = 1.0 / (1.0 + np.exp(-x))
    w = w - w.min()
    s = w.sum()
    if s > 0:
        w = w / s
    return w.astype(np.float64)


def low_shelf_weights(freqs_hz: Iterable[float], end_hz: float, slope_oct: float) -> np.ndarray:
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    x = (np.log2(freqs_hz) - np.log2(float(end_hz))) / float(slope_oct)
    w = 1.0 / (1.0 + np.exp(x))
    w = w - w.min()
    s = w.sum()
    if s > 0:
        w = w / s
    return w.astype(np.float64)


def build_weight_bank_8d(freqs_hz: Iterable[float]) -> dict[str, np.ndarray]:
    """Weighted perceptual feature bank.

    Normal features:
        value = sum(curve * normalized_weights)

    brightness:
        value = sum(curve * high_weights) - sum(curve * low_weights)
    """
    weights: dict[str, np.ndarray] = {}

    weights["sub_bass"] = gaussian_weights(freqs_hz, center_hz=45, sigma_oct=0.75)
    weights["bass"] = gaussian_weights(freqs_hz, center_hz=120, sigma_oct=0.70)
    weights["lowmid"] = gaussian_weights(freqs_hz, center_hz=350, sigma_oct=0.75)
    weights["warmth"] = gaussian_weights(freqs_hz, center_hz=700, sigma_oct=0.75)
    weights["presence"] = gaussian_weights(freqs_hz, center_hz=2500, sigma_oct=0.80)
    weights["clarity"] = gaussian_weights(freqs_hz, center_hz=5500, sigma_oct=0.75)
    weights["air"] = high_shelf_weights(freqs_hz, start_hz=8000, slope_oct=0.75)

    low_w = low_shelf_weights(freqs_hz, end_hz=500, slope_oct=0.85)
    high_w = high_shelf_weights(freqs_hz, start_hz=4000, slope_oct=0.85)
    weights["brightness"] = high_w - low_w

    return weights


def extract_8d_from_curve(
    curve: Iterable[float],
    freqs_hz: Iterable[float],
    weight_bank: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    if weight_bank is None:
        weight_bank = build_weight_bank_8d(freqs_hz)

    values = []
    for name in FEATURE_NAMES_8D:
        w = weight_bank[name]
        values.append(float(np.sum(curve * w)))
    return np.asarray(values, dtype=np.float32)


def add_8d_features(
    df: pd.DataFrame,
    freqs: Iterable[float],
    curve_col: str = "curve_23",
    prefix: str = "z8_raw",
) -> pd.DataFrame:
    df = df.copy()
    weight_bank = build_weight_bank_8d(freqs)
    z_values = np.stack([
        extract_8d_from_curve(curve, freqs_hz=freqs, weight_bank=weight_bank)
        for curve in df[curve_col].values
    ]).astype(np.float32)

    df[f"{prefix}_vector"] = list(z_values)
    for i, name in enumerate(FEATURE_NAMES_8D):
        df[f"{prefix}_{name}"] = z_values[:, i]
    return df


@dataclass
class FeatureScaleNormalizer:
    """Affine normalizer from raw 8D dB features to personalization scale.

    transform:
        z_scaled = clip((z_raw - center) / scale, -clip_value, clip_value)

    A percentile fit with p5->-2 and p95->+2 is a safe default:
        scale = (p95 - p5) / 4
    """

    center: list[float]
    scale: list[float]
    clip_value: float = 2.0
    feature_names: list[str] | None = None
    fit_info: dict[str, Any] | None = None

    @classmethod
    def fit_percentile(
        cls,
        z_raw: np.ndarray,
        lower: float = 5.0,
        upper: float = 95.0,
        clip_value: float = 2.0,
        min_scale: float = 1e-3,
        feature_names: list[str] | None = None,
        manual_scale_multipliers: dict[str, float] | None = None,
    ) -> "FeatureScaleNormalizer":
        z_raw = np.asarray(z_raw, dtype=np.float64)
        p_low = np.percentile(z_raw, lower, axis=0)
        p_mid = np.percentile(z_raw, 50.0, axis=0)
        p_high = np.percentile(z_raw, upper, axis=0)
        scale = (p_high - p_low) / (2.0 * clip_value)
        scale = np.maximum(scale, min_scale)

        names = feature_names or FEATURE_NAMES_8D
        if manual_scale_multipliers:
            scale = scale.copy()
            for name, mult in manual_scale_multipliers.items():
                if name in names:
                    scale[names.index(name)] *= float(mult)

        return cls(
            center=p_mid.tolist(),
            scale=scale.tolist(),
            clip_value=float(clip_value),
            feature_names=list(names),
            fit_info={
                "method": "percentile",
                "lower": lower,
                "upper": upper,
                "p_low": p_low.tolist(),
                "p_mid": p_mid.tolist(),
                "p_high": p_high.tolist(),
                "manual_scale_multipliers": manual_scale_multipliers or {},
            },
        )


    @classmethod
    def fit_contract(
        cls,
        extreme_db: dict[str, float] | None = None,
        clip_value: float = 2.0,
        feature_names: list[str] | None = None,
        center: np.ndarray | list[float] | None = None,
    ) -> "FeatureScaleNormalizer":
        """Build the product-scale normalizer discussed for personalization.

        Unlike percentile scaling, this does NOT map dataset percentiles to [-2, 2].
        It maps raw weighted dB values to a perceptual contract:

            raw_feature_db = +extreme_db[feature] -> z = +2
            raw_feature_db = -extreme_db[feature] -> z = -2

        Therefore a small real curve such as bass=-2 dB becomes about -0.29
        when the bass extreme is 14 dB.
        """
        names = feature_names or FEATURE_NAMES_8D
        extreme_vec = contract_extreme_db_vector(names, extreme_db)
        scale = extreme_vec / float(clip_value)
        if center is None:
            center_vec = np.zeros(len(names), dtype=np.float64)
        else:
            center_vec = np.asarray(center, dtype=np.float64)
        return cls(
            center=center_vec.tolist(),
            scale=scale.tolist(),
            clip_value=float(clip_value),
            feature_names=list(names),
            fit_info={
                "method": "contract_db",
                "extreme_db": {name: float(extreme_vec[i]) for i, name in enumerate(names)},
                "meaning": "raw weighted feature dB divided by extreme_db/2; z=2 means an explicit extreme dB value, not a dataset percentile",
            },
        )

    def transform(self, z_raw: np.ndarray) -> np.ndarray:
        z_raw = np.asarray(z_raw, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        z = (z_raw - center) / scale
        z = np.clip(z, -self.clip_value, self.clip_value)
        return z.astype(np.float32)

    def inverse_transform(self, z_scaled: np.ndarray) -> np.ndarray:
        z_scaled = np.asarray(z_scaled, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        return (z_scaled * scale + center).astype(np.float32)

    def to_frame(self) -> pd.DataFrame:
        names = self.feature_names or FEATURE_NAMES_8D
        df = pd.DataFrame({
            "feature": names,
            "center_raw": self.center,
            "scale_raw_per_z_unit": self.scale,
        })
        if self.fit_info:
            for key in ["p_low", "p_mid", "p_high"]:
                if key in self.fit_info:
                    df[key] = self.fit_info[key]
        return df

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "FeatureScaleNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def add_scaled_8d_features(
    df: pd.DataFrame,
    normalizer: FeatureScaleNormalizer,
    raw_col: str = "z8_raw_vector",
    prefix: str = "z8_scaled",
) -> pd.DataFrame:
    df = df.copy()
    raw = np.stack([parse_array_value(v, dtype=np.float32) for v in df[raw_col].values])
    scaled = normalizer.transform(raw)
    df[f"{prefix}_vector"] = list(scaled.astype(np.float32))
    names = normalizer.feature_names or FEATURE_NAMES_8D
    for i, name in enumerate(names):
        df[f"{prefix}_{name}"] = scaled[:, i]
    return df


def extract_8d_scaled_from_curve(
    curve: Iterable[float],
    freqs_hz: Iterable[float],
    normalizer: FeatureScaleNormalizer,
    weight_bank: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    z_raw = extract_8d_from_curve(curve, freqs_hz=freqs_hz, weight_bank=weight_bank)
    return normalizer.transform(z_raw[None, :])[0]


def prepare_curve_columns(
    df: pd.DataFrame,
    curve_col: str = "curve_23",
    freqs_col: str = "freqs_23",
    default_freqs: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse curve/frequency columns from older mapper notebooks."""
    df = df.copy()

    if f"{curve_col}_json" in df.columns and curve_col not in df.columns:
        df[curve_col] = df[f"{curve_col}_json"]
    if curve_col not in df.columns:
        raise ValueError(f"DataFrame must contain {curve_col} or {curve_col}_json")
    df[curve_col] = df[curve_col].apply(lambda x: parse_array_value(x, dtype=np.float32))

    if f"{freqs_col}_json" in df.columns and freqs_col not in df.columns:
        df[freqs_col] = df[f"{freqs_col}_json"]
    if freqs_col in df.columns:
        df[freqs_col] = df[freqs_col].apply(lambda x: parse_array_value(x, dtype=float))
        freqs = np.asarray(df.iloc[0][freqs_col], dtype=np.float64)
    elif default_freqs is not None:
        freqs = np.asarray(default_freqs, dtype=np.float64)
        df[freqs_col] = [freqs.copy() for _ in range(len(df))]
    else:
        freqs = FREQS_23_DEFAULT.copy()
        df[freqs_col] = [freqs.copy() for _ in range(len(df))]
    return df, freqs
