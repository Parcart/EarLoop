"""Metrics for scale-aligned learned/interpretable mappers."""

from __future__ import annotations

from typing import Callable, Iterable
import numpy as np
import pandas as pd

from .feature_space import FEATURE_NAMES_8D, extract_8d_scaled_from_curve, FeatureScaleNormalizer
from .mapper_basis import build_mapper_basis_8d, make_axis_sweep


def safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < eps:
        return float("nan")
    return float(np.dot(a, b) / den)


def normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    return safe_corr(a, b)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def make_zone_masks(freqs_hz: Iterable[float]) -> dict[str, np.ndarray]:
    f = np.asarray(freqs_hz, dtype=float)
    return {
        "low_end": f <= 160,
        "low_mid": (f > 160) & (f <= 630),
        "mid": (f > 630) & (f <= 2000),
        "presence": (f > 2000) & (f <= 5000),
        "air": f > 5000,
    }


def expected_target_mask(feature: str, freqs_hz: Iterable[float]) -> np.ndarray:
    f = np.asarray(freqs_hz, dtype=float)
    if feature == "sub_bass":
        return f <= 80
    if feature == "bass":
        return (f >= 50) & (f <= 200)
    if feature == "lowmid":
        return (f >= 160) & (f <= 500)
    if feature == "warmth":
        return (f >= 250) & (f <= 1000)
    if feature == "presence":
        return (f >= 1000) & (f <= 4000)
    if feature == "clarity":
        return (f >= 3000) & (f <= 8000)
    if feature == "air":
        return f >= 8000
    if feature == "brightness":
        return (f <= 500) | (f >= 4000)
    return np.ones_like(f, dtype=bool)


def predict_with_callable(mapper_predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> np.ndarray:
    Y = mapper_predict(np.asarray(X, dtype=np.float32))
    return np.asarray(Y, dtype=np.float32)


def compute_cycle_consistency(
    mapper_predict: Callable[[np.ndarray], np.ndarray],
    z_samples: np.ndarray,
    freqs_hz: Iterable[float],
    normalizer: FeatureScaleNormalizer,
) -> pd.DataFrame:
    z_samples = np.asarray(z_samples, dtype=np.float32)
    curves = predict_with_callable(mapper_predict, z_samples)
    z_recon = np.stack([
        extract_8d_scaled_from_curve(curve, freqs_hz, normalizer=normalizer)
        for curve in curves
    ]).astype(np.float32)
    err = z_recon - z_samples
    rows = []
    for i, name in enumerate(normalizer.feature_names or FEATURE_NAMES_8D):
        rows.append({
            "feature": name,
            "cycle_mae": float(np.mean(np.abs(err[:, i]))),
            "cycle_mse": float(np.mean(err[:, i] ** 2)),
            "cycle_corr": safe_corr(z_samples[:, i], z_recon[:, i]),
        })
    rows.append({
        "feature": "__overall__",
        "cycle_mae": float(np.mean(np.abs(err))),
        "cycle_mse": float(np.mean(err ** 2)),
        "cycle_corr": safe_corr(z_samples.reshape(-1), z_recon.reshape(-1)),
    })
    return pd.DataFrame(rows)


def compute_basis_alignment(
    mapper_predict: Callable[[np.ndarray], np.ndarray],
    freqs_hz: Iterable[float],
    feature_names: list[str] | None = None,
    values: Iterable[float] | None = None,
) -> pd.DataFrame:
    names = feature_names or FEATURE_NAMES_8D
    if values is None:
        values = np.linspace(-2.0, 2.0, 9)
    values = np.asarray(list(values), dtype=np.float32)
    basis = build_mapper_basis_8d(freqs_hz, names)

    rows = []
    for feature in names:
        X = make_axis_sweep(feature, values, names)
        curves = predict_with_callable(mapper_predict, X)
        center_idx = int(np.argmin(np.abs(values)))
        center_curve = curves[center_idx]
        expected = basis[feature]
        alignments = []
        for i, value in enumerate(values):
            if i == center_idx:
                continue
            diff = curves[i] - center_curve
            direction = np.sign(value - values[center_idx])
            corr = normalized_corr(diff, direction * expected)
            if not np.isnan(corr):
                alignments.append(corr)
        rows.append({
            "feature": feature,
            "basis_alignment": float(np.mean(alignments)) if alignments else 0.0,
        })
    return pd.DataFrame(rows)


def collect_sweep_slopes(
    mapper_predict: Callable[[np.ndarray], np.ndarray],
    freqs_hz: Iterable[float],
    feature_names: list[str] | None = None,
    values: Iterable[float] | None = None,
) -> pd.DataFrame:
    names = feature_names or FEATURE_NAMES_8D
    if values is None:
        values = np.linspace(-2.0, 2.0, 9)
    values = np.asarray(list(values), dtype=np.float32)
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    rows = []
    for feature in names:
        X = make_axis_sweep(feature, values, names)
        curves = predict_with_callable(mapper_predict, X)
        slopes = np.asarray([np.polyfit(values, curves[:, j], deg=1)[0] for j in range(curves.shape[1])])
        row = {"feature": feature}
        for freq, slope in zip(freqs_hz, slopes):
            row[f"slope_{freq:g}"] = float(slope)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sweep_sensitivity(df_slopes: pd.DataFrame, freqs_hz: Iterable[float]) -> pd.DataFrame:
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    slope_cols = [c for c in df_slopes.columns if c.startswith("slope_")]
    rows = []
    for _, row in df_slopes.iterrows():
        feature = row["feature"]
        slopes = row[slope_cols].to_numpy(dtype=float)
        target_mask = expected_target_mask(feature, freqs_hz)
        off_mask = ~target_mask
        target_response = float(np.mean(np.abs(slopes[target_mask]))) if np.any(target_mask) else 0.0
        off_response = float(np.mean(np.abs(slopes[off_mask]))) if np.any(off_mask) else 0.0
        ratio = target_response / (off_response + 1e-8)
        target_slopes = slopes[target_mask]
        if np.mean(np.abs(target_slopes)) < 1e-8:
            sign_consistency = 0.0
        else:
            pos_frac = float(np.mean(target_slopes > 0))
            neg_frac = float(np.mean(target_slopes < 0))
            sign_consistency = max(pos_frac, neg_frac)
        rows.append({
            "feature": feature,
            "target_response": target_response,
            "off_response": off_response,
            "target_off_ratio": float(ratio),
            "sign_consistency": float(sign_consistency),
            "max_abs_slope": float(np.max(np.abs(slopes))),
        })
    return pd.DataFrame(rows)


def compute_scale_sweep_metrics(
    mapper_predict: Callable[[np.ndarray], np.ndarray],
    freqs_hz: Iterable[float],
    feature_names: list[str] | None = None,
    values: Iterable[float] | None = None,
) -> pd.DataFrame:
    names = feature_names or FEATURE_NAMES_8D
    if values is None:
        values = [0.0, 0.5, 1.0, 1.5, 2.0]
    values = np.asarray(values, dtype=np.float32)
    rows = []
    for feature in names:
        X = make_axis_sweep(feature, values, names)
        curves = predict_with_callable(mapper_predict, X)
        mask = expected_target_mask(feature, freqs_hz)
        target_energy = np.asarray([float(np.mean(np.abs(c[mask]))) for c in curves])
        diffs = np.diff(target_energy)
        monotonicity = float(np.mean(diffs >= -1e-5)) if len(diffs) else 1.0
        rows.append({
            "feature": feature,
            "monotonicity_score": monotonicity,
            "target_energy_z0": float(target_energy[0]),
            "target_energy_z2": float(target_energy[-1]),
            "scale_gain_z2_minus_z0": float(target_energy[-1] - target_energy[0]),
        })
    return pd.DataFrame(rows)


def compute_extreme_coverage(
    mapper_predict: Callable[[np.ndarray], np.ndarray],
    freqs_hz: Iterable[float],
    feature_names: list[str] | None = None,
    extreme_value: float = 2.0,
) -> pd.DataFrame:
    names = feature_names or FEATURE_NAMES_8D
    rows = []
    for feature in names:
        X = make_axis_sweep(feature, [0.0, extreme_value], names)
        curves = predict_with_callable(mapper_predict, X)
        diff = curves[1] - curves[0]
        mask = expected_target_mask(feature, freqs_hz)
        rows.append({
            "feature": feature,
            "extreme_peak_abs_db": float(np.max(np.abs(diff[mask])) if np.any(mask) else np.max(np.abs(diff))),
            "extreme_mean_abs_target_db": float(np.mean(np.abs(diff[mask])) if np.any(mask) else np.mean(np.abs(diff))),
            "extreme_mean_abs_off_db": float(np.mean(np.abs(diff[~mask])) if np.any(~mask) else 0.0),
        })
    return pd.DataFrame(rows)


def summarize_mapper_quality(
    model_name: str,
    mapper_predict: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    freqs_hz: Iterable[float],
    normalizer: FeatureScaleNormalizer | None = None,
) -> dict[str, float | str]:
    Y_pred = predict_with_callable(mapper_predict, X_test)
    align = compute_basis_alignment(mapper_predict, freqs_hz)
    slopes = collect_sweep_slopes(mapper_predict, freqs_hz)
    sens = summarize_sweep_sensitivity(slopes, freqs_hz)
    scale = compute_scale_sweep_metrics(mapper_predict, freqs_hz)
    extreme = compute_extreme_coverage(mapper_predict, freqs_hz)
    row: dict[str, float | str] = {
        "model": model_name,
        "test_mse": mse(Y_test, Y_pred),
        "test_mae": mae(Y_test, Y_pred),
        "mean_basis_alignment": float(align["basis_alignment"].mean()),
        "mean_target_off_ratio": float(sens["target_off_ratio"].mean()),
        "mean_sign_consistency": float(sens["sign_consistency"].mean()),
        "mean_monotonicity_score": float(scale["monotonicity_score"].mean()),
        "mean_scale_gain_z2_minus_z0": float(scale["scale_gain_z2_minus_z0"].mean()),
        "mean_extreme_peak_abs_db": float(extreme["extreme_peak_abs_db"].mean()),
        "mean_extreme_target_abs_db": float(extreme["extreme_mean_abs_target_db"].mean()),
        "mean_extreme_off_abs_db": float(extreme["extreme_mean_abs_off_db"].mean()),
    }
    if normalizer is not None:
        cycle = compute_cycle_consistency(mapper_predict, X_test, freqs_hz, normalizer)
        overall = cycle[cycle["feature"] == "__overall__"].iloc[0]
        row["cycle_mse"] = float(overall["cycle_mse"])
        row["cycle_mae"] = float(overall["cycle_mae"])
        row["cycle_corr"] = float(overall["cycle_corr"])
    return row
