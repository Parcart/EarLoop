from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .contract_space import DEFAULT_CONTRACT_SCALE, ContractScale, clip_contract_z
from .state import FEATURE_NAMES_8D


FREQS_23_DEFAULT = np.asarray([
    20.0, 50.0, 83.0, 120.0, 159.5102997, 200.0437622, 254.0482178,
    308.5627136, 383.0, 443.8639832, 622.0435181, 798.0671387,
    1000.0, 1485.9825439, 1875.0, 2368.0808105, 3389.6481934,
    4365.3632812, 6934.2607422, 8568.9951172, 12000.0, 14000.0, 16000.0,
], dtype=np.float64)


def _log2_freqs(freqs_hz: np.ndarray) -> np.ndarray:
    return np.log2(np.asarray(freqs_hz, dtype=np.float64))


def gaussian_shape(freqs_hz: np.ndarray, center_hz: float, sigma_oct: float) -> np.ndarray:
    log_f = _log2_freqs(freqs_hz)
    center = np.log2(float(center_hz))
    x = (log_f - center) / float(sigma_oct)
    y = np.exp(-0.5 * x * x)
    peak = float(np.max(np.abs(y)))
    return y / (peak + 1e-8)


def high_shelf_shape(freqs_hz: np.ndarray, start_hz: float, slope_oct: float) -> np.ndarray:
    x = (_log2_freqs(freqs_hz) - np.log2(float(start_hz))) / float(slope_oct)
    y = 1.0 / (1.0 + np.exp(-x))
    y = y - float(np.min(y))
    return y / (float(np.max(np.abs(y))) + 1e-8)


def low_shelf_shape(freqs_hz: np.ndarray, end_hz: float, slope_oct: float) -> np.ndarray:
    x = (_log2_freqs(freqs_hz) - np.log2(float(end_hz))) / float(slope_oct)
    y = 1.0 / (1.0 + np.exp(x))
    y = y - float(np.min(y))
    return y / (float(np.max(np.abs(y))) + 1e-8)


def moving_average_smooth(curve: np.ndarray, strength: float = 0.15, passes: int = 1) -> np.ndarray:
    """Tiny smoothing that keeps the original shape mostly intact."""
    curve = np.asarray(curve, dtype=np.float64).copy()
    if strength <= 0 or passes <= 0 or len(curve) < 3:
        return curve
    strength = float(np.clip(strength, 0.0, 1.0))
    for _ in range(int(passes)):
        padded = np.pad(curve, (1, 1), mode="edge")
        avg = (padded[:-2] + padded[1:-1] + padded[2:]) / 3.0
        curve = (1.0 - strength) * curve + strength * avg
    return curve


@dataclass
class MapperSafetyConfig:
    max_abs_db: float = 16.0
    max_boost_db: float = 12.0
    max_cut_db: float = 16.0
    smoothing_strength: float = 0.10
    smoothing_passes: int = 1
    subtract_headroom: bool = False


def apply_curve_safety(curve: np.ndarray, cfg: MapperSafetyConfig | None = None) -> np.ndarray:
    cfg = MapperSafetyConfig() if cfg is None else cfg
    y = np.asarray(curve, dtype=np.float64).copy()
    y = np.clip(y, -float(cfg.max_cut_db), float(cfg.max_boost_db))
    y = np.clip(y, -float(cfg.max_abs_db), float(cfg.max_abs_db))
    y = moving_average_smooth(y, strength=cfg.smoothing_strength, passes=cfg.smoothing_passes)
    if cfg.subtract_headroom:
        max_boost = float(np.max(y))
        if max_boost > 0:
            y = y - max_boost
    return y.astype(np.float32)


class InterpretableContractMapper8D:
    """Manual z-contract -> 23-band EQ mapper used as a stable prior.

    This mapper is intentionally not a reconstruction model. It encodes the
    product contract for the live personalization loop. A learned mapper can be
    compared against it or replace it behind the same interface.
    """

    def __init__(
        self,
        freqs_hz: Iterable[float] | None = None,
        contract_scale: ContractScale = DEFAULT_CONTRACT_SCALE,
        safety: bool = True,
        safety_config: MapperSafetyConfig | None = None,
    ) -> None:
        self.freqs_hz = FREQS_23_DEFAULT if freqs_hz is None else np.asarray(freqs_hz, dtype=np.float64)
        self.contract_scale = contract_scale
        self.safety = bool(safety)
        self.safety_config = MapperSafetyConfig() if safety_config is None else safety_config
        self.basis_matrix = self._build_basis_matrix()

    def _build_basis_matrix(self) -> np.ndarray:
        f = self.freqs_hz
        basis = {}
        basis["sub_bass"] = gaussian_shape(f, center_hz=45.0, sigma_oct=0.75)
        basis["bass"] = gaussian_shape(f, center_hz=120.0, sigma_oct=0.70)
        basis["lowmid"] = gaussian_shape(f, center_hz=350.0, sigma_oct=0.75)
        basis["warmth"] = gaussian_shape(f, center_hz=700.0, sigma_oct=0.75)
        basis["presence"] = gaussian_shape(f, center_hz=2500.0, sigma_oct=0.80)
        basis["clarity"] = gaussian_shape(f, center_hz=5500.0, sigma_oct=0.75)
        basis["air"] = high_shelf_shape(f, start_hz=8000.0, slope_oct=0.75)

        low = low_shelf_shape(f, end_hz=500.0, slope_oct=0.85)
        high = high_shelf_shape(f, start_hz=4000.0, slope_oct=0.85)
        tilt = high - low
        basis["brightness"] = tilt / (float(np.max(np.abs(tilt))) + 1e-8)

        return np.stack([basis[name] for name in FEATURE_NAMES_8D], axis=0).astype(np.float64)

    def map_batch(self, z_contract: np.ndarray) -> np.ndarray:
        z = np.asarray(z_contract, dtype=np.float64)
        if z.ndim == 1:
            z = z[None, :]
        z = clip_contract_z(z, self.contract_scale.clip_value)
        amplitudes = self.contract_scale.to_raw_feature_db(z)
        curves = amplitudes @ self.basis_matrix
        if self.safety:
            curves = np.stack([apply_curve_safety(c, self.safety_config) for c in curves], axis=0)
        return curves.astype(np.float32)

    def map_one(self, z_contract: np.ndarray) -> np.ndarray:
        return self.map_batch(np.asarray(z_contract, dtype=np.float64)[None, :])[0]


class CallableContractMapper:
    """Adapter for a learned mapper callable.

    The callable must accept a [N, 8] numpy float32 array and return [N, 23].
    This keeps the personalization loop independent from a specific PyTorch
    architecture used in mapper_v2.
    """

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray], freqs_hz: Iterable[float] | None = None) -> None:
        self.predict_fn = predict_fn
        self.freqs_hz = FREQS_23_DEFAULT if freqs_hz is None else np.asarray(freqs_hz, dtype=np.float64)

    def map_batch(self, z_contract: np.ndarray) -> np.ndarray:
        x = np.asarray(z_contract, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return np.asarray(self.predict_fn(x), dtype=np.float32)

    def map_one(self, z_contract: np.ndarray) -> np.ndarray:
        return self.map_batch(np.asarray(z_contract, dtype=np.float32)[None, :])[0]


class TorchScriptContractMapper:
    """Optional runtime for a TorchScript mapper.

    Prefer exporting Model B to TorchScript/ONNX for product integration. The
    regular .pt checkpoint from mapper_v2 may require the original Python model
    class; TorchScript avoids that dependency.
    """

    def __init__(self, model_path: str | Path, freqs_hz: Iterable[float] | None = None, device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        self.freqs_hz = FREQS_23_DEFAULT if freqs_hz is None else np.asarray(freqs_hz, dtype=np.float64)

    def map_batch(self, z_contract: np.ndarray) -> np.ndarray:
        x_np = np.asarray(z_contract, dtype=np.float32)
        if x_np.ndim == 1:
            x_np = x_np[None, :]
        with self.torch.no_grad():
            x = self.torch.from_numpy(x_np).to(self.device)
            y = self.model(x).detach().cpu().numpy()
        return y.astype(np.float32)

    def map_one(self, z_contract: np.ndarray) -> np.ndarray:
        return self.map_batch(np.asarray(z_contract, dtype=np.float32)[None, :])[0]
