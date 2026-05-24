from __future__ import annotations

import numpy as np

from ..interfaces import EqMapper
from ..types import EqCurve, PerceptualProfile


class ParametricEqMapper(EqMapper):
    """
    Маппер из perceptual-space в 23-band EQ curve.

    V1 использует 6 интерпретируемых параметров:
        [bass, tilt, presence, air, lowmid, sparkle]

    Идея:
    - не генерировать gain каждой полосы отдельно;
    - строить гладкую кривую как сумму нескольких базисных форм
      на логарифмической частотной шкале;
    - затем ограничивать её по допустимым gain'ам.

    Это даёт:
    - интерпретируемость;
    - гладкость;
    - меньше артефактов;
    - удобное расширение до 7-го параметра (shape/warmth и т.п.).
    """

    DEFAULT_FEATURES = ["bass", "tilt", "presence", "air", "lowmid", "sparkle"]

    def __init__(
        self,
        freqs_hz: np.ndarray,
        *,
        feature_names: list[str] | None = None,
        max_abs_gain_db: float = 12.0,
        preamp_scale_db: float = 0.4,
        preamp_threshold_db: float = 1.0,
        tilt_scale_db: float = 4.0,
        bass_scale_db: float = 5.0,
        lowmid_scale_db: float = 4.0,
        presence_scale_db: float = 4.0,
        air_scale_db: float = 3.5,
        sparkle_scale_db: float = 3.0,
        smooth_sigma_oct: float = 0.55,
    ) -> None:
        freqs = np.asarray(freqs_hz, dtype=np.float32)
        if freqs.ndim != 1 or freqs.shape[0] < 3:
            raise ValueError("freqs_hz must be a 1D array with at least 3 bands")
        if np.any(freqs <= 0):
            raise ValueError("freqs_hz must contain positive values")
        if max_abs_gain_db <= 0:
            raise ValueError("max_abs_gain_db must be positive")
        if smooth_sigma_oct <= 0:
            raise ValueError("smooth_sigma_oct must be positive")

        self.freqs_hz = freqs.copy()
        self.feature_names = list(feature_names or self.DEFAULT_FEATURES)
        self.dim = len(self.feature_names)

        self.max_abs_gain_db = float(max_abs_gain_db)
        self.preamp_scale_db = float(preamp_scale_db)
        self.preamp_threshold_db = float(preamp_threshold_db)

        self._log_f = np.log2(self.freqs_hz.astype(np.float64))
        self._log_f_center = float(np.mean(self._log_f))
        self._tilt_basis = self._build_tilt_basis()

        # Базисные формы. Каждая имеет максимум около 1.0 в своей области.
        self._basis = {
            "bass": bass_scale_db * self._gaussian_basis(center_hz=90.0, sigma_oct=0.80),
            "lowmid": lowmid_scale_db * self._gaussian_basis(center_hz=280.0, sigma_oct=0.65),
            "presence": presence_scale_db * self._gaussian_basis(center_hz=3200.0, sigma_oct=0.60),
            "air": air_scale_db * self._gaussian_basis(center_hz=9500.0, sigma_oct=0.75),
            "sparkle": sparkle_scale_db * self._high_shelf_like(start_hz=6500.0, slope_oct=0.55),
            "tilt": tilt_scale_db * self._tilt_basis,
        }

        self._smoothing_kernel = self._build_smoothing_kernel(sigma_oct=smooth_sigma_oct)

    def map_profile(self, profile: PerceptualProfile) -> EqCurve:
        z = np.asarray(profile.values, dtype=np.float32)
        if z.ndim != 1 or z.shape[0] != self.dim:
            raise ValueError(
                f"Profile dimension mismatch: expected {self.dim}, got {z.shape}"
            )

        curve = np.zeros_like(self.freqs_hz, dtype=np.float64)

        for idx, feature_name in enumerate(self.feature_names):
            if feature_name not in self._basis:
                raise ValueError(f"Unsupported feature name: {feature_name}")
            curve += float(z[idx]) * self._basis[feature_name]

        curve = self._smooth_curve(curve)
        curve = np.clip(curve, -self.max_abs_gain_db, self.max_abs_gain_db)

        # Не штрафуем почти нулевую кривую по громкости.
        # Компенсация начинается только после заметного boost.
        max_boost = float(max(0.0, np.max(curve)))
        effective_boost = max(0.0, max_boost - self.preamp_threshold_db)
        preamp_db = -min(self.preamp_scale_db * effective_boost, 9.0)

        return EqCurve(
            freqs_hz=self.freqs_hz,
            gains_db=curve.astype(np.float32),
            preamp_db=float(preamp_db),
        )

    def _gaussian_basis(self, *, center_hz: float, sigma_oct: float) -> np.ndarray:
        c = np.log2(float(center_hz))
        x = (self._log_f - c) / float(sigma_oct)
        basis = np.exp(-0.5 * (x ** 2))
        return basis.astype(np.float64)

    def _high_shelf_like(self, *, start_hz: float, slope_oct: float) -> np.ndarray:
        x = (self._log_f - np.log2(float(start_hz))) / float(slope_oct)
        basis = 1.0 / (1.0 + np.exp(-x))
        basis = basis - basis.min()
        if basis.max() > 0:
            basis = basis / basis.max()
        return basis.astype(np.float64)

    def _build_tilt_basis(self) -> np.ndarray:
        x = self._log_f - self._log_f_center
        x = x / max(1e-8, np.max(np.abs(x)))
        return x.astype(np.float64)

    def _build_smoothing_kernel(self, *, sigma_oct: float) -> np.ndarray:
        # Оцениваем шаг по лог-частоте и строим короткое ядро Гаусса
        diffs = np.diff(self._log_f)
        step = float(np.median(diffs)) if diffs.size > 0 else 0.25
        radius = max(1, int(np.ceil(2.5 * sigma_oct / max(step, 1e-6))))
        grid = np.arange(-radius, radius + 1, dtype=np.float64) * step
        kernel = np.exp(-0.5 * (grid / sigma_oct) ** 2)
        kernel /= np.sum(kernel)
        return kernel

    def _smooth_curve(self, curve: np.ndarray) -> np.ndarray:
        if self._smoothing_kernel.size <= 1:
            return curve
        padded = np.pad(curve, (self._smoothing_kernel.size // 2,), mode="edge")
        smoothed = np.convolve(padded, self._smoothing_kernel, mode="valid")
        return smoothed.astype(np.float64)
