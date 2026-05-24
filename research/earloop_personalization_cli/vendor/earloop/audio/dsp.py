import numpy as np
from scipy.signal import sosfilt


def peaking_eq_coeffs(fs: float, f0: float, q: float, gain_db: float):
    """
    Коэффициенты biquad peaking EQ (RBJ Audio EQ Cookbook).

    fs: sample rate
    f0: center frequency (Hz)
    q: Q-factor
    gain_db: gain in dB
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * (f0 / fs)
    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def peaking_eq_sos(fs: float, f0: float, q: float, gain_db: float):
    """
    Возвращает SOS строку: [b0, b1, b2, a0(=1), a1, a2]
    """
    b, a = peaking_eq_coeffs(fs, f0, q, gain_db)
    return np.array([b[0], b[1], b[2], 1.0, a[1], a[2]], dtype=np.float64)


def q_from_neighbors(freqs_hz: np.ndarray, i: int) -> float:
    """
    Приближённо выбираем Q по ширине полосы между соседними частотами (лог-сетка).
    """
    f = np.asarray(freqs_hz, float).ravel()
    if len(f) < 2:
        return 1.0

    if i == 0:
        f_lo = f[0] / np.sqrt(f[1] / f[0])
        f_hi = np.sqrt(f[0] * f[1])
    elif i == len(f) - 1:
        f_lo = np.sqrt(f[-2] * f[-1])
        f_hi = f[-1] * np.sqrt(f[-1] / f[-2])
    else:
        f_lo = np.sqrt(f[i - 1] * f[i])
        f_hi = np.sqrt(f[i] * f[i + 1])

    bw = max(1e-6, f_hi - f_lo)
    q = float(f[i] / bw)
    return float(np.clip(q, 0.3, 12.0))


class SOSCascade:
    """
    Каскад SOS (biquad) фильтров для стерео.

    Важно:
    - строим peaking-фильтры на каждой полосе
    - Q оцениваем из соседей, чтобы получить адекватную ширину полосы
    """

    def __init__(
        self,
        fs: int,
        freqs_hz: np.ndarray,
        gains_db: np.ndarray,
        max_abs_gain_db: float = 24.0,
    ):
        self.fs = int(fs)

        f = np.asarray(freqs_hz, float).ravel()
        g = np.asarray(gains_db, float).ravel()
        g = np.clip(g, -max_abs_gain_db, max_abs_gain_db)

        rows = []
        for i, (f0, gdb) in enumerate(zip(f, g)):
            if float(f0) <= 0.0:
                continue
            q = q_from_neighbors(f, i)
            rows.append(peaking_eq_sos(self.fs, float(f0), float(q), float(gdb)))

        self.sos = np.vstack(rows) if rows else np.zeros((0, 6), dtype=np.float64)

        # zi: (n_sections, 2 states, n_channels=2)
        self.zi = np.zeros((self.sos.shape[0], 2, 2), dtype=np.float64)

    def reset(self):
        self.zi.fill(0.0)

    def process(self, x_stereo: np.ndarray) -> np.ndarray:
        """
        x_stereo: shape [N, 2]
        """
        if self.sos.shape[0] == 0:
            return x_stereo

        x = np.asarray(x_stereo, dtype=np.float64)
        yL, self.zi[:, :, 0] = sosfilt(self.sos, x[:, 0], zi=self.zi[:, :, 0])
        yR, self.zi[:, :, 1] = sosfilt(self.sos, x[:, 1], zi=self.zi[:, :, 1])
        return np.column_stack([yL, yR])
