import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from scipy.signal import sosfilt


# --------------------------
# RBJ Peaking EQ biquad
# --------------------------
def peaking_eq_coeffs(fs: float, f0: float, q: float, gain_db: float):
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


def q_from_neighbors(freqs_hz: np.ndarray, i: int) -> float:
    f = freqs_hz
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


def peaking_eq_sos(fs: float, f0: float, q: float, gain_db: float):
    b, a = peaking_eq_coeffs(fs, f0, q, gain_db)
    # a0=1 (нормализовано), SOS row: [b0 b1 b2 a0 a1 a2]
    return np.array([b[0], b[1], b[2], 1.0, a[1], a[2]], dtype=np.float64)

class SOSCascade:
    def __init__(self, fs: int, freqs_hz: np.ndarray, gains_db: np.ndarray, max_abs_gain_db: float = 24.0):
        self.fs = fs
        f = np.asarray(freqs_hz, float).ravel()
        g = np.clip(np.asarray(gains_db, float).ravel(), -max_abs_gain_db, max_abs_gain_db)

        rows = []
        for i, (f0, gdb) in enumerate(zip(f, g)):
            if f0 <= 0:
                continue
            q = q_from_neighbors(f, i)
            rows.append(peaking_eq_sos(fs, float(f0), float(q), float(gdb)))

        self.sos = np.vstack(rows) if rows else np.zeros((0, 6), dtype=np.float64)
        self.zi = np.zeros((self.sos.shape[0], 2, 2), dtype=np.float64)  # (sections, 2, channels)

    def reset(self):
        self.zi.fill(0.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        if self.sos.shape[0] == 0:
            return x
        yL, self.zi[:, :, 0] = sosfilt(self.sos, x[:, 0], zi=self.zi[:, :, 0])
        yR, self.zi[:, :, 1] = sosfilt(self.sos, x[:, 1], zi=self.zi[:, :, 1])
        return np.column_stack([yL, yR])


# --------------------------
# Preset loading
# --------------------------
def load_preset_csv(path: Path):
    df = np.read_csv(path) if False else None  # placeholder to avoid pandas dependency


def read_preset(path: Path):
    # expected columns: fc_hz,gain_db
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    freqs = data["fc_hz"].astype(float)
    gains = data["gain_db"].astype(float)
    return freqs, gains


def discover_presets(preset_dir: Path):
    # files like: preset_smile_40.csv, preset_smile_23.csv, preset_smile_11_knee.csv ...
    all_files = sorted(preset_dir.glob("preset_*_40.csv"))
    names = [f.name.replace("preset_", "").replace("_40.csv", "") for f in all_files]
    return names


def preset_paths(preset_dir: Path, name: str):
    return {
        "40": preset_dir / f"preset_{name}_40.csv",
        "23": preset_dir / f"preset_{name}_23.csv",
        "11_knee": preset_dir / f"preset_{name}_11_knee.csv",
        "11_greedy": preset_dir / f"preset_{name}_11_greedy.csv",
    }


# --------------------------
# Player
# --------------------------
class EQPlayer:
    def __init__(self, wav_path: Path, preset_dir: Path, preamp_db: float = -6.0, block: int = 1024):
        self.wav_path = wav_path
        self.preset_dir = preset_dir
        self.block = block
        self.preamp = 10 ** (preamp_db / 20.0)

        self.audio, self.fs = sf.read(str(wav_path), always_2d=True, dtype="float32")
        if self.audio.shape[1] == 1:
            self.audio = np.repeat(self.audio, 2, axis=1)
        else:
            self.audio = self.audio[:, :2]
        self.n = self.audio.shape[0]
        self.pos = 0

        self.preset_names = discover_presets(preset_dir)
        if not self.preset_names:
            raise FileNotFoundError(f"No presets found in {preset_dir} (expected preset_*_40.csv).")

        self.current_preset_idx = 0
        self.mode = "bypass"  # bypass, 40, 23, 11_knee, 11_greedy

        self.cascade = None
        self.lock = threading.Lock()

        # prepare initial
        self._load_cascade()

    def _load_cascade(self):
        name = self.preset_names[self.current_preset_idx]
        paths = preset_paths(self.preset_dir, name)

        casc = None
        if self.mode != "bypass":
            p = paths.get(self.mode)
            if p is None or (not p.exists()):
                # if greedy missing, fall back
                if self.mode == "11_greedy":
                    print("No 11_greedy for this preset; falling back to 11_knee")
                    p = paths["11_knee"]
                else:
                    raise FileNotFoundError(f"Missing preset file: {p}")

            freqs, gains = read_preset(p)
            casc = SOSCascade(self.fs, freqs, gains)

        self.cascade = casc
        if self.cascade is not None:
            self.cascade.reset()

        print(f"Preset: {name} | Mode: {self.mode}")

    def set_mode(self, mode: str):
        with self.lock:
            self.mode = mode
            self._load_cascade()

    def next_preset(self):
        with self.lock:
            self.current_preset_idx = (self.current_preset_idx + 1) % len(self.preset_names)
            self._load_cascade()

    def prev_preset(self):
        with self.lock:
            self.current_preset_idx = (self.current_preset_idx - 1) % len(self.preset_names)
            self._load_cascade()

    def callback(self, outdata, frames, time, status):
        # loop wav
        end = self.pos + frames
        if end <= self.n:
            block = self.audio[self.pos:end]
            self.pos = end
        else:
            tail = self.audio[self.pos:]
            head = self.audio[: end - self.n]
            block = np.vstack([tail, head])
            self.pos = end - self.n

        x = block.astype(np.float64) * self.preamp

        with self.lock:
            casc = self.cascade

        if casc is not None:
            y = casc.process(x)
        else:
            y = x

        # safety clip
        y = np.clip(y, -0.98, 0.98).astype(np.float32)
        outdata[:] = y

        if status:
            print(status)


def keyboard_loop(player: EQPlayer):
    print("\nControls:")
    print("  1: 40 bands")
    print("  2: 23 bands")
    print("  3: 11 knee")
    print("  4: 11 greedy (if exists)")
    print("  b: bypass")
    print("  n/p: next/prev preset")
    print("  q: quit\n")

    while True:
        ch = sys.stdin.read(1)
        if ch == "1":
            player.set_mode("40")
        elif ch == "2":
            player.set_mode("23")
        elif ch == "3":
            player.set_mode("11_knee")
        elif ch == "4":
            player.set_mode("11_greedy")
        elif ch.lower() == "b":
            player.set_mode("bypass")
        elif ch.lower() == "n":
            player.next_preset()
        elif ch.lower() == "p":
            player.prev_preset()
        elif ch.lower() == "q":
            break


def main():
    wav_path = Path("test.wav")
    if not wav_path.exists():
        raise FileNotFoundError("Put a WAV named test.wav next to this script.")

    preset_dir = Path("exports_listening/presets_projected")
    if not preset_dir.exists():
        raise FileNotFoundError("Missing exports_listening/presets_projected. Run the notebook projection first.")

    # preamp -6 dB to reduce clipping risk
    player = EQPlayer(wav_path, preset_dir, preamp_db=-6.0, block=1024)

    # Start in bypass
    player.set_mode("bypass")

    with sd.OutputStream(
        samplerate=player.fs,
        blocksize=player.block,
        channels=2,
        dtype="float32",
        callback=player.callback,
    ):
        keyboard_loop(player)


if __name__ == "__main__":
    main()
