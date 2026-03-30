from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .dsp import SOSCascade


@dataclass(frozen=True)
class EqProfile:
    profile_id: int
    freqs_hz: np.ndarray
    gains_db: np.ndarray
    preamp_db: float = 0.0
    name: str = ""


class AudioProcessor:
    def process(self, chunk: np.ndarray) -> np.ndarray:
        return chunk

    def reset(self) -> None:
        return None


class PassthroughProcessor(AudioProcessor):
    pass


class EqualizerProcessor(AudioProcessor):
    """Thread-safe EQ processor based on SOSCascade."""

    def __init__(self, samplerate: int, channels: int = 2, initial_profile: Optional[EqProfile] = None):
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self._lock = threading.RLock()
        self._cascade: Optional[SOSCascade] = None
        self._gain = 1.0
        self._active_profile: Optional[EqProfile] = None
        if initial_profile is not None:
            self.set_profile(initial_profile)

    @property
    def active_profile(self) -> Optional[EqProfile]:
        with self._lock:
            return self._active_profile

    def set_profile(self, profile: EqProfile) -> None:
        cascade = SOSCascade(self.samplerate, profile.freqs_hz, profile.gains_db)
        cascade.reset()
        linear_gain = float(10 ** (profile.preamp_db / 20.0))
        with self._lock:
            self._cascade = cascade
            self._gain = linear_gain
            self._active_profile = profile

    def clear_profile(self) -> None:
        with self._lock:
            self._cascade = None
            self._gain = 1.0
            self._active_profile = None

    def reset(self) -> None:
        with self._lock:
            if self._cascade is not None:
                self._cascade.reset()

    def process(self, chunk: np.ndarray) -> np.ndarray:
        x = np.asarray(chunk, dtype=np.float32)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] == 1 and self.channels == 2:
            x = np.repeat(x, 2, axis=1)
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")

        with self._lock:
            cascade = self._cascade
            gain = self._gain

        y = x.astype(np.float64) * gain
        if cascade is not None:
            y = cascade.process(y)
        return np.clip(y, -0.98, 0.98).astype(np.float32)