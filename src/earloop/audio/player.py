import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .dsp import SOSCascade


@dataclass
class SimpleProfile:
    """
    Минимальный профиль для воспроизведения.
    (Если у тебя есть более сложный объект профиля — можно адаптировать.)
    """
    profile_id: int
    freqs_hz: np.ndarray
    gains_db: np.ndarray


class ABPlayer:
    """
    Реалтайм A/B плеер:
    - один WAV
    - два EQ-профиля (A и B)
    - мгновенное переключение A/B
    """

    def __init__(
        self,
        wav_path: Path,
        profile_a: SimpleProfile,
        profile_b: SimpleProfile,
        preamp_db: float = -6.0,
        block: int = 1024,
    ):
        self.wav_path = Path(wav_path)
        self.block = int(block)

        audio, fs = sf.read(str(self.wav_path), always_2d=True, dtype="float32")
        self.fs = int(fs)

        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        else:
            audio = audio[:, :2]

        self.audio = audio
        self.n = self.audio.shape[0]
        self.pos = 0

        # Общий предусилитель плеера
        self.preamp = float(10 ** (float(preamp_db) / 20.0))

        self.lock = threading.Lock()
        self.which = "A"

        self.set_pair(profile_a, profile_b)

    def set_pair(self, profile_a: SimpleProfile, profile_b: SimpleProfile):
        with self.lock:
            self.profile_a = profile_a
            self.profile_b = profile_b

            self.casc_a = SOSCascade(self.fs, profile_a.freqs_hz, profile_a.gains_db)
            self.casc_b = SOSCascade(self.fs, profile_b.freqs_hz, profile_b.gains_db)

            self.casc_a.reset()
            self.casc_b.reset()

            self.which = "A"

    def set_which(self, which: str):
        with self.lock:
            self.which = "A" if which.upper() == "A" else "B"

    def callback(self, outdata, frames, time, status):
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
            casc = self.casc_a if self.which == "A" else self.casc_b

        y = casc.process(x)

        # safety clip
        y = np.clip(y, -0.98, 0.98).astype(np.float32)
        outdata[:] = y

        if status:
            print(status)

    def run_forever(self):
        """
        Запускает OutputStream и держит его живым (до Ctrl+C / выхода процесса).
        """
        with sd.OutputStream(
            samplerate=self.fs,
            blocksize=self.block,
            channels=2,
            dtype="float32",
            callback=self.callback,
        ):
            while True:
                sd.sleep(250)
