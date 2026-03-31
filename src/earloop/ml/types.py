from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


class PreferenceChoice(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    REJECT_BOTH = "reject_both"
    SKIP = "skip"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class PerceptualProfile:
    """
    Компактное представление профиля в perceptual-space.

    Пример:
        values = np.array([bass, lowmid, mid, presence, air, tilt], dtype=np.float32)
    """
    values: FloatArray
    profile_id: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float32)

        if arr.ndim != 1:
            raise ValueError("PerceptualProfile.values must be a 1D array")

        arr = arr.copy()
        arr.setflags(write=False)

        object.__setattr__(self, "values", arr)

    @property
    def dim(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True, slots=True)
class PerceptualPair:
    """
    Пара профилей в perceptual-space до перевода в EQ.
    """
    left: PerceptualProfile
    right: PerceptualProfile
    pair_id: str | None = None
    iteration: int | None = None


@dataclass(frozen=True, slots=True)
class EqCurve:
    """
    Профиль на сетке эквалайзера.
    freqs_hz и gains_db должны быть одной длины.
    """
    freqs_hz: FloatArray
    gains_db: FloatArray
    preamp_db: float = 0.0

    def __post_init__(self) -> None:
        freqs = np.asarray(self.freqs_hz, dtype=np.float32)
        gains = np.asarray(self.gains_db, dtype=np.float32)

        if freqs.ndim != 1 or gains.ndim != 1:
            raise ValueError("EqCurve fields must be 1D arrays")
        if freqs.shape[0] != gains.shape[0]:
            raise ValueError("freqs_hz and gains_db must have the same length")

        freqs = freqs.copy()
        gains = gains.copy()
        freqs.setflags(write=False)
        gains.setflags(write=False)

        object.__setattr__(self, "freqs_hz", freqs)
        object.__setattr__(self, "gains_db", gains)

    @property
    def bands_count(self) -> int:
        return int(self.freqs_hz.shape[0])


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    """
    Кандидат после маппинга в EQ.
    """
    perceptual: PerceptualProfile
    eq_curve: EqCurve
    candidate_id: str | None = None
    origin: str | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """
    Пара кандидатов, готовых к прослушиванию/сравнению.
    """
    left: CandidateProfile
    right: CandidateProfile
    pair_id: str | None = None
    iteration: int | None = None


@dataclass(frozen=True, slots=True)
class PreferenceObservation:
    """
    Результат выбора пользователя по паре.
    """
    pair_id: str | None
    choice: PreferenceChoice
    iteration: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """
    Результат проверки кандидата или EQ-кривой.
    """
    is_valid: bool
    score: float | None = None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelState:
    model_type: str
    dim: int
    learning_rate: float
    l2_reg: float
    weights: tuple[float, ...]
    version: int = 1


@dataclass(frozen=True, slots=True)
class PreferenceEvent:
    user_id: str
    session_id: str
    pair_id: str | None
    iteration: int | None
    choice: PreferenceChoice
    left_profile: tuple[float, ...]
    right_profile: tuple[float, ...]
    left_score: float | None = None
    right_score: float | None = None
    model_version: str | None = None
    mapper_version: str | None = None
    timestamp_utc: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    user_id: str
    session_id: str
    model_state: ModelState
    iteration: int = 0
    best_profile: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)