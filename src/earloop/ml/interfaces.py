from __future__ import annotations

from abc import ABC, abstractmethod

from .types import (
    CandidatePair,
    EqCurve,
    PerceptualPair,
    PerceptualProfile,
    PreferenceChoice,
    ValidationResult,
)


class PreferenceModel(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def score(self, profile: PerceptualProfile) -> float:
        raise NotImplementedError

    @abstractmethod
    def prefer_left_probability(
        self,
        left: PerceptualProfile,
        right: PerceptualProfile,
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def update_from_choice(
        self,
        pair: PerceptualPair,
        choice: PreferenceChoice,
    ) -> None:
        raise NotImplementedError


class CandidateGenerator(ABC):
    @abstractmethod
    def generate_pair(
        self,
        model: PreferenceModel,
        *,
        iteration: int | None = None,
    ) -> PerceptualPair:
        raise NotImplementedError

    @abstractmethod
    def generate_next_pair_after_reject(
        self,
        model: PreferenceModel,
        rejected_pair: PerceptualPair,
        *,
        iteration: int | None = None,
    ) -> PerceptualPair:
        raise NotImplementedError


class EqMapper(ABC):
    @abstractmethod
    def map_profile(self, profile: PerceptualProfile) -> EqCurve:
        raise NotImplementedError


class EqValidator(ABC):
    @abstractmethod
    def validate_curve(self, curve: EqCurve) -> ValidationResult:
        raise NotImplementedError

    @abstractmethod
    def validate_pair(self, pair: CandidatePair) -> ValidationResult:
        raise NotImplementedError