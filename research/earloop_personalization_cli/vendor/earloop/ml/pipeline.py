from __future__ import annotations

from datetime import datetime, timezone

from .interfaces import CandidateGenerator, EqMapper, EqValidator, PreferenceModel
from .types import (
    CandidatePair,
    CandidateProfile,
    EqCurve,
    PerceptualPair,
    PreferenceChoice,
    PreferenceEvent,
    PreferenceObservation,
    ValidationResult,
)


class PersonalizationPipeline:
    """
    Простой orchestration-слой для personalization-сессии.

    Что делает:
    - запрашивает следующую perceptual-пару у генератора;
    - при наличии mapper превращает её в CandidatePair;
    - при наличии validator проверяет pair;
    - принимает выбор пользователя;
    - обновляет preference-модель;
    - пишет event log.

    Это не "умная" часть, а клей между компонентами.
    """

    def __init__(
        self,
        preference_model: PreferenceModel,
        candidate_generator: CandidateGenerator,
        *,
        eq_mapper: EqMapper | None = None,
        eq_validator: EqValidator | None = None,
        user_id: str = "local_user",
        session_id: str = "local_session",
        model_version: str = "linear_v1",
        mapper_version: str | None = None,
    ) -> None:
        self.preference_model = preference_model
        self.candidate_generator = candidate_generator
        self.eq_mapper = eq_mapper
        self.eq_validator = eq_validator

        self.user_id = user_id
        self.session_id = session_id
        self.model_version = model_version
        self.mapper_version = mapper_version

        self.iteration = 0

        self.current_perceptual_pair: PerceptualPair | None = None
        self.current_candidate_pair: CandidatePair | None = None

        self.observations: list[PreferenceObservation] = []
        self.event_log: list[PreferenceEvent] = []

        self.last_choice: PreferenceChoice | None = None

    def next_perceptual_pair(self) -> PerceptualPair:
        """
        Вернуть следующую пару в perceptual-space.

        Если предыдущий выбор был REJECT_BOTH, просим у генератора
        более разнообразную следующую пару.
        """
        self.iteration += 1

        if (
            self.last_choice == PreferenceChoice.REJECT_BOTH
            and self.current_perceptual_pair is not None
        ):
            pair = self.candidate_generator.generate_next_pair_after_reject(
                self.preference_model,
                self.current_perceptual_pair,
                iteration=self.iteration,
            )
        else:
            pair = self.candidate_generator.generate_pair(
                self.preference_model,
                iteration=self.iteration,
            )

        self.current_perceptual_pair = pair
        self.current_candidate_pair = None
        return pair

    def next_candidate_pair(self) -> CandidatePair:
        """
        Вернуть следующую пару уже после маппинга в EQ.

        Требует подключённый eq_mapper.
        """
        if self.eq_mapper is None:
            raise RuntimeError("eq_mapper is not configured")

        perceptual_pair = self.next_perceptual_pair()
        candidate_pair = self._build_candidate_pair(perceptual_pair)

        if self.eq_validator is not None:
            validation = self.eq_validator.validate_pair(candidate_pair)
            if not validation.is_valid:
                raise RuntimeError(
                    f"Generated pair is invalid: {', '.join(validation.reasons)}"
                )

        self.current_candidate_pair = candidate_pair
        return candidate_pair

    def submit_choice(self, choice: PreferenceChoice) -> PreferenceObservation:
        """
        Принять выбор пользователя для текущей perceptual-пары.
        """
        if self.current_perceptual_pair is None:
            raise RuntimeError("No active pair. Call next_perceptual_pair() or next_candidate_pair() first.")

        observation = PreferenceObservation(
            pair_id=self.current_perceptual_pair.pair_id,
            choice=choice,
            iteration=self.current_perceptual_pair.iteration,
        )
        self.observations.append(observation)

        event = self._build_event(self.current_perceptual_pair, choice)
        self.event_log.append(event)

        self.preference_model.update_from_choice(self.current_perceptual_pair, choice)

        self.last_choice = choice
        return observation

    def get_best_profile(self):
        """
        Вернуть лучший профиль из генератора, если генератор это поддерживает.
        """
        if hasattr(self.candidate_generator, "best_profile"):
            return self.candidate_generator.best_profile(self.preference_model)
        raise RuntimeError("Current generator does not support best_profile().")

    def get_model_state(self):
        """
        Вернуть состояние модели, если модель это поддерживает.
        """
        if hasattr(self.preference_model, "get_state"):
            return self.preference_model.get_state()
        raise RuntimeError("Current preference model does not support get_state().")

    def reset_session(self) -> None:
        self.iteration = 0
        self.current_perceptual_pair = None
        self.current_candidate_pair = None
        self.observations.clear()
        self.event_log.clear()
        self.last_choice = None

    def _build_candidate_pair(self, perceptual_pair: PerceptualPair) -> CandidatePair:
        if self.eq_mapper is None:
            raise RuntimeError("eq_mapper is not configured")

        left_curve = self.eq_mapper.map_profile(perceptual_pair.left)
        right_curve = self.eq_mapper.map_profile(perceptual_pair.right)

        left_score = self.preference_model.score(perceptual_pair.left)
        right_score = self.preference_model.score(perceptual_pair.right)

        left_candidate = CandidateProfile(
            perceptual=perceptual_pair.left,
            eq_curve=left_curve,
            candidate_id=f"{perceptual_pair.pair_id}_left" if perceptual_pair.pair_id else None,
            origin="pipeline",
            score=left_score,
        )
        right_candidate = CandidateProfile(
            perceptual=perceptual_pair.right,
            eq_curve=right_curve,
            candidate_id=f"{perceptual_pair.pair_id}_right" if perceptual_pair.pair_id else None,
            origin="pipeline",
            score=right_score,
        )

        return CandidatePair(
            left=left_candidate,
            right=right_candidate,
            pair_id=perceptual_pair.pair_id,
            iteration=perceptual_pair.iteration,
        )

    def _build_event(
        self,
        pair: PerceptualPair,
        choice: PreferenceChoice,
    ) -> PreferenceEvent:
        left_score = self.preference_model.score(pair.left)
        right_score = self.preference_model.score(pair.right)

        return PreferenceEvent(
            user_id=self.user_id,
            session_id=self.session_id,
            pair_id=pair.pair_id,
            iteration=pair.iteration,
            choice=choice,
            left_profile=tuple(float(x) for x in pair.left.values.tolist()),
            right_profile=tuple(float(x) for x in pair.right.values.tolist()),
            left_score=float(left_score),
            right_score=float(right_score),
            model_version=self.model_version,
            mapper_version=self.mapper_version,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )