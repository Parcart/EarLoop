from __future__ import annotations

import uuid
from typing import Sequence

import numpy as np

from ..interfaces import CandidateGenerator, PreferenceModel
from ..types import PerceptualPair, PerceptualProfile


class PoolCandidateGenerator(CandidateGenerator):
    """
    Генератор кандидатов из заранее подготовленного пула perceptual-профилей. Взято из ноутбука generate_synth_eq_dataset_with_params.ipynb

    Логика близка к уже существующему прототипу:
    - считаем score для всего пула;
    - лучший профиль становится LEFT;
    - RIGHT берётся либо случайно (exploration),
      либо из top-K по score (exploitation).

    Это хороший baseline для первой версии системы:
    - воспроизводимо;
    - просто отлаживать;
    - хорошо сочетается с синтетическим датасетом.
    """

    def __init__(
        self,
        profiles: Sequence[PerceptualProfile],
        *,
        explore_prob: float = 0.25,
        top_k: int = 20,
        reject_sample_size: int = 50,
        random_seed: int | None = None,
    ) -> None:
        if not profiles:
            raise ValueError("profiles must not be empty")
        if len(profiles) < 2:
            raise ValueError("at least 2 profiles are required")
        if not (0.0 <= explore_prob <= 1.0):
            raise ValueError("explore_prob must be in [0, 1]")
        if top_k <= 1:
            raise ValueError("top_k must be > 1")
        if reject_sample_size <= 1:
            raise ValueError("reject_sample_size must be > 1")

        self._profiles = list(profiles)
        self.dim = self._profiles[0].dim

        for idx, profile in enumerate(self._profiles):
            if profile.dim != self.dim:
                raise ValueError(
                    f"All profiles must have the same dim. "
                    f"Profile {idx} has dim={profile.dim}, expected {self.dim}"
                )

        self.explore_prob = float(explore_prob)
        self.top_k = int(top_k)
        self.reject_sample_size = int(reject_sample_size)
        self._rng = np.random.default_rng(random_seed)

        # Матрица признаков для быстрого скоринга
        self._X = np.stack(
            [np.asarray(profile.values, dtype=np.float32) for profile in self._profiles],
            axis=0,
        ).astype(np.float32)

    @property
    def profiles(self) -> list[PerceptualProfile]:
        return list(self._profiles)

    def generate_pair(
        self,
        model: PreferenceModel,
        *,
        iteration: int | None = None,
    ) -> PerceptualPair:
        self._check_model_dim(model)

        scores = self._score_all(model)

        best_idx = int(np.argmax(scores))
        left_idx = best_idx

        if self._rng.random() < self.explore_prob:
            right_idx = self._sample_random_excluding(best_idx)
        else:
            right_idx = self._sample_from_top_k(scores, exclude_idx=best_idx)

        return PerceptualPair(
            left=self._profiles[left_idx],
            right=self._profiles[right_idx],
            pair_id=self._make_pair_id(),
            iteration=iteration,
        )

    def generate_next_pair_after_reject(
        self,
        model: PreferenceModel,
        rejected_pair: PerceptualPair,
        *,
        iteration: int | None = None,
    ) -> PerceptualPair:
        """
        После REJECT_BOTH стараемся:
        - не показывать ту же пару;
        - взять кандидатов подальше от отвергнутых;
        - при этом не терять слишком перспективные варианты.

        Простая эвристика:
        1) считаем score для всего пула;
        2) берём top-N перспективных;
        3) среди них ищем пару, которая максимально далека
           от обоих отвергнутых профилей.
        """
        self._check_model_dim(model)

        scores = self._score_all(model)

        left_rej = np.asarray(rejected_pair.left.values, dtype=np.float32)
        right_rej = np.asarray(rejected_pair.right.values, dtype=np.float32)

        n = len(self._profiles)
        sample_size = min(self.reject_sample_size, n)
        candidate_idx = np.argsort(scores)[-sample_size:]

        # "Антипохожесть" к отвергнутой паре:
        # чем дальше профиль от обоих rejected, тем лучше.
        dist_left = np.linalg.norm(self._X[candidate_idx] - left_rej[None, :], axis=1)
        dist_right = np.linalg.norm(self._X[candidate_idx] - right_rej[None, :], axis=1)
        anti_reject_score = dist_left + dist_right

        # Первый кандидат — наиболее перспективный среди "непохожих"
        order = np.argsort(anti_reject_score)[::-1]
        first_idx = int(candidate_idx[order[0]])

        # Второй кандидат — тоже непохожий, но не тот же самый
        second_idx = None
        for ord_idx in order[1:]:
            idx = int(candidate_idx[ord_idx])
            if idx != first_idx:
                second_idx = idx
                break

        if second_idx is None:
            second_idx = self._sample_random_excluding(first_idx)

        # Слева всё же ставим того, кто по модели выглядит лучше
        if scores[second_idx] > scores[first_idx]:
            first_idx, second_idx = second_idx, first_idx

        return PerceptualPair(
            left=self._profiles[first_idx],
            right=self._profiles[second_idx],
            pair_id=self._make_pair_id(),
            iteration=iteration,
        )

    def best_profile(self, model: PreferenceModel) -> PerceptualProfile:
        self._check_model_dim(model)
        scores = self._score_all(model)
        return self._profiles[int(np.argmax(scores))]

    def rank_profiles(
        self,
        model: PreferenceModel,
    ) -> list[tuple[PerceptualProfile, float]]:
        self._check_model_dim(model)
        scores = self._score_all(model)
        items = list(zip(self._profiles, scores.tolist()))
        items.sort(key=lambda item: item[1], reverse=True)
        return items

    def _score_all(self, model: PreferenceModel) -> np.ndarray:
        return np.asarray(
            [model.score(profile) for profile in self._profiles],
            dtype=np.float32,
        )

    def _sample_random_excluding(self, exclude_idx: int) -> int:
        n = len(self._profiles)
        if n < 2:
            raise ValueError("Need at least 2 profiles")

        idx = int(self._rng.integers(0, n))
        if idx == exclude_idx:
            idx = (idx + 1) % n
        return idx

    def _sample_from_top_k(self, scores: np.ndarray, *, exclude_idx: int) -> int:
        n = len(scores)
        k = min(self.top_k, n)

        top_idx = np.argsort(scores)[-k:]
        if len(top_idx) == 1:
            return self._sample_random_excluding(exclude_idx)

        for _ in range(16):
            idx = int(self._rng.choice(top_idx))
            if idx != exclude_idx:
                return idx

        # fallback
        for idx in reversed(top_idx.tolist()):
            if idx != exclude_idx:
                return int(idx)

        return self._sample_random_excluding(exclude_idx)

    def _check_model_dim(self, model: PreferenceModel) -> None:
        if model.dim != self.dim:
            raise ValueError(
                f"Dimension mismatch: generator dim={self.dim}, model dim={model.dim}"
            )

    @staticmethod
    def _make_pair_id() -> str:
        return str(uuid.uuid4())