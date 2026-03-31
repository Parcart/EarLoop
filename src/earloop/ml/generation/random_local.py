from __future__ import annotations

import uuid

import numpy as np

from ..interfaces import CandidateGenerator, PreferenceModel
from ..types import PerceptualPair, PerceptualProfile


class RandomLocalCandidateGenerator(CandidateGenerator):
    """
    Простой генератор кандидатов в perceptual-space.

    Идея:
    - если модель ещё не обучена, генерируем случайные пары по всему пространству;
    - если модель уже что-то знает, берём "лучший" центр и генерируем рядом с ним;
    - после REJECT_BOTH увеличиваем радиус поиска, чтобы получить более разнообразную пару.
    """

    def __init__(
        self,
        dim: int,
        *,
        low: float = -1.0,
        high: float = 1.0,
        local_std: float = 0.20,
        reject_std_multiplier: float = 1.75,
        min_pair_distance: float = 0.15,
        random_seed: int | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if low >= high:
            raise ValueError("low must be less than high")
        if local_std <= 0:
            raise ValueError("local_std must be positive")
        if reject_std_multiplier <= 1.0:
            raise ValueError("reject_std_multiplier must be > 1.0")
        if min_pair_distance <= 0:
            raise ValueError("min_pair_distance must be positive")

        self.dim = int(dim)
        self.low = float(low)
        self.high = float(high)
        self.local_std = float(local_std)
        self.reject_std_multiplier = float(reject_std_multiplier)
        self.min_pair_distance = float(min_pair_distance)

        self._rng = np.random.default_rng(random_seed)
        self._best_profile: PerceptualProfile | None = None

    def set_best_profile(self, profile: PerceptualProfile | None) -> None:
        self._best_profile = profile

    def generate_pair(
        self,
        model: PreferenceModel,
        *,
        iteration: int | None = None,
    ) -> PerceptualPair:
        self._check_model_dim(model)

        if self._best_profile is None:
            left = self._random_profile()
            right = self._random_profile_far_from(left)
        else:
            center = self._best_profile
            left = self._sample_near(center, std=self.local_std)
            right = self._sample_near(center, std=self.local_std)

            if self._distance(left, right) < self.min_pair_distance:
                right = self._sample_near(center, std=self.local_std * 1.5)

        # Упорядочим пару так, чтобы слева был чуть более перспективный кандидат.
        if model.score(right) > model.score(left):
            left, right = right, left

        return PerceptualPair(
            left=left,
            right=right,
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
        self._check_model_dim(model)

        center = self._pair_midpoint(rejected_pair)
        std = self.local_std * self.reject_std_multiplier

        left = self._sample_near(center, std=std)
        right = self._sample_near(center, std=std)

        if self._distance(left, right) < self.min_pair_distance:
            right = self._sample_near(center, std=std * 1.5)

        if model.score(right) > model.score(left):
            left, right = right, left

        return PerceptualPair(
            left=left,
            right=right,
            pair_id=self._make_pair_id(),
            iteration=iteration,
        )

    def update_best_from_choice(
        self,
        pair: PerceptualPair,
        choice: str,
    ) -> None:
        """
        Утилитарный метод: обновить текущий лучший профиль по выбору пользователя.
        Это не часть общего интерфейса, а удобная логика для простого pipeline.
        """
        if choice == "left":
            self._best_profile = pair.left
        elif choice == "right":
            self._best_profile = pair.right

    def _random_profile(self) -> PerceptualProfile:
        values = self._rng.uniform(self.low, self.high, size=self.dim).astype(np.float32)
        return PerceptualProfile(values=values)

    def _random_profile_far_from(self, other: PerceptualProfile) -> PerceptualProfile:
        for _ in range(32):
            candidate = self._random_profile()
            if self._distance(candidate, other) >= self.min_pair_distance:
                return candidate
        return self._random_profile()

    def _sample_near(
        self,
        center: PerceptualProfile,
        *,
        std: float,
    ) -> PerceptualProfile:
        center_vec = np.asarray(center.values, dtype=np.float32)
        noise = self._rng.normal(loc=0.0, scale=std, size=self.dim).astype(np.float32)
        values = np.clip(center_vec + noise, self.low, self.high).astype(np.float32)
        return PerceptualProfile(values=values)

    def _pair_midpoint(self, pair: PerceptualPair) -> PerceptualProfile:
        left = np.asarray(pair.left.values, dtype=np.float32)
        right = np.asarray(pair.right.values, dtype=np.float32)
        midpoint = ((left + right) / 2.0).astype(np.float32)
        return PerceptualProfile(values=midpoint)

    def _distance(
        self,
        a: PerceptualProfile,
        b: PerceptualProfile,
    ) -> float:
        av = np.asarray(a.values, dtype=np.float32)
        bv = np.asarray(b.values, dtype=np.float32)
        return float(np.linalg.norm(av - bv))

    def _check_model_dim(self, model: PreferenceModel) -> None:
        if model.dim != self.dim:
            raise ValueError(
                f"Dimension mismatch: generator dim={self.dim}, model dim={model.dim}"
            )

    @staticmethod
    def _make_pair_id() -> str:
        return str(uuid.uuid4())