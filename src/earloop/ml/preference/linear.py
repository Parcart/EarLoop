from __future__ import annotations

import numpy as np

from ..types import PerceptualPair, PerceptualProfile, PreferenceChoice, ModelState


class LinearPreferenceModel:
    """
    Простая линейная модель пользовательских предпочтений
    в perceptual-space.

    Идея:
        score(z) = w · z

    где:
        z — perceptual-профиль
        w — вектор пользовательских предпочтений

    Обновление делается по pairwise-выбору пользователя:
        если выбран LEFT, то веса смещаются в сторону (left - right)
        если выбран RIGHT, то веса смещаются в сторону (right - left)

    ez baseline
    """

    def __init__(
        self,
        dim: int,
        learning_rate: float = 0.1,
        l2_reg: float = 0.0,
        initial_weights: np.ndarray | None = None,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if l2_reg < 0:
            raise ValueError("l2_reg must be non-negative")

        self.dim = int(dim)
        self.learning_rate = float(learning_rate)
        self.l2_reg = float(l2_reg)

        if initial_weights is None:
            self._weights = np.zeros(self.dim, dtype=np.float32)
        else:
            w = np.asarray(initial_weights, dtype=np.float32)
            if w.ndim != 1 or w.shape[0] != self.dim:
                raise ValueError("initial_weights must be a 1D array of shape (dim,)")
            self._weights = w.copy()

    @property
    def weights(self) -> np.ndarray:
        """
        Возвращает копию весов, чтобы их нельзя было случайно
        изменить снаружи.
        """
        return self._weights.copy()

    def reset(self) -> None:
        self._weights.fill(0.0)

    def set_weights(self, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim != 1 or w.shape[0] != self.dim:
            raise ValueError("weights must be a 1D array of shape (dim,)")
        self._weights = w.copy()

    def score(self, profile: PerceptualProfile) -> float:
        z = self._as_vector(profile)
        return float(np.dot(self._weights, z))

    def prefer_left_probability(
        self,
        left: PerceptualProfile,
        right: PerceptualProfile,
    ) -> float:
        """
        Вероятность выбора left над right.

        P(left > right) = sigmoid(score(left) - score(right))
        """
        diff = self.score(left) - self.score(right)
        return float(self._sigmoid(diff))

    def update_from_choice(
        self,
        pair: PerceptualPair,
        choice: PreferenceChoice,
    ) -> None:
        """
        Обновление модели по выбору пользователя.

        LEFT:
            w <- w + lr * (left - right)

        RIGHT:
            w <- w + lr * (right - left)

        REJECT_BOTH / SKIP / STOP:
            пока не обновляют модель

        Дополнительно может применяться L2-регуляризация.
        """
        left = self._as_vector(pair.left)
        right = self._as_vector(pair.right)

        if choice == PreferenceChoice.LEFT:
            delta = left - right
        elif choice == PreferenceChoice.RIGHT:
            delta = right - left
        elif choice in (
            PreferenceChoice.REJECT_BOTH,
            PreferenceChoice.SKIP,
            PreferenceChoice.STOP,
        ):
            return
        else:
            raise ValueError(f"Unsupported choice: {choice}")

        self._weights = self._weights + self.learning_rate * delta

        if self.l2_reg > 0.0:
            self._weights = self._weights * (1.0 - self.learning_rate * self.l2_reg)

        self._weights = self._weights.astype(np.float32, copy=False)

    def get_state(self) -> ModelState:
        return ModelState(
            model_type="linear_preference",
            dim=self.dim,
            learning_rate=self.learning_rate,
            l2_reg=self.l2_reg,
            weights=tuple(float(x) for x in self._weights.tolist()),
            version=1,
        )

    def load_state(self, state: ModelState) -> None:
        if state.model_type != "linear_preference":
            raise ValueError(f"Unsupported model_type: {state.model_type}")
        if state.dim != self.dim:
            raise ValueError(f"State dim mismatch: expected {self.dim}, got {state.dim}")

        self.learning_rate = float(state.learning_rate)
        self.l2_reg = float(state.l2_reg)
        self._weights = np.asarray(state.weights, dtype=np.float32)


    def rank_profiles(
        self,
        profiles: list[PerceptualProfile],
    ) -> list[tuple[PerceptualProfile, float]]:
        """
        Удобный вспомогательный метод:
        отсортировать профили по score от лучшего к худшему.
        """
        scored = [(profile, self.score(profile)) for profile in profiles]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def best_of(self, profiles: list[PerceptualProfile]) -> PerceptualProfile:
        if not profiles:
            raise ValueError("profiles must not be empty")
        return max(profiles, key=self.score)

    def _as_vector(self, profile: PerceptualProfile) -> np.ndarray:
        z = np.asarray(profile.values, dtype=np.float32)
        if z.ndim != 1 or z.shape[0] != self.dim:
            raise ValueError(
                f"Profile dimension mismatch: expected {self.dim}, got {z.shape}"
            )
        return z

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Чуть более стабильная версия для больших значений
        if x >= 0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        z = np.exp(x)
        return z / (1.0 + z)