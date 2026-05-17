from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import clip_vector


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr, dtype=np.float64)
    positive = x_arr >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x_arr[positive]))
    exp_x = np.exp(x_arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


@dataclass
class LogisticDistancePreferenceModel:
    """
    Online pairwise Preference Model in compact 8D space.

    The model keeps an estimate z_pref of the user's preferred point and defines
    a predicted utility as a negative squared distance:

        U_hat(z) = -sum_j feature_weight_j * (z_j - z_pref_j)^2

    The probability that candidate A is preferred to candidate B is:

        P(A > B) = sigmoid((U_hat(A) - U_hat(B)) / temperature)

    This is a lightweight V4a model. It is trained online from A/B choices and
    does not require any real user logs at this stage. The synthetic z_target is
    used only for evaluation, not for training.
    """

    dim: int = 8
    lr: float = 0.06
    temperature: float = 0.75
    l2: float = 0.003
    clip_value: float | None = 2.0
    feature_weight: np.ndarray | None = None
    z_pref: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.z_pref is None:
            self.z_pref = np.zeros(int(self.dim), dtype=np.float64)
        else:
            self.z_pref = np.asarray(self.z_pref, dtype=np.float64).copy()
            self.dim = int(len(self.z_pref))

        if self.feature_weight is None:
            self.feature_weight = np.ones(int(self.dim), dtype=np.float64)
        else:
            self.feature_weight = np.asarray(self.feature_weight, dtype=np.float64).copy()
            if self.feature_weight.shape != self.z_pref.shape:
                raise ValueError("feature_weight must have the same shape as z_pref")
            self.feature_weight = self.feature_weight / (self.feature_weight.mean() + 1e-8)

        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    def utility(self, z: np.ndarray) -> float:
        """Predicted negative squared distance utility."""
        z = np.asarray(z, dtype=np.float64)
        diff = z - self.z_pref
        return -float(np.sum(self.feature_weight * diff * diff))

    def preference_logit(self, z_a: np.ndarray, z_b: np.ndarray) -> float:
        """Logit for P(A > B)."""
        return float((self.utility(z_a) - self.utility(z_b)) / float(self.temperature))

    def predict_proba_a(self, z_a: np.ndarray, z_b: np.ndarray) -> float:
        """Predict probability that A is preferred to B."""
        return float(sigmoid(self.preference_logit(z_a, z_b)))

    def predict_choice(self, z_a: np.ndarray, z_b: np.ndarray) -> str:
        """Predict the most likely A/B choice."""
        return "A" if self.predict_proba_a(z_a, z_b) >= 0.5 else "B"

    def log_loss(self, z_a: np.ndarray, z_b: np.ndarray, choice: str, eps: float = 1e-8) -> float:
        """Binary log loss for an observed choice."""
        if choice not in {"A", "B"}:
            raise ValueError("choice must be 'A' or 'B'")
        p = np.clip(self.predict_proba_a(z_a, z_b), eps, 1.0 - eps)
        y = 1.0 if choice == "A" else 0.0
        return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def update(self, z_a: np.ndarray, z_b: np.ndarray, choice: str) -> dict:
        """
        Update z_pref from one A/B observation using logistic loss.

        Returns diagnostic values computed before and after the update.
        """
        if choice not in {"A", "B"}:
            raise ValueError("choice must be 'A' or 'B'")

        z_a = np.asarray(z_a, dtype=np.float64)
        z_b = np.asarray(z_b, dtype=np.float64)
        y = 1.0 if choice == "A" else 0.0

        p_before = self.predict_proba_a(z_a, z_b)
        pred_before = "A" if p_before >= 0.5 else "B"
        loss_before = self.log_loss(z_a, z_b, choice)

        # d(logit)/d(z_pref) = 2 * feature_weight * (z_a - z_b) / temperature
        delta = z_a - z_b
        dlogit_dz = 2.0 * self.feature_weight * delta / float(self.temperature)

        # Binary cross-entropy gradient: (p - y) * dlogit/dz + L2.
        grad = (p_before - y) * dlogit_dz + float(self.l2) * self.z_pref
        self.z_pref = self.z_pref - float(self.lr) * grad
        self.z_pref = clip_vector(self.z_pref, self.clip_value)

        p_after = self.predict_proba_a(z_a, z_b)
        loss_after = self.log_loss(z_a, z_b, choice)

        record = {
            "choice": choice,
            "p_before": float(p_before),
            "p_after": float(p_after),
            "pred_before": pred_before,
            "correct_before": bool(pred_before == choice),
            "loss_before": float(loss_before),
            "loss_after": float(loss_after),
            "grad_norm": float(np.linalg.norm(grad)),
            "z_pref_after": self.z_pref.copy(),
        }
        self.history.append(record)
        return record

    def copy(self) -> "LogisticDistancePreferenceModel":
        """Return a copy of the model."""
        return LogisticDistancePreferenceModel(
            dim=int(self.dim),
            lr=float(self.lr),
            temperature=float(self.temperature),
            l2=float(self.l2),
            clip_value=self.clip_value,
            feature_weight=self.feature_weight.copy(),
            z_pref=self.z_pref.copy(),
            history=list(self.history),
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity with safe zero handling."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)
