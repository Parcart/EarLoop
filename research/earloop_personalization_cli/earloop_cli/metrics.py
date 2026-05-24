from __future__ import annotations

import numpy as np

from .vendor_path import ensure_vendor_path
ensure_vendor_path()
from personalization.contract_metrics import curve_metrics, mapped_pair_metrics, mapped_curve_metrics
from personalization.contract_space import contract_pair_distance


def rms(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def vector_norm(x) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=np.float64)))


def cosine(a, b, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def flatten_vector(prefix: str, names: list[str], values) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {f"{prefix}_{name}": float(arr[i]) for i, name in enumerate(names)}
