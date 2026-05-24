from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .vendor_path import ensure_vendor_path

ensure_vendor_path()
from personalization.contract_mapper import (
    FREQS_23_DEFAULT,
    InterpretableContractMapper8D,
    TorchScriptContractMapper,
    apply_curve_safety,
    MapperSafetyConfig,
)


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU approximation used by most PyTorch GELU defaults closely enough for inference."""
    # tanh approximation is enough for this small tester runtime and avoids scipy dependency here.
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


class NumpyMLPContractMapper:
    """Tiny NumPy runtime for exported Model B contract controllable MLP.

    This avoids bundling PyTorch into the tester exe. The model is expected as an
    .npz file exported from the PyTorch state_dict with keys:
    net_0_weight, net_0_bias, net_2_weight, net_2_bias, net_4_weight,
    net_4_bias, net_6_weight, net_6_bias.
    """

    def __init__(self, model_path: str | Path, max_abs_db: float = 12.0) -> None:
        self.model_path = Path(model_path)
        data = np.load(self.model_path)
        self.weights = [
            data["net_0_weight"].astype(np.float32),
            data["net_2_weight"].astype(np.float32),
            data["net_4_weight"].astype(np.float32),
            data["net_6_weight"].astype(np.float32),
        ]
        self.biases = [
            data["net_0_bias"].astype(np.float32),
            data["net_2_bias"].astype(np.float32),
            data["net_4_bias"].astype(np.float32),
            data["net_6_bias"].astype(np.float32),
        ]
        self.freqs_hz = data["freqs"].astype(np.float64) if "freqs" in data else FREQS_23_DEFAULT
        self.safety_config = MapperSafetyConfig(max_abs_db=float(max_abs_db))
        self.mapper_version = f"numpy_model_b_contract_controllable_mlp:{self.model_path.name}"

    def map_batch(self, z_contract: np.ndarray) -> np.ndarray:
        x = np.asarray(z_contract, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != 8:
            raise ValueError(f"Expected z_contract shape [N, 8], got {x.shape}")

        h = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = h @ w.T + b
            h = _gelu(h)
        y = h @ self.weights[-1].T + self.biases[-1]

        # Final runtime safety for tester builds: no extreme EQ values.
        y = np.asarray([apply_curve_safety(row, self.safety_config) for row in y], dtype=np.float32)
        return y

    def map_one(self, z_contract: np.ndarray) -> np.ndarray:
        return self.map_batch(np.asarray(z_contract, dtype=np.float32)[None, :])[0]


def _resolve_model_path(model_path_raw: str) -> Path | None:
    raw = str(model_path_raw or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path

    # First resolve relative to current working directory, then relative to package root.
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    package_root = Path(__file__).resolve().parents[1]
    return package_root / path


def make_mapper(cfg: Any):
    """Create runtime mapper for the CLI.

    Modes:
    - npz: learned Model B/v4 mapper exported to NumPy weights, no PyTorch dependency.
    - torchscript: learned mapper exported as TorchScript (.pt/.ts), requires torch.
    - interpretable: deterministic hand-built contract mapper, safe fallback.
    - auto: use .npz if model_path exists, then TorchScript for .pt/.ts, otherwise fallback.
    """
    mapper_cfg = getattr(cfg, "mapper", None)
    mode = str(getattr(mapper_cfg, "mode", "auto") or "auto").lower()
    model_path = _resolve_model_path(str(getattr(mapper_cfg, "model_path", "") or ""))
    allow_fallback = bool(getattr(mapper_cfg, "allow_interpretable_fallback", True))
    device = str(getattr(mapper_cfg, "device", "cpu") or "cpu")
    max_abs_db = float(getattr(cfg, "max_abs_db", 12.0))

    try:
        if mode in {"auto", "npz", "numpy"} and model_path is not None and model_path.exists() and model_path.suffix.lower() == ".npz":
            return NumpyMLPContractMapper(model_path=model_path, max_abs_db=max_abs_db)

        if mode in {"npz", "numpy"}:
            raise FileNotFoundError(f"NumPy mapper not found: {model_path}")

        use_torchscript = mode == "torchscript" or (
            mode == "auto"
            and model_path is not None
            and model_path.exists()
            and model_path.suffix.lower() in {".pt", ".ts", ".torchscript"}
        )
        if use_torchscript:
            if model_path is None or not model_path.exists():
                raise FileNotFoundError(f"TorchScript mapper not found: {model_path}")
            mapper = TorchScriptContractMapper(model_path=model_path, device=device)
            mapper.mapper_version = f"torchscript_contract_mapper:{model_path.name}"
            return mapper

        if mode == "interpretable" or mode == "auto":
            mapper = InterpretableContractMapper8D()
            mapper.mapper_version = "interpretable_contract_mapper_8d_runtime"
            return mapper

        raise ValueError(f"Unknown mapper mode: {mode}")

    except Exception:
        if not allow_fallback:
            raise
        mapper = InterpretableContractMapper8D()
        mapper.mapper_version = f"interpretable_contract_mapper_8d_fallback_after_{mode}_failure"
        return mapper
