"""Reusable PyTorch mapper models and training utilities.

This module keeps the architecture close to the previous mapper notebooks:
- MapperMLP baseline
- controllability alignment loss
- BasisResidualMapper / hybrid-style mapper
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

from .feature_space import FEATURE_NAMES_8D
from .mapper_basis import build_basis_matrix_8d


def reset_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class MapperMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128, 64),
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        if activation == "relu":
            act_layer = nn.ReLU
        elif activation == "gelu":
            act_layer = nn.GELU
        elif activation == "leaky_relu":
            act_layer = nn.LeakyReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasisResidualMapper(nn.Module):
    """Base interpretable linear basis + residual MLP.

    This keeps the learned mapper close to the expected 8D basis, while allowing
    residual corrections from data.
    """

    def __init__(
        self,
        expected_basis: np.ndarray,
        input_dim: int = 8,
        output_dim: int = 23,
        hidden_dims: tuple[int, ...] = (64, 64),
        activation: str = "gelu",
        dropout: float = 0.0,
        residual_scale: float = 0.5,
        basis_scale: float = 1.0,
        trainable_basis_scale: bool = False,
    ):
        super().__init__()
        expected_basis_t = torch.tensor(expected_basis, dtype=torch.float32)
        self.register_buffer("expected_basis", expected_basis_t)
        if trainable_basis_scale:
            self.basis_scale = nn.Parameter(torch.tensor(float(basis_scale), dtype=torch.float32))
        else:
            self.register_buffer("basis_scale", torch.tensor(float(basis_scale), dtype=torch.float32))
        self.residual_scale = float(residual_scale)
        self.residual_mlp = MapperMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )

    def forward_with_parts(self, x: torch.Tensor):
        base = self.basis_scale * (x @ self.expected_basis)
        residual = self.residual_mlp(x)
        curve = base + self.residual_scale * residual
        return curve, base, residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        curve, _, _ = self.forward_with_parts(x)
        return curve


def normalize_vector_torch(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def controllability_alignment_loss(
    model: nn.Module,
    expected_basis: np.ndarray | torch.Tensor,
    feature_names: list[str] = FEATURE_NAMES_8D,
    sweep_value: float = 1.0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    num_features = len(feature_names)
    expected_basis_t = torch.tensor(expected_basis, dtype=torch.float32, device=device)

    z_zero = torch.zeros(1, num_features, dtype=torch.float32, device=device)
    y_zero = model(z_zero)
    losses = []
    for i in range(num_features):
        z_plus = torch.zeros(1, num_features, dtype=torch.float32, device=device)
        z_minus = torch.zeros(1, num_features, dtype=torch.float32, device=device)
        z_plus[0, i] = sweep_value
        z_minus[0, i] = -sweep_value
        y_plus = model(z_plus)
        y_minus = model(z_minus)
        diff_plus = y_plus - y_zero
        diff_minus = y_minus - y_zero
        basis_i = expected_basis_t[i].unsqueeze(0)
        diff_plus_n = normalize_vector_torch(diff_plus)
        diff_minus_n = normalize_vector_torch(diff_minus)
        basis_n = normalize_vector_torch(basis_i)
        cos_plus = torch.sum(diff_plus_n * basis_n, dim=-1)
        cos_minus = torch.sum(diff_minus_n * (-basis_n), dim=-1)
        losses.append(1.0 - cos_plus)
        losses.append(1.0 - cos_minus)
    return torch.mean(torch.cat(losses))


@dataclass
class MapperTrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    lambda_smooth: float = 0.0
    lambda_ctrl: float = 0.0
    lambda_residual: float = 0.0
    ctrl_sweep_value: float = 1.0
    num_epochs: int = 250
    verbose_every: int = 25
    seed: int = 42
    use_tqdm: bool = True


class MapperExperiment:
    def __init__(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        freqs: Iterable[float],
        expected_basis: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        config: MapperTrainConfig | None = None,
        device: str | None = None,
    ):
        self.model = model
        self.freqs = np.asarray(freqs, dtype=float)
        self.feature_names = feature_names or FEATURE_NAMES_8D
        self.expected_basis = expected_basis
        self.config = config or MapperTrainConfig()
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val
        self.X_test, self.Y_test = X_test, Y_test
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.history: list[dict[str, float]] = []
        self.train_loader = self._make_loader(X_train, Y_train, shuffle=True)
        self.val_loader = self._make_loader(X_val, Y_val, shuffle=False)
        self.test_loader = self._make_loader(X_test, Y_test, shuffle=False)

    def _make_loader(self, X: np.ndarray, Y: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            EqDataset(X, Y),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
        )

    @staticmethod
    def smoothness_penalty(y_pred: torch.Tensor) -> torch.Tensor:
        diff = y_pred[:, 1:] - y_pred[:, :-1]
        return torch.mean(diff ** 2)

    def _run_epoch(self, loader: DataLoader, optimizer=None) -> dict[str, float]:
        is_train = optimizer is not None
        self.model.train(is_train)
        total_loss_sum = 0.0
        total_mse_sum = 0.0
        total_smooth_sum = 0.0
        total_ctrl_sum = 0.0
        total_residual_sum = 0.0
        total_count = 0
        cfg = self.config

        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            if hasattr(self.model, "forward_with_parts"):
                y_pred, _, residual = self.model.forward_with_parts(xb)
                residual_penalty = torch.mean(residual ** 2)
            else:
                y_pred = self.model(xb)
                residual_penalty = torch.tensor(0.0, device=self.device)

            mse = torch.mean((y_pred - yb) ** 2)
            smooth = self.smoothness_penalty(y_pred)

            if self.expected_basis is not None and cfg.lambda_ctrl > 0:
                ctrl = controllability_alignment_loss(
                    self.model,
                    expected_basis=self.expected_basis,
                    feature_names=self.feature_names,
                    sweep_value=cfg.ctrl_sweep_value,
                    device=self.device,
                )
            else:
                ctrl = torch.tensor(0.0, device=self.device)

            loss = (
                mse
                + cfg.lambda_smooth * smooth
                + cfg.lambda_ctrl * ctrl
                + cfg.lambda_residual * residual_penalty
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = xb.size(0)
            total_loss_sum += loss.item() * batch_size
            total_mse_sum += mse.item() * batch_size
            total_smooth_sum += smooth.item() * batch_size
            total_ctrl_sum += ctrl.item() * batch_size
            total_residual_sum += residual_penalty.item() * batch_size
            total_count += batch_size

        return {
            "loss": total_loss_sum / total_count,
            "mse": total_mse_sum / total_count,
            "smooth": total_smooth_sum / total_count,
            "ctrl": total_ctrl_sum / total_count,
            "residual_penalty": total_residual_sum / total_count,
        }

    def fit(self) -> pd.DataFrame:
        """Train mapper and return per-epoch metrics.

        The objective is intentionally kept close to the original mapper notebooks:
        MSE + lambda_smooth * smoothness + lambda_ctrl * controllability +
        lambda_residual * residual_penalty. tqdm only changes progress rendering.
        """
        cfg = self.config
        reset_seeds(cfg.seed)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        best_val_mse = float("inf")
        best_state = None

        epoch_iter = range(1, cfg.num_epochs + 1)
        progress = None
        if cfg.use_tqdm and tqdm is not None:
            progress = tqdm(epoch_iter, desc="training", dynamic_ncols=True, leave=True)
            epoch_iter = progress

        for epoch in epoch_iter:
            train_metrics = self._run_epoch(self.train_loader, optimizer=optimizer)
            val_metrics = self._run_epoch(self.val_loader, optimizer=None)
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(row)
            if val_metrics["mse"] < best_val_mse:
                best_val_mse = val_metrics["mse"]
                best_state = deepcopy(self.model.state_dict())

            postfix = {
                "train_mse": f"{train_metrics['mse']:.5f}",
                "val_mse": f"{val_metrics['mse']:.5f}",
            }
            if cfg.lambda_ctrl > 0:
                postfix["ctrl"] = f"{train_metrics['ctrl']:.4f}"
            if progress is not None:
                progress.set_postfix(postfix)
            elif cfg.verbose_every and epoch % cfg.verbose_every == 0:
                print(
                    f"Epoch {epoch:04d} | "
                    f"train_mse={train_metrics['mse']:.6f} | "
                    f"val_mse={val_metrics['mse']:.6f}"
                )
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return pd.DataFrame(self.history)

    def evaluate(self, split: str = "test") -> dict[str, float]:
        if split == "train":
            return self._run_epoch(self.train_loader, optimizer=None)
        if split == "val":
            return self._run_epoch(self.val_loader, optimizer=None)
        if split == "test":
            return self._run_epoch(self.test_loader, optimizer=None)
        raise ValueError(f"Unknown split: {split}")

    @torch.no_grad()
    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = []
        for start in range(0, len(X_t), batch_size):
            xb = X_t[start:start+batch_size].to(self.device)
            preds.append(self.model(xb).detach().cpu().numpy())
        return np.concatenate(preds, axis=0).astype(np.float32)

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "freqs": self.freqs,
            "feature_names": self.feature_names,
            "config": self.config.__dict__,
            "history": self.history,
            "extra": extra or {},
        }
        torch.save(payload, path)


def split_train_val_test(
    X: np.ndarray,
    Y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_frac)
    n_val = int(len(idx) * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], X[test_idx], Y[test_idx]


def make_mapper_model(
    model_type: str,
    input_dim: int,
    output_dim: int,
    freqs: Iterable[float],
    hidden_dims: tuple[int, ...] = (128, 128, 64),
    activation: str = "gelu",
    dropout: float = 0.0,
    residual_scale: float = 0.4,
    basis_scale: float = 1.0,
) -> tuple[nn.Module, np.ndarray | None]:
    model_type = model_type.lower()
    if model_type == "mlp":
        return MapperMLP(input_dim, output_dim, hidden_dims, activation, dropout), None
    if model_type == "basis_residual":
        expected_basis = build_basis_matrix_8d(freqs)
        model = BasisResidualMapper(
            expected_basis=expected_basis,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            residual_scale=residual_scale,
            basis_scale=basis_scale,
        )
        return model, expected_basis
    if model_type == "controllable_mlp":
        expected_basis = build_basis_matrix_8d(freqs)
        return MapperMLP(input_dim, output_dim, hidden_dims, activation, dropout), expected_basis
    raise ValueError(f"Unknown model_type: {model_type}")
