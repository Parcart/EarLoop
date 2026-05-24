from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .control_basis import CONTROL_BASIS_6D_TO_8D, CONTROL_NAMES_6D
from .loop import PairStrategy
from .preference_model import LogisticDistancePreferenceModel
from .preference_model_eval import (
    PreferenceModelSessionResult,
    run_preference_model_learning_session_v4a,
)
from .state import PreferenceState, clip_vector
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser


@dataclass
class HeldoutPairRecord:
    """One unseen A/B pair used to evaluate a trained per-user Preference Model."""

    pair_id: int
    source: str
    choice: str
    predicted_choice: str
    correct: bool
    p_a: float
    log_loss: float
    u_a: float
    u_b: float
    utility_margin: float
    z_a: np.ndarray
    z_b: np.ndarray


def _sigmoid_safe_log_loss(p_a: float, choice: str, eps: float = 1e-8) -> float:
    """Binary log loss for an A/B choice from already computed P(A>B)."""
    p = float(np.clip(p_a, eps, 1.0 - eps))
    y = 1.0 if choice == "A" else 0.0
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + eps)


def _make_symmetric_pair(
    center: np.ndarray,
    direction: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    clip_value: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one held-out A/B pair around a fixed center."""
    d = _normalize(direction)
    if rng.random() < 0.5:
        d = -d
    z_a = center + float(scale) * d
    z_b = center - float(scale) * d
    z_a = clip_vector(z_a, clip_value)
    z_b = clip_vector(z_b, clip_value)
    return z_a, z_b


def generate_heldout_pair(
    center: np.ndarray,
    rng: np.random.Generator,
    source_probs: dict[str, float] | None = None,
    step_scale: float = 0.6,
    clip_value: float | None = 2.0,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Generate one held-out A/B pair that was not used during training.

    The pair is generated around the final heuristic center from a source mix:
    random 8D directions, semantic 6D directions, and one-axis directions.
    This tests whether the trained model generalizes to new questions near the
    personalization region instead of only explaining the training trajectory.
    """
    center = np.asarray(center, dtype=np.float64)
    dim = len(center)

    if source_probs is None:
        source_probs = {
            "random": 0.40,
            "semantic6d": 0.40,
            "axis": 0.20,
        }

    sources = list(source_probs.keys())
    probs = np.array([source_probs[name] for name in sources], dtype=np.float64)
    probs = probs / (probs.sum() + 1e-8)
    source = str(rng.choice(sources, p=probs))

    if source == "semantic6d":
        idx = int(rng.integers(0, len(CONTROL_NAMES_6D)))
        direction = CONTROL_BASIS_6D_TO_8D[idx]
        # A small range of semantic question widths.
        scale = float(step_scale * rng.choice([0.45, 0.65, 0.85]))
        source_label = f"semantic6d:{CONTROL_NAMES_6D[idx]}"

    elif source == "axis":
        axis = int(rng.integers(0, dim))
        direction = np.zeros(dim, dtype=np.float64)
        direction[axis] = 1.0
        scale = float(step_scale * rng.choice([0.45, 0.70]))
        source_label = f"axis:{axis}"

    elif source == "random":
        direction = rng.normal(size=dim)
        scale = float(step_scale * rng.choice([0.45, 0.70, 0.95]))
        source_label = "random"

    else:
        raise ValueError(f"Unknown held-out source: {source}")

    z_a, z_b = _make_symmetric_pair(
        center=center,
        direction=direction,
        scale=scale,
        rng=rng,
        clip_value=clip_value,
    )
    return z_a, z_b, source_label


def evaluate_model_on_heldout_pairs(
    model: LogisticDistancePreferenceModel,
    synthetic_user: SyntheticUser,
    center: np.ndarray,
    n_pairs: int = 100,
    step_scale: float = 0.6,
    clip_value: float | None = 2.0,
    seed: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate a trained per-user model on new A/B pairs.

    The synthetic user answers these pairs, but the Preference Model is not
    updated. This is a post-session generalization test.
    """
    rng = np.random.default_rng(seed)
    records: list[HeldoutPairRecord] = []

    for pair_id in range(int(n_pairs)):
        z_a, z_b, source = generate_heldout_pair(
            center=center,
            rng=rng,
            step_scale=step_scale,
            clip_value=clip_value,
        )
        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)
        p_a = float(model.predict_proba_a(z_a, z_b))
        predicted_choice = "A" if p_a >= 0.5 else "B"
        correct = bool(predicted_choice == choice)
        log_loss = _sigmoid_safe_log_loss(p_a, choice)
        records.append(HeldoutPairRecord(
            pair_id=pair_id,
            source=source,
            choice=choice,
            predicted_choice=predicted_choice,
            correct=correct,
            p_a=p_a,
            log_loss=log_loss,
            u_a=float(u_a),
            u_b=float(u_b),
            utility_margin=float(abs(u_a - u_b)),
            z_a=z_a.copy(),
            z_b=z_b.copy(),
        ))

    rows = []
    for r in records:
        rows.append({
            "pair_id": int(r.pair_id),
            "source": r.source,
            "choice": r.choice,
            "predicted_choice": r.predicted_choice,
            "correct": bool(r.correct),
            "p_a": float(r.p_a),
            "log_loss": float(r.log_loss),
            "u_a": float(r.u_a),
            "u_b": float(r.u_b),
            "utility_margin": float(r.utility_margin),
        })
    pairs = pd.DataFrame(rows)

    if len(pairs) == 0:
        summary = {
            "heldout_n_pairs": 0,
            "heldout_accuracy": float("nan"),
            "heldout_log_loss": float("nan"),
            "heldout_mean_margin": float("nan"),
            "heldout_mean_confidence": float("nan"),
        }
    else:
        confidence = np.abs(pairs["p_a"].to_numpy(dtype=np.float64) - 0.5) * 2.0
        summary = {
            "heldout_n_pairs": int(len(pairs)),
            "heldout_accuracy": float(pairs["correct"].mean()),
            "heldout_log_loss": float(pairs["log_loss"].mean()),
            "heldout_mean_margin": float(pairs["utility_margin"].mean()),
            "heldout_mean_confidence": float(np.mean(confidence)),
        }

    return summary, pairs


def run_preference_model_heldout_session_v4a1(
    synthetic_user: SyntheticUser,
    n_steps: int = 25,
    n_heldout_pairs: int = 100,
    pair_strategy: PairStrategy = "semantic_active_v21",
    step_scale: float = 0.6,
    heuristic_lr: float = 0.25,
    model_lr: float = 0.06,
    model_temperature: float = 0.75,
    model_l2: float = 0.003,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    train_seed: int | None = None,
    heldout_seed: int | None = None,
    model_feature_weight: Literal["uniform", "oracle"] = "uniform",
) -> tuple[PreferenceModelSessionResult, dict, pd.DataFrame]:
    """Train one per-user V4a model and evaluate it on unseen held-out pairs."""
    result = run_preference_model_learning_session_v4a(
        synthetic_user=synthetic_user,
        n_steps=n_steps,
        step_scale=step_scale,
        heuristic_lr=heuristic_lr,
        model_lr=model_lr,
        model_temperature=model_temperature,
        model_l2=model_l2,
        init_std=init_std,
        std_decay=std_decay,
        min_std=min_std,
        clip_value=clip_value,
        pair_strategy=pair_strategy,
        seed=train_seed,
        model_feature_weight=model_feature_weight,
    )

    # Evaluate around the final heuristic preference state: this is the area the
    # personalization loop would likely continue exploring after the session.
    center = result.final_heuristic_state.z_mean.copy()
    heldout_summary, heldout_pairs = evaluate_model_on_heldout_pairs(
        model=result.final_model,
        synthetic_user=synthetic_user,
        center=center,
        n_pairs=n_heldout_pairs,
        step_scale=step_scale,
        clip_value=clip_value,
        seed=heldout_seed,
    )
    return result, heldout_summary, heldout_pairs


def run_preference_model_heldout_batch_v4a1(
    dataset: pd.DataFrame,
    n_steps: int = 25,
    n_heldout_pairs: int = 100,
    pair_strategy: PairStrategy = "semantic_active_v21",
    step_scale: float = 0.6,
    heuristic_lr: float = 0.25,
    model_lr: float = 0.06,
    model_temperature: float = 0.75,
    model_l2: float = 0.003,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    train_seed_base: int = 30_000,
    heldout_seed_base: int = 70_000,
    user_seed_base: int = 10_000,
    model_feature_weight: Literal["uniform", "oracle"] = "uniform",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run V4a.1 held-out evaluation on a fixed synthetic user dataset.

    A new Logistic Preference Model is initialized for every user. It is trained
    online on the session pairs and then evaluated on unseen held-out pairs.
    """
    session_rows: list[dict] = []
    train_step_rows: list[dict] = []
    heldout_rows: list[dict] = []

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        user = row_to_synthetic_user(row, seed=user_seed_base + user_id)

        result, heldout_summary, heldout_pairs = run_preference_model_heldout_session_v4a1(
            synthetic_user=user,
            n_steps=n_steps,
            n_heldout_pairs=n_heldout_pairs,
            pair_strategy=pair_strategy,
            step_scale=step_scale,
            heuristic_lr=heuristic_lr,
            model_lr=model_lr,
            model_temperature=model_temperature,
            model_l2=model_l2,
            init_std=init_std,
            std_decay=std_decay,
            min_std=min_std,
            clip_value=clip_value,
            train_seed=train_seed_base + user_id,
            heldout_seed=heldout_seed_base + user_id,
            model_feature_weight=model_feature_weight,
        )

        initial_distance = float(np.linalg.norm(z_target))
        heuristic_final = float(result.heuristic_distances[-1]) if len(result.heuristic_distances) else initial_distance
        model_final = float(result.model_distances[-1]) if len(result.model_distances) else initial_distance
        final_cosine = float(result.model_cosines[-1]) if len(result.model_cosines) else float("nan")

        session_row = {
            "user_id": user_id,
            "target_mode": target_mode,
            "n_steps": int(n_steps),
            "n_heldout_pairs": int(n_heldout_pairs),
            "initial_distance": initial_distance,
            "heuristic_final_distance": heuristic_final,
            "model_final_distance": model_final,
            "heuristic_best_distance": float(np.min(np.concatenate([[initial_distance], result.heuristic_distances]))),
            "model_best_distance": float(np.min(np.concatenate([[initial_distance], result.model_distances]))),
            "train_prediction_accuracy": float(result.prediction_accuracy),
            "train_log_loss_before": float(result.mean_log_loss_before),
            "train_log_loss_after": float(result.mean_log_loss_after),
            "final_model_cosine_to_target": final_cosine,
        }
        session_row.update(heldout_summary)
        session_rows.append(session_row)

        for record in result.records:
            train_step_rows.append({
                "user_id": user_id,
                "target_mode": target_mode,
                "step": int(record.step),
                "choice": record.choice,
                "p_a_before": float(record.p_a_before),
                "p_a_after": float(record.p_a_after),
                "pred_before": record.pred_before,
                "correct_before": bool(record.correct_before),
                "log_loss_before": float(record.log_loss_before),
                "log_loss_after": float(record.log_loss_after),
                "heuristic_distance_to_target": float(record.heuristic_distance_to_target),
                "model_distance_to_target": float(record.model_distance_to_target),
                "model_cosine_to_target": float(record.model_cosine_to_target),
                "pair_strategy": record.pair_strategy,
                "pair_sub_strategy": record.pair_meta.get("sub_strategy"),
                "control_name": record.pair_meta.get("control_name"),
            })

        for _, pair_row in heldout_pairs.iterrows():
            payload = pair_row.to_dict()
            payload.update({
                "user_id": user_id,
                "target_mode": target_mode,
            })
            heldout_rows.append(payload)

    sessions = pd.DataFrame(session_rows)
    train_steps = pd.DataFrame(train_step_rows)
    heldout_pairs = pd.DataFrame(heldout_rows)
    return sessions, train_steps, heldout_pairs


def summarize_v4a1_by_target_mode(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out Preference Model results by target_mode."""
    return (
        sessions
        .groupby("target_mode")
        .agg(
            users=("user_id", "count"),
            mean_initial_distance=("initial_distance", "mean"),
            heuristic_mean_final_distance=("heuristic_final_distance", "mean"),
            model_mean_final_distance=("model_final_distance", "mean"),
            mean_final_model_cosine_to_target=("final_model_cosine_to_target", "mean"),
            mean_train_accuracy=("train_prediction_accuracy", "mean"),
            mean_train_log_loss_before=("train_log_loss_before", "mean"),
            mean_train_log_loss_after=("train_log_loss_after", "mean"),
            mean_heldout_accuracy=("heldout_accuracy", "mean"),
            mean_heldout_log_loss=("heldout_log_loss", "mean"),
            mean_heldout_confidence=("heldout_mean_confidence", "mean"),
            mean_heldout_margin=("heldout_mean_margin", "mean"),
        )
        .reset_index()
    )


def summarize_v4a1_heldout_by_source(heldout_pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out accuracy/log-loss by target mode and question source."""
    if heldout_pairs.empty:
        return pd.DataFrame()
    source_group = heldout_pairs.copy()
    source_group["source_group"] = source_group["source"].astype(str).str.split(":").str[0]
    return (
        source_group
        .groupby(["target_mode", "source_group"])
        .agg(
            pairs=("pair_id", "count"),
            heldout_accuracy=("correct", "mean"),
            heldout_log_loss=("log_loss", "mean"),
            mean_margin=("utility_margin", "mean"),
            mean_p_a=("p_a", "mean"),
        )
        .reset_index()
    )


def save_v4a1_outputs(
    sessions: pd.DataFrame,
    train_steps: pd.DataFrame,
    heldout_pairs: pd.DataFrame,
    summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "v4a1_preference_model_heldout",
) -> None:
    """Save V4a.1 held-out evaluation tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    train_steps.to_csv(output_dir / f"{prefix}_train_steps.csv", index=False)
    heldout_pairs.to_csv(output_dir / f"{prefix}_heldout_pairs.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    source_summary.to_csv(output_dir / f"{prefix}_heldout_by_source.csv", index=False)
