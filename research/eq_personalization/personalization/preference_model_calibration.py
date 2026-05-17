from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .metrics import distance_to_target
from .preference_model import LogisticDistancePreferenceModel, cosine_similarity, sigmoid
from .preference_model_eval import PreferenceModelSessionResult, run_preference_model_learning_session_v4a
from .preference_model_heldout import generate_heldout_pair
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser


CALIBRATION_DISPLAY_NAMES = {
    "heuristic_update": "Heuristic update",
    "raw_preference_model": "Raw Preference Model",
    "norm_calibrated_model": "Norm-calibrated PM",
    "train_scale_model": "Train-scale PM",
    "blend_70h_30m": "Blend 70% heuristic + 30% PM",
    "blend_50h_50m": "Blend 50% heuristic + 50% PM",
}


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + eps)


def _clip_vector(v: np.ndarray, clip_value: float | None) -> np.ndarray:
    if clip_value is None:
        return np.asarray(v, dtype=np.float64)
    return np.clip(np.asarray(v, dtype=np.float64), -float(clip_value), float(clip_value))


def _distance_utility(z: np.ndarray, z_pref: np.ndarray, feature_weight: np.ndarray) -> float:
    diff = np.asarray(z, dtype=np.float64) - np.asarray(z_pref, dtype=np.float64)
    return -float(np.sum(feature_weight * diff * diff))


def predict_proba_for_z_pref(
    z_pref: np.ndarray,
    z_a: np.ndarray,
    z_b: np.ndarray,
    feature_weight: np.ndarray,
    temperature: float,
) -> float:
    """P(A>B) under a negative-distance Preference Model with a fixed z_pref."""
    u_a = _distance_utility(z_a, z_pref, feature_weight)
    u_b = _distance_utility(z_b, z_pref, feature_weight)
    return float(sigmoid((u_a - u_b) / float(temperature)))


def log_loss_for_z_pref(
    z_pref: np.ndarray,
    records: list,
    feature_weight: np.ndarray,
    temperature: float,
    eps: float = 1e-8,
) -> float:
    """Mean binary log loss on a list of training records for a fixed z_pref."""
    if len(records) == 0:
        return float("nan")
    losses: list[float] = []
    for record in records:
        p = predict_proba_for_z_pref(
            z_pref=z_pref,
            z_a=record.z_a,
            z_b=record.z_b,
            feature_weight=feature_weight,
            temperature=temperature,
        )
        p = float(np.clip(p, eps, 1.0 - eps))
        y = 1.0 if record.choice == "A" else 0.0
        losses.append(float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    return float(np.mean(losses))


def calibrate_by_heuristic_norm(
    model_vector: np.ndarray,
    heuristic_vector: np.ndarray,
    clip_value: float | None = 2.0,
) -> np.ndarray:
    """Use the model direction but set its norm to the heuristic state norm."""
    direction = _normalize(model_vector)
    scale = float(np.linalg.norm(heuristic_vector))
    return _clip_vector(scale * direction, clip_value)


def calibrate_by_train_log_loss(
    model_vector: np.ndarray,
    records: list,
    feature_weight: np.ndarray,
    temperature: float,
    heuristic_vector: np.ndarray | None = None,
    clip_value: float | None = 2.0,
    n_grid: int = 81,
) -> tuple[np.ndarray, float, float]:
    """
    Use the model direction and choose only the scalar length that minimizes
    train log loss on the observed A/B choices.

    Returns: calibrated_vector, best_scale, best_train_log_loss.
    """
    direction = _normalize(model_vector)
    raw_norm = float(np.linalg.norm(model_vector))
    heuristic_norm = float(np.linalg.norm(heuristic_vector)) if heuristic_vector is not None else raw_norm

    dim = int(len(model_vector))
    if clip_value is None:
        max_scale = max(2.0, 2.0 * raw_norm, 2.0 * heuristic_norm)
    else:
        # Maximum Euclidean norm after per-coordinate clipping.
        max_scale = float(np.sqrt(dim) * float(clip_value))
        # Keep the grid focused around plausible session scales.
        max_scale = min(max_scale, max(1.0, 2.5 * raw_norm, 2.5 * heuristic_norm))

    scales = np.linspace(0.0, max_scale, int(n_grid))
    best_vec = np.zeros_like(model_vector, dtype=np.float64)
    best_scale = 0.0
    best_loss = float("inf")

    for scale in scales:
        candidate = _clip_vector(float(scale) * direction, clip_value)
        loss = log_loss_for_z_pref(candidate, records, feature_weight, temperature)
        if loss < best_loss:
            best_loss = float(loss)
            best_scale = float(scale)
            best_vec = candidate

    return best_vec, best_scale, best_loss


def build_calibrated_vectors(
    result: PreferenceModelSessionResult,
    clip_value: float | None = 2.0,
) -> dict[str, np.ndarray]:
    """Build alternative final preference vectors from the same V4a session."""
    heuristic = result.final_heuristic_state.z_mean.copy()
    raw = result.final_model.z_pref.copy()
    feature_weight = result.final_model.feature_weight.copy()
    temperature = float(result.final_model.temperature)

    norm_calibrated = calibrate_by_heuristic_norm(raw, heuristic, clip_value=clip_value)
    train_scaled, _, _ = calibrate_by_train_log_loss(
        model_vector=raw,
        records=result.records,
        feature_weight=feature_weight,
        temperature=temperature,
        heuristic_vector=heuristic,
        clip_value=clip_value,
    )

    blend_70 = _clip_vector(0.70 * heuristic + 0.30 * train_scaled, clip_value)
    blend_50 = _clip_vector(0.50 * heuristic + 0.50 * train_scaled, clip_value)

    return {
        "heuristic_update": heuristic,
        "raw_preference_model": raw,
        "norm_calibrated_model": norm_calibrated,
        "train_scale_model": train_scaled,
        "blend_70h_30m": blend_70,
        "blend_50h_50m": blend_50,
    }


def make_model_from_vector(
    vector: np.ndarray,
    template_model: LogisticDistancePreferenceModel,
) -> LogisticDistancePreferenceModel:
    """Create a Preference Model object with a fixed z_pref vector."""
    return LogisticDistancePreferenceModel(
        dim=len(vector),
        lr=float(template_model.lr),
        temperature=float(template_model.temperature),
        l2=float(template_model.l2),
        clip_value=template_model.clip_value,
        feature_weight=template_model.feature_weight.copy(),
        z_pref=np.asarray(vector, dtype=np.float64).copy(),
    )


def generate_fixed_heldout_pairs(
    synthetic_user: SyntheticUser,
    center: np.ndarray,
    n_pairs: int = 100,
    step_scale: float = 0.6,
    clip_value: float | None = 2.0,
    seed: int | None = None,
) -> list[dict]:
    """
    Generate held-out pairs once, including z vectors, so all calibrated
    preference vectors are evaluated on exactly the same unseen questions.
    """
    rng = np.random.default_rng(seed)
    pairs: list[dict] = []
    for pair_id in range(int(n_pairs)):
        z_a, z_b, source = generate_heldout_pair(
            center=center,
            rng=rng,
            step_scale=step_scale,
            clip_value=clip_value,
        )
        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)
        pairs.append({
            "pair_id": int(pair_id),
            "source": str(source).split(":", 1)[0],
            "source_detail": str(source),
            "z_a": z_a.copy(),
            "z_b": z_b.copy(),
            "choice": str(choice),
            "u_a": float(u_a),
            "u_b": float(u_b),
            "utility_margin": float(abs(u_a - u_b)),
        })
    return pairs


def evaluate_vector_on_heldout_pairs(
    vector: np.ndarray,
    pairs: list[dict],
    template_model: LogisticDistancePreferenceModel,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate one fixed z_pref vector on a fixed held-out pair list."""
    feature_weight = template_model.feature_weight.copy()
    temperature = float(template_model.temperature)
    rows: list[dict] = []
    for pair in pairs:
        p_a = predict_proba_for_z_pref(
            z_pref=vector,
            z_a=pair["z_a"],
            z_b=pair["z_b"],
            feature_weight=feature_weight,
            temperature=temperature,
        )
        predicted_choice = "A" if p_a >= 0.5 else "B"
        correct = bool(predicted_choice == pair["choice"])
        p = float(np.clip(p_a, 1e-8, 1.0 - 1e-8))
        y = 1.0 if pair["choice"] == "A" else 0.0
        loss = float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
        rows.append({
            "pair_id": int(pair["pair_id"]),
            "source": pair["source"],
            "source_detail": pair["source_detail"],
            "choice": pair["choice"],
            "predicted_choice": predicted_choice,
            "correct": correct,
            "p_a": float(p_a),
            "log_loss": loss,
            "utility_margin": float(pair["utility_margin"]),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        summary = {
            "heldout_accuracy": float("nan"),
            "heldout_log_loss": float("nan"),
            "heldout_mean_confidence": float("nan"),
            "heldout_n_pairs": 0,
        }
    else:
        confidence = np.abs(df["p_a"].to_numpy(dtype=np.float64) - 0.5) * 2.0
        summary = {
            "heldout_accuracy": float(df["correct"].mean()),
            "heldout_log_loss": float(df["log_loss"].mean()),
            "heldout_mean_confidence": float(np.mean(confidence)),
            "heldout_n_pairs": int(len(df)),
        }
    return summary, df


def run_preference_model_calibration_session_v4a2(
    synthetic_user: SyntheticUser,
    n_steps: int = 25,
    n_heldout_pairs: int = 100,
    pair_strategy: str = "semantic_active_v21",
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
) -> tuple[PreferenceModelSessionResult, pd.DataFrame, pd.DataFrame]:
    """
    Train one online Preference Model, then evaluate several final-vector
    calibration methods on the same train history and held-out pairs.
    """
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
        pair_strategy=pair_strategy,  # type: ignore[arg-type]
        seed=train_seed,
        model_feature_weight=model_feature_weight,
    )

    vectors = build_calibrated_vectors(result, clip_value=clip_value)
    heldout_pairs = generate_fixed_heldout_pairs(
        synthetic_user=synthetic_user,
        center=result.final_heuristic_state.z_mean.copy(),
        n_pairs=n_heldout_pairs,
        step_scale=step_scale,
        clip_value=clip_value,
        seed=heldout_seed,
    )

    summary_rows: list[dict] = []
    pair_rows: list[dict] = []
    feature_weight = result.final_model.feature_weight.copy()
    temperature = float(result.final_model.temperature)

    # Pre-compute train-scale diagnostics.
    _, train_best_scale, train_best_loss = calibrate_by_train_log_loss(
        result.final_model.z_pref,
        result.records,
        feature_weight,
        temperature,
        heuristic_vector=result.final_heuristic_state.z_mean,
        clip_value=clip_value,
    )

    for method, vector in vectors.items():
        heldout_summary, heldout_df = evaluate_vector_on_heldout_pairs(
            vector=vector,
            pairs=heldout_pairs,
            template_model=result.final_model,
        )
        train_loss = log_loss_for_z_pref(vector, result.records, feature_weight, temperature)
        summary_rows.append({
            "method": method,
            "method_display": CALIBRATION_DISPLAY_NAMES.get(method, method),
            "final_distance": float(distance_to_target(vector, synthetic_user.z_target)),
            "cosine_to_target": float(cosine_similarity(vector, synthetic_user.z_target)),
            "vector_norm": float(np.linalg.norm(vector)),
            "train_log_loss": float(train_loss),
            "train_best_scale": float(train_best_scale) if method == "train_scale_model" else float("nan"),
            "train_best_log_loss": float(train_best_loss) if method == "train_scale_model" else float("nan"),
            **heldout_summary,
        })
        heldout_df = heldout_df.copy()
        heldout_df.insert(0, "method", method)
        heldout_df.insert(1, "method_display", CALIBRATION_DISPLAY_NAMES.get(method, method))
        pair_rows.extend(heldout_df.to_dict(orient="records"))

    return result, pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def run_preference_model_calibration_batch_v4a2(
    dataset: pd.DataFrame,
    n_steps: int = 25,
    n_heldout_pairs: int = 100,
    pair_strategy: str = "semantic_active_v21",
    step_scale: float = 0.6,
    heuristic_lr: float = 0.25,
    model_lr: float = 0.06,
    model_temperature: float = 0.75,
    model_l2: float = 0.003,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    train_seed_base: int = 70_000,
    heldout_seed_base: int = 90_000,
    user_seed_base: int = 10_000,
    model_feature_weight: Literal["uniform", "oracle"] = "uniform",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run V4a.2 scale calibration over a fixed synthetic user dataset."""
    session_rows: list[dict] = []
    heldout_rows: list[dict] = []
    train_rows: list[dict] = []

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        user = row_to_synthetic_user(row, seed=user_seed_base + user_id)

        result, method_summary, heldout_df = run_preference_model_calibration_session_v4a2(
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

        for method_row in method_summary.to_dict(orient="records"):
            method_row.update({
                "user_id": user_id,
                "target_mode": target_mode,
                "initial_distance": float(np.linalg.norm(z_target)),
                "n_steps": int(n_steps),
            })
            # Add user metadata when available.
            for col in ["intensity_label", "main_archetype", "secondary_archetype"]:
                if col in row.index:
                    method_row[col] = row.get(col)
            session_rows.append(method_row)

        heldout_df = heldout_df.copy()
        heldout_df.insert(0, "user_id", user_id)
        heldout_df.insert(1, "target_mode", target_mode)
        heldout_rows.extend(heldout_df.to_dict(orient="records"))

        for record in result.records:
            train_rows.append({
                "user_id": user_id,
                "target_mode": target_mode,
                "step": int(record.step),
                "choice": record.choice,
                "p_a_before": float(record.p_a_before),
                "p_a_after": float(record.p_a_after),
                "correct_before": bool(record.correct_before),
                "log_loss_before": float(record.log_loss_before),
                "log_loss_after": float(record.log_loss_after),
                "heuristic_distance_to_target": float(record.heuristic_distance_to_target),
                "model_distance_to_target": float(record.model_distance_to_target),
                "model_cosine_to_target": float(record.model_cosine_to_target),
                "pair_sub_strategy": record.pair_meta.get("sub_strategy"),
                "control_name": record.pair_meta.get("control_name"),
            })

    return pd.DataFrame(session_rows), pd.DataFrame(heldout_rows), pd.DataFrame(train_rows)


def summarize_calibration_by_target_mode(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate V4a.2 calibration methods by target mode and method."""
    return (
        sessions
        .groupby(["target_mode", "method", "method_display"])
        .agg(
            users=("user_id", "nunique"),
            mean_initial_distance=("initial_distance", "mean"),
            mean_final_distance=("final_distance", "mean"),
            std_final_distance=("final_distance", "std"),
            mean_cosine_to_target=("cosine_to_target", "mean"),
            mean_vector_norm=("vector_norm", "mean"),
            mean_train_log_loss=("train_log_loss", "mean"),
            mean_heldout_accuracy=("heldout_accuracy", "mean"),
            mean_heldout_log_loss=("heldout_log_loss", "mean"),
            mean_heldout_confidence=("heldout_mean_confidence", "mean"),
        )
        .reset_index()
    )


def summarize_calibration_heldout_by_source(heldout_pairs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate held-out accuracy/log loss by target mode, method and source."""
    return (
        heldout_pairs
        .groupby(["target_mode", "method", "method_display", "source"])
        .agg(
            pairs=("pair_id", "count"),
            accuracy=("correct", "mean"),
            log_loss=("log_loss", "mean"),
            mean_confidence=("p_a", lambda x: float(np.mean(np.abs(np.asarray(x, dtype=np.float64) - 0.5) * 2.0))),
            mean_margin=("utility_margin", "mean"),
        )
        .reset_index()
    )


def save_v4a2_outputs(
    sessions: pd.DataFrame,
    heldout_pairs: pd.DataFrame,
    train_steps: pd.DataFrame,
    summary: pd.DataFrame,
    source_summary: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "notebook_v4a2_preference_model_calibration",
) -> None:
    """Save V4a.2 result tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    heldout_pairs.to_csv(output_dir / f"{prefix}_heldout_pairs.csv", index=False)
    train_steps.to_csv(output_dir / f"{prefix}_train_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    source_summary.to_csv(output_dir / f"{prefix}_heldout_by_source.csv", index=False)
