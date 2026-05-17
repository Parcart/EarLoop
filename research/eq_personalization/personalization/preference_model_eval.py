from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .loop import PairStrategy, generate_pair_by_strategy
from .metrics import distance_to_target
from .pair_generator import PairGenerator
from .preference_model import LogisticDistancePreferenceModel, cosine_similarity
from .preference_update import update_state_from_choice
from .state import PreferenceState, init_preference_state
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser


@dataclass
class PreferenceModelStepRecord:
    step: int
    choice: str
    z_a: np.ndarray
    z_b: np.ndarray
    u_a: float
    u_b: float
    p_a_before: float
    p_a_after: float
    pred_before: str
    correct_before: bool
    log_loss_before: float
    log_loss_after: float
    heuristic_z_mean_after: np.ndarray
    model_z_pref_after: np.ndarray
    heuristic_distance_to_target: float
    model_distance_to_target: float
    model_cosine_to_target: float
    pair_strategy: str
    pair_meta: dict


@dataclass
class PreferenceModelSessionResult:
    final_heuristic_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[PreferenceModelStepRecord]
    heuristic_distances: np.ndarray
    model_distances: np.ndarray
    model_cosines: np.ndarray
    prediction_accuracy: float
    mean_log_loss_before: float
    mean_log_loss_after: float


def run_preference_model_learning_session_v4a(
    synthetic_user: SyntheticUser,
    n_steps: int = 25,
    step_scale: float = 0.6,
    heuristic_lr: float = 0.25,
    model_lr: float = 0.06,
    model_temperature: float = 0.75,
    model_l2: float = 0.003,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_strategy: PairStrategy = "semantic_active_v21",
    seed: int | None = None,
    model_feature_weight: Literal["uniform", "oracle"] = "uniform",
) -> PreferenceModelSessionResult:
    """
    V4a experiment: learn a lightweight Preference Model from A/B answers.

    Pair generation is fixed to a chosen Pair Generator strategy. The Preference
    Model observes the same A/B choices and learns an internal z_pref online.

    This experiment answers: can a model learn useful preference information
    from the A/B history before it is allowed to control pair generation?
    """
    rng = np.random.default_rng(seed)
    state = init_preference_state(dim=len(synthetic_user.z_target), init_std=init_std)
    pair_generator = PairGenerator(step_scale=step_scale, clip_value=clip_value, rng=rng)

    feature_weight = None
    if model_feature_weight == "oracle":
        # Diagnostic upper-bound mode: the model knows feature sensitivity but not z_target.
        feature_weight = synthetic_user.feature_importance.copy()

    model = LogisticDistancePreferenceModel(
        dim=len(synthetic_user.z_target),
        lr=model_lr,
        temperature=model_temperature,
        l2=model_l2,
        clip_value=clip_value,
        feature_weight=feature_weight,
    )

    records: list[PreferenceModelStepRecord] = []
    heuristic_distances: list[float] = []
    model_distances: list[float] = []
    model_cosines: list[float] = []
    correct: list[bool] = []
    losses_before: list[float] = []
    losses_after: list[float] = []

    for step in range(1, int(n_steps) + 1):
        z_a, z_b, _direction, pair_meta = generate_pair_by_strategy(
            pair_generator,
            state,
            pair_strategy,
        )

        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)
        update_info = model.update(z_a, z_b, choice)

        state = update_state_from_choice(
            state=state,
            z_a=z_a,
            z_b=z_b,
            choice=choice,
            lr=heuristic_lr,
            std_decay=std_decay,
            min_std=min_std,
            clip_value=clip_value,
            pair_meta=pair_meta,
        )

        heuristic_dist = distance_to_target(state.z_mean, synthetic_user.z_target)
        model_dist = distance_to_target(model.z_pref, synthetic_user.z_target)
        model_cos = cosine_similarity(model.z_pref, synthetic_user.z_target)

        heuristic_distances.append(heuristic_dist)
        model_distances.append(model_dist)
        model_cosines.append(model_cos)
        correct.append(bool(update_info["correct_before"]))
        losses_before.append(float(update_info["loss_before"]))
        losses_after.append(float(update_info["loss_after"]))

        records.append(PreferenceModelStepRecord(
            step=step,
            choice=choice,
            z_a=z_a.copy(),
            z_b=z_b.copy(),
            u_a=float(u_a),
            u_b=float(u_b),
            p_a_before=float(update_info["p_before"]),
            p_a_after=float(update_info["p_after"]),
            pred_before=str(update_info["pred_before"]),
            correct_before=bool(update_info["correct_before"]),
            log_loss_before=float(update_info["loss_before"]),
            log_loss_after=float(update_info["loss_after"]),
            heuristic_z_mean_after=state.z_mean.copy(),
            model_z_pref_after=model.z_pref.copy(),
            heuristic_distance_to_target=float(heuristic_dist),
            model_distance_to_target=float(model_dist),
            model_cosine_to_target=float(model_cos),
            pair_strategy=pair_strategy,
            pair_meta=pair_meta,
        ))

    return PreferenceModelSessionResult(
        final_heuristic_state=state,
        final_model=model,
        records=records,
        heuristic_distances=np.asarray(heuristic_distances, dtype=np.float64),
        model_distances=np.asarray(model_distances, dtype=np.float64),
        model_cosines=np.asarray(model_cosines, dtype=np.float64),
        prediction_accuracy=float(np.mean(correct)) if correct else float("nan"),
        mean_log_loss_before=float(np.mean(losses_before)) if losses_before else float("nan"),
        mean_log_loss_after=float(np.mean(losses_after)) if losses_after else float("nan"),
    )


def distances_with_initial(z_target: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Prepend true distance at step 0 to a distance curve."""
    initial = float(np.linalg.norm(np.zeros_like(z_target) - z_target))
    return np.concatenate([[initial], np.asarray(distances, dtype=np.float64)])


def make_v4a_summary_row(
    user_id: int,
    target_mode: str,
    result: PreferenceModelSessionResult,
    z_target: np.ndarray,
) -> dict:
    heuristic_d = distances_with_initial(z_target, result.heuristic_distances)
    model_d = distances_with_initial(z_target, result.model_distances)

    return {
        "user_id": int(user_id),
        "target_mode": str(target_mode),
        "n_steps": int(len(result.records)),
        "initial_distance": float(heuristic_d[0]),
        "heuristic_final_distance": float(heuristic_d[-1]),
        "model_final_distance": float(model_d[-1]),
        "heuristic_best_distance": float(np.min(heuristic_d)),
        "model_best_distance": float(np.min(model_d)),
        "heuristic_mean_distance": float(np.mean(heuristic_d)),
        "model_mean_distance": float(np.mean(model_d)),
        "heuristic_improvement_pct": float(100.0 * (heuristic_d[0] - heuristic_d[-1]) / (heuristic_d[0] + 1e-8)),
        "model_improvement_pct": float(100.0 * (model_d[0] - model_d[-1]) / (model_d[0] + 1e-8)),
        "prediction_accuracy": float(result.prediction_accuracy),
        "mean_log_loss_before": float(result.mean_log_loss_before),
        "mean_log_loss_after": float(result.mean_log_loss_after),
        "final_model_cosine_to_target": float(result.model_cosines[-1]) if len(result.model_cosines) else float("nan"),
    }


def run_preference_model_batch_v4a(
    dataset: pd.DataFrame,
    n_steps: int = 25,
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
    pair_seed_base: int = 30_000,
    user_seed_base: int = 10_000,
    model_feature_weight: Literal["uniform", "oracle"] = "uniform",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """
    Run V4a learning experiment on a fixed synthetic user dataset.

    Returns:
        sessions: one row per user;
        steps: one row per user/step;
        curves[target_mode]["heuristic"|"model"|"cosine"] arrays.
    """
    session_rows: list[dict] = []
    step_rows: list[dict] = []
    curve_store: dict[str, dict[str, list[np.ndarray]]] = {}

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        user = row_to_synthetic_user(row, seed=user_seed_base + user_id)

        result = run_preference_model_learning_session_v4a(
            synthetic_user=user,
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
            seed=pair_seed_base + user_id,
            model_feature_weight=model_feature_weight,
        )

        session_rows.append(make_v4a_summary_row(user_id, target_mode, result, z_target))

        curve_store.setdefault(target_mode, {"heuristic": [], "model": [], "cosine": []})
        curve_store[target_mode]["heuristic"].append(distances_with_initial(z_target, result.heuristic_distances))
        curve_store[target_mode]["model"].append(distances_with_initial(z_target, result.model_distances))
        curve_store[target_mode]["cosine"].append(np.concatenate([[0.0], result.model_cosines]))

        for record in result.records:
            step_rows.append({
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

    sessions = pd.DataFrame(session_rows)
    steps = pd.DataFrame(step_rows)
    curves: dict[str, dict[str, np.ndarray]] = {}
    for target_mode, by_name in curve_store.items():
        curves[target_mode] = {name: np.asarray(items, dtype=np.float64) for name, items in by_name.items()}

    return sessions, steps, curves


def summarize_v4a_by_target_mode(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate V4a sessions by target_mode."""
    return (
        sessions
        .groupby("target_mode")
        .agg(
            users=("user_id", "count"),
            mean_initial_distance=("initial_distance", "mean"),
            heuristic_mean_final_distance=("heuristic_final_distance", "mean"),
            model_mean_final_distance=("model_final_distance", "mean"),
            heuristic_mean_best_distance=("heuristic_best_distance", "mean"),
            model_mean_best_distance=("model_best_distance", "mean"),
            heuristic_mean_improvement_pct=("heuristic_improvement_pct", "mean"),
            model_mean_improvement_pct=("model_improvement_pct", "mean"),
            mean_prediction_accuracy=("prediction_accuracy", "mean"),
            mean_log_loss_before=("mean_log_loss_before", "mean"),
            mean_log_loss_after=("mean_log_loss_after", "mean"),
            mean_final_model_cosine_to_target=("final_model_cosine_to_target", "mean"),
        )
        .reset_index()
    )


def save_v4a_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "v4a_preference_model",
) -> None:
    """Save V4a result tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    steps.to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
