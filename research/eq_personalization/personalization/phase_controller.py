from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .control_basis import CONTROL_BASIS_6D_TO_8D, CONTROL_NAMES_6D
from .loop import generate_pair_by_strategy
from .metrics import distance_to_target
from .pair_generator import PairGenerator, normalize
from .preference_model import LogisticDistancePreferenceModel, cosine_similarity
from .preference_model_calibration import calibrate_by_heuristic_norm, build_calibrated_vectors
from .preference_update import update_state_from_choice
from .state import FEATURE_NAMES_8D, PreferenceState, clip_vector, init_preference_state
from .synthetic_dataset import row_to_synthetic_user, row_to_target
from .synthetic_user import SyntheticUser


PhaseName = Literal[
    "exploration",
    "direction_locked",
    "zone_refinement",
    "ready_to_finalize",
    "saturated",
]


PHASE_DISPLAY_NAMES = {
    "exploration": "Exploration",
    "direction_locked": "Direction locked",
    "zone_refinement": "Zone refinement",
    "ready_to_finalize": "Ready to finalize",
    "saturated": "Saturated / constrained",
}

FINAL_VECTOR_DISPLAY_NAMES = {
    "heuristic_update": "Heuristic update",
    "raw_preference_model": "Raw PM",
    "norm_calibrated_model": "Norm-calibrated PM",
    "train_scale_model": "Train-scale PM",
    "blend_70h_30m": "Selected: Blend 70/30",
    "blend_50h_50m": "Blend 50/50",
}

ZONE_REFINEMENT_DIRECTIONS = {
    "sub_bass_depth": np.array([1.00, 0.25, 0.00, 0.00, -0.05, 0.00, 0.00, -0.05], dtype=np.float64),
    "bass_punch": np.array([0.20, 1.00, -0.15, -0.05, -0.05, 0.05, 0.00, 0.00], dtype=np.float64),
    "clean_bass_detail": np.array([0.65, 0.80, -0.25, -0.20, -0.05, 0.15, 0.05, -0.05], dtype=np.float64),
    "warmth_body_local": np.array([0.00, 0.10, 0.65, 0.80, -0.15, -0.10, 0.00, -0.15], dtype=np.float64),
    "vocal_presence": np.array([0.00, 0.00, -0.10, 0.00, 0.85, 0.55, 0.05, 0.00], dtype=np.float64),
    "clarity_without_harshness": np.array([-0.05, -0.05, -0.10, -0.10, 0.20, 0.85, 0.30, -0.20], dtype=np.float64),
    "air_detail": np.array([-0.10, -0.15, 0.00, -0.10, 0.05, 0.25, 0.90, 0.45], dtype=np.float64),
    "brightness_smoothing": np.array([0.05, 0.05, 0.00, 0.10, -0.20, -0.20, 0.10, -0.90], dtype=np.float64),
}

ZONE_GROUPS = {
    "low_end": ["sub_bass_depth", "bass_punch", "clean_bass_detail"],
    "body_mid": ["warmth_body_local", "vocal_presence"],
    "detail_high": ["clarity_without_harshness", "air_detail", "brightness_smoothing"],
}


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return 0.0
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def _pair_from_direction(
    state: PreferenceState,
    direction: np.ndarray,
    scale: float,
    clip_value: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    direction = normalize(direction)
    if rng.random() < 0.5:
        direction = -direction

    raw_a = state.z_mean + float(scale) * direction
    raw_b = state.z_mean - float(scale) * direction
    z_a = clip_vector(raw_a, clip_value)
    z_b = clip_vector(raw_b, clip_value)

    raw_step = np.linalg.norm(raw_a - state.z_mean) + np.linalg.norm(raw_b - state.z_mean) + 1e-8
    safe_step = np.linalg.norm(z_a - state.z_mean) + np.linalg.norm(z_b - state.z_mean)
    clip_ratio = float(np.clip(1.0 - safe_step / raw_step, 0.0, 1.0))
    return z_a, z_b, direction, clip_ratio


@dataclass
class PhaseControllerConfig:
    min_lock_step: int = 16
    lock_confidence_threshold: float = 0.90
    ready_confidence_threshold: float = 0.88
    ready_update_norm_threshold: float = 0.32
    ready_mean_std_threshold: float = 0.25
    ready_patience: int = 3
    saturation_clip_ratio_threshold: float = 0.35
    saturation_patience: int = 3
    zone_step_scale: float = 0.42
    zone_scale_multipliers: tuple[float, ...] = (0.55, 0.75, 1.0)
    distance_threshold: float = 0.40
    synthetic_convergence_patience: int = 2


@dataclass
class PhaseSessionResult:
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[dict]
    distances: np.ndarray
    phases: list[str]
    direction_lock_step: int | None
    ready_step: int | None
    synthetic_threshold_step: int | None
    saturation_step: int | None
    final_status: str


def make_zone_refinement_pair(
    state: PreferenceState,
    locked_direction: np.ndarray,
    rng: np.random.Generator,
    clip_value: float | None = 2.0,
    base_step_scale: float = 0.42,
    scale_multipliers: tuple[float, ...] = (0.55, 0.75, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Generate a local zone-refinement question after the global direction is locked.

    The selector still does not predict preference. It chooses a zone question that:
    - aligns with the locked preference direction;
    - covers remaining uncertainty;
    - does not repeat the same local question too often;
    - stays inside compact-vector bounds.
    """
    locked_direction = normalize(locked_direction)
    candidates: list[dict] = []

    for zone_name, direction_names in ZONE_GROUPS.items():
        for direction_name in direction_names:
            direction = normalize(ZONE_REFINEMENT_DIRECTIONS[direction_name])
            alignment = abs(float(np.dot(np.abs(direction), np.abs(locked_direction))))
            uncertainty = float(np.sum(np.abs(direction) * state.z_std))

            recent = state.history[-3:]
            repeat_count = 0
            for item in recent:
                meta = item.get("pair_meta") or {}
                if meta.get("zone_direction") == direction_name:
                    repeat_count += 1
            repetition_penalty = float(repeat_count / max(1, len(recent))) if recent else 0.0

            for multiplier in scale_multipliers:
                scale = float(base_step_scale) * float(multiplier)
                raw_a = state.z_mean + scale * direction
                raw_b = state.z_mean - scale * direction
                overflow_a = np.maximum(np.abs(raw_a) - float(clip_value), 0.0) if clip_value is not None else 0.0
                overflow_b = np.maximum(np.abs(raw_b) - float(clip_value), 0.0) if clip_value is not None else 0.0
                safety_penalty = float(np.mean(overflow_a * overflow_a + overflow_b * overflow_b)) if clip_value is not None else 0.0
                diversity = float(np.linalg.norm(raw_a - raw_b))

                score = (
                    0.65 * alignment
                    + 0.65 * uncertainty
                    + 0.15 * diversity
                    - 1.00 * safety_penalty
                    - 0.30 * repetition_penalty
                )
                candidates.append({
                    "zone_name": zone_name,
                    "zone_direction": direction_name,
                    "direction": direction,
                    "scale": scale,
                    "scale_multiplier": float(multiplier),
                    "score": float(score),
                    "alignment_score": float(alignment),
                    "uncertainty_score": float(uncertainty),
                    "diversity_score": float(diversity),
                    "safety_penalty": float(safety_penalty),
                    "repetition_penalty": float(repetition_penalty),
                })

    best = max(candidates, key=lambda item: item["score"])
    z_a, z_b, signed_direction, clip_ratio = _pair_from_direction(
        state=state,
        direction=best["direction"],
        scale=best["scale"],
        clip_value=clip_value,
        rng=rng,
    )

    meta = {
        "strategy": "phase_aware_controller",
        "sub_strategy": "zone_refinement",
        "zone_name": best["zone_name"],
        "zone_direction": best["zone_direction"],
        "control_name": best["zone_direction"],
        "control_direction": signed_direction.copy(),
        "scale": best["scale"],
        "scale_multiplier": best["scale_multiplier"],
        "score": best["score"],
        "alignment_score": best["alignment_score"],
        "uncertainty_score": best["uncertainty_score"],
        "diversity_score": best["diversity_score"],
        "safety_penalty": best["safety_penalty"],
        "repetition_penalty": best["repetition_penalty"],
        "clip_ratio": clip_ratio,
    }
    return z_a, z_b, signed_direction, meta


def _recent_update_norm(z_history: list[np.ndarray], lookback: int = 3) -> float:
    if len(z_history) <= int(lookback):
        return float("inf")
    return float(np.linalg.norm(z_history[-1] - z_history[-1 - int(lookback)]))


def _first_stable_threshold_step(values: list[float], threshold: float, patience: int) -> int | None:
    if len(values) < int(patience):
        return None
    for idx in range(0, len(values) - int(patience) + 1):
        window = values[idx: idx + int(patience)]
        if all(v <= threshold for v in window):
            return int(idx + 1)  # steps are 1-indexed
    return None


def run_phase_aware_session(
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
    seed: int | None = None,
    config: PhaseControllerConfig | None = None,
    stop_on_ready: bool = False,
) -> PhaseSessionResult:
    """
    Phase-aware personalization loop.

    Exploration uses Semantic active v3. After the heuristic state and parallel
    Preference Model agree on a direction, the loop switches to local zone
    refinement questions. The monitor can also mark ready_to_finalize or
    saturated states. This module is intended to estimate convergence speed and
    session status, not to replace the final architecture yet.
    """
    cfg = PhaseControllerConfig() if config is None else config
    rng = np.random.default_rng(seed)
    state = init_preference_state(dim=len(synthetic_user.z_target), init_std=init_std)
    pair_generator = PairGenerator(step_scale=step_scale, clip_value=clip_value, rng=rng)
    model = LogisticDistancePreferenceModel(
        dim=len(synthetic_user.z_target),
        lr=model_lr,
        temperature=model_temperature,
        l2=model_l2,
        clip_value=clip_value,
    )

    records: list[dict] = []
    distances: list[float] = []
    phases: list[str] = []
    z_history: list[np.ndarray] = [state.z_mean.copy()]
    confidence_history: list[float] = []
    clip_ratio_history: list[float] = []

    direction_lock_step: int | None = None
    ready_step: int | None = None
    saturation_step: int | None = None
    phase: PhaseName = "exploration"
    locked_direction: np.ndarray | None = None

    for step in range(1, int(n_steps) + 1):
        model_calibrated = calibrate_by_heuristic_norm(
            model_vector=model.z_pref,
            heuristic_vector=state.z_mean,
            clip_value=clip_value,
        )
        direction_confidence = _safe_cosine(state.z_mean, model_calibrated)
        confidence_history.append(direction_confidence)
        recent_update_norm = _recent_update_norm(z_history, lookback=3)
        mean_std = float(np.mean(state.z_std))

        if direction_lock_step is None:
            has_model_signal = float(np.linalg.norm(model.z_pref)) > 0.10
            if (
                step >= int(cfg.min_lock_step)
                and has_model_signal
                and direction_confidence >= float(cfg.lock_confidence_threshold)
            ):
                direction_lock_step = int(step)
                locked_direction = normalize(0.70 * state.z_mean + 0.30 * model_calibrated)
                phase = "direction_locked"

        if phase in {"direction_locked", "zone_refinement"}:
            # One step may be marked as direction_locked; afterwards use zone refinement.
            if step > int(direction_lock_step or step):
                phase = "zone_refinement"

        if phase == "zone_refinement" and locked_direction is not None:
            z_a, z_b, _direction, pair_meta = make_zone_refinement_pair(
                state=state,
                locked_direction=locked_direction,
                rng=rng,
                clip_value=clip_value,
                base_step_scale=cfg.zone_step_scale,
                scale_multipliers=cfg.zone_scale_multipliers,
            )
        else:
            z_a, z_b, _direction, pair_meta = generate_pair_by_strategy(
                pair_generator,
                state,
                "semantic_active_v21",
            )
            pair_meta = dict(pair_meta)
            pair_meta["sub_strategy"] = "semantic_active_v21"
            pair_meta.setdefault("clip_ratio", 0.0)

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

        z_history.append(state.z_mean.copy())
        dist = distance_to_target(state.z_mean, synthetic_user.z_target)
        distances.append(float(dist))
        phases.append(str(phase))

        clip_ratio = float(pair_meta.get("clip_ratio", 0.0) or 0.0)
        if clip_ratio == 0.0 and pair_meta.get("safety_penalty", 0.0):
            clip_ratio = min(1.0, float(pair_meta.get("safety_penalty", 0.0)) * 4.0)
        clip_ratio_history.append(clip_ratio)

        if saturation_step is None and len(clip_ratio_history) >= cfg.saturation_patience:
            recent_clip = clip_ratio_history[-int(cfg.saturation_patience):]
            if np.mean(recent_clip) >= float(cfg.saturation_clip_ratio_threshold):
                saturation_step = int(step - cfg.saturation_patience + 1)

        model_calibrated_after = calibrate_by_heuristic_norm(
            model_vector=model.z_pref,
            heuristic_vector=state.z_mean,
            clip_value=clip_value,
        )
        direction_confidence_after = _safe_cosine(state.z_mean, model_calibrated_after)
        recent_update_norm_after = _recent_update_norm(z_history, lookback=3)
        mean_std_after = float(np.mean(state.z_std))

        ready_now = (
            direction_lock_step is not None
            and direction_confidence_after >= float(cfg.ready_confidence_threshold)
            and recent_update_norm_after <= float(cfg.ready_update_norm_threshold)
            and mean_std_after <= float(cfg.ready_mean_std_threshold)
            and saturation_step is None
        )

        if ready_now and ready_step is None:
            # Require condition to stay true for a short window via records below.
            ready_step = int(step)

        records.append({
            "step": int(step),
            "phase": str(phase),
            "choice": str(choice),
            "u_a": float(u_a),
            "u_b": float(u_b),
            "distance_to_target": float(dist),
            "z_mean_after": state.z_mean.copy(),
            "z_std_mean_after": mean_std_after,
            "direction_confidence": float(direction_confidence_after),
            "recent_update_norm": float(recent_update_norm_after),
            "clip_ratio": float(clip_ratio),
            "p_a_before": float(update_info["p_before"]),
            "correct_before": bool(update_info["correct_before"]),
            "pair_meta": pair_meta,
            "pair_source": pair_meta.get("sub_strategy", pair_meta.get("strategy", "unknown")),
            "control_name": pair_meta.get("control_name"),
            "zone_direction": pair_meta.get("zone_direction"),
        })

        if stop_on_ready and ready_step is not None:
            phase = "ready_to_finalize"
            break

    synthetic_threshold_step = _first_stable_threshold_step(
        values=distances,
        threshold=float(cfg.distance_threshold),
        patience=int(cfg.synthetic_convergence_patience),
    )

    if saturation_step is not None:
        final_status = "saturated"
    elif ready_step is not None:
        final_status = "ready_to_finalize"
    elif direction_lock_step is not None:
        final_status = "direction_locked"
    else:
        final_status = "exploration"

    return PhaseSessionResult(
        final_state=state,
        final_model=model,
        records=records,
        distances=np.asarray(distances, dtype=np.float64),
        phases=phases,
        direction_lock_step=direction_lock_step,
        ready_step=ready_step,
        synthetic_threshold_step=synthetic_threshold_step,
        saturation_step=saturation_step,
        final_status=final_status,
    )


def _run_semantic_baseline_with_pm(
    synthetic_user: SyntheticUser,
    n_steps: int,
    seed: int | None,
    step_scale: float,
    heuristic_lr: float,
    model_lr: float,
    model_temperature: float,
    model_l2: float,
    init_std: float,
    std_decay: float,
    min_std: float,
    clip_value: float | None,
) -> PhaseSessionResult:
    """Same record schema as phase-aware loop, but always Semantic active v3."""
    result = run_phase_aware_session(
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
        seed=seed,
        config=PhaseControllerConfig(min_lock_step=10_000),
        stop_on_ready=False,
    )
    for rec in result.records:
        rec["phase"] = "exploration"
    result.phases = ["exploration"] * len(result.records)
    result.direction_lock_step = None
    result.ready_step = None
    result.saturation_step = None
    result.final_status = "fixed_semantic_active"
    return result


def phase_session_to_summary_row(
    user_id: int,
    target_mode: str,
    strategy: str,
    result: PhaseSessionResult,
    z_target: np.ndarray,
    clip_value: float | None = 2.0,
) -> dict:
    initial_distance = float(np.linalg.norm(z_target))
    distances = np.concatenate([[initial_distance], result.distances])

    pseudo_records = []
    # build_calibrated_vectors expects record objects with z_a/z_b/choice; for this
    # summary only final heuristic distance is needed, so do direct calculations here.
    heuristic = result.final_state.z_mean.copy()
    norm_calibrated = calibrate_by_heuristic_norm(result.final_model.z_pref, heuristic, clip_value=clip_value)
    selected_blend = clip_vector(0.70 * heuristic + 0.30 * norm_calibrated, clip_value)

    return {
        "user_id": int(user_id),
        "target_mode": str(target_mode),
        "strategy": str(strategy),
        "n_steps_completed": int(len(result.records)),
        "initial_distance": float(initial_distance),
        "heuristic_final_distance": float(distance_to_target(heuristic, z_target)),
        "selected_blend_final_distance": float(distance_to_target(selected_blend, z_target)),
        "best_distance": float(np.min(distances)),
        "mean_distance": float(np.mean(distances)),
        "distance_auc": float(np.mean(distances)),
        "direction_lock_step": result.direction_lock_step,
        "ready_step": result.ready_step,
        "synthetic_threshold_step": result.synthetic_threshold_step,
        "saturation_step": result.saturation_step,
        "final_status": result.final_status,
        "direction_locked": bool(result.direction_lock_step is not None),
        "ready_to_finalize": bool(result.ready_step is not None),
        "synthetic_converged": bool(result.synthetic_threshold_step is not None),
        "saturated": bool(result.saturation_step is not None),
        "final_direction_confidence": float(result.records[-1]["direction_confidence"]) if result.records else float("nan"),
        "final_mean_std": float(result.records[-1]["z_std_mean_after"]) if result.records else float("nan"),
        "final_update_norm": float(result.records[-1]["recent_update_norm"]) if result.records else float("nan"),
    }


def run_phase_controller_batch(
    dataset: pd.DataFrame,
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
    seed_base: int = 80_000,
    user_seed_base: int = 10_000,
    config: PhaseControllerConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run fixed Semantic active v3 and phase-aware loop on a dataset."""
    session_rows: list[dict] = []
    step_rows: list[dict] = []
    status_rows: list[dict] = []

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        for strategy in ["semantic_active_v3_fixed", "phase_aware_v1"]:
            # Re-create the synthetic user per strategy so stochastic choice noise is comparable.
            user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
            seed = int(seed_base + user_id + (0 if strategy == "semantic_active_v3_fixed" else 100_000))
            if strategy == "semantic_active_v3_fixed":
                result = _run_semantic_baseline_with_pm(
                    synthetic_user=user,
                    n_steps=n_steps,
                    seed=seed,
                    step_scale=step_scale,
                    heuristic_lr=heuristic_lr,
                    model_lr=model_lr,
                    model_temperature=model_temperature,
                    model_l2=model_l2,
                    init_std=init_std,
                    std_decay=std_decay,
                    min_std=min_std,
                    clip_value=clip_value,
                )
            else:
                result = run_phase_aware_session(
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
                    seed=seed,
                    config=config,
                    stop_on_ready=False,
                )

            summary = phase_session_to_summary_row(
                user_id=user_id,
                target_mode=target_mode,
                strategy=strategy,
                result=result,
                z_target=z_target,
                clip_value=clip_value,
            )
            session_rows.append(summary)

            for rec in result.records:
                pair_meta = rec.get("pair_meta") or {}
                step_rows.append({
                    "user_id": user_id,
                    "target_mode": target_mode,
                    "strategy": strategy,
                    "step": int(rec["step"]),
                    "phase": rec["phase"],
                    "distance_to_target": float(rec["distance_to_target"]),
                    "direction_confidence": float(rec["direction_confidence"]),
                    "recent_update_norm": float(rec["recent_update_norm"]),
                    "z_std_mean_after": float(rec["z_std_mean_after"]),
                    "clip_ratio": float(rec["clip_ratio"]),
                    "pair_source": rec.get("pair_source"),
                    "control_name": rec.get("control_name"),
                    "zone_direction": rec.get("zone_direction"),
                    "choice": rec["choice"],
                })

            status_rows.append({
                "user_id": user_id,
                "target_mode": target_mode,
                "strategy": strategy,
                "final_status": summary["final_status"],
                "direction_lock_step": summary["direction_lock_step"],
                "ready_step": summary["ready_step"],
                "synthetic_threshold_step": summary["synthetic_threshold_step"],
                "saturation_step": summary["saturation_step"],
            })

    return pd.DataFrame(session_rows), pd.DataFrame(step_rows), pd.DataFrame(status_rows)


def summarize_phase_results(sessions: pd.DataFrame) -> pd.DataFrame:
    grouped = sessions.groupby(["target_mode", "strategy"], dropna=False)
    return grouped.agg(
        users=("user_id", "nunique"),
        mean_heuristic_final_distance=("heuristic_final_distance", "mean"),
        mean_selected_blend_final_distance=("selected_blend_final_distance", "mean"),
        mean_best_distance=("best_distance", "mean"),
        mean_distance_auc=("distance_auc", "mean"),
        direction_lock_rate=("direction_locked", "mean"),
        ready_rate=("ready_to_finalize", "mean"),
        synthetic_convergence_rate=("synthetic_converged", "mean"),
        saturation_rate=("saturated", "mean"),
        mean_direction_lock_step=("direction_lock_step", "mean"),
        mean_ready_step=("ready_step", "mean"),
        mean_synthetic_threshold_step=("synthetic_threshold_step", "mean"),
        mean_final_direction_confidence=("final_direction_confidence", "mean"),
    ).reset_index()
