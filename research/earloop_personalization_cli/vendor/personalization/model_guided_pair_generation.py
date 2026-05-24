from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .control_basis import CONTROL_BASIS_6D_TO_8D, CONTROL_NAMES_6D
from .loop import run_personalization_session_v0
from .metrics import distance_to_target
from .pair_generator import PairGenerator, normalize
from .preference_model import LogisticDistancePreferenceModel, cosine_similarity
from .preference_model_calibration import calibrate_by_heuristic_norm, calibrate_by_train_log_loss
from .preference_update import update_state_from_choice
from .state import FEATURE_NAMES_8D, PreferenceState, clip_vector, init_preference_state
from .synthetic_dataset import (
    generate_synthetic_users_dataset,
    row_to_importance,
    row_to_synthetic_user,
    row_to_target,
)
from .synthetic_user import SyntheticUser


MODEL_GUIDED_STRATEGY_DISPLAY_NAMES = {
    "semantic_active_v21": "Semantic active v3",
    "candidate_pool_active": "Candidate pool active",
    "model_only_cold": "Model-only cold",
    "model_only_warmup": "Model-only warmup",
    "model_only_pretrained": "Model-only pretrained prior",
    "hybrid_model_guided": "Hybrid model-guided",
    "hybrid_model_guided_pretrained": "Hybrid model-guided pretrained",
}

TARGET_MODE_DISPLAY_NAMES = {
    "random8d": "Random 8D",
    "semantic4d": "Semantic 4D",
    "semantic6d": "Semantic 6D",
    "archetype8d": "Archetype 8D",
}

ModelGuidedStrategy = Literal[
    "model_only_cold",
    "model_only_warmup",
    "model_only_pretrained",
    "hybrid_model_guided",
    "hybrid_model_guided_pretrained",
]


@dataclass
class PopulationPreferencePrior:
    """
    Population-level prior for Preference Model initialization.

    This is not an individual user's target. It is a weak prior estimated from a
    separate synthetic population, analogous to pretraining/calibrating model
    hyperparameters before a new personal A/B session.
    """

    z_pref: np.ndarray
    feature_weight: np.ndarray
    n_users: int
    modes: list[str]
    seed: int


def _normalize01(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < eps:
        return np.ones_like(values) * 0.5
    return (values - lo) / (hi - lo + eps)


def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)) + eps)


def build_population_preference_prior(
    n_per_mode: int = 250,
    seed: int = 2026,
    modes: tuple[str, ...] = ("random8d", "semantic4d", "semantic6d", "archetype8d"),
    use_zero_target_prior: bool = True,
) -> PopulationPreferencePrior:
    """
    Build a weak population prior from an independent synthetic population.

    For individual personalization, the target prior should remain weak. By
    default, z_pref starts at zero and only feature_weight is pretrained from
    population feature-importance statistics. If use_zero_target_prior=False,
    the global mean target vector is also used as initialization, but this is
    intentionally optional because it can over-bias the first A/B questions.
    """
    dataset = generate_synthetic_users_dataset(
        n_per_mode=int(n_per_mode),
        modes=modes,
        seed=int(seed),
        noise_std=0.05,
    )

    targets = np.stack([row_to_target(row) for _, row in dataset.iterrows()], axis=0)
    importances = np.stack([row_to_importance(row) for _, row in dataset.iterrows()], axis=0)

    if use_zero_target_prior:
        z_pref = np.zeros(targets.shape[1], dtype=np.float64)
    else:
        z_pref = targets.mean(axis=0).astype(np.float64)

    feature_weight = importances.mean(axis=0).astype(np.float64)
    feature_weight = feature_weight / (feature_weight.mean() + 1e-8)

    return PopulationPreferencePrior(
        z_pref=z_pref,
        feature_weight=feature_weight,
        n_users=int(len(dataset)),
        modes=list(modes),
        seed=int(seed),
    )


def _safety_penalty(z_a: np.ndarray, z_b: np.ndarray, clip_value: float | None = 2.0) -> float:
    if clip_value is None:
        return 0.0
    overflow_a = np.maximum(np.abs(z_a) - float(clip_value), 0.0)
    overflow_b = np.maximum(np.abs(z_b) - float(clip_value), 0.0)
    return float(np.mean(overflow_a * overflow_a + overflow_b * overflow_b))


def _recent_repetition_penalty(state: PreferenceState, label: str | None, lookback: int = 3) -> float:
    if label is None or not state.history:
        return 0.0
    recent = state.history[-int(lookback):]
    matches = 0
    for item in recent:
        meta = item.get("pair_meta") or {}
        recent_label = meta.get("control_name") or meta.get("axis_name") or meta.get("source")
        if recent_label == label:
            matches += 1
    return float(matches / max(len(recent), 1))


def _make_pair_from_direction(
    state: PreferenceState,
    direction: np.ndarray,
    scale: float,
    rng: np.random.Generator,
    clip_value: float | None = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = normalize(direction)
    if rng.random() < 0.5:
        d = -d
    z_a = clip_vector(state.z_mean + float(scale) * d, clip_value)
    z_b = clip_vector(state.z_mean - float(scale) * d, clip_value)
    return z_a, z_b, d


def _generate_model_guided_candidates(
    state: PreferenceState,
    preference_model: LogisticDistancePreferenceModel,
    rng: np.random.Generator,
    step_scale: float = 0.6,
    clip_value: float | None = 2.0,
    n_random: int = 18,
    semantic_scales: tuple[float, ...] = (0.35, 0.60, 0.85, 1.10),
    axis_scales: tuple[float, ...] = (0.45, 0.75),
    random_scales: tuple[float, ...] = (0.55, 0.85),
    model_scales: tuple[float, ...] = (0.40, 0.70, 1.00),
) -> list[dict]:
    """Fast candidate generation as a Python list of dicts."""
    z_std = np.asarray(state.z_std, dtype=np.float64)
    rows: list[dict] = []
    dim = len(state.z_mean)

    def add_candidate(
        source: str,
        label: str,
        direction: np.ndarray,
        scale: float,
        scale_multiplier: float | None = None,
        control_name: str | None = None,
        axis_name: str | None = None,
    ) -> None:
        d = normalize(direction)
        z_a, z_b, signed_d = _make_pair_from_direction(
            state=state,
            direction=d,
            scale=scale,
            rng=rng,
            clip_value=clip_value,
        )
        touched = np.abs(signed_d)
        uncertainty_coverage = float(np.sum(touched * z_std) / (np.sum(touched) + 1e-8))
        diversity = float(np.linalg.norm(z_a - z_b) / (2.0 * step_scale * 1.10 + 1e-8))
        diversity = float(np.clip(diversity, 0.0, 1.5))

        p_a = float(preference_model.predict_proba_a(z_a, z_b))
        model_uncertainty = float(1.0 - 2.0 * abs(p_a - 0.5))
        model_confidence = float(1.0 - model_uncertainty)

        u_a = preference_model.utility(z_a)
        u_b = preference_model.utility(z_b)
        mean_model_dist = 0.5 * (
            np.linalg.norm(z_a - preference_model.z_pref) + np.linalg.norm(z_b - preference_model.z_pref)
        )
        model_quality = float(1.0 / (1.0 + mean_model_dist))

        safety = _safety_penalty(z_a, z_b, clip_value=clip_value)
        repetition_label = control_name or axis_name or source
        repetition = _recent_repetition_penalty(state, repetition_label)

        semantic_bonus = 1.0 if source == "semantic_v21" else 0.0
        axis_bonus = 1.0 if source == "axis" else 0.0
        random_bonus = 1.0 if source == "random" else 0.0
        model_bonus = 1.0 if source == "model_direction" else 0.0

        proxy_score = (
            1.00 * uncertainty_coverage
            + 0.18 * diversity
            + 0.12 * semantic_bonus
            + 0.04 * axis_bonus
            + 0.06 * random_bonus
            + 0.08 * model_bonus
            - 1.00 * safety
            - 0.25 * repetition
        )

        rows.append({
            "source": source,
            "label": label,
            "control_name": control_name,
            "axis_name": axis_name,
            "scale": float(scale),
            "scale_multiplier": None if scale_multiplier is None else float(scale_multiplier),
            "direction": signed_d.copy(),
            "z_a": z_a.copy(),
            "z_b": z_b.copy(),
            "p_a": p_a,
            "model_uncertainty": model_uncertainty,
            "model_confidence": model_confidence,
            "model_quality": model_quality,
            "model_utility_a": float(u_a),
            "model_utility_b": float(u_b),
            "uncertainty_coverage": uncertainty_coverage,
            "diversity": diversity,
            "safety_penalty": safety,
            "repetition_penalty": repetition,
            "semantic_bonus": semantic_bonus,
            "axis_bonus": axis_bonus,
            "random_bonus": random_bonus,
            "model_bonus": model_bonus,
            "proxy_score": float(proxy_score),
        })

    for i, control_name in enumerate(CONTROL_NAMES_6D):
        direction = normalize(CONTROL_BASIS_6D_TO_8D[i])
        touched = np.abs(direction)
        uncertainty_scale = float(np.sum(touched * z_std) / (np.sum(touched) + 1e-8))
        for mult in semantic_scales:
            scale = step_scale * float(mult) * max(uncertainty_scale, 0.30)
            add_candidate("semantic_v21", control_name, direction, scale, float(mult), control_name=control_name)

    for axis, axis_name in enumerate(FEATURE_NAMES_8D):
        direction = np.zeros(dim, dtype=np.float64)
        direction[axis] = 1.0
        for mult in axis_scales:
            scale = step_scale * float(mult) * max(float(z_std[axis]), 0.30)
            add_candidate("axis", f"axis:{axis_name}", direction, scale, float(mult), axis_name=axis_name)

    for i in range(int(n_random)):
        direction = normalize(rng.normal(size=dim) * z_std)
        for mult in random_scales:
            scale = step_scale * float(mult)
            add_candidate("random", f"random:{i}", direction, scale, float(mult))

    model_direction = np.asarray(preference_model.z_pref, dtype=np.float64) - np.asarray(state.z_mean, dtype=np.float64)
    if np.linalg.norm(model_direction) > 1e-6:
        for mult in model_scales:
            scale = step_scale * float(mult)
            add_candidate("model_direction", "model_direction", model_direction, scale, float(mult))

    return rows


def generate_model_guided_candidate_pool(
    state: PreferenceState,
    preference_model: LogisticDistancePreferenceModel,
    rng: np.random.Generator,
    step_scale: float = 0.6,
    clip_value: float | None = 2.0,
    n_random: int = 18,
    semantic_scales: tuple[float, ...] = (0.35, 0.60, 0.85, 1.10),
    axis_scales: tuple[float, ...] = (0.45, 0.75),
    random_scales: tuple[float, ...] = (0.55, 0.85),
    model_scales: tuple[float, ...] = (0.40, 0.70, 1.00),
) -> pd.DataFrame:
    """
    Generate A/B candidate pairs from semantic, axis, random and model-based directions.
    Returned DataFrame stores z_a/z_b arrays and diagnostic scores.
    """
    return pd.DataFrame(_generate_model_guided_candidates(
        state=state,
        preference_model=preference_model,
        rng=rng,
        step_scale=step_scale,
        clip_value=clip_value,
        n_random=n_random,
        semantic_scales=semantic_scales,
        axis_scales=axis_scales,
        random_scales=random_scales,
        model_scales=model_scales,
    ))


def select_pair_from_model_guided_pool(
    state: PreferenceState,
    preference_model: LogisticDistancePreferenceModel,
    rng: np.random.Generator,
    selection_mode: Literal["model_only", "hybrid"] = "hybrid",
    step_scale: float = 0.6,
    clip_value: float | None = 2.0,
    n_steps: int = 25,
    max_model_weight: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Select an A/B pair from a candidate pool using model-only or hybrid acquisition."""
    pool = _generate_model_guided_candidates(
        state=state,
        preference_model=preference_model,
        rng=rng,
        step_scale=step_scale,
        clip_value=clip_value,
    )
    if not pool:
        raise RuntimeError("Candidate pool is empty")

    proxy_values = np.asarray([item["proxy_score"] for item in pool], dtype=np.float64)
    proxy_norm_values = _normalize01(proxy_values)

    if selection_mode == "model_only":
        model_weight = 1.0
        proxy_weight = 0.0
        for item, proxy_norm in zip(pool, proxy_norm_values):
            item["proxy_score_norm"] = float(proxy_norm)
            item["acquisition_score"] = float(
                1.00 * item["model_uncertainty"]
                + 0.12 * item["diversity"]
                + 0.15 * item["model_quality"]
                - 1.00 * item["safety_penalty"]
                - 0.20 * item["repetition_penalty"]
            )
    elif selection_mode == "hybrid":
        progress = float(min(1.0, max(0.0, state.step / max(int(n_steps), 1))))
        model_weight = float(min(max_model_weight, max_model_weight * progress))
        proxy_weight = float(1.0 - model_weight)
        for item, proxy_norm in zip(pool, proxy_norm_values):
            item["proxy_score_norm"] = float(proxy_norm)
            item["acquisition_score"] = float(
                proxy_weight * float(proxy_norm)
                + model_weight * item["model_uncertainty"]
                + 0.12 * item["model_quality"]
                - 0.60 * item["safety_penalty"]
                - 0.15 * item["repetition_penalty"]
            )
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    best = max(pool, key=lambda item: item["acquisition_score"])
    top_candidates = []
    for item in sorted(pool, key=lambda x: x["acquisition_score"], reverse=True)[:8]:
        top_candidates.append({
            "source": item["source"],
            "label": item["label"],
            "scale_multiplier": item["scale_multiplier"],
            "acquisition_score": item["acquisition_score"],
            "proxy_score": item["proxy_score"],
            "proxy_score_norm": item["proxy_score_norm"],
            "model_uncertainty": item["model_uncertainty"],
            "model_quality": item["model_quality"],
            "p_a": item["p_a"],
            "uncertainty_coverage": item["uncertainty_coverage"],
            "diversity": item["diversity"],
        })

    meta = {
        "strategy": f"model_guided_{selection_mode}",
        "sub_strategy": best["source"],
        "source": best["source"],
        "label": best["label"],
        "control_name": best["control_name"],
        "axis_name": best["axis_name"],
        "scale": float(best["scale"]),
        "scale_multiplier": best["scale_multiplier"],
        "selection_mode": selection_mode,
        "acquisition_score": float(best["acquisition_score"]),
        "proxy_score": float(best["proxy_score"]),
        "proxy_score_norm": float(best["proxy_score_norm"]),
        "model_uncertainty": float(best["model_uncertainty"]),
        "model_quality": float(best["model_quality"]),
        "model_p_a": float(best["p_a"]),
        "uncertainty_coverage": float(best["uncertainty_coverage"]),
        "diversity": float(best["diversity"]),
        "safety_penalty": float(best["safety_penalty"]),
        "repetition_penalty": float(best["repetition_penalty"]),
        "model_weight": float(model_weight),
        "proxy_weight": float(proxy_weight),
        "candidate_count": int(len(pool)),
        "top_candidates": top_candidates,
        "control_direction": best["direction"].copy(),
    }

    return best["z_a"].copy(), best["z_b"].copy(), best["direction"].copy(), meta


@dataclass
class ModelGuidedStepRecord:
    step: int
    strategy: str
    choice: str
    z_a: np.ndarray
    z_b: np.ndarray
    u_a: float
    u_b: float
    z_mean_after: np.ndarray
    heuristic_distance_to_target: float
    model_distance_to_target: float
    model_cosine_to_target: float
    p_a_before: float
    p_a_after: float
    correct_before: bool
    loss_before: float
    loss_after: float
    pair_meta: dict


@dataclass
class ModelGuidedSessionResult:
    final_state: PreferenceState
    final_model: LogisticDistancePreferenceModel
    records: list[ModelGuidedStepRecord]
    heuristic_distances: np.ndarray
    model_distances: np.ndarray


def run_model_guided_pair_session_v4b(
    synthetic_user: SyntheticUser,
    strategy: ModelGuidedStrategy = "hybrid_model_guided",
    n_steps: int = 25,
    warmup_steps: int = 5,
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
    prior: PopulationPreferencePrior | None = None,
) -> ModelGuidedSessionResult:
    """Run one model-guided Pair Generator session."""
    rng = np.random.default_rng(seed)
    state = init_preference_state(dim=len(synthetic_user.z_target), init_std=init_std)
    pair_generator = PairGenerator(step_scale=step_scale, clip_value=clip_value, rng=rng)

    use_prior = strategy in {"model_only_pretrained", "hybrid_model_guided_pretrained"}
    if use_prior and prior is not None:
        init_z = prior.z_pref.copy()
        feature_weight = prior.feature_weight.copy()
    else:
        init_z = np.zeros(len(synthetic_user.z_target), dtype=np.float64)
        feature_weight = np.ones(len(synthetic_user.z_target), dtype=np.float64)

    model = LogisticDistancePreferenceModel(
        dim=len(synthetic_user.z_target),
        lr=model_lr,
        temperature=model_temperature,
        l2=model_l2,
        clip_value=clip_value,
        feature_weight=feature_weight,
        z_pref=init_z,
    )

    records: list[ModelGuidedStepRecord] = []
    heuristic_distances: list[float] = []
    model_distances: list[float] = []

    for step in range(1, int(n_steps) + 1):
        if strategy == "model_only_warmup" and step <= int(warmup_steps):
            z_a, z_b, _direction, pair_meta = pair_generator.semantic_active_v21(state)
            pair_meta = dict(pair_meta)
            pair_meta["strategy"] = strategy
            pair_meta["selection_mode"] = "warmup_semantic_active"
        else:
            if strategy in {"model_only_cold", "model_only_warmup", "model_only_pretrained"}:
                selection_mode = "model_only"
            elif strategy in {"hybrid_model_guided", "hybrid_model_guided_pretrained"}:
                selection_mode = "hybrid"
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            z_a, z_b, _direction, pair_meta = select_pair_from_model_guided_pool(
                state=state,
                preference_model=model,
                rng=rng,
                selection_mode=selection_mode,  # type: ignore[arg-type]
                step_scale=step_scale,
                clip_value=clip_value,
                n_steps=n_steps,
            )
            pair_meta = dict(pair_meta)
            pair_meta["strategy"] = strategy

        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)

        p_before = float(model.predict_proba_a(z_a, z_b))
        pred_before = "A" if p_before >= 0.5 else "B"
        correct_before = bool(pred_before == choice)
        loss_before = float(model.log_loss(z_a, z_b, choice))

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
        update_info = model.update(z_a, z_b, choice)

        h_dist = float(distance_to_target(state.z_mean, synthetic_user.z_target))
        m_dist = float(distance_to_target(model.z_pref, synthetic_user.z_target))
        cos = float(cosine_similarity(model.z_pref, synthetic_user.z_target))
        heuristic_distances.append(h_dist)
        model_distances.append(m_dist)

        records.append(ModelGuidedStepRecord(
            step=int(step),
            strategy=strategy,
            choice=choice,
            z_a=z_a.copy(),
            z_b=z_b.copy(),
            u_a=float(u_a),
            u_b=float(u_b),
            z_mean_after=state.z_mean.copy(),
            heuristic_distance_to_target=h_dist,
            model_distance_to_target=m_dist,
            model_cosine_to_target=cos,
            p_a_before=p_before,
            p_a_after=float(update_info["p_after"]),
            correct_before=correct_before,
            loss_before=loss_before,
            loss_after=float(update_info["loss_after"]),
            pair_meta=pair_meta,
        ))

    return ModelGuidedSessionResult(
        final_state=state,
        final_model=model,
        records=records,
        heuristic_distances=np.asarray(heuristic_distances, dtype=np.float64),
        model_distances=np.asarray(model_distances, dtype=np.float64),
    )


def _calibrated_final_vectors(result: ModelGuidedSessionResult, clip_value: float | None = 2.0) -> dict[str, np.ndarray]:
    heuristic = result.final_state.z_mean.copy()
    raw = result.final_model.z_pref.copy()
    norm_cal = calibrate_by_heuristic_norm(raw, heuristic, clip_value=clip_value)
    # Reuse training history from model-guided records for scale optimization.
    train_scaled, _, _ = calibrate_model_vector_from_guided_records(
        raw,
        result.records,
        result.final_model.feature_weight,
        result.final_model.temperature,
        heuristic_vector=heuristic,
        clip_value=clip_value,
    )
    blend_70 = clip_vector(0.70 * heuristic + 0.30 * train_scaled, clip_value)
    return {
        "heuristic_update": heuristic,
        "raw_preference_model": raw,
        "norm_calibrated_model": norm_cal,
        "train_scale_model": train_scaled,
        "blend_70h_30m": blend_70,
    }


def calibrate_model_vector_from_guided_records(
    model_vector: np.ndarray,
    records: list[ModelGuidedStepRecord],
    feature_weight: np.ndarray,
    temperature: float,
    heuristic_vector: np.ndarray | None = None,
    clip_value: float | None = 2.0,
    n_grid: int = 81,
) -> tuple[np.ndarray, float, float]:
    """Train-loss scale calibration for model-guided session records."""
    from .preference_model_calibration import predict_proba_for_z_pref

    direction = model_vector / (np.linalg.norm(model_vector) + 1e-8)
    raw_norm = float(np.linalg.norm(model_vector))
    heuristic_norm = float(np.linalg.norm(heuristic_vector)) if heuristic_vector is not None else raw_norm
    dim = int(len(model_vector))
    if clip_value is None:
        max_scale = max(2.0, 2.0 * raw_norm, 2.0 * heuristic_norm)
    else:
        max_scale = min(float(np.sqrt(dim) * float(clip_value)), max(1.0, 2.5 * raw_norm, 2.5 * heuristic_norm))
    scales = np.linspace(0.0, max_scale, int(n_grid))
    best_vec = np.zeros_like(model_vector, dtype=np.float64)
    best_scale = 0.0
    best_loss = float("inf")
    for scale in scales:
        candidate = clip_vector(float(scale) * direction, clip_value)
        losses = []
        for record in records:
            p = predict_proba_for_z_pref(candidate, record.z_a, record.z_b, feature_weight, temperature)
            p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
            y = 1.0 if record.choice == "A" else 0.0
            losses.append(float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
        loss = float(np.mean(losses)) if losses else float("inf")
        if loss < best_loss:
            best_loss = loss
            best_scale = float(scale)
            best_vec = candidate
    return best_vec, best_scale, best_loss


def run_model_guided_pair_batch_v4b(
    dataset: pd.DataFrame,
    strategies: tuple[str, ...] = (
        "semantic_active_v21",
        "candidate_pool_active",
        "model_only_cold",
        "model_only_warmup",
        "model_only_pretrained",
        "hybrid_model_guided",
        "hybrid_model_guided_pretrained",
    ),
    n_steps: int = 25,
    warmup_steps: int = 5,
    step_scale: float = 0.6,
    heuristic_lr: float = 0.25,
    model_lr: float = 0.06,
    model_temperature: float = 0.75,
    model_l2: float = 0.003,
    clip_value: float | None = 2.0,
    seed_base: int = 120_000,
    user_seed_base: int = 10_000,
    prior: PopulationPreferencePrior | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline and V4b model-guided strategies over a fixed dataset."""
    session_rows: list[dict] = []
    step_rows: list[dict] = []

    for _, row in dataset.iterrows():
        user_id = int(row["user_id"])
        target_mode = str(row["target_mode"])
        z_target = row_to_target(row)
        user = row_to_synthetic_user(row, seed=user_seed_base + user_id)
        initial_distance = float(np.linalg.norm(z_target))

        for s_idx, strategy in enumerate(strategies):
            seed = int(seed_base + 10_000 * s_idx + user_id)

            if strategy in {"semantic_active_v21", "candidate_pool_active"}:
                baseline = run_personalization_session_v0(
                    synthetic_user=user,
                    n_steps=n_steps,
                    step_scale=step_scale,
                    lr=heuristic_lr,
                    clip_value=clip_value,
                    pair_strategy=strategy,  # type: ignore[arg-type]
                    seed=seed,
                )
                final_vector = baseline.final_state.z_mean.copy()
                distances = baseline.distances
                final_distance = float(distance_to_target(final_vector, z_target))
                best_distance = float(np.min(distances))
                mean_distance = float(np.mean(distances))
                session_rows.append({
                    "user_id": user_id,
                    "target_mode": target_mode,
                    "strategy": strategy,
                    "strategy_display": MODEL_GUIDED_STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                    "final_vector_method": "heuristic_update",
                    "initial_distance": initial_distance,
                    "final_distance": final_distance,
                    "best_distance": best_distance,
                    "mean_distance": mean_distance,
                    "improvement_abs": initial_distance - final_distance,
                    "improvement_pct": 100.0 * (initial_distance - final_distance) / (initial_distance + 1e-8),
                    "cosine_to_target": float(cosine_similarity(final_vector, z_target)),
                    "heldout_accuracy_proxy": float("nan"),
                    "n_steps": int(n_steps),
                })
                for record in baseline.records:
                    step_rows.append({
                        "user_id": user_id,
                        "target_mode": target_mode,
                        "strategy": strategy,
                        "step": int(record.step),
                        "distance_to_target": float(record.distance_to_target),
                        "pair_source": record.pair_meta.get("sub_strategy") or record.pair_meta.get("strategy"),
                        "control_name": record.pair_meta.get("control_name"),
                        "model_p_a_before": float("nan"),
                        "model_uncertainty": float("nan"),
                    })
                continue

            result = run_model_guided_pair_session_v4b(
                synthetic_user=user,
                strategy=strategy,  # type: ignore[arg-type]
                n_steps=n_steps,
                warmup_steps=warmup_steps,
                step_scale=step_scale,
                heuristic_lr=heuristic_lr,
                model_lr=model_lr,
                model_temperature=model_temperature,
                model_l2=model_l2,
                clip_value=clip_value,
                seed=seed,
                prior=prior,
            )
            vectors = _calibrated_final_vectors(result, clip_value=clip_value)

            for method, final_vector in vectors.items():
                final_distance = float(distance_to_target(final_vector, z_target))
                session_rows.append({
                    "user_id": user_id,
                    "target_mode": target_mode,
                    "strategy": strategy,
                    "strategy_display": MODEL_GUIDED_STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                    "final_vector_method": method,
                    "initial_distance": initial_distance,
                    "final_distance": final_distance,
                    "best_distance": float(np.min(result.heuristic_distances)),
                    "mean_distance": float(np.mean(result.heuristic_distances)),
                    "improvement_abs": initial_distance - final_distance,
                    "improvement_pct": 100.0 * (initial_distance - final_distance) / (initial_distance + 1e-8),
                    "cosine_to_target": float(cosine_similarity(final_vector, z_target)),
                    "heldout_accuracy_proxy": float(np.mean([r.correct_before for r in result.records])),
                    "n_steps": int(n_steps),
                })

            for record in result.records:
                step_rows.append({
                    "user_id": user_id,
                    "target_mode": target_mode,
                    "strategy": strategy,
                    "step": int(record.step),
                    "distance_to_target": float(record.heuristic_distance_to_target),
                    "model_distance_to_target": float(record.model_distance_to_target),
                    "model_cosine_to_target": float(record.model_cosine_to_target),
                    "pair_source": record.pair_meta.get("source") or record.pair_meta.get("sub_strategy"),
                    "control_name": record.pair_meta.get("control_name"),
                    "selection_mode": record.pair_meta.get("selection_mode"),
                    "model_p_a_before": float(record.p_a_before),
                    "model_uncertainty": record.pair_meta.get("model_uncertainty", float("nan")),
                    "proxy_score": record.pair_meta.get("proxy_score", float("nan")),
                    "acquisition_score": record.pair_meta.get("acquisition_score", float("nan")),
                })

    sessions = pd.DataFrame(session_rows)
    steps = pd.DataFrame(step_rows)
    for col in ["intensity_label", "main_archetype", "secondary_archetype"]:
        if col in dataset.columns and col not in sessions.columns:
            meta = dataset[["user_id", col]].copy()
            sessions = sessions.merge(meta, on="user_id", how="left")
            steps = steps.merge(meta, on="user_id", how="left")
    return sessions, steps


def summarize_model_guided_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate V4b model-guided strategy results."""
    return (
        sessions
        .groupby(["target_mode", "strategy", "strategy_display", "final_vector_method"])
        .agg(
            users=("user_id", "nunique"),
            mean_initial_distance=("initial_distance", "mean"),
            mean_final_distance=("final_distance", "mean"),
            std_final_distance=("final_distance", "std"),
            mean_best_distance=("best_distance", "mean"),
            mean_mean_distance=("mean_distance", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            mean_cosine_to_target=("cosine_to_target", "mean"),
            mean_train_accuracy_proxy=("heldout_accuracy_proxy", "mean"),
        )
        .reset_index()
    )


def save_v4b_outputs(
    sessions: pd.DataFrame,
    steps: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "notebook_v4b_model_guided_pair_generator",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    steps.to_csv(output_dir / f"{prefix}_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
