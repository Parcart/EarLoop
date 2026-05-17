from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .contract_space import clip_contract_z
from .state import FEATURE_NAMES_8D, PreferenceState
from .preference_model import LogisticDistancePreferenceModel, cosine_similarity

FeedbackLabel = Literal[
    "too_much_bass",
    "not_enough_bass",
    "too_bright",
    "too_dark",
    "too_muddy",
    "too_thin",
    "more_presence",
    "less_presence",
    "more_air",
    "less_air",
    "too_boomy",
    "too_harsh",
    "vocal_hidden",
    "too_sharp_s",
    "make_stronger",
    "make_weaker",
]


FEEDBACK_DELTAS: dict[str, np.ndarray] = {
    "too_much_bass": np.asarray([-0.16, -0.22, -0.05, 0.00, 0.00, 0.00, 0.00, 0.08], dtype=np.float64),
    "not_enough_bass": np.asarray([0.16, 0.22, 0.03, 0.00, 0.00, 0.00, 0.00, -0.05], dtype=np.float64),
    "too_bright": np.asarray([0.02, 0.03, 0.02, 0.03, -0.06, -0.12, -0.18, -0.22], dtype=np.float64),
    "too_dark": np.asarray([-0.02, -0.02, -0.02, -0.03, 0.06, 0.12, 0.18, 0.22], dtype=np.float64),
    "too_muddy": np.asarray([0.00, -0.06, -0.18, -0.12, 0.04, 0.08, 0.03, 0.08], dtype=np.float64),
    "too_thin": np.asarray([0.08, 0.12, 0.12, 0.10, -0.02, -0.02, 0.00, -0.04], dtype=np.float64),
    "more_presence": np.asarray([0.00, 0.00, -0.03, 0.00, 0.18, 0.08, 0.00, 0.05], dtype=np.float64),
    "less_presence": np.asarray([0.00, 0.00, 0.03, 0.00, -0.18, -0.08, 0.00, -0.05], dtype=np.float64),
    "more_air": np.asarray([0.00, 0.00, 0.00, 0.00, 0.02, 0.08, 0.22, 0.12], dtype=np.float64),
    "less_air": np.asarray([0.00, 0.00, 0.00, 0.00, -0.02, -0.08, -0.22, -0.12], dtype=np.float64),
    # More user-facing labels for live prototype and simulation.
    "too_boomy": np.asarray([0.00, -0.12, -0.18, -0.08, 0.03, 0.05, 0.00, 0.06], dtype=np.float64),
    "too_harsh": np.asarray([0.00, 0.02, 0.02, 0.03, -0.16, -0.12, -0.04, -0.10], dtype=np.float64),
    "vocal_hidden": np.asarray([0.00, -0.04, -0.10, -0.02, 0.18, 0.10, 0.03, 0.05], dtype=np.float64),
    "too_sharp_s": np.asarray([0.00, 0.00, 0.00, 0.00, -0.06, -0.14, -0.18, -0.16], dtype=np.float64),
}

FEEDBACK_SIMULATION_LABELS = tuple(FEEDBACK_DELTAS.keys())


@dataclass
class FeedbackDecision:
    use_feedback: bool
    label: str | None = None
    strength: float = 1.0
    probability: float = 0.0
    best_label_cosine: float = 0.0
    badness_score: float = 0.0
    ambiguity_score: float = 0.0
    misalignment_score: float = 0.0
    best_candidate_distance_z: float = np.nan
    state_distance_z: float = np.nan
    reason: str = "none"


def feedback_delta(label: str, strength: float = 1.0) -> np.ndarray:
    if label == "make_weaker":
        raise ValueError("make_weaker needs current z; use apply_feedback_to_z")
    if label == "make_stronger":
        raise ValueError("make_stronger needs current z; use apply_feedback_to_z")
    if label not in FEEDBACK_DELTAS:
        raise ValueError(f"Unknown feedback label: {label}")
    return FEEDBACK_DELTAS[label] * float(strength)


def apply_feedback_to_z(z_contract: np.ndarray, label: str, strength: float = 1.0, clip_value: float = 2.0) -> np.ndarray:
    z = np.asarray(z_contract, dtype=np.float64)
    if label == "make_weaker":
        out = z * (1.0 - 0.18 * float(strength))
    elif label == "make_stronger":
        out = z * (1.0 + 0.16 * float(strength))
    else:
        out = z + feedback_delta(label, strength=strength)
    return clip_contract_z(out, clip_value)


def apply_feedback_to_state(
    state: PreferenceState,
    label: str,
    strength: float = 1.0,
    clip_value: float = 2.0,
    std_decay: float = 0.88,
    min_std: float = 0.08,
) -> PreferenceState:
    new_state = state.copy()
    before = new_state.z_mean.copy()
    new_state.z_mean = apply_feedback_to_z(new_state.z_mean, label, strength=strength, clip_value=clip_value)

    # Feedback is directional information. Reduce uncertainty mainly on affected axes.
    delta = np.abs(new_state.z_mean - before)
    if float(delta.max()) > 1e-9:
        affected = delta / (float(delta.max()) + 1e-8)
        new_state.z_std = np.maximum(
            new_state.z_std * (1.0 - (1.0 - float(std_decay)) * affected),
            float(min_std),
        )
    else:
        new_state.z_std = np.maximum(new_state.z_std * 0.97, float(min_std))

    new_state.history.append({
        "type": "directional_feedback",
        "feedback_label": label,
        "strength": float(strength),
        "z_mean_before": before.copy(),
        "z_mean_after": new_state.z_mean.copy(),
    })
    return new_state


def apply_feedback_to_model(
    model: LogisticDistancePreferenceModel,
    state_before: PreferenceState,
    state_after: PreferenceState,
    feedback_model_lr: float = 0.35,
) -> dict:
    """Nudge the online Preference Model in the same direction as feedback.

    This is not mapper training. It only updates the per-session online preference
    estimate so that the next pair generator sees feedback as directional evidence.
    """
    delta = np.asarray(state_after.z_mean - state_before.z_mean, dtype=np.float64)
    before = model.z_pref.copy()
    model.z_pref = clip_contract_z(model.z_pref + float(feedback_model_lr) * delta, model.clip_value)
    record = {
        "type": "directional_feedback_model_update",
        "feedback_model_lr": float(feedback_model_lr),
        "model_delta_norm": float(np.linalg.norm(model.z_pref - before)),
        "z_pref_after": model.z_pref.copy(),
    }
    model.history.append(record)
    return record


def feedback_direction_for_label(label: str, z_reference: np.ndarray | None = None) -> np.ndarray:
    if label == "make_weaker":
        if z_reference is None:
            raise ValueError("make_weaker requires z_reference")
        return -np.asarray(z_reference, dtype=np.float64)
    if label == "make_stronger":
        if z_reference is None:
            raise ValueError("make_stronger requires z_reference")
        return np.asarray(z_reference, dtype=np.float64)
    return np.asarray(FEEDBACK_DELTAS[label], dtype=np.float64)


def select_feedback_label_by_cosine(
    desired_delta: np.ndarray,
    labels: tuple[str, ...] = FEEDBACK_SIMULATION_LABELS,
) -> tuple[str, float]:
    desired_delta = np.asarray(desired_delta, dtype=np.float64)
    best_label = labels[0]
    best_cos = -1.0
    for label in labels:
        delta = feedback_direction_for_label(label)
        cos = cosine_similarity(delta, desired_delta)
        if cos > best_cos:
            best_label = label
            best_cos = float(cos)
    return best_label, float(best_cos)


def _weighted_norm(x: np.ndarray, feature_importance: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=np.float64)
    if feature_importance is None:
        return float(np.linalg.norm(x))
    w = np.asarray(feature_importance, dtype=np.float64)
    w = w / (float(np.mean(w)) + 1e-8)
    return float(np.sqrt(np.sum(w * x * x)))


def _sigmoid01(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def synthetic_feedback_decision(
    *,
    z_a: np.ndarray,
    z_b: np.ndarray,
    state: PreferenceState,
    z_target: np.ndarray,
    feature_importance: np.ndarray | None,
    rng: np.random.Generator,
    step: int,
    policy: str = "hybrid",
    min_step: int = 3,
    cooldown_active: bool = False,
    max_probability: float = 0.38,
    base_probability: float = 0.02,
    random_probability: float = 0.16,
    bad_distance_z: float = 0.95,
    cosine_threshold: float = 0.45,
    ambiguity_margin: float = 0.18,
    misalignment_weight: float = 0.35,
    strength: float = 1.0,
) -> FeedbackDecision:
    """Simulate whether a virtual user rejects both A/B options and gives feedback.

    The policy is deliberately stochastic: even if both options are poor, feedback is
    only selected with a bounded probability. This approximates a real user who often
    still picks the better option, but sometimes says why both options are wrong.
    """
    if policy in {"none", "off", "disabled"}:
        return FeedbackDecision(False, reason="disabled")
    if int(step) < int(min_step):
        return FeedbackDecision(False, reason="too_early")
    if cooldown_active:
        return FeedbackDecision(False, reason="cooldown")

    z_a = np.asarray(z_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64)
    z_target = np.asarray(z_target, dtype=np.float64)
    z_center = 0.5 * (z_a + z_b)
    z_state = np.asarray(state.z_mean, dtype=np.float64)

    dist_a = _weighted_norm(z_a - z_target, feature_importance)
    dist_b = _weighted_norm(z_b - z_target, feature_importance)
    best_dist = min(dist_a, dist_b)
    state_dist = _weighted_norm(z_state - z_target, feature_importance)

    # If both candidates are similarly good/bad, A/B is less informative and feedback is more plausible.
    utility_gap_like = abs(dist_a - dist_b)
    ambiguity_score = float(np.exp(-utility_gap_like / max(float(ambiguity_margin), 1e-6)))

    desired_delta = z_target - z_center
    label, best_cos = select_feedback_label_by_cosine(desired_delta)

    pair_direction = z_b - z_a
    target_direction = z_target - z_state
    pair_alignment = abs(cosine_similarity(pair_direction, target_direction))
    misalignment_score = float(np.clip(1.0 - pair_alignment, 0.0, 1.0))

    # Smooth probability that best available candidate is still too far from target.
    badness_score = _sigmoid01((best_dist - float(bad_distance_z)) / 0.22)
    worse_than_state = 1.0 if best_dist > 0.98 * state_dist else 0.0

    if policy == "random":
        probability = float(random_probability) if badness_score > 0.45 else 0.0
        label = str(rng.choice(np.asarray(FEEDBACK_SIMULATION_LABELS, dtype=object)))
        best_cos = cosine_similarity(feedback_direction_for_label(label), desired_delta)
    elif policy == "cosine":
        if best_cos < float(cosine_threshold):
            return FeedbackDecision(
                False,
                label=label,
                probability=0.0,
                best_label_cosine=best_cos,
                badness_score=badness_score,
                ambiguity_score=ambiguity_score,
                misalignment_score=misalignment_score,
                best_candidate_distance_z=best_dist,
                state_distance_z=state_dist,
                reason="low_cosine",
            )
        probability = float(base_probability) + float(max_probability) * badness_score * (
            0.45 * ambiguity_score + float(misalignment_weight) * misalignment_score + 0.20 * worse_than_state
        )
    else:  # hybrid = cosine label + stochastic trigger; sometimes explore top alternatives.
        if best_cos < float(cosine_threshold):
            probability = float(base_probability) * badness_score
        else:
            probability = float(base_probability) + float(max_probability) * badness_score * (
                0.40 * ambiguity_score + float(misalignment_weight) * misalignment_score + 0.25 * worse_than_state
            )
            # Rarely choose another plausible label among top cosine matches.
            if rng.random() < 0.18:
                scored = []
                for cand in FEEDBACK_SIMULATION_LABELS:
                    scored.append((cosine_similarity(feedback_direction_for_label(cand), desired_delta), cand))
                scored.sort(reverse=True)
                top = [cand for cos, cand in scored[:3] if cos > 0.15]
                if top:
                    label = str(rng.choice(np.asarray(top, dtype=object)))
                    best_cos = cosine_similarity(feedback_direction_for_label(label), desired_delta)

    probability = float(np.clip(probability, 0.0, float(max_probability)))
    use_feedback = bool(rng.random() < probability)
    return FeedbackDecision(
        use_feedback=use_feedback,
        label=label if use_feedback else label,
        strength=float(strength),
        probability=probability,
        best_label_cosine=float(best_cos),
        badness_score=float(badness_score),
        ambiguity_score=float(ambiguity_score),
        misalignment_score=float(misalignment_score),
        best_candidate_distance_z=float(best_dist),
        state_distance_z=float(state_dist),
        reason="sampled_feedback" if use_feedback else "sampled_choice",
    )


def feedback_labels_table():
    import pandas as pd
    rows = []
    for label, delta in FEEDBACK_DELTAS.items():
        row = {"feedback_label": label}
        row.update({name: float(value) for name, value in zip(FEATURE_NAMES_8D, delta)})
        rows.append(row)
    rows.append({"feedback_label": "make_weaker"})
    rows.append({"feedback_label": "make_stronger"})
    return pd.DataFrame(rows)
