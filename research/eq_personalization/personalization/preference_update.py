from __future__ import annotations

import numpy as np

from .state import PreferenceState, clip_vector


def _update_uncertainty(
    state: PreferenceState,
    pair_meta: dict | None,
    std_decay: float,
    min_std: float,
) -> None:
    """
    Update z_std after an A/B question.

    For random questions, all dimensions decay equally.
    For axis questions, the selected axis decays more strongly.
    For semantic-control questions, dimensions touched by the semantic direction
    decay proportionally to the absolute direction weight.
    """
    if pair_meta is None:
        state.z_std = np.maximum(state.z_std * float(std_decay), float(min_std))
        return

    axis = pair_meta.get("axis")
    control_direction = pair_meta.get("control_direction")

    if axis is not None:
        # One concrete feature was tested. Reduce uncertainty mostly there.
        state.z_std = state.z_std * 0.99
        state.z_std[int(axis)] *= 0.75
        state.z_std = np.maximum(state.z_std, float(min_std))
        return

    if control_direction is not None:
        # A semantic direction tested several 8D features. Reduce uncertainty
        # more on dimensions that were strongly involved in the question.
        d = np.abs(np.asarray(control_direction, dtype=np.float64))
        if d.max() > 0:
            information = d / (d.max() + 1e-8)
            state.z_std = state.z_std * 0.995
            state.z_std = state.z_std * (1.0 - 0.25 * information)
            state.z_std = np.maximum(state.z_std, float(min_std))
            return

    # Fallback for random/hybrid-random questions.
    state.z_std = np.maximum(state.z_std * float(std_decay), float(min_std))


def update_state_from_choice(
    state: PreferenceState,
    z_a: np.ndarray,
    z_b: np.ndarray,
    choice: str,
    lr: float = 0.25,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_meta: dict | None = None,
) -> PreferenceState:
    """
    Baseline preference update.

    If user chose A, move z_mean toward A away from B.
    If user chose B, move z_mean toward B away from A.
    """
    if choice not in {"A", "B"}:
        raise ValueError("choice must be 'A' or 'B'")

    z_a = np.asarray(z_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64)

    preferred = z_a if choice == "A" else z_b
    rejected = z_b if choice == "A" else z_a

    direction = preferred - rejected
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    state.z_mean = clip_vector(state.z_mean + float(lr) * direction, clip_value)
    _update_uncertainty(state, pair_meta=pair_meta, std_decay=std_decay, min_std=min_std)

    state.step += 1
    state.history.append({
        "type": "ab_choice",
        "step": state.step,
        "choice": choice,
        "z_a": z_a.copy(),
        "z_b": z_b.copy(),
        "preferred": preferred.copy(),
        "rejected": rejected.copy(),
        "pair_meta": pair_meta,
        "z_mean_after": state.z_mean.copy(),
        "z_std_after": state.z_std.copy(),
    })
    return state


def update_state_toward_point(
    state: PreferenceState,
    z_target_point: np.ndarray,
    lr: float = 0.20,
    clip_value: float | None = 2.0,
) -> PreferenceState:
    """Optional direct update toward a point, useful for experiments."""
    z_target_point = np.asarray(z_target_point, dtype=np.float64)
    direction = z_target_point - state.z_mean
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    state.z_mean = clip_vector(state.z_mean + float(lr) * direction, clip_value)
    state.step += 1
    return state
