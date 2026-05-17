from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .metrics import distance_to_target
from .pair_generator import PairGenerator
from .preference_update import update_state_from_choice
from .synthetic_user import SyntheticUser
from .state import FEATURE_NAMES_8D, PreferenceState, init_preference_state


PairStrategy = Literal[
    "random",
    "uncertainty_axis",
    "semantic_control",
    "semantic_control_v21",
    "semantic_active_v21",
    "candidate_pool_active",
    "adaptive_router_v32",
    "hybrid",
    "hybrid_v21",
    "hybrid_active_v21",
]


@dataclass
class StepRecord:
    step: int
    choice: str
    z_a: np.ndarray
    z_b: np.ndarray
    u_a: float
    u_b: float
    z_mean_after: np.ndarray
    distance_to_target: float
    pair_strategy: str
    pair_meta: dict


@dataclass
class SessionResult:
    final_state: PreferenceState
    records: list[StepRecord]
    distances: np.ndarray


def generate_pair_by_strategy(
    generator: PairGenerator,
    state: PreferenceState,
    strategy: PairStrategy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if strategy == "random":
        z_a, z_b, direction = generator.random_symmetric(state)
        return z_a, z_b, direction, {
            "strategy": "random",
            "axis": None,
            "axis_name": None,
            "control_index": None,
            "control_name": None,
        }

    if strategy == "uncertainty_axis":
        z_a, z_b, direction, axis = generator.uncertainty_axis(state)
        return z_a, z_b, direction, {
            "strategy": "uncertainty_axis",
            "axis": axis,
            "axis_name": FEATURE_NAMES_8D[axis],
            "control_index": None,
            "control_name": None,
        }

    if strategy == "semantic_control":
        z_a, z_b, direction, meta = generator.semantic_control(state)
        return z_a, z_b, direction, meta

    if strategy == "semantic_control_v21":
        z_a, z_b, direction, meta = generator.semantic_control_v21(state)
        return z_a, z_b, direction, meta

    if strategy == "semantic_active_v21":
        z_a, z_b, direction, meta = generator.semantic_active_v21(state)
        return z_a, z_b, direction, meta

    if strategy == "candidate_pool_active":
        z_a, z_b, direction, meta = generator.candidate_pool_active(state)
        return z_a, z_b, direction, meta

    if strategy == "adaptive_router_v32":
        z_a, z_b, direction, meta = generator.adaptive_router_v32(state)
        return z_a, z_b, direction, meta

    if strategy == "hybrid":
        z_a, z_b, direction, meta = generator.hybrid(state)
        return z_a, z_b, direction, meta

    if strategy == "hybrid_v21":
        z_a, z_b, direction, meta = generator.hybrid_v21(state)
        return z_a, z_b, direction, meta

    if strategy == "hybrid_active_v21":
        z_a, z_b, direction, meta = generator.hybrid_active_v21(state)
        return z_a, z_b, direction, meta

    raise ValueError(f"Unknown pair strategy: {strategy}")


def run_personalization_session_v0(
    synthetic_user: SyntheticUser,
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
    init_std: float = 1.0,
    std_decay: float = 0.95,
    min_std: float = 0.15,
    clip_value: float | None = 2.0,
    pair_strategy: PairStrategy = "random",
    seed: int | None = None,
) -> SessionResult:
    """
    Run baseline A/B personalization loop in compact 8D space.

    This is an offline simulator: the synthetic user has a hidden z_target,
    chooses between A and B, and the state moves toward the chosen candidate.
    """
    rng = np.random.default_rng(seed)
    state = init_preference_state(dim=len(synthetic_user.z_target), init_std=init_std)
    pair_generator = PairGenerator(step_scale=step_scale, clip_value=clip_value, rng=rng)

    records: list[StepRecord] = []
    distances: list[float] = []

    for step in range(1, int(n_steps) + 1):
        z_a, z_b, _direction, pair_meta = generate_pair_by_strategy(
            pair_generator,
            state,
            pair_strategy,
        )

        choice, u_a, u_b = synthetic_user.choose(z_a, z_b)

        state = update_state_from_choice(
            state=state,
            z_a=z_a,
            z_b=z_b,
            choice=choice,
            lr=lr,
            std_decay=std_decay,
            min_std=min_std,
            clip_value=clip_value,
            pair_meta=pair_meta,
        )

        dist = distance_to_target(state.z_mean, synthetic_user.z_target)
        distances.append(dist)
        records.append(StepRecord(
            step=step,
            choice=choice,
            z_a=z_a.copy(),
            z_b=z_b.copy(),
            u_a=float(u_a),
            u_b=float(u_b),
            z_mean_after=state.z_mean.copy(),
            distance_to_target=dist,
            pair_strategy=pair_strategy,
            pair_meta=pair_meta,
        ))

    return SessionResult(
        final_state=state,
        records=records,
        distances=np.asarray(distances, dtype=np.float64),
    )


def run_many_sessions_v0(
    users: list[SyntheticUser],
    **kwargs,
) -> list[SessionResult]:
    """Run the same loop for multiple synthetic users."""
    results = []
    base_seed = kwargs.pop("seed", None)
    for i, user in enumerate(users):
        seed = None if base_seed is None else int(base_seed) + i
        results.append(run_personalization_session_v0(user, seed=seed, **kwargs))
    return results
