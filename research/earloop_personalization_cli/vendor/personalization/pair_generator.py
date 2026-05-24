from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .control_basis import (
    CONTROL_BASIS_4D_TO_8D,
    CONTROL_BASIS_6D_TO_8D,
    CONTROL_NAMES_4D,
    CONTROL_NAMES_6D,
    control_probabilities,
)
from .state import FEATURE_NAMES_8D, PreferenceState, clip_vector


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a unit-norm vector."""
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + eps)


@dataclass
class PairGenerator:
    """
    A/B pair generator around the current preference state.

    Strategies:
    - random_symmetric: random multidimensional exploration direction;
    - uncertainty_axis: one 8D axis chosen by current uncertainty z_std;
    - semantic_control: v2 semantic 4D control direction mapped into 8D;
    - semantic_control_v21: v2.1 extended 6D semantic control direction;
    - semantic_active_v21: V3 active selector over v2.1 semantic directions;
    - candidate_pool_active: V3.1 active selector over a mixed pool of random, axis and semantic questions;
    - adaptive_router_v32: V3.2 lightweight router between semantic-active, candidate-pool and uncertainty-axis;
    - hybrid: mixture of random, v2 semantic and axis questions;
    - hybrid_v21: mixture of random, v2.1 semantic and axis questions;
    - hybrid_active_v21: mixture of random, V3 semantic-active and axis questions.

    Important: semantic_active_v21 does not predict what the user will like.
    It scores question usefulness: uncertainty coverage, diversity, safety and
    repetition. adaptive_router_v32 is still heuristic routing, not a learned
    Preference Model. A full Preference Model should be added as a later V4 layer.
    """

    step_scale: float = 0.6
    clip_value: float | None = 2.0
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng()

    def _make_pair(
        self,
        state: PreferenceState,
        direction: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        direction = normalize(direction)

        # Randomly flip so A is not always the positive direction.
        if self.rng.random() < 0.5:
            direction = -direction

        z_a = clip_vector(state.z_mean + float(scale) * direction, self.clip_value)
        z_b = clip_vector(state.z_mean - float(scale) * direction, self.clip_value)
        return z_a, z_b, direction

    def _safety_penalty(self, state: PreferenceState, direction: np.ndarray, scale: float) -> float:
        """Penalty for proposed candidates outside compact-space safety bounds."""
        if self.clip_value is None:
            return 0.0
        direction = normalize(direction)
        proposed_a = state.z_mean + float(scale) * direction
        proposed_b = state.z_mean - float(scale) * direction
        overflow_a = np.maximum(np.abs(proposed_a) - float(self.clip_value), 0.0)
        overflow_b = np.maximum(np.abs(proposed_b) - float(self.clip_value), 0.0)
        return float(np.mean(overflow_a * overflow_a + overflow_b * overflow_b))

    def _repetition_penalty(self, state: PreferenceState, control_name: str, lookback: int = 3) -> float:
        """Penalty for asking too many recent questions about the same semantic control."""
        if not state.history:
            return 0.0
        recent = state.history[-int(lookback):]
        if not recent:
            return 0.0
        count = 0
        for item in recent:
            meta = item.get("pair_meta") or {}
            if meta.get("control_name") == control_name:
                count += 1
        return float(count / len(recent))

    def random_symmetric(self, state: PreferenceState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a symmetric pair along a random 8D direction."""
        dim = len(state.z_mean)
        direction = normalize(self.rng.normal(size=dim))

        # Make the question wider on uncertain dimensions.
        scaled_direction = normalize(direction * state.z_std)
        z_a, z_b, signed_direction = self._make_pair(
            state=state,
            direction=scaled_direction,
            scale=self.step_scale,
        )
        return z_a, z_b, signed_direction

    def uncertainty_axis(self, state: PreferenceState) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Generate a pair along one weighted 8D axis.

        The axis is sampled with probability proportional to z_std. Higher
        uncertainty means a higher probability of asking about that feature.
        """
        dim = len(state.z_mean)
        std = np.asarray(state.z_std, dtype=np.float64)
        probs = std / (std.sum() + 1e-8)
        axis = int(self.rng.choice(dim, p=probs))

        direction = np.zeros(dim, dtype=np.float64)
        direction[axis] = 1.0
        scale = self.step_scale * float(state.z_std[axis])

        z_a, z_b, signed_direction = self._make_pair(
            state=state,
            direction=direction,
            scale=scale,
        )
        return z_a, z_b, signed_direction, axis

    def _semantic_control_from_basis(
        self,
        state: PreferenceState,
        basis: np.ndarray,
        control_names: list[str],
        strategy_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Generate a pair along one semantic control basis direction."""
        probs = control_probabilities(state.z_std, basis=basis)
        control_index = int(self.rng.choice(len(control_names), p=probs))
        control_name = control_names[control_index]

        direction = np.asarray(basis[control_index], dtype=np.float64).copy()

        # Wider semantic questions when the touched axes are uncertain.
        uncertainty_scale = float(
            np.sum(np.abs(direction) * state.z_std) / (np.sum(np.abs(direction)) + 1e-8)
        )
        scale = self.step_scale * max(uncertainty_scale, 0.15)

        z_a, z_b, signed_direction = self._make_pair(
            state=state,
            direction=direction,
            scale=scale,
        )

        meta = {
            "strategy": strategy_name,
            "control_index": control_index,
            "control_name": control_name,
            "control_direction": signed_direction.copy(),
            "control_probabilities": probs.copy(),
            "basis_dim": int(len(control_names)),
            "selection_method": "probability_by_uncertainty",
        }
        return z_a, z_b, signed_direction, meta

    def _semantic_active_from_basis(
        self,
        state: PreferenceState,
        basis: np.ndarray,
        control_names: list[str],
        strategy_name: str,
        scales: tuple[float, ...] = (0.40, 0.60, 0.85),
        uncertainty_weight: float = 1.00,
        diversity_weight: float = 0.15,
        safety_weight: float = 1.00,
        repetition_weight: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        V3 active semantic pair selection.

        This function generates several semantic A/B questions and selects the
        one with the highest question-usefulness score. The score is heuristic:
        it uses uncertainty coverage, pair diversity, safety and repetition.
        It is not a Preference Model and does not predict the user's choice.
        """
        basis = np.asarray(basis, dtype=np.float64)
        z_std = np.asarray(state.z_std, dtype=np.float64)

        candidates: list[dict] = []
        max_scale = max(float(s) for s in scales)

        for control_index, control_name in enumerate(control_names):
            direction = normalize(basis[control_index])
            touched = np.abs(direction)

            # How much this semantic direction probes uncertain 8D features.
            uncertainty_score = float(np.sum(touched * z_std) / (np.sum(touched) + 1e-8))

            for scale_mult in scales:
                scale = self.step_scale * float(scale_mult)
                diversity_score = float(scale / (self.step_scale * max_scale + 1e-8))
                safety_penalty = self._safety_penalty(state, direction, scale)
                repetition_penalty = self._repetition_penalty(state, control_name)

                total_score = (
                    uncertainty_weight * uncertainty_score
                    + diversity_weight * diversity_score
                    - safety_weight * safety_penalty
                    - repetition_weight * repetition_penalty
                )

                candidates.append({
                    "score": float(total_score),
                    "control_index": int(control_index),
                    "control_name": control_name,
                    "direction": direction.copy(),
                    "scale": float(scale),
                    "scale_multiplier": float(scale_mult),
                    "uncertainty_score": float(uncertainty_score),
                    "diversity_score": float(diversity_score),
                    "safety_penalty": float(safety_penalty),
                    "repetition_penalty": float(repetition_penalty),
                })

        best = max(candidates, key=lambda item: item["score"])
        z_a, z_b, signed_direction = self._make_pair(
            state=state,
            direction=best["direction"],
            scale=best["scale"],
        )

        # Keep compact metadata in records; full candidates would make CSVs too noisy.
        top_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:5]
        top_candidates_meta = [
            {
                "control_name": item["control_name"],
                "scale_multiplier": item["scale_multiplier"],
                "score": item["score"],
                "uncertainty_score": item["uncertainty_score"],
                "diversity_score": item["diversity_score"],
                "safety_penalty": item["safety_penalty"],
                "repetition_penalty": item["repetition_penalty"],
            }
            for item in top_candidates
        ]

        meta = {
            "strategy": strategy_name,
            "sub_strategy": "semantic_active",
            "selection_method": "active_question_score",
            "control_index": best["control_index"],
            "control_name": best["control_name"],
            "control_direction": signed_direction.copy(),
            "basis_dim": int(len(control_names)),
            "scale": best["scale"],
            "scale_multiplier": best["scale_multiplier"],
            "score": best["score"],
            "uncertainty_score": best["uncertainty_score"],
            "diversity_score": best["diversity_score"],
            "safety_penalty": best["safety_penalty"],
            "repetition_penalty": best["repetition_penalty"],
            "top_candidates": top_candidates_meta,
        }
        return z_a, z_b, signed_direction, meta

    def semantic_control(self, state: PreferenceState) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Generate a pair along one original v2 4D semantic control direction."""
        return self._semantic_control_from_basis(
            state=state,
            basis=CONTROL_BASIS_4D_TO_8D,
            control_names=CONTROL_NAMES_4D,
            strategy_name="semantic_control",
        )

    def semantic_control_v21(self, state: PreferenceState) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Generate a pair along one extended v2.1 6D semantic control direction."""
        return self._semantic_control_from_basis(
            state=state,
            basis=CONTROL_BASIS_6D_TO_8D,
            control_names=CONTROL_NAMES_6D,
            strategy_name="semantic_control_v21",
        )

    def semantic_active_v21(self, state: PreferenceState) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """V3: actively select the most useful semantic question from v2.1 6D basis."""
        return self._semantic_active_from_basis(
            state=state,
            basis=CONTROL_BASIS_6D_TO_8D,
            control_names=CONTROL_NAMES_6D,
            strategy_name="semantic_active_v21",
        )

    def candidate_pool_active(
        self,
        state: PreferenceState,
        n_random: int = 18,
        semantic_scales: tuple[float, ...] = (0.35, 0.60, 0.85, 1.10),
        axis_scales: tuple[float, ...] = (0.45, 0.75),
        random_scales: tuple[float, ...] = (0.55, 0.85),
        uncertainty_weight: float = 1.00,
        diversity_weight: float = 0.18,
        semantic_bonus_weight: float = 0.12,
        axis_bonus_weight: float = 0.04,
        random_bonus_weight: float = 0.06,
        safety_weight: float = 1.00,
        repetition_weight: float = 0.25,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        V3.1: active selection from a mixed candidate pool.

        The strategy generates many possible A/B questions from several sources:
        - extended semantic basis v2.1 directions;
        - one-axis uncertainty diagnostic questions;
        - random multidimensional exploration directions.

        It then chooses the question with the highest question-usefulness score.
        This is still not a Preference Model: it does not predict whether A or B
        will be preferred. It only selects a useful next question based on
        uncertainty coverage, diversity, semantic structure, safety and history.
        """
        z_std = np.asarray(state.z_std, dtype=np.float64)
        candidates: list[dict] = []

        def add_candidate(
            source: str,
            direction: np.ndarray,
            scale: float,
            label: str,
            control_index: int | None = None,
            control_name: str | None = None,
            axis: int | None = None,
            axis_name: str | None = None,
            basis_dim: int | None = None,
            scale_multiplier: float | None = None,
        ) -> None:
            d = normalize(direction)
            touched = np.abs(d)
            uncertainty_score = float(np.sum(touched * z_std) / (np.sum(touched) + 1e-8))
            diversity_score = float(scale / (self.step_scale * 1.10 + 1e-8))
            safety_penalty = self._safety_penalty(state, d, scale)

            # Penalize recently repeated explicit semantic controls or axis labels.
            if control_name is not None:
                repetition_label = control_name
            elif axis_name is not None:
                repetition_label = f"axis:{axis_name}"
            else:
                repetition_label = None

            repetition_penalty = 0.0
            if repetition_label is not None and state.history:
                recent = state.history[-3:]
                matches = 0
                for item in recent:
                    meta = item.get("pair_meta") or {}
                    recent_label = meta.get("control_name") or (
                        f"axis:{meta.get('axis_name')}" if meta.get("axis_name") is not None else None
                    )
                    if recent_label == repetition_label:
                        matches += 1
                repetition_penalty = float(matches / max(len(recent), 1))

            semantic_bonus = 1.0 if source == "semantic_v21" else 0.0
            axis_bonus = 1.0 if source == "axis" else 0.0
            random_bonus = 1.0 if source == "random" else 0.0

            score = (
                uncertainty_weight * uncertainty_score
                + diversity_weight * diversity_score
                + semantic_bonus_weight * semantic_bonus
                + axis_bonus_weight * axis_bonus
                + random_bonus_weight * random_bonus
                - safety_weight * safety_penalty
                - repetition_weight * repetition_penalty
            )

            candidates.append({
                "score": float(score),
                "source": source,
                "label": label,
                "direction": d.copy(),
                "scale": float(scale),
                "scale_multiplier": None if scale_multiplier is None else float(scale_multiplier),
                "uncertainty_score": float(uncertainty_score),
                "diversity_score": float(diversity_score),
                "semantic_bonus": float(semantic_bonus),
                "axis_bonus": float(axis_bonus),
                "random_bonus": float(random_bonus),
                "safety_penalty": float(safety_penalty),
                "repetition_penalty": float(repetition_penalty),
                "control_index": control_index,
                "control_name": control_name,
                "axis": axis,
                "axis_name": axis_name,
                "basis_dim": basis_dim,
            })

        # 1) Semantic v2.1 candidates.
        for control_index, control_name in enumerate(CONTROL_NAMES_6D):
            direction = normalize(CONTROL_BASIS_6D_TO_8D[control_index])
            # Make the scale mildly uncertainty-aware for that direction.
            touched = np.abs(direction)
            uncertainty_scale = float(np.sum(touched * z_std) / (np.sum(touched) + 1e-8))
            for scale_mult in semantic_scales:
                scale = self.step_scale * float(scale_mult) * max(uncertainty_scale, 0.30)
                add_candidate(
                    source="semantic_v21",
                    direction=direction,
                    scale=scale,
                    label=control_name,
                    control_index=int(control_index),
                    control_name=control_name,
                    basis_dim=len(CONTROL_NAMES_6D),
                    scale_multiplier=float(scale_mult),
                )

        # 2) Axis candidates.
        for axis, axis_name in enumerate(FEATURE_NAMES_8D):
            direction = np.zeros(len(state.z_mean), dtype=np.float64)
            direction[axis] = 1.0
            for scale_mult in axis_scales:
                scale = self.step_scale * float(scale_mult) * max(float(z_std[axis]), 0.30)
                add_candidate(
                    source="axis",
                    direction=direction,
                    scale=scale,
                    label=f"axis:{axis_name}",
                    axis=int(axis),
                    axis_name=axis_name,
                    scale_multiplier=float(scale_mult),
                )

        # 3) Random multidimensional candidates.
        for i in range(int(n_random)):
            direction = normalize(self.rng.normal(size=len(state.z_mean)) * z_std)
            for scale_mult in random_scales:
                scale = self.step_scale * float(scale_mult)
                add_candidate(
                    source="random",
                    direction=direction,
                    scale=scale,
                    label=f"random:{i}",
                    scale_multiplier=float(scale_mult),
                )

        best = max(candidates, key=lambda item: item["score"])
        z_a, z_b, signed_direction = self._make_pair(
            state=state,
            direction=best["direction"],
            scale=best["scale"],
        )

        top_candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)[:8]
        top_candidates_meta = [
            {
                "source": item["source"],
                "label": item["label"],
                "score": item["score"],
                "scale_multiplier": item["scale_multiplier"],
                "uncertainty_score": item["uncertainty_score"],
                "diversity_score": item["diversity_score"],
                "safety_penalty": item["safety_penalty"],
                "repetition_penalty": item["repetition_penalty"],
            }
            for item in top_candidates
        ]

        meta = {
            "strategy": "candidate_pool_active",
            "sub_strategy": best["source"],
            "selection_method": "mixed_candidate_pool_question_score",
            "axis": best["axis"],
            "axis_name": best["axis_name"],
            "control_index": best["control_index"],
            "control_name": best["control_name"],
            "control_direction": signed_direction.copy(),
            "basis_dim": best["basis_dim"],
            "scale": best["scale"],
            "scale_multiplier": best["scale_multiplier"],
            "score": best["score"],
            "uncertainty_score": best["uncertainty_score"],
            "diversity_score": best["diversity_score"],
            "semantic_bonus": best["semantic_bonus"],
            "axis_bonus": best["axis_bonus"],
            "random_bonus": best["random_bonus"],
            "safety_penalty": best["safety_penalty"],
            "repetition_penalty": best["repetition_penalty"],
            "candidate_count": int(len(candidates)),
            "top_candidates": top_candidates_meta,
        }
        return z_a, z_b, signed_direction, meta


    def _semantic_subspace_fit(self, vector: np.ndarray, basis: np.ndarray | None = None) -> float:
        """
        Estimate how well a vector lies in the span of the semantic basis.

        1.0 means the vector is almost fully represented by semantic controls;
        values closer to 0 mean the vector is less semantic-basis-like.
        This is a routing heuristic, not a user-preference prediction.
        """
        basis = CONTROL_BASIS_6D_TO_8D if basis is None else np.asarray(basis, dtype=np.float64)
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            return 0.0

        # Project vector to the row span of the basis using least squares.
        # basis: [controls, features], so basis.T maps control coefficients to 8D.
        coeffs, *_ = np.linalg.lstsq(basis.T, vector, rcond=None)
        projected = basis.T @ coeffs
        return float(np.linalg.norm(projected) / (norm + 1e-8))

    def _recent_displacement(self, state: PreferenceState, lookback: int = 4) -> np.ndarray:
        """Return recent movement of z_mean over the last few updates."""
        if not state.history:
            return np.zeros_like(state.z_mean)
        recent = state.history[-int(lookback):]
        if not recent:
            return np.zeros_like(state.z_mean)
        start = np.asarray(recent[0].get("z_mean_after", state.z_mean), dtype=np.float64)
        end = np.asarray(recent[-1].get("z_mean_after", state.z_mean), dtype=np.float64)
        return end - start

    def _axis_concentration(self, state: PreferenceState) -> float:
        """Return how concentrated uncertainty is on a single raw 8D axis."""
        z_std = np.asarray(state.z_std, dtype=np.float64)
        return float(np.max(z_std) / (np.sum(z_std) + 1e-8))

    def adaptive_router_v32(
        self,
        state: PreferenceState,
        warmup_steps: int = 4,
        semantic_fit_threshold: float = 0.68,
        recent_fit_threshold: float = 0.72,
        axis_concentration_threshold: float = 0.22,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        V3.2: lightweight adaptive router over existing Pair Generator strategies.

        The router chooses which question-generator to use on the current step:
        - early warmup uses candidate_pool_active for broad mixed exploration;
        - if current movement/state looks semantic-basis-like, use semantic_active_v21;
        - if uncertainty collapses onto one raw feature, use uncertainty_axis;
        - otherwise use candidate_pool_active as a universal fallback.

        This is still Pair Generation logic. It does not predict the user's
        preference and does not use a learned Preference Model.
        """
        fit_mean = self._semantic_subspace_fit(state.z_mean)
        recent_delta = self._recent_displacement(state, lookback=4)
        fit_recent = self._semantic_subspace_fit(recent_delta)
        axis_conc = self._axis_concentration(state)
        mean_std = float(np.mean(state.z_std))
        max_std = float(np.max(state.z_std))

        if state.step < int(warmup_steps):
            routed_strategy = "candidate_pool_active"
            route_reason = "warmup_mixed_exploration"
        elif fit_mean >= semantic_fit_threshold or fit_recent >= recent_fit_threshold:
            routed_strategy = "semantic_active_v21"
            route_reason = "semantic_subspace_fit"
        elif axis_conc >= axis_concentration_threshold:
            routed_strategy = "uncertainty_axis"
            route_reason = "axis_uncertainty_concentration"
        else:
            routed_strategy = "candidate_pool_active"
            route_reason = "mixed_pool_fallback"

        if routed_strategy == "semantic_active_v21":
            z_a, z_b, direction, meta = self.semantic_active_v21(state)
        elif routed_strategy == "uncertainty_axis":
            z_a, z_b, direction, axis = self.uncertainty_axis(state)
            meta = {
                "strategy": "uncertainty_axis",
                "axis": axis,
                "axis_name": FEATURE_NAMES_8D[axis],
                "control_index": None,
                "control_name": None,
            }
        elif routed_strategy == "candidate_pool_active":
            z_a, z_b, direction, meta = self.candidate_pool_active(state)
        else:
            raise RuntimeError(f"Unexpected routed strategy: {routed_strategy}")

        meta = dict(meta)
        meta["strategy"] = "adaptive_router_v32"
        meta["routed_strategy"] = routed_strategy
        meta["sub_strategy"] = routed_strategy
        meta["route_reason"] = route_reason
        meta["route_scores"] = {
            "semantic_fit_mean": float(fit_mean),
            "semantic_fit_recent": float(fit_recent),
            "axis_concentration": float(axis_conc),
            "mean_std": float(mean_std),
            "max_std": float(max_std),
            "warmup_steps": int(warmup_steps),
        }
        return z_a, z_b, direction, meta

    def hybrid(
        self,
        state: PreferenceState,
        random_prob: float = 0.60,
        semantic_prob: float = 0.30,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Generate a pair using v2 mixture of strategies.

        Default mixture:
        - 60% random multidimensional exploration;
        - 30% v2 4D semantic control direction;
        - 10% uncertainty axis diagnostic question.
        """
        r = float(self.rng.random())

        if r < random_prob:
            z_a, z_b, direction = self.random_symmetric(state)
            return z_a, z_b, direction, {
                "strategy": "hybrid",
                "sub_strategy": "random",
                "axis": None,
                "axis_name": None,
                "control_index": None,
                "control_name": None,
                "basis_dim": None,
            }

        if r < random_prob + semantic_prob:
            z_a, z_b, direction, meta = self.semantic_control(state)
            meta = dict(meta)
            meta["strategy"] = "hybrid"
            meta["sub_strategy"] = "semantic_control"
            return z_a, z_b, direction, meta

        z_a, z_b, direction, axis = self.uncertainty_axis(state)
        return z_a, z_b, direction, {
            "strategy": "hybrid",
            "sub_strategy": "uncertainty_axis",
            "axis": axis,
            "axis_name": FEATURE_NAMES_8D[axis],
            "control_index": None,
            "control_name": None,
            "basis_dim": None,
        }

    def hybrid_v21(
        self,
        state: PreferenceState,
        random_prob: float = 0.55,
        semantic_prob: float = 0.35,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Generate a pair using v2.1 mixture of strategies.

        Default mixture:
        - 55% random multidimensional exploration;
        - 35% v2.1 6D semantic control direction;
        - 10% uncertainty axis diagnostic question.
        """
        r = float(self.rng.random())

        if r < random_prob:
            z_a, z_b, direction = self.random_symmetric(state)
            return z_a, z_b, direction, {
                "strategy": "hybrid_v21",
                "sub_strategy": "random",
                "axis": None,
                "axis_name": None,
                "control_index": None,
                "control_name": None,
                "basis_dim": None,
            }

        if r < random_prob + semantic_prob:
            z_a, z_b, direction, meta = self.semantic_control_v21(state)
            meta = dict(meta)
            meta["strategy"] = "hybrid_v21"
            meta["sub_strategy"] = "semantic_control_v21"
            return z_a, z_b, direction, meta

        z_a, z_b, direction, axis = self.uncertainty_axis(state)
        return z_a, z_b, direction, {
            "strategy": "hybrid_v21",
            "sub_strategy": "uncertainty_axis",
            "axis": axis,
            "axis_name": FEATURE_NAMES_8D[axis],
            "control_index": None,
            "control_name": None,
            "basis_dim": None,
        }

    def hybrid_active_v21(
        self,
        state: PreferenceState,
        random_prob: float = 0.45,
        semantic_active_prob: float = 0.45,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Generate a pair using V3 mixture of strategies.

        Default mixture:
        - 45% random multidimensional exploration;
        - 45% V3 semantic-active selection;
        - 10% uncertainty axis diagnostic question.
        """
        r = float(self.rng.random())

        if r < random_prob:
            z_a, z_b, direction = self.random_symmetric(state)
            return z_a, z_b, direction, {
                "strategy": "hybrid_active_v21",
                "sub_strategy": "random",
                "axis": None,
                "axis_name": None,
                "control_index": None,
                "control_name": None,
                "basis_dim": None,
            }

        if r < random_prob + semantic_active_prob:
            z_a, z_b, direction, meta = self.semantic_active_v21(state)
            meta = dict(meta)
            meta["strategy"] = "hybrid_active_v21"
            meta["sub_strategy"] = "semantic_active_v21"
            return z_a, z_b, direction, meta

        z_a, z_b, direction, axis = self.uncertainty_axis(state)
        return z_a, z_b, direction, {
            "strategy": "hybrid_active_v21",
            "sub_strategy": "uncertainty_axis",
            "axis": axis,
            "axis_name": FEATURE_NAMES_8D[axis],
            "control_index": None,
            "control_name": None,
            "basis_dim": None,
        }
