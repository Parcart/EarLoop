from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .contract_mapper import InterpretableContractMapper8D
from .contract_metrics import mapped_pair_metrics
from .contract_space import contract_pair_distance, clip_contract_z
from .pair_generator import PairGenerator, normalize
from .preference_model import LogisticDistancePreferenceModel
from .state import FEATURE_NAMES_8D, PreferenceState
from .contract_feedback import feedback_direction_for_label

PairSource = Literal["semantic", "candidate_pool", "direct"]


@dataclass
class ContractPairConfig:
    """Safety-aware pair generation settings for z-contract space."""

    step_scale: float = 0.48
    clip_value: float = 2.0
    max_pair_distance_z: float = 1.35
    min_pair_distance_z: float = 0.16
    max_pair_distance_db_rms: float = 7.5
    max_candidate_abs_db: float = 14.0
    max_shrink_steps: int = 8

    # Direct refinement after soft-stop / readiness marker.
    direct_min_step: float = 0.16
    direct_max_step: float = 0.46
    direct_step_fraction: float = 0.45
    direct_trust_mid_weight: float = 0.45
    direct_trust_presence_weight: float = 0.65
    direct_trust_high_weight: float = 0.90


class ContractPairGenerator:
    """Wrapper around the old pair generators with dB-aware safety.

    The old pair generators already know how to ask useful semantic questions.
    This class keeps their logic, but interprets all vectors as z_contract and
    shrinks proposals that would create overly aggressive mapped EQ curves.
    """

    def __init__(
        self,
        config: ContractPairConfig | None = None,
        mapper: Any | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = ContractPairConfig() if config is None else config
        self.rng = np.random.default_rng() if rng is None else rng
        self.mapper = InterpretableContractMapper8D() if mapper is None else mapper
        self.base = PairGenerator(
            step_scale=self.config.step_scale,
            clip_value=self.config.clip_value,
            rng=self.rng,
        )

    def _safety_ok(self, z_a: np.ndarray, z_b: np.ndarray) -> tuple[bool, dict]:
        metrics = mapped_pair_metrics(z_a, z_b, self.mapper)
        ok = (
            metrics["pair_distance_z"] <= self.config.max_pair_distance_z
            and metrics["pair_distance_z"] >= self.config.min_pair_distance_z
            and metrics["pair_distance_db_rms"] <= self.config.max_pair_distance_db_rms
            and metrics["pair_max_abs_db"] <= self.config.max_candidate_abs_db
        )
        return bool(ok), metrics

    def _shrink_pair(
        self,
        state: PreferenceState,
        z_a: np.ndarray,
        z_b: np.ndarray,
        meta: dict,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        z_center = np.asarray(state.z_mean, dtype=np.float64)
        z_a0 = np.asarray(z_a, dtype=np.float64)
        z_b0 = np.asarray(z_b, dtype=np.float64)
        meta = dict(meta)

        shrink = 1.0
        for _ in range(self.config.max_shrink_steps + 1):
            za = clip_contract_z(z_center + shrink * (z_a0 - z_center), self.config.clip_value)
            zb = clip_contract_z(z_center + shrink * (z_b0 - z_center), self.config.clip_value)
            ok, metrics = self._safety_ok(za, zb)
            if ok:
                meta.update(metrics)
                meta["safety_shrink"] = float(shrink)
                meta["safety_ok"] = True
                return za, zb, meta
            shrink *= 0.78

        meta.update(metrics)
        meta["safety_shrink"] = float(shrink)
        meta["safety_ok"] = False
        return za, zb, meta

    def semantic_active(self, state: PreferenceState) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        z_a, z_b, direction, meta = self.base.semantic_active_v21(state)
        meta = dict(meta)
        meta["source_group"] = "semantic_contract"
        meta["contract_mode"] = "semantic_active_v21"
        z_a, z_b, meta = self._shrink_pair(state, z_a, z_b, meta)
        return z_a, z_b, direction, meta

    def candidate_pool(self, state: PreferenceState) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        z_a, z_b, direction, meta = self.base.candidate_pool_active(state)
        meta = dict(meta)
        meta["source_group"] = "candidate_pool_contract"
        meta["contract_mode"] = "candidate_pool_active"
        z_a, z_b, meta = self._shrink_pair(state, z_a, z_b, meta)
        return z_a, z_b, direction, meta


    def axis_refinement(self, state: PreferenceState) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Ask a focused raw 8D axis question in contract space.

        This is useful in late refinement when semantic controls already gave a
        rough direction, but one uncertain axis still needs confirmation.
        """
        z_a, z_b, direction, axis = self.base.uncertainty_axis(state)
        meta = {
            "strategy": "contract_axis_refinement",
            "contract_mode": "uncertainty_axis",
            "source_group": "axis_contract",
            "axis": int(axis),
            "axis_name": FEATURE_NAMES_8D[int(axis)],
            "control_direction": direction.copy(),
        }
        z_a, z_b, meta = self._shrink_pair(state, z_a, z_b, meta)
        return z_a, z_b, direction, meta

    def _trust_mask(self, dim: int) -> np.ndarray:
        if dim != 8:
            return np.ones(dim, dtype=np.float64)
        cfg = self.config
        return np.asarray([
            1.00,
            1.00,
            cfg.direct_trust_mid_weight,
            cfg.direct_trust_mid_weight,
            cfg.direct_trust_presence_weight,
            cfg.direct_trust_high_weight,
            cfg.direct_trust_high_weight,
            cfg.direct_trust_high_weight,
        ], dtype=np.float64)


    def feedback_recovery(
        self,
        state: PreferenceState,
        feedback_label: str,
        strength: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Generate the first A/B pair after a user rejects both options.

        The pair compares the corrected anchor against a slightly stronger move in
        the same feedback direction. This lets the system ask: "was that correction
        enough, or should we continue moving away from the rejected region?"
        """
        z_current = np.asarray(state.z_mean, dtype=np.float64)
        direction_raw = feedback_direction_for_label(feedback_label, z_reference=z_current)
        norm = float(np.linalg.norm(direction_raw))
        if norm <= 1e-8:
            return self.semantic_active(state)

        direction = direction_raw / (norm + 1e-8)
        # Feedback recovery should be audible even in late refinement, but still
        # safer than a full semantic exploration step.
        step = float(np.clip(0.22 + 0.10 * float(strength), 0.22, 0.42))
        z_anchor = z_current.copy()
        z_probe = clip_contract_z(z_current + step * direction, self.config.clip_value)

        if self.rng.random() < 0.5:
            z_a, z_b = z_anchor, z_probe
            probe_role = "B"
        else:
            z_a, z_b = z_probe, z_anchor
            probe_role = "A"

        meta = {
            "strategy": "contract_feedback_recovery",
            "contract_mode": "feedback_recovery",
            "source_group": "feedback_recovery_contract",
            "feedback_label": feedback_label,
            "feedback_strength": float(strength),
            "feedback_probe_role": probe_role,
            "feedback_direction_norm": norm,
            "feedback_probe_step": step,
            "control_direction": direction.copy(),
        }
        z_a, z_b, meta = self._shrink_pair(state, z_a, z_b, meta)
        return z_a, z_b, direction, meta

    def direct_refinement(
        self,
        state: PreferenceState,
        model: LogisticDistancePreferenceModel,
        mode: Literal["blend", "pm", "trust"] = "trust",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        z_current = np.asarray(state.z_mean, dtype=np.float64)
        z_pm = clip_contract_z(np.asarray(model.z_pref, dtype=np.float64), self.config.clip_value)
        # Conservative blend: online model proposes direction, state anchors it.
        z_blend = clip_contract_z(0.55 * z_current + 0.45 * z_pm, self.config.clip_value)

        if mode == "pm":
            target = z_pm
            source_group = "direct_pm_contract"
        else:
            target = z_blend
            source_group = "direct_blend_contract"

        delta = np.asarray(target - z_current, dtype=np.float64)
        if mode == "trust":
            mask = self._trust_mask(len(delta))
            std = np.asarray(state.z_std, dtype=np.float64)
            std_boost = 0.75 + 0.25 * (std / (float(np.max(std)) + 1e-8))
            delta = delta * mask * std_boost
            source_group = "direct_trust_contract"

        norm = float(np.linalg.norm(delta))
        if norm <= 1e-8:
            # Explicit fallback; caller can still treat this as semantic.
            return self.semantic_active(state)

        direction = delta / (norm + 1e-8)
        step = float(np.clip(norm * self.config.direct_step_fraction, self.config.direct_min_step, self.config.direct_max_step))
        z_anchor = z_current.copy()
        z_direct = clip_contract_z(z_current + step * direction, self.config.clip_value)

        # Randomize A/B order.
        if self.rng.random() < 0.5:
            z_a, z_b = z_anchor, z_direct
            direct_role = "B"
        else:
            z_a, z_b = z_direct, z_anchor
            direct_role = "A"

        meta = {
            "strategy": "contract_direct_refinement",
            "contract_mode": f"direct_{mode}",
            "source_group": source_group,
            "selected_direct_role": direct_role,
            "raw_delta_norm": norm,
            "direct_step": step,
            "target_pm_norm": float(np.linalg.norm(z_pm)),
            "target_blend_norm": float(np.linalg.norm(z_blend)),
            "control_direction": direction.copy(),
        }
        z_a, z_b, meta = self._shrink_pair(state, z_a, z_b, meta)
        return z_a, z_b, direction, meta
