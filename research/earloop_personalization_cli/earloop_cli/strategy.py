from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


StrategyName = Literal[
    "semantic_contract_v6",
    "phase_mixed_contract_v6",
    "direct_contract_v6",
    "phase_aware_feedback",
]


@dataclass(frozen=True)
class StrategyPreset:
    name: str
    label: str
    description: str
    contract_strategy: str
    manual_feedback_enabled: bool = True
    phase_aware_feedback: bool = False
    default_budget: int = 24
    direct_mode: str = "trust"


STRATEGIES: dict[str, StrategyPreset] = {
    "semantic_contract_v6": StrategyPreset(
        name="semantic_contract_v6",
        label="Semantic contract v6",
        description="Стабильный semantic A/B backbone без позднего phase routing.",
        contract_strategy="semantic_contract_v6",
        manual_feedback_enabled=True,
        phase_aware_feedback=False,
        default_budget=24,
    ),
    "phase_mixed_contract_v6": StrategyPreset(
        name="phase_mixed_contract_v6",
        label="Phase mixed contract v6",
        description="Семантический backbone + поздние direct/axis/candidate probes после readiness marker.",
        contract_strategy="phase_mixed_contract_v6",
        manual_feedback_enabled=True,
        phase_aware_feedback=False,
        default_budget=24,
    ),
    "direct_contract_v6": StrategyPreset(
        name="direct_contract_v6",
        label="Direct contract v6",
        description="Более прямое уточнение preference state через online model.",
        contract_strategy="direct_contract_v6",
        manual_feedback_enabled=True,
        phase_aware_feedback=False,
        default_budget=24,
    ),
    "phase_aware_feedback": StrategyPreset(
        name="phase_aware_feedback",
        label="Phase-aware Directional Feedback",
        description="Оптимальный режим по умолчанию: phase_mixed contour + ручной Directional Feedback + late refinement after soft-stop.",
        contract_strategy="phase_mixed_contract_v6",
        manual_feedback_enabled=True,
        phase_aware_feedback=True,
        default_budget=24,
    ),
}

DEFAULT_STRATEGY = "phase_aware_feedback"


def get_strategy(name: str | None) -> StrategyPreset:
    if not name:
        name = DEFAULT_STRATEGY
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {', '.join(STRATEGIES)}")
    return STRATEGIES[name]


def strategy_help() -> str:
    rows = []
    for key, preset in STRATEGIES.items():
        mark = "*" if key == DEFAULT_STRATEGY else " "
        rows.append(f"{mark} {key}: {preset.description}")
    return "\n".join(rows)
