from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .vendor_path import ensure_vendor_path
ensure_vendor_path()
from personalization.contract_mapper import InterpretableContractMapper8D
from personalization.contract_metrics import summarize_contract_sessions
from personalization.contract_session import ContractSessionConfig, run_contract_personalization_session, contract_summary_dataframe, contract_records_dataframe
from personalization.contract_space import CONTRACT_ARCHETYPES_8D
from personalization.synthetic_user import SyntheticUser, make_random_synthetic_user


EVAL_STRATEGY_PRESETS = {
    "semantic_contract_v6": dict(strategy="semantic_contract_v6", enable_feedback=False),
    "phase_mixed_contract_v6": dict(strategy="phase_mixed_contract_v6", enable_feedback=False),
    "direct_contract_v6": dict(strategy="direct_contract_v6", enable_feedback=False),
    "phase_aware_feedback": dict(
        strategy="phase_mixed_contract_v6",
        experiment_label="phase_aware_feedback",
        enable_feedback=True,
        feedback_policy="cosine",
        feedback_phase_aware=True,
        feedback_max_probability=0.34,
        feedback_base_probability=0.015,
        feedback_cosine_threshold=0.42,
    ),
}


def make_eval_users(n_users: int, seed: int = 42, noise_std: float = 0.03) -> list[tuple[str, SyntheticUser]]:
    rng = np.random.default_rng(seed)
    users: list[tuple[str, SyntheticUser]] = []
    archetypes = list(CONTRACT_ARCHETYPES_8D.items())
    for name, z in archetypes:
        users.append((name, SyntheticUser(z_target=z.copy(), noise_std=noise_std, seed=int(rng.integers(0, 1_000_000)))))
    while len(users) < int(n_users):
        u = make_random_synthetic_user(dim=8, target_scale=0.95, noise_std=noise_std, seed=int(rng.integers(0, 1_000_000)))
        users.append(("random", u))
    return users[: int(n_users)]


def evaluate_strategies(
    *,
    out_dir: Path,
    strategies: list[str],
    n_users: int,
    budget: int,
    seed: int,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapper = InterpretableContractMapper8D()
    users = make_eval_users(n_users=n_users, seed=seed)
    all_results = []
    for strategy in strategies:
        if strategy not in EVAL_STRATEGY_PRESETS:
            raise ValueError(f"Unknown eval strategy {strategy}")
        preset = EVAL_STRATEGY_PRESETS[strategy]
        for user_idx, (target_mode, user) in enumerate(users):
            cfg = ContractSessionConfig(
                n_steps=budget,
                seed=seed + user_idx * 1009 + abs(hash(strategy)) % 997,
                **preset,
            )
            result = run_contract_personalization_session(
                synthetic_user=user,
                target_mode=target_mode,
                config=cfg,
                mapper=mapper,
            )
            all_results.append(result)

    summary = contract_summary_dataframe(all_results)
    records = contract_records_dataframe(all_results)
    summary.to_csv(out_dir / "session_summary.csv", index=False)
    records.to_csv(out_dir / "session_steps.csv", index=False)

    agg = summary.groupby("strategy").agg(
        n_sessions=("strategy", "size"),
        final_distance_z_mean=("final_distance_z", "mean"),
        final_distance_z_median=("final_distance_z", "median"),
        final_distance_db_rms_mean=("final_distance_db_rms", "mean"),
        final_distance_db_rms_median=("final_distance_db_rms", "median"),
        ready_rate=("ready_step", lambda s: float(s.notna().mean())),
        ready_step_mean=("ready_step", "mean"),
        feedback_count_mean=("feedback_count", "mean"),
        feedback_rate_mean=("feedback_rate", "mean"),
        final_mapped_max_abs_db_mean=("final_mapped_max_abs_db", "mean"),
        final_mean_z_std_mean=("final_mean_z_std", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / "strategy_comparison.csv", index=False)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "strategies": strategies,
        "n_users": n_users,
        "budget": budget,
        "seed": seed,
    }
    (out_dir / "eval_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir / "strategy_comparison.csv"
