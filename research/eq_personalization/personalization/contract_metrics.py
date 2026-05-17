from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contract_space import contract_distance, contract_pair_distance


@dataclass
class CurveMetrics:
    max_abs_db: float
    max_boost_db: float
    max_cut_db: float
    mean_abs_db: float
    rms_db: float
    smoothness: float


def curve_metrics(curve: np.ndarray) -> CurveMetrics:
    y = np.asarray(curve, dtype=np.float64)
    diff2 = np.diff(y, n=2) if len(y) >= 3 else np.asarray([0.0])
    return CurveMetrics(
        max_abs_db=float(np.max(np.abs(y))),
        max_boost_db=float(np.max(y)),
        max_cut_db=float(-np.min(y)),
        mean_abs_db=float(np.mean(np.abs(y))),
        rms_db=float(np.sqrt(np.mean(y * y))),
        smoothness=float(np.sqrt(np.mean(diff2 * diff2))),
    )


def mapped_curve_metrics(z_contract: np.ndarray, mapper: Any) -> dict[str, float]:
    curve = mapper.map_one(z_contract)
    m = curve_metrics(curve)
    return {
        "mapped_max_abs_db": m.max_abs_db,
        "mapped_max_boost_db": m.max_boost_db,
        "mapped_max_cut_db": m.max_cut_db,
        "mapped_mean_abs_db": m.mean_abs_db,
        "mapped_rms_db": m.rms_db,
        "mapped_smoothness": m.smoothness,
    }


def mapped_pair_metrics(z_a: np.ndarray, z_b: np.ndarray, mapper: Any) -> dict[str, float]:
    curves = mapper.map_batch(np.stack([z_a, z_b], axis=0))
    ca, cb = curves[0], curves[1]
    diff = ca - cb
    out = {
        "pair_distance_z": contract_pair_distance(z_a, z_b),
        "pair_distance_db_rms": float(np.sqrt(np.mean(diff * diff))),
        "pair_distance_db_mae": float(np.mean(np.abs(diff))),
        "pair_distance_db_max_abs": float(np.max(np.abs(diff))),
    }
    ma, mb = curve_metrics(ca), curve_metrics(cb)
    out.update({
        "a_max_abs_db": ma.max_abs_db,
        "b_max_abs_db": mb.max_abs_db,
        "pair_max_abs_db": max(ma.max_abs_db, mb.max_abs_db),
        "pair_mean_abs_db": float((ma.mean_abs_db + mb.mean_abs_db) / 2.0),
    })
    return out


def mapped_distance_to_target(z: np.ndarray, z_target: np.ndarray, mapper: Any) -> dict[str, float]:
    curves = mapper.map_batch(np.stack([z, z_target], axis=0))
    diff = curves[0] - curves[1]
    return {
        "distance_to_target_z": contract_distance(z, z_target),
        "distance_to_target_db_rms": float(np.sqrt(np.mean(diff * diff))),
        "distance_to_target_db_mae": float(np.mean(np.abs(diff))),
        "distance_to_target_db_max_abs": float(np.max(np.abs(diff))),
    }


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {}
        for key, value in rec.items():
            if isinstance(value, np.ndarray):
                continue
            if isinstance(value, (list, tuple, dict)):
                continue
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_contract_sessions(results: list[Any]) -> pd.DataFrame:
    rows = []
    for result in results:
        final_record = result.records[-1] if result.records else {}
        row = {
            "strategy": result.strategy_name,
            "target_mode": result.target_mode,
            "used_steps": result.used_steps,
            "ready_step": result.ready_step,
            "final_status": result.final_status,
            "final_distance_z": float(result.final_distance_z),
            "final_distance_db_rms": float(result.final_distance_db_rms),
            "final_mapped_max_abs_db": float(final_record.get("state_mapped_max_abs_db", np.nan)),
            "final_mean_z_std": float(final_record.get("mean_z_std", np.nan)),
        }
        rows.append(row)
    return pd.DataFrame(rows)
