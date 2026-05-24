from __future__ import annotations

import csv
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).round(8).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class SessionLogger:
    def __init__(self, data_root: Path, strategy: str) -> None:
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.root = Path(data_root) / "sessions" / self.session_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        self.events_path = self.root / "events.jsonl"
        self.steps_jsonl_path = self.root / "steps.jsonl"
        self.vectors_csv_path = self.root / "step_vectors.csv"
        self.eq_csv_path = self.root / "eq_curves.csv"
        self.model_jsonl_path = self.root / "model_state.jsonl"
        self.summary_path = self.root / "session_summary.json"
        self._step_rows: list[dict[str, Any]] = []
        self._vector_rows: list[dict[str, Any]] = []
        self._eq_rows: list[dict[str, Any]] = []

    def write_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")

    def event(self, event: str, **payload: Any) -> None:
        self.write_jsonl(self.events_path, {"ts": now_iso(), "event": event, **payload})

    def log_step(self, step: dict[str, Any]) -> None:
        step = {"ts": now_iso(), **step}
        self.write_jsonl(self.steps_jsonl_path, step)
        flat = {}
        for k, v in step.items():
            if isinstance(v, np.ndarray) or isinstance(v, (list, tuple, dict)):
                continue
            flat[k] = json_safe(v)
        self._step_rows.append(flat)

    def log_vectors(self, row: dict[str, Any]) -> None:
        self._vector_rows.append(json_safe(row))

    def log_eq_curve(self, step: int, label: str, freqs, gains) -> None:
        freqs = np.asarray(freqs, dtype=float)
        gains = np.asarray(gains, dtype=float)
        row = {"step": int(step), "label": str(label)}
        for i, (f, g) in enumerate(zip(freqs, gains)):
            row[f"freq_{i:02d}_hz"] = float(f)
            row[f"gain_{i:02d}_db"] = float(g)
        self._eq_rows.append(row)

    def log_model_state(self, payload: dict[str, Any]) -> None:
        self.write_jsonl(self.model_jsonl_path, {"ts": now_iso(), **payload})

    def save_summary(self, summary: dict[str, Any]) -> None:
        self.summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    def flush_tables(self) -> None:
        def write_rows(path: Path, rows: list[dict[str, Any]]):
            if not rows:
                return
            keys = sorted({k for row in rows for k in row.keys()})
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: json_safe(row.get(k, "")) for k in keys})
        write_rows(self.root / "session_steps_flat.csv", self._step_rows)
        write_rows(self.vectors_csv_path, self._vector_rows)
        write_rows(self.eq_csv_path, self._eq_rows)

    def export_zip(self, exports_dir: Path | None = None) -> Path:
        self.flush_tables()
        exports_dir = self.root.parent.parent / "exports" if exports_dir is None else Path(exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)
        zip_base = exports_dir / f"{self.session_id}_logs"
        archive = shutil.make_archive(str(zip_base), "zip", root_dir=self.root)
        return Path(archive)
