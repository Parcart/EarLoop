from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .vendor_path import ensure_vendor_path
ensure_vendor_path()
from personalization.contract_mapper import FREQS_23_DEFAULT
from personalization.state import FEATURE_NAMES_8D


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class SavedEqProfile:
    profile_id: str
    name: str
    created_at: str
    strategy: str
    mapper_version: str
    z_contract: list[float]
    eq23_db: list[float]
    freqs_23: list[float]
    session_id: str | None = None
    steps_count: int = 0
    feedback_count: int = 0
    soft_stop_triggered: bool = False
    user_finished_after_soft_stop: bool | None = None
    final_rating: int | None = None
    comment: str | None = None
    profile_source: str | None = None
    source_step: int | None = None
    source_choice: str | None = None
    final_state_z_contract: list[float] | None = None
    final_state_eq23_db: list[float] | None = None
    last_selected_step: int | None = None
    last_selected_choice: str | None = None
    last_selected_z_contract: list[float] | None = None
    last_selected_eq23_db: list[float] | None = None
    saved_at_step: int | None = None

    @classmethod
    def create(
        cls,
        *,
        name: str,
        strategy: str,
        mapper_version: str,
        z_contract,
        eq23_db,
        freqs_23=FREQS_23_DEFAULT,
        session_id: str | None = None,
        steps_count: int = 0,
        feedback_count: int = 0,
        soft_stop_triggered: bool = False,
        user_finished_after_soft_stop: bool | None = None,
        final_rating: int | None = None,
        comment: str | None = None,
        profile_source: str | None = None,
        source_step: int | None = None,
        source_choice: str | None = None,
        final_state_z_contract=None,
        final_state_eq23_db=None,
        last_selected_step: int | None = None,
        last_selected_choice: str | None = None,
        last_selected_z_contract=None,
        last_selected_eq23_db=None,
        saved_at_step: int | None = None,
    ) -> "SavedEqProfile":
        def _maybe_list(value):
            if value is None:
                return None
            return np.asarray(value, dtype=float).round(6).tolist()

        return cls(
            profile_id=f"profile_{uuid.uuid4().hex[:10]}",
            name=name,
            created_at=now_iso(),
            strategy=strategy,
            mapper_version=mapper_version,
            z_contract=np.asarray(z_contract, dtype=float).round(6).tolist(),
            eq23_db=np.asarray(eq23_db, dtype=float).round(6).tolist(),
            freqs_23=np.asarray(freqs_23, dtype=float).round(6).tolist(),
            session_id=session_id,
            steps_count=int(steps_count),
            feedback_count=int(feedback_count),
            soft_stop_triggered=bool(soft_stop_triggered),
            user_finished_after_soft_stop=user_finished_after_soft_stop,
            final_rating=final_rating,
            comment=comment,
            profile_source=profile_source,
            source_step=None if source_step is None else int(source_step),
            source_choice=source_choice,
            final_state_z_contract=_maybe_list(final_state_z_contract),
            final_state_eq23_db=_maybe_list(final_state_eq23_db),
            last_selected_step=None if last_selected_step is None else int(last_selected_step),
            last_selected_choice=last_selected_choice,
            last_selected_z_contract=_maybe_list(last_selected_z_contract),
            last_selected_eq23_db=_maybe_list(last_selected_eq23_db),
            saved_at_step=None if saved_at_step is None else int(saved_at_step),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class ProfileManager:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def save(self, profile: SavedEqProfile) -> Path:
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in profile.name.strip())[:48] or "profile"
        path = self.profiles_dir / f"{profile.created_at.replace(':','-')}_{safe_name}_{profile.profile_id}.json"
        path.write_text(profile.to_json(), encoding="utf-8")
        return path

    def list_profiles(self) -> list[Path]:
        return sorted(self.profiles_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    def load(self, path: str | Path) -> SavedEqProfile:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # Be tolerant to profiles from older/newer CLI builds. Unknown fields are ignored;
        # missing fields get dataclass defaults.
        allowed = set(SavedEqProfile.__dataclass_fields__.keys())
        payload = {k: v for k, v in payload.items() if k in allowed}
        return SavedEqProfile(**payload)

    def print_profiles(self) -> list[Path]:
        profiles = self.list_profiles()
        if not profiles:
            print("No saved profiles yet.")
            return []
        print("\nSaved profiles:")
        for idx, path in enumerate(profiles, 1):
            try:
                p = self.load(path)
                print(f"[{idx}] {p.name} | {p.created_at} | strategy={p.strategy} | steps={p.steps_count} | rating={p.final_rating}")
            except Exception:
                print(f"[{idx}] {path.name}")
        return profiles

    @staticmethod
    def feature_table(z_contract: list[float]) -> str:
        rows = []
        for name, val in zip(FEATURE_NAMES_8D, z_contract):
            rows.append(f"{name:>11}: {float(val): .3f}")
        return "\n".join(rows)
