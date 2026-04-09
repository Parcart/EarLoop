from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_runtime_root() -> Path:
    configured = os.environ.get("EARLOOP_RUNTIME_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    if getattr(sys, "frozen", False):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data).expanduser().resolve() / "EarLoop"
        return Path.home().resolve() / "AppData" / "Local" / "EarLoop"

    return resolve_project_root() / "data" / "engine"


def resolve_runtime_data_dir() -> Path:
    return resolve_runtime_root() / "data"


def resolve_runtime_logs_dir() -> Path:
    return resolve_runtime_root() / "logs"


def ensure_runtime_directories() -> tuple[Path, Path, Path]:
    root = resolve_runtime_root()
    data_dir = resolve_runtime_data_dir()
    logs_dir = resolve_runtime_logs_dir()
    root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return root, data_dir, logs_dir
