from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from colorama import Fore, Style, init as colorama_init
from earloop.utils.runtime_paths import ensure_runtime_directories, resolve_runtime_logs_dir

colorama_init()

_FMT = "[%(name)s] %(levelname)s %(asctime)s - %(message)s"
_DATE = "%d-%m-%Y %H:%M:%S"

_LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.WHITE,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.RED + Style.BRIGHT,
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        try:
            color = _LEVEL_COLORS.get(record.levelno, "")
            record.levelname = f"{color}{original_levelname}{Style.RESET_ALL}"
            return super().format(record)
        finally:
            record.levelname = original_levelname


def _resolve_runtime_log_path() -> Path:
    try:
        import os

        configured_value = os.environ.get("EARLOOP_RUNTIME_LOG_PATH")
        runtime_log_path = Path(configured_value).expanduser().resolve() if configured_value else None
    except Exception:
        runtime_log_path = None
    if runtime_log_path is not None:
        runtime_log_path.parent.mkdir(parents=True, exist_ok=True)
        return runtime_log_path
    ensure_runtime_directories()
    return resolve_runtime_logs_dir() / "runtime.log"


def setup_logger(name: str, *, level: int = logging.INFO, worker_id: int | None = None) -> logging.Logger:
    logger_name = f"{name}-{worker_id}" if worker_id is not None else name
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(ColorFormatter(_FMT, datefmt=_DATE))
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

        try:
            file_handler = logging.FileHandler(_resolve_runtime_log_path(), encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except Exception as exc:
            fallback_handler = logging.StreamHandler(stream=sys.stderr)
            fallback_handler.setFormatter(ColorFormatter(_FMT, datefmt=_DATE))
            fallback_handler.setLevel(logging.WARNING)
            logger.addHandler(fallback_handler)
            logger.warning("failed to initialize runtime file logger: %s", exc)
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
            else:
                handler.setFormatter(ColorFormatter(_FMT, datefmt=_DATE))
            handler.setLevel(level)

    return logger


def append_jsonl(path: str | Path, event_name: str, **payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
        **payload,
    }

    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
