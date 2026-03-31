from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from colorama import Fore, Style, init as colorama_init

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


def setup_logger(name: str, *, level: int = logging.INFO, worker_id: int | None = None) -> logging.Logger:
    logger_name = f"{name}-{worker_id}" if worker_id is not None else name
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(ColorFormatter(_FMT, datefmt=_DATE))
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
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
