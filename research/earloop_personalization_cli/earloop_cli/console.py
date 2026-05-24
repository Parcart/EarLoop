from __future__ import annotations

import os
import sys


def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        try:
            os.system("")
        except Exception:
            pass


def _colors_enabled() -> bool:
    raw = os.environ.get("EARLOOP_CLI_COLOR", "auto").strip().lower()
    if raw in {"0", "false", "no", "off", "none"}:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


_enable_windows_ansi()
USE_COLOR = _colors_enabled()

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"


def c(text: object, code: str = "") -> str:
    value = str(text)
    if not USE_COLOR or not code:
        return value
    return f"{code}{value}{RESET}"


def title(text: object) -> str:
    return c(text, BOLD + CYAN)


def info(text: object) -> str:
    return c(text, CYAN)


def ok(text: object) -> str:
    return c(text, GREEN)


def warn(text: object) -> str:
    return c(text, YELLOW)


def error(text: object) -> str:
    return c(text, RED)


def muted(text: object) -> str:
    return c(text, DIM)


def key(text: object) -> str:
    return c(text, BOLD + WHITE)
