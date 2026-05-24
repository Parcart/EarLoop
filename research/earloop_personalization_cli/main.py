from __future__ import annotations

import sys
import traceback

from earloop_cli.cli import main


def _pause_if_frozen(message: str = "Нажмите Enter для выхода...") -> None:
    if getattr(sys, "frozen", False):
        try:
            input(f"\n{message}")
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        _pause_if_frozen()
    except SystemExit as exc:
        # argparse exits through SystemExit. Keep double-clicked exe readable.
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            _pause_if_frozen()
        raise
    except Exception:
        print("\n=== EarLoop Personalization CLI crashed ===")
        traceback.print_exc()
        _pause_if_frozen()
        raise
