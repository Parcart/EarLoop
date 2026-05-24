from __future__ import annotations

import sys
from pathlib import Path


def ensure_vendor_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    vendor = root / "vendor"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    return vendor
