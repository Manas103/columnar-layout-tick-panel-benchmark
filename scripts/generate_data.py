"""Builds all 4 layouts of the 1.2B-row simulated tick panel from scratch.

    python scripts/generate_data.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

os.environ.setdefault("POLARS_MAX_THREADS", "8")

from tickpanel import layouts  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    layouts.build_all()
    elapsed = time.time() - t0
    print(f"built all layouts in {elapsed:.1f}s")
    for name in ("none_unsorted", "monthly_unsorted", "daily_unsorted", "daily_sorted"):
        print(name, layouts.layout_total_bytes(name), "bytes")
