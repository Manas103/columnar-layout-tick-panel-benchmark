"""Pins the 30 as-of-join analytic outputs from one layout to a committed
JSON snapshot, and diffs a fresh run (normally against a different layout,
after a storage re-layout) against that snapshot. This is the regression
gate itself: "the storage changed, prove the 30 outputs did not."
"""

from __future__ import annotations

import json
import pathlib

PINNED_PATH = pathlib.Path(__file__).resolve().parent / "regression" / "pinned_asof_outputs.json"

_FLOAT_TOLERANCE = 1e-9


def write_pinned(rows: list[dict], path: pathlib.Path | None = None) -> None:
    path = path if path is not None else PINNED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True))


def load_pinned(path: pathlib.Path | None = None) -> list[dict]:
    path = path if path is not None else PINNED_PATH
    return json.loads(path.read_text())


def diff_against_pinned(rows: list[dict], pinned: list[dict] | None = None, path: pathlib.Path | None = None) -> dict:
    pinned = pinned if pinned is not None else load_pinned(path)

    pinned_by_key = {(r["instrument_id"], r["day"]): r for r in pinned}
    rows_by_key = {(r["instrument_id"], r["day"]): r for r in rows}

    mismatches = []
    for key in sorted(pinned_by_key):
        if key not in rows_by_key:
            mismatches.append({"key": key, "reason": "missing_in_fresh_run"})
            continue
        expected = pinned_by_key[key]
        actual = rows_by_key[key]
        for field in ("n_trades", "avg_slippage", "avg_prevailing_spread"):
            e, a = expected[field], actual[field]
            if isinstance(e, float) or isinstance(a, float):
                if abs(float(e) - float(a)) > _FLOAT_TOLERANCE:
                    mismatches.append({"key": key, "field": field, "expected": e, "actual": a})
            elif e != a:
                mismatches.append({"key": key, "field": field, "expected": e, "actual": a})

    return {
        "n_pinned": len(pinned_by_key),
        "n_checked": len(rows_by_key),
        "n_mismatches": len(mismatches),
        "all_identical": len(mismatches) == 0 and len(pinned_by_key) == len(rows_by_key),
        "mismatches": mismatches,
    }
