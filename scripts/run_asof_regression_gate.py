"""Builds the as-of join fixture (trades and quotes, 20 instruments x 5
days, in both unsorted and daily-sorted-by-instrument layouts), computes
the 30 pinned analytic outputs against the unsorted layout (the pin, if
not already committed), then recomputes the same 30 outputs against the
daily-sorted layout and diffs them against the pin: the storage changed,
this proves the 30 outputs did not.

Run: python scripts/run_asof_regression_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tickpanel import asof_fixture, asof_join, regression_gate


def main() -> None:
    asof_fixture.build_layouts()

    baseline_rows = asof_join.run_asof_join_aggregates("unsorted")
    print(f"Computed {len(baseline_rows)} analytic outputs (10 instruments x 3 days) from the unsorted layout.")

    if not regression_gate.PINNED_PATH.exists():
        regression_gate.write_pinned(baseline_rows)
        print(f"No pinned snapshot found; wrote {regression_gate.PINNED_PATH} as the pin.")
    else:
        print(f"Pinned snapshot already exists at {regression_gate.PINNED_PATH}; not overwriting.")

    pinned = regression_gate.load_pinned()
    self_diff = regression_gate.diff_against_pinned(baseline_rows, pinned)
    print(f"Unsorted layout vs. pin (sanity check): {self_diff['n_mismatches']} mismatches "
          f"of {self_diff['n_pinned']} pinned outputs, all_identical={self_diff['all_identical']}")

    sorted_rows = asof_join.run_asof_join_aggregates("daily_sorted")
    gate_result = regression_gate.diff_against_pinned(sorted_rows, pinned)

    print()
    print("=== Regression Gate: unsorted -> daily-sorted-by-instrument ===")
    print(f"Pinned analytic outputs: {gate_result['n_pinned']}")
    print(f"Outputs recomputed after the storage re-layout: {gate_result['n_checked']}")
    print(f"Mismatches: {gate_result['n_mismatches']}")
    print(f"Every pinned result stayed identical: {gate_result['all_identical']}")
    if gate_result["mismatches"]:
        for m in gate_result["mismatches"]:
            print(f"  MISMATCH: {m}")

    print()
    print("Sample pinned rows (first 3 of 30):")
    for row in sorted_rows[:3]:
        print(f"  {row}")


if __name__ == "__main__":
    main()
