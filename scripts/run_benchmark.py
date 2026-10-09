"""Runs every claim measurement against the already-built layouts and
prints the full result as JSON.

    python scripts/run_benchmark.py > docs/benchmark_output.txt
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

os.environ.setdefault("POLARS_MAX_THREADS", "8")

from tickpanel import measurements, query  # noqa: E402


def main() -> None:
    start, end = query.query_window()
    print("=== query definition ===")
    print(
        json.dumps(
            {
                "instrument_ids": list(query.query_instrument_ids()),
                "date_range": [start, end],
                "side_filter": 1,
            },
            indent=2,
        )
    )
    print()

    scale = measurements.measure_scale()
    print("=== scale ===")
    print(json.dumps(scale, indent=2, default=str))
    print()

    bench = measurements.measure_query_benchmark(repeats=5)
    print("=== query_benchmark ===")
    printable_bench = {
        "value": bench["value"],
        "baseline_layout": bench["baseline_layout"],
        "best_layout": bench["best_layout"],
        "repeats": bench["repeats"],
        "by_layout": {
            name: {
                "median_seconds": r["median_seconds"],
                "min_seconds": r["min_seconds"],
                "max_seconds": r["max_seconds"],
                "median_bytes_read": r["median_bytes_read"],
                "layout_total_bytes": r["layout_total_bytes"],
                "scan_footprint_fraction": r["scan_footprint_fraction"],
                "rows_returned": r["rows_returned"],
            }
            for name, r in bench["by_layout"].items()
        },
    }
    print(json.dumps(printable_bench, indent=2, default=str))
    print()

    agreement = measurements.measure_polars_streaming_agreement()
    print("=== polars_streaming_agreement ===")
    printable_agreement = {
        "value": agreement["value"],
        "all_layouts_within_2pct": agreement["all_layouts_within_2pct"],
        "all_layouts_under_4gb_ceiling": agreement["all_layouts_under_4gb_ceiling"],
        "max_peak_rss_gb_across_layouts": agreement["max_peak_rss_gb_across_layouts"],
        "by_layout": {
            name: {
                "max_relative_difference": r["max_relative_difference"],
                "within_2pct": r["within_2pct"],
                "polars_seconds": r["polars_seconds"],
                "polars_peak_rss_gb": r["polars_peak_rss_gb"],
            }
            for name, r in agreement["by_layout"].items()
        },
    }
    print(json.dumps(printable_agreement, indent=2, default=str))


if __name__ == "__main__":
    main()
