"""Measures the one multi-filter query against one layout: wall-clock time
(median of repeats) and scan footprint (bytes DuckDB's own profiler reports
actually read from disk to answer the query), both from the real DuckDB
engine, not estimated.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import time

import duckdb

from tickpanel import config, layouts, query


def run_duckdb_once(glob_path: str, threads: int = config.MAX_THREADS) -> tuple[float, int, int]:
    """One run: (wall_seconds, total_bytes_read, rows_returned)."""
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={threads}")
    with tempfile.TemporaryDirectory() as tmp:
        prof_path = str(pathlib.Path(tmp) / "prof.json")
        con.execute("PRAGMA enable_profiling='json'")
        con.execute(f"PRAGMA profiling_output='{prof_path}'")
        sql = query.duckdb_sql(glob_path)
        t0 = time.perf_counter()
        result = con.execute(sql).fetchall()
        elapsed = time.perf_counter() - t0
        profile = json.loads(pathlib.Path(prof_path).read_text())
    con.close()
    return elapsed, int(profile.get("total_bytes_read", 0)), len(result)


def benchmark_layout(layout: str, repeats: int = 5) -> dict:
    glob_path = layouts.layout_glob(layout)
    timings = []
    bytes_read_values = []
    rows_returned = None
    for _ in range(repeats):
        elapsed, bytes_read, n_rows = run_duckdb_once(glob_path)
        timings.append(elapsed)
        bytes_read_values.append(bytes_read)
        rows_returned = n_rows

    timings_sorted = sorted(timings)
    median = timings_sorted[len(timings_sorted) // 2]
    total_layout_bytes = layouts.layout_total_bytes(layout)
    median_bytes_read = sorted(bytes_read_values)[len(bytes_read_values) // 2]

    return {
        "layout": layout,
        "repeats": repeats,
        "timings_seconds": timings,
        "median_seconds": median,
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "median_bytes_read": median_bytes_read,
        "layout_total_bytes": total_layout_bytes,
        "scan_footprint_fraction": median_bytes_read / total_layout_bytes if total_layout_bytes else None,
        "rows_returned": rows_returned,
    }


def run_all(repeats: int = 5) -> dict[str, dict]:
    return {layout: benchmark_layout(layout, repeats=repeats) for layout in config.LAYOUTS}
