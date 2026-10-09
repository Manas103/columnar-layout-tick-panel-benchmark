"""Cross-checks the DuckDB result against polars' own streaming engine on
the identical files, and measures the streaming engine's peak process
memory while it runs.

Peak memory is measured, not enforced: polars' public API has no hard
memory cap to pass in, so this samples the running process's actual RSS on
a background thread while `collect(engine="streaming")` runs and reports
the measured peak against the declared 4 GB ceiling, the same
measured-not-enforced pattern this portfolio uses elsewhere for claims a
library does not expose a knob for.
"""

from __future__ import annotations

import threading
import time

import psutil

from tickpanel import layouts, query


def _sample_peak_rss(stop_event: threading.Event, proc: psutil.Process, peak: list[int]) -> None:
    while not stop_event.is_set():
        try:
            peak[0] = max(peak[0], proc.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.05)


def run_polars_streaming(layout: str, root=None) -> dict:
    glob_path = layouts.layout_glob(layout, root=root)
    lf = query.polars_lazy_query(glob_path)

    proc = psutil.Process()
    peak = [proc.memory_info().rss]
    stop_event = threading.Event()
    sampler = threading.Thread(target=_sample_peak_rss, args=(stop_event, proc, peak), daemon=True)
    sampler.start()

    t0 = time.perf_counter()
    df = lf.collect(engine="streaming")
    elapsed = time.perf_counter() - t0

    stop_event.set()
    sampler.join()

    return {
        "layout": layout,
        "seconds": elapsed,
        "peak_rss_bytes": peak[0],
        "n_rows": df.height,
        "result": df.sort(["instrument_id", "day"]),
    }


def compare_to_duckdb(layout: str, duckdb_rows: list[tuple], root=None) -> dict:
    polars_result = run_polars_streaming(layout, root=root)
    df = polars_result["result"]

    duckdb_map = {(r[0], str(r[1])): (float(r[2]), int(r[3]), int(r[4])) for r in duckdb_rows}
    polars_map = {
        (row["instrument_id"], str(row["day"])): (float(row["avg_price"]), int(row["total_size"]), int(row["n_ticks"]))
        for row in df.iter_rows(named=True)
    }

    same_keys = set(duckdb_map) == set(polars_map)
    max_rel_diff = 0.0
    for key, (d_price, d_size, d_n) in duckdb_map.items():
        p_price, p_size, p_n = polars_map.get(key, (float("nan"), -1, -1))
        rel_price = abs(d_price - p_price) / max(abs(d_price), abs(p_price), 1e-9)
        rel_size = abs(d_size - p_size) / max(abs(d_size), abs(p_size), 1)
        max_rel_diff = max(max_rel_diff, rel_price, rel_size)
        if d_n != p_n:
            max_rel_diff = max(max_rel_diff, 1.0)

    return {
        "layout": layout,
        "same_group_keys": same_keys,
        "max_relative_difference": max_rel_diff,
        "within_2pct": bool(max_rel_diff <= 0.02),
        "polars_seconds": polars_result["seconds"],
        "polars_peak_rss_bytes": polars_result["peak_rss_bytes"],
        "polars_peak_rss_gb": polars_result["peak_rss_bytes"] / (1024**3),
    }
