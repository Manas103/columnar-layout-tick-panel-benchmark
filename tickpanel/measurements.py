"""The claim measurements, each an honest measured number, not a target."""

from __future__ import annotations

import duckdb

from tickpanel import benchmark, config, layouts, polars_check, query


def measure_scale() -> dict:
    """Claim: a 1.2B-row simulated tick panel, identical across every layout."""
    counts = {}
    for layout in config.LAYOUTS:
        glob_path = layouts.layout_glob(layout)
        con = duckdb.connect()
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob_path}', hive_partitioning=true)").fetchone()[0]
        con.close()
        counts[layout] = int(n)
    all_equal = len(set(counts.values())) == 1
    total = next(iter(counts.values()))
    return {
        "value": bool(all_equal and total == config.TOTAL_ROWS),
        "row_counts_by_layout": counts,
        "expected_total_rows": config.TOTAL_ROWS,
    }


def measure_query_benchmark(repeats: int = 5) -> dict:
    """Claims: median query time cut by partitioning + sorting, scan
    footprint cut, measured across every layout with the one query."""
    results = benchmark.run_all(repeats=repeats)
    baseline = results["none_unsorted"]
    best = results["daily_sorted"]

    time_speedup = baseline["median_seconds"] / best["median_seconds"] if best["median_seconds"] else None
    footprint_cut_pct = (
        (1 - best["median_bytes_read"] / baseline["median_bytes_read"]) * 100
        if baseline["median_bytes_read"]
        else None
    )

    return {
        "value": {
            "median_query_time_speedup_x": round(time_speedup, 2) if time_speedup else None,
            "scan_footprint_cut_pct": round(footprint_cut_pct, 1) if footprint_cut_pct is not None else None,
        },
        "by_layout": results,
        "baseline_layout": "none_unsorted",
        "best_layout": "daily_sorted",
        "repeats": repeats,
    }


def measure_polars_streaming_agreement() -> dict:
    """Claim: polars streaming agrees with the DuckDB result within 2%,
    under a declared 4 GB memory ceiling, checked on every layout."""
    results = {}
    for layout in config.LAYOUTS:
        con = duckdb.connect()
        con.execute(f"PRAGMA threads={config.MAX_THREADS}")
        sql = query.duckdb_sql(layouts.layout_glob(layout))
        duckdb_rows = con.execute(sql).fetchall()
        con.close()
        results[layout] = polars_check.compare_to_duckdb(layout, duckdb_rows)

    all_within_2pct = all(r["within_2pct"] for r in results.values())
    all_under_ceiling = all(
        r["polars_peak_rss_gb"] <= config.STREAMING_MEMORY_CEILING_GB for r in results.values()
    )
    max_rss_gb = max(r["polars_peak_rss_gb"] for r in results.values())

    return {
        "value": bool(all_within_2pct),
        "all_layouts_within_2pct": all_within_2pct,
        "all_layouts_under_4gb_ceiling": all_under_ceiling,
        "max_peak_rss_gb_across_layouts": round(max_rss_gb, 3),
        "by_layout": results,
    }
