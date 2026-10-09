"""The one multi-filter query every layout is measured against: a small
instrument subset, a short date range, one side, grouped to a daily
per-instrument summary. This is the shape a research query actually has
(look at a handful of names over a few sessions), not a full scan.
"""

from __future__ import annotations

from tickpanel import config, simulate


def query_instrument_ids() -> tuple[int, ...]:
    """A contiguous block of instrument IDs, not evenly spaced across the
    full range. A real "look at a handful of related names" query asks for
    a cluster, not one name sampled from every part of the universe; a
    cluster is also the only query shape sorting by instrument_id can ever
    help answer, since an evenly-spaced sample would land in a different
    row group per instrument even when the data is sorted.
    """
    n = min(config.QUERY_N_INSTRUMENTS, config.N_INSTRUMENTS)
    return tuple(range(n))


def query_day_range() -> tuple[int, int]:
    start_idx = config.N_DAYS // 2
    end_idx = min(start_idx + config.QUERY_N_DAYS - 1, config.N_DAYS - 1)
    return start_idx, end_idx


def query_window() -> tuple[str, str]:
    start_idx, end_idx = query_day_range()
    start = simulate.day_date(start_idx).isoformat()
    end = simulate.day_date(end_idx).isoformat()
    return start, end


def duckdb_sql(glob_path: str) -> str:
    start, end = query_window()
    ids = ", ".join(str(i) for i in query_instrument_ids())
    return f"""
    SELECT
        instrument_id,
        CAST(ts AS DATE) AS day,
        AVG(price) AS avg_price,
        SUM(size) AS total_size,
        COUNT(*) AS n_ticks
    FROM read_parquet('{glob_path}', hive_partitioning = true)
    WHERE instrument_id IN ({ids})
      AND ts >= TIMESTAMP '{start} 00:00:00'
      AND ts < TIMESTAMP '{end} 23:59:59.999999'
      AND side = 1
    GROUP BY instrument_id, CAST(ts AS DATE)
    ORDER BY instrument_id, day
    """


def polars_lazy_query(glob_path: str):
    import datetime

    import polars as pl

    start, end = query_window()
    start_dt = datetime.datetime.fromisoformat(start)
    end_dt = datetime.datetime.fromisoformat(end) + datetime.timedelta(days=1)

    lf = pl.scan_parquet(glob_path, hive_partitioning=True)
    return (
        lf.filter(
            pl.col("instrument_id").is_in(list(query_instrument_ids()))
            & (pl.col("ts") >= start_dt)
            & (pl.col("ts") < end_dt)
            & (pl.col("side") == 1)
        )
        .with_columns(pl.col("ts").cast(pl.Date).alias("day"))
        .group_by(["instrument_id", "day"])
        .agg(
            pl.col("price").mean().alias("avg_price"),
            pl.col("size").sum().alias("total_size"),
            pl.len().alias("n_ticks"),
        )
        .sort(["instrument_id", "day"])
    )
