"""One trading day's worth of simulated ticks, vectorized across all 500
instruments at once.

Everything here is synthetic, stated as such in the README. Each
instrument has a fixed, deterministic tick count per day (exactly
`AVG_TICKS_PER_INSTRUMENT_DAY`, not Poisson-varied), a declared
simplification that keeps the total row count exactly 1.2B and every
per-layout row count independently checkable, at the cost of not modeling
real intraday volume variation across instruments.
"""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl

from tickpanel import config


def day_date(day_index: int) -> datetime.date:
    return config.FIRST_DAY + datetime.timedelta(days=day_index)


def simulate_day(day_index: int) -> pl.DataFrame:
    """Every tick for every instrument on this day, in arrival (unsorted)
    order: ticks interleave across instruments the way a real consolidated
    tape would, not grouped by instrument.
    """
    rng = np.random.default_rng(config.SEED + day_index)
    n_inst = config.N_INSTRUMENTS
    n_ticks = config.AVG_TICKS_PER_INSTRUMENT_DAY
    span_ns = config.TRADING_HOURS_END_NS - config.TRADING_HOURS_START_NS

    # Intraday offsets, one row per (instrument, tick), sorted per instrument
    # so each instrument's own price path is a genuine walk forward in time.
    offsets = rng.integers(0, span_ns, size=(n_inst, n_ticks), dtype=np.int64)
    offsets.sort(axis=1)
    ts_ns = (offsets + config.TRADING_HOURS_START_NS).reshape(-1)

    base_price = 10.0 + rng.permutation(n_inst).astype(np.float64) * 0.73 + (day_index % 7)
    steps = rng.normal(0.0, 0.015, size=(n_inst, n_ticks)).astype(np.float64)
    walk = np.cumsum(steps, axis=1)
    price = np.maximum(0.01, base_price[:, None] + walk).astype(np.float32).reshape(-1)

    size = rng.integers(1, 501, size=n_inst * n_ticks, dtype=np.uint32)
    side = rng.integers(0, 2, size=n_inst * n_ticks, dtype=np.uint8)
    instrument_id = np.repeat(np.arange(n_inst, dtype=np.uint16), n_ticks)

    date = day_date(day_index)
    ts_datetime = np.datetime64(date, "ns") + ts_ns.astype("timedelta64[ns]")

    df = pl.DataFrame(
        {
            "instrument_id": instrument_id,
            "ts": ts_datetime,
            "price": price,
            "size": size,
            "side": side,
        }
    )

    # Arrival order across instruments: interleaved by timestamp, the shape
    # a real consolidated tape has and the "unsorted" baseline layouts keep.
    return df.sort("ts")


def expected_row_count() -> int:
    return config.N_INSTRUMENTS * config.N_DAYS * config.AVG_TICKS_PER_INSTRUMENT_DAY
