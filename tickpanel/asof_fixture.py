"""A second, smaller synthetic fixture for the trade-to-prevailing-quote
as-of join and the regression gate: trades and a separate quote stream,
20 instruments over 5 days, written out in both arrival order and a
daily-partitioned, instrument-sorted layout.

This is deliberately a different, smaller panel than the flagship
1.2B-row, 4-layout benchmark in `layouts.py`/`simulate.py`: that panel has
no quote stream at all (tick.side only labels trade side), and the point
of this fixture is to exercise the as-of join and the regression gate
honestly at a scale that regenerates in seconds, not to repeat the 1.2B-row
measurement a second time. Both facts are stated here and in the README so
neither is mistaken for the other.
"""

from __future__ import annotations

import datetime
import pathlib

import numpy as np
import polars as pl
import pyarrow.parquet as pq

FIXTURE_SEED = 20261023
N_INSTRUMENTS = 20
N_DAYS = 5
TRADES_PER_INSTRUMENT_DAY = 5_000
QUOTES_PER_INSTRUMENT_DAY = 5_000
FIRST_DAY = datetime.date(2024, 6, 3)  # a Monday, distinct from the flagship panel's calendar

TRADING_HOURS_START_NS = 9 * 3_600_000_000_000 + 30 * 60_000_000_000
TRADING_HOURS_END_NS = 16 * 3_600_000_000_000

FIXTURE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "asof_fixture"

# The 10-instrument x 3-day window the as-of join's 30 pinned outputs cover.
QUERY_N_INSTRUMENTS = 10
QUERY_N_DAYS = 3


def day_date(day_index: int) -> datetime.date:
    return FIRST_DAY + datetime.timedelta(days=day_index)


def query_instrument_ids() -> tuple[int, ...]:
    return tuple(range(QUERY_N_INSTRUMENTS))


def query_day_indices() -> tuple[int, ...]:
    start = N_DAYS // 2
    return tuple(range(start, min(start + QUERY_N_DAYS, N_DAYS)))


def simulate_day_pair(day_index: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Trades and quotes for one day, both derived from the same underlying
    per-instrument "true mid" random-walk path so that a trade's price and
    the quote prevailing at that instant are genuinely related (a trade
    executes near the true mid plus small execution noise; a quote is a
    bid/ask straddling the true mid with a small spread). Generating them
    from two independent walks, as an earlier version of this fixture did,
    produced an avg_slippage on the order of several points on a ~$20-40
    instrument, an economically nonsensical "prevailing quote" that was
    really just two unrelated price series; see the README Findings.
    """
    rng = np.random.default_rng(FIXTURE_SEED + day_index)
    n_inst = N_INSTRUMENTS
    n_trades = TRADES_PER_INSTRUMENT_DAY
    n_quotes = QUOTES_PER_INSTRUMENT_DAY
    total = n_trades + n_quotes
    span_ns = TRADING_HOURS_END_NS - TRADING_HOURS_START_NS

    offsets = rng.integers(0, span_ns, size=(n_inst, total), dtype=np.int64)
    offsets.sort(axis=1)
    ts_ns = offsets + TRADING_HOURS_START_NS

    base_price = 20.0 + rng.permutation(n_inst).astype(np.float64) * 1.1
    steps = rng.normal(0.0, 0.01, size=(n_inst, total))
    true_mid = np.maximum(0.05, base_price[:, None] + np.cumsum(steps, axis=1))

    date = day_date(day_index)
    ts_datetime = np.datetime64(date, "ns") + ts_ns.astype("timedelta64[ns]")

    trade_frames = []
    quote_frames = []
    for i in range(n_inst):
        perm = rng.permutation(total)
        trade_idx = np.sort(perm[:n_trades])
        quote_idx = np.sort(perm[n_trades:])

        exec_noise = rng.normal(0.0, 0.015, size=n_trades)
        trade_price = np.maximum(0.01, true_mid[i, trade_idx] + exec_noise)
        trade_size = rng.integers(1, 501, size=n_trades, dtype=np.int64)

        half_spread = rng.uniform(0.01, 0.05, size=n_quotes)
        quote_mid = true_mid[i, quote_idx]

        trade_frames.append(pl.DataFrame({
            "instrument_id": np.full(n_trades, i, dtype=np.int32),
            "ts": ts_datetime[i, trade_idx],
            "price": trade_price,
            "size": trade_size,
        }))
        quote_frames.append(pl.DataFrame({
            "instrument_id": np.full(n_quotes, i, dtype=np.int32),
            "ts": ts_datetime[i, quote_idx],
            "bid": quote_mid - half_spread,
            "ask": quote_mid + half_spread,
        }))

    trades_df = pl.concat(trade_frames).sort("ts")
    quotes_df = pl.concat(quote_frames).sort("ts")
    return trades_df, quotes_df


def _write_unsorted(table, path: pathlib.Path, writer_cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path) not in writer_cache:
        writer_cache[str(path)] = pq.ParquetWriter(str(path), table.schema)
    writer_cache[str(path)].write_table(table)


def build_layouts(root: pathlib.Path | None = None, n_days: int | None = None) -> None:
    """Writes trades and quotes, each in two layouts: arrival-order
    (unsorted, single file per stream) and daily-partitioned, sorted by
    instrument within each day's partition."""
    root = root if root is not None else FIXTURE_ROOT
    n_days = n_days if n_days is not None else N_DAYS
    root.mkdir(parents=True, exist_ok=True)

    trades_unsorted_path = root / "unsorted" / "trades.parquet"
    quotes_unsorted_path = root / "unsorted" / "quotes.parquet"
    writer_cache: dict = {}

    for day_index in range(n_days):
        date = day_date(day_index)
        trades, quotes = simulate_day_pair(day_index)

        _write_unsorted(trades.to_arrow(), trades_unsorted_path, writer_cache)
        _write_unsorted(quotes.to_arrow(), quotes_unsorted_path, writer_cache)

        trades_sorted_dir = root / "daily_sorted" / f"day={date.isoformat()}"
        trades_sorted_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(trades.sort(["instrument_id", "ts"]).to_arrow(), str(trades_sorted_dir / "trades.parquet"))

        quotes_sorted_dir = root / "daily_sorted" / f"day={date.isoformat()}"
        quotes_sorted_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(quotes.sort(["instrument_id", "ts"]).to_arrow(), str(quotes_sorted_dir / "quotes.parquet"))

    for w in writer_cache.values():
        w.close()


def trades_glob(layout: str, root: pathlib.Path | None = None) -> str:
    root = root if root is not None else FIXTURE_ROOT
    if layout == "unsorted":
        return str(root / "unsorted" / "trades.parquet")
    return str(root / layout / "**" / "trades.parquet")


def quotes_glob(layout: str, root: pathlib.Path | None = None) -> str:
    root = root if root is not None else FIXTURE_ROOT
    if layout == "unsorted":
        return str(root / "unsorted" / "quotes.parquet")
    return str(root / layout / "**" / "quotes.parquet")
