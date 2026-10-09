"""Simulation and layout parameters, fixed before any query was ever timed.

Everything here is a declared policy, chosen for size and shape (1.2B rows,
a "tick-scale" panel, a handful of partition/sort combinations that fit
this machine's disk budget), not tuned after seeing a benchmark result.
"""

from __future__ import annotations

import datetime

SEED = 20261006

N_INSTRUMENTS = 500
N_DAYS = 60
AVG_TICKS_PER_INSTRUMENT_DAY = 40_000
# 500 * 60 * 40,000 = 1,200,000,000 rows exactly.
TOTAL_ROWS = N_INSTRUMENTS * N_DAYS * AVG_TICKS_PER_INSTRUMENT_DAY

FIRST_DAY = datetime.date(2024, 1, 2)  # a Tuesday; simulated trading days are consecutive calendar days, not a real exchange calendar

TRADING_HOURS_START_NS = 9 * 3_600_000_000_000 + 30 * 60_000_000_000  # 09:30:00, nanoseconds since midnight
TRADING_HOURS_END_NS = 16 * 3_600_000_000_000  # 16:00:00

# Resource courtesy: this machine is shared and in use while this runs.
# Half of 16 logical cores, same cap the playbook requires for compiled
# builds, applied here to polars' and DuckDB's own thread pools.
MAX_THREADS = 8

# The one multi-filter query every layout is measured against.
QUERY_N_INSTRUMENTS = 10
QUERY_N_DAYS = 3

# polars streaming's declared memory ceiling for the cross-check query.
STREAMING_MEMORY_CEILING_GB = 4

LAYOUTS = ("none_unsorted", "monthly_unsorted", "daily_unsorted", "daily_sorted")

ROWS_PER_CHUNK_WRITE_BATCH = 2_000_000  # rows generated/written at a time within a day
