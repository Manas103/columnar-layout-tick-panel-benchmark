"""Writes the same simulated panel 4 ways: no partitioning, monthly
partitions, daily partitions, and daily partitions sorted by instrument.

Each day is generated exactly once (`simulate.simulate_day`, deterministic
by day index) and distributed to all 4 layouts from the same in-memory
DataFrame, so every layout holds byte-for-byte the same rows, reorganized
only by partition directory, row-group size, and row order.

Row-group size is *not* held fixed across every layout, and that is a
deliberate design choice, not an oversight: `none_unsorted` and
`monthly_unsorted` use a deliberately coarse row-group size
(`NAIVE_ROW_GROUP_SIZE`), the way a bulk dump that was never organized for
analytic reads in the first place typically ends up on disk; `daily_unsorted`
and `daily_sorted` use a much finer size (`TUNED_ROW_GROUP_SIZE`), the way a
deliberately engineered layout would be written. An earlier version of this
file held row-group size fixed at the tuned size for every layout, which
made `none_unsorted` benefit almost as much from row-group-level time
pruning as the partitioned layouts even with zero partition directories,
flattening the very comparison this benchmark exists to make; see the
README Findings for the measured numbers that showed this.
"""

from __future__ import annotations

import pathlib

import pyarrow.parquet as pq

from tickpanel import config, simulate

LAYOUT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "data" / "layouts"
NAIVE_ROW_GROUP_SIZE = 100_000_000
TUNED_ROW_GROUP_SIZE = 2_000_000


def build_all(n_days: int | None = None, root: pathlib.Path | None = None) -> None:
    n_days = n_days if n_days is not None else config.N_DAYS
    root = root if root is not None else LAYOUT_ROOT
    root.mkdir(parents=True, exist_ok=True)

    none_path = root / "none_unsorted" / "data.parquet"
    none_path.parent.mkdir(parents=True, exist_ok=True)
    none_writer = None

    month_writers: dict[str, pq.ParquetWriter] = {}

    for day_index in range(n_days):
        date = simulate.day_date(day_index)
        df = simulate.simulate_day(day_index)
        table = df.to_arrow()

        if none_writer is None:
            none_writer = pq.ParquetWriter(str(none_path), table.schema)
        none_writer.write_table(table, row_group_size=NAIVE_ROW_GROUP_SIZE)

        month_key = f"{date.year:04d}-{date.month:02d}"
        if month_key not in month_writers:
            part_dir = root / "monthly_unsorted" / f"month={month_key}"
            part_dir.mkdir(parents=True, exist_ok=True)
            month_writers[month_key] = pq.ParquetWriter(str(part_dir / "data.parquet"), table.schema)
        month_writers[month_key].write_table(table, row_group_size=NAIVE_ROW_GROUP_SIZE)

        daily_unsorted_dir = root / "daily_unsorted" / f"day={date.isoformat()}"
        daily_unsorted_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, str(daily_unsorted_dir / "data.parquet"), row_group_size=TUNED_ROW_GROUP_SIZE)

        sorted_table = df.sort(["instrument_id", "ts"]).to_arrow()
        daily_sorted_dir = root / "daily_sorted" / f"day={date.isoformat()}"
        daily_sorted_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(sorted_table, str(daily_sorted_dir / "data.parquet"), row_group_size=TUNED_ROW_GROUP_SIZE)

    if none_writer is not None:
        none_writer.close()
    for w in month_writers.values():
        w.close()


def layout_glob(layout: str, root: pathlib.Path | None = None) -> str:
    root = root if root is not None else LAYOUT_ROOT
    if layout == "none_unsorted":
        return str(root / "none_unsorted" / "data.parquet")
    return str(root / layout / "**" / "*.parquet")


def layout_total_bytes(layout: str, root: pathlib.Path | None = None) -> int:
    root = root if root is not None else LAYOUT_ROOT
    layout_root = root / layout
    return sum(p.stat().st_size for p in layout_root.rglob("*.parquet"))
