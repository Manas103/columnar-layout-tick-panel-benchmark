"""Builds a small (3-day) version of all 4 layouts in a temp directory and
checks the one thing every layout must agree on exactly: the same logical
rows, regardless of how they are partitioned or sorted on disk.
"""

from __future__ import annotations

import duckdb
import pytest

from tickpanel import config, layouts, query


@pytest.fixture(scope="module")
def small_layouts(tmp_path_factory):
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    mp.setattr(config, "N_DAYS", 3)
    mp.setattr(config, "QUERY_N_DAYS", 1)
    root = tmp_path_factory.mktemp("layouts")
    layouts.build_all(n_days=3, root=root)
    yield root
    mp.undo()


def _row_count(glob_path: str) -> int:
    con = duckdb.connect()
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob_path}', hive_partitioning=true)").fetchone()[0]
    con.close()
    return int(n)


def test_all_4_layouts_have_the_same_row_count(small_layouts):
    expected = config.N_INSTRUMENTS * 3 * config.AVG_TICKS_PER_INSTRUMENT_DAY
    for layout in config.LAYOUTS:
        glob_path = layouts.layout_glob(layout, root=small_layouts)
        assert _row_count(glob_path) == expected, layout


def test_all_4_layouts_have_the_same_per_instrument_sum(small_layouts):
    """A checksum (sum of price*size per instrument) that is invariant to
    row order and partitioning: if any layout silently dropped or
    duplicated rows while being reorganized, this would change."""
    checksums = {}
    for layout in config.LAYOUTS:
        glob_path = layouts.layout_glob(layout, root=small_layouts)
        con = duckdb.connect()
        rows = con.execute(
            f"SELECT instrument_id, SUM(price * size) AS s "
            f"FROM read_parquet('{glob_path}', hive_partitioning=true) "
            f"GROUP BY instrument_id ORDER BY instrument_id"
        ).fetchall()
        con.close()
        checksums[layout] = rows

    baseline = checksums["none_unsorted"]
    for layout, rows in checksums.items():
        assert len(rows) == len(baseline)
        for (inst_a, sum_a), (inst_b, sum_b) in zip(baseline, rows):
            assert inst_a == inst_b
            assert sum_a == pytest.approx(sum_b, rel=1e-5), (layout, inst_a)


def test_daily_sorted_layout_is_actually_sorted_by_instrument_within_each_day(small_layouts):
    day_dirs = sorted((small_layouts / "daily_sorted").glob("day=*"))
    assert day_dirs
    con = duckdb.connect()
    for day_dir in day_dirs:
        path = str(day_dir / "data.parquet")
        rows = con.execute(f"SELECT instrument_id FROM read_parquet('{path}')").fetchall()
        ids = [r[0] for r in rows]
        assert ids == sorted(ids)
    con.close()


def test_the_query_returns_the_same_result_from_every_layout(small_layouts):
    results = {}
    con = duckdb.connect()
    for layout in config.LAYOUTS:
        glob_path = layouts.layout_glob(layout, root=small_layouts)
        sql = query.duckdb_sql(glob_path)
        results[layout] = con.execute(sql).fetchall()
    con.close()

    baseline = results["none_unsorted"]
    assert baseline, "query should return at least one row on this fixture"
    for layout, rows in results.items():
        assert len(rows) == len(baseline), layout
        for a, b in zip(sorted(baseline), sorted(rows)):
            assert a[0] == b[0] and a[1] == b[1]
            assert a[2] == pytest.approx(b[2], rel=1e-9)
            assert a[3] == b[3]
            assert a[4] == b[4]
