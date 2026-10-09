"""Sanity checks for the timing/scan-footprint harness and the polars
streaming cross-check, on the same small (3-day) fixture as
test_layouts_and_query.py, not a performance assertion (this module only
checks the harness measures *something* sane, not any particular speed).
"""

from __future__ import annotations

import pytest

from tickpanel import benchmark, config, layouts, polars_check, query


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


def test_run_duckdb_once_reports_positive_time_and_bytes(small_layouts):
    glob_path = layouts.layout_glob("daily_sorted", root=small_layouts)
    elapsed, bytes_read, n_rows = benchmark.run_duckdb_once(glob_path, threads=2)
    assert elapsed > 0
    assert bytes_read > 0
    assert n_rows > 0


def test_daily_sorted_scans_fewer_bytes_than_none_unsorted_on_this_fixture(small_layouts):
    none_glob = layouts.layout_glob("none_unsorted", root=small_layouts)
    sorted_glob = layouts.layout_glob("daily_sorted", root=small_layouts)
    _, none_bytes, _ = benchmark.run_duckdb_once(none_glob, threads=2)
    _, sorted_bytes, _ = benchmark.run_duckdb_once(sorted_glob, threads=2)
    assert sorted_bytes <= none_bytes


def test_polars_streaming_matches_duckdb_on_every_layout(small_layouts):
    import duckdb

    con = duckdb.connect()
    for layout in config.LAYOUTS:
        glob_path = layouts.layout_glob(layout, root=small_layouts)
        sql = query.duckdb_sql(glob_path)
        duckdb_rows = con.execute(sql).fetchall()
        comparison = polars_check.compare_to_duckdb(layout, duckdb_rows, root=small_layouts)
        assert comparison["same_group_keys"], layout
        assert comparison["within_2pct"], layout
    con.close()
