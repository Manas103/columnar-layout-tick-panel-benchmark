"""The trade-to-prevailing-quote as-of join and the regression gate that
pins its 30 analytic outputs across a storage re-layout. Uses a 3-day,
tmp_path copy of the fixture, not the committed data/asof_fixture, so
these tests never depend on, or mutate, the real pinned snapshot.
"""

from __future__ import annotations

import pytest

from tickpanel import asof_fixture, asof_join, regression_gate


@pytest.fixture(scope="module")
def small_fixture(tmp_path_factory):
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    mp.setattr(asof_fixture, "N_DAYS", 3)
    mp.setattr(asof_fixture, "QUERY_N_DAYS", 2)
    root = tmp_path_factory.mktemp("asof_fixture")
    asof_fixture.build_layouts(root=root, n_days=3)
    yield root
    mp.undo()


def test_asof_join_every_trade_gets_a_prevailing_quote_not_a_future_one():
    """A hand-built mini case: the as-of join must pick the most recent
    quote at or before the trade, never a later one."""
    import datetime

    import duckdb

    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE t (instrument_id INT, ts TIMESTAMP, price DOUBLE, size BIGINT);
        CREATE TABLE q (instrument_id INT, ts TIMESTAMP, bid DOUBLE, ask DOUBLE);
        INSERT INTO t VALUES (0, '2024-01-01 09:30:10', 10.02, 100);
        INSERT INTO q VALUES
            (0, '2024-01-01 09:30:00', 9.98, 10.02),
            (0, '2024-01-01 09:30:05', 9.99, 10.01),
            (0, '2024-01-01 09:30:20', 10.10, 10.20);
        """
    )
    row = con.execute(
        """
        SELECT q.ts FROM t ASOF JOIN q
          ON t.instrument_id = q.instrument_id AND t.ts >= q.ts
        """
    ).fetchone()
    con.close()
    assert row[0] == datetime.datetime(2024, 1, 1, 9, 30, 5)


def test_run_asof_join_aggregates_returns_30_rows(small_fixture):
    rows = asof_join.run_asof_join_aggregates("unsorted", root=small_fixture)
    assert len(rows) == 10 * 2  # QUERY_N_INSTRUMENTS x QUERY_N_DAYS (monkeypatched to 2 days)


def test_aggregates_agree_between_unsorted_and_daily_sorted_layouts(small_fixture):
    unsorted_rows = asof_join.run_asof_join_aggregates("unsorted", root=small_fixture)
    sorted_rows = asof_join.run_asof_join_aggregates("daily_sorted", root=small_fixture)
    diff = regression_gate.diff_against_pinned(sorted_rows, pinned=unsorted_rows)
    assert diff["all_identical"], diff["mismatches"]


def test_slippage_is_small_not_a_multi_point_drift(small_fixture):
    """Regression guard for the two-independent-walks bug (see README
    Findings): average slippage must stay within a few cents of zero, not
    drift by whole points, which would mean the trade and quote streams
    are not actually tracking the same underlying price."""
    rows = asof_join.run_asof_join_aggregates("unsorted", root=small_fixture)
    for row in rows:
        assert abs(row["avg_slippage"]) < 0.5, row


def test_regression_gate_detects_a_real_mismatch():
    pinned = [{"instrument_id": 0, "day": "2024-01-01", "n_trades": 10, "avg_slippage": 0.01, "avg_prevailing_spread": 0.02}]
    tampered = [{"instrument_id": 0, "day": "2024-01-01", "n_trades": 10, "avg_slippage": 0.05, "avg_prevailing_spread": 0.02}]
    diff = regression_gate.diff_against_pinned(tampered, pinned=pinned)
    assert not diff["all_identical"]
    assert diff["n_mismatches"] == 1


def test_regression_gate_write_and_load_round_trip(tmp_path):
    path = tmp_path / "pinned.json"
    rows = [{"instrument_id": 1, "day": "2024-01-02", "n_trades": 5, "avg_slippage": 0.0, "avg_prevailing_spread": 0.03}]
    regression_gate.write_pinned(rows, path=path)
    loaded = regression_gate.load_pinned(path=path)
    assert loaded == rows
