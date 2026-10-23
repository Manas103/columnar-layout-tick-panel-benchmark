"""The trade-to-prevailing-quote as-of join: for every trade, the
prevailing quote is the most recent quote for that same instrument with a
timestamp at or before the trade's own timestamp, DuckDB's native ASOF
JOIN (`ON t.instrument_id = q.instrument_id AND t.ts >= q.ts`), not a
window function or a self-join simulating one. This is the as-of join
kdb/q is named for; the resume names the operation rather than claiming q,
since claiming q here would be false (see README).
"""

from __future__ import annotations

import duckdb

from tickpanel import asof_fixture


def _window_predicate(instrument_ids: tuple[int, ...], day_indices: tuple[int, ...]) -> tuple[str, list]:
    ids_sql = ", ".join(str(i) for i in instrument_ids)
    day_strs = [asof_fixture.day_date(d).isoformat() for d in day_indices]
    return ids_sql, day_strs


def run_asof_join_aggregates(layout: str, root=None) -> list[dict]:
    """Returns 30 analytic output rows (10 instruments x 3 days): per
    instrument-day, the count of trades, the average signed slippage of
    each trade's price against its prevailing quote midpoint, and the
    average prevailing spread, all computed from one ASOF JOIN."""
    instrument_ids = asof_fixture.query_instrument_ids()
    day_indices = asof_fixture.query_day_indices()
    ids_sql, day_strs = _window_predicate(instrument_ids, day_indices)
    day_list_sql = ", ".join(f"DATE '{d}'" for d in day_strs)

    trades_glob = asof_fixture.trades_glob(layout, root)
    quotes_glob = asof_fixture.quotes_glob(layout, root)

    con = duckdb.connect()
    try:
        sql = f"""
        WITH joined AS (
            SELECT
                t.instrument_id AS instrument_id,
                CAST(t.ts AS DATE) AS day,
                t.price AS trade_price,
                (q.bid + q.ask) / 2.0 AS mid,
                (q.ask - q.bid) AS spread
            FROM read_parquet('{trades_glob}', hive_partitioning = true) t
            ASOF JOIN read_parquet('{quotes_glob}', hive_partitioning = true) q
                ON t.instrument_id = q.instrument_id AND t.ts >= q.ts
            WHERE t.instrument_id IN ({ids_sql})
              AND CAST(t.ts AS DATE) IN ({day_list_sql})
        )
        SELECT
            instrument_id,
            day,
            COUNT(*) AS n_trades,
            AVG(trade_price - mid) AS avg_slippage,
            AVG(spread) AS avg_prevailing_spread
        FROM joined
        GROUP BY instrument_id, day
        ORDER BY instrument_id, day
        """
        rows = con.execute(sql).fetchall()
        cols = [d[0] for d in con.description]
    finally:
        con.close()

    out = []
    for row in rows:
        record = dict(zip(cols, row))
        record["day"] = record["day"].isoformat()
        out.append(record)
    return out
