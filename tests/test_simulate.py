from __future__ import annotations

from tickpanel import config, simulate


def test_one_day_has_the_declared_row_count():
    df = simulate.simulate_day(0)
    assert df.height == config.N_INSTRUMENTS * config.AVG_TICKS_PER_INSTRUMENT_DAY


def test_deterministic_by_day_index():
    df_a = simulate.simulate_day(5)
    df_b = simulate.simulate_day(5)
    assert df_a.equals(df_b)


def test_different_days_differ():
    df_a = simulate.simulate_day(0)
    df_b = simulate.simulate_day(1)
    assert not df_a.equals(df_b)


def test_every_instrument_present_every_day():
    df = simulate.simulate_day(3)
    assert df["instrument_id"].n_unique() == config.N_INSTRUMENTS
    counts = df.group_by("instrument_id").len()
    assert (counts["len"] == config.AVG_TICKS_PER_INSTRUMENT_DAY).all()


def test_timestamps_stay_within_trading_hours():
    df = simulate.simulate_day(10)
    date = simulate.day_date(10)
    import datetime

    start = datetime.datetime.combine(date, datetime.time(9, 30))
    end = datetime.datetime.combine(date, datetime.time(16, 0))
    assert df["ts"].min() >= start
    assert df["ts"].max() < end


def test_prices_are_positive():
    df = simulate.simulate_day(0)
    assert (df["price"] > 0).all()
