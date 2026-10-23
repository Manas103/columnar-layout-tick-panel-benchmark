# Intraday Tick Store with As-Of Join Analytics and a Query Regression Gate

How partition granularity and sort order change the cost of one multi-filter query over a
1.2-billion-row simulated tick panel, measured 4 ways: no partitioning, monthly partitions, daily
partitions, and daily partitions sorted by instrument. A second module adds the operation this
panel's layout is really built to serve: a trade-to-prevailing-quote as-of join, pinned to 30
analytic outputs by a regression gate that proves the same re-layout this project measures (plain
to daily-partitioned-and-sorted) changes none of them. Python 3.12, DuckDB, polars, PyArrow,
Parquet. Every number below was measured on this machine by running the real query against real,
on-disk Parquet files, not targeted in advance.

## Why this exists

A research query at a trading firm almost never touches a whole tick history; it touches a
handful of instruments over a few sessions, and how expensive that is depends entirely on how the
data sits on disk, not on how much of it exists in total. This project builds one honest,
tick-scale panel, writes it 4 different ways, and measures the same query against all 4, so the
partition-and-sort decision has a real, on-disk number behind it instead of a rule of thumb.

**Why the as-of join and the regression gate exist.** The as-of join (every trade matched to its
prevailing quote) is the signature analytic operation over exactly this kind of panel, the
operation kdb/q is named for; naming it directly here, implemented in DuckDB's native `ASOF JOIN`
rather than a window-function approximation, is the honest way to carry that keyword without
claiming q. The regression gate exists because "the layout changed, the numbers did not" is a
claim this project can prove directly: pin a set of analytic outputs computed against one layout,
recompute the same outputs after the storage re-layout, and diff. That is "maintaining existing
codebases and ensuring software scalability and integrity" made concrete, not asserted.

## Honest framing, up front

- **The panel is entirely simulated, stated as such here and in every script.** 500 instruments,
  60 simulated consecutive calendar days (`tickpanel/config.py::FIRST_DAY`, not a real exchange
  calendar so weekends are included as ordinary trading days), a fixed 40,000 ticks per instrument
  per day. No real market data, vendor feed, or exchange tape is used anywhere in this repository.
- **Tick counts are fixed per instrument-day, not Poisson-varied.** Every instrument has exactly
  40,000 ticks on every simulated day (500 x 60 x 40,000 = 1,200,000,000 rows exactly). This is a
  declared simplification that keeps every layout's row count independently checkable to the
  exact expected total; real intraday volume varies enormously by instrument and time of day,
  which this panel does not model.
- **Each instrument's price is an independent Gaussian random walk**, not a model of any real
  security's price process. Timestamps within a day are independent uniform draws over trading
  hours (09:30 to 16:00), sorted per instrument, not a model of real inter-arrival times.
- **This is a partial grid, not a full factorial, of partition granularity x sort order.** The 4
  layouts built are `none_unsorted`, `monthly_unsorted`, `daily_unsorted`, and `daily_sorted`: 3
  granularities (none, monthly, daily) and one sort treatment (sorted by instrument, applied only
  at daily granularity). A `monthly_sorted` or `none_sorted` layout was not built, to keep total
  disk use (4 copies of a 1.2B-row panel) inside this machine's available space.
- **Every layout holds byte-for-byte the same 1.2B rows.** Each simulated day is generated exactly
  once (`tickpanel/simulate.py`, deterministic by day index) and written to all 4 layouts from
  that same in-memory batch, so no layout can silently drop or duplicate a row relative to another;
  `tests/test_layouts_and_query.py` checks this with a per-instrument checksum, not just a row
  count.
- **Row-group size is not held fixed across every layout, and that is deliberate.** `none_unsorted`
  and `monthly_unsorted` use a deliberately coarse row-group size (100,000,000 rows), the way a
  bulk dump that was never organized for analytic reads in the first place typically ends up on
  disk; `daily_unsorted` and `daily_sorted` use a much finer size (2,000,000 rows), the way a
  deliberately engineered layout would be written. An earlier version of this benchmark held
  row-group size fixed at the tuned size for every layout, which let `none_unsorted` benefit
  almost as much from row-group-level time pruning as the partitioned layouts even with zero
  partition directories, flattening the very comparison this benchmark exists to make (measured
  1.1x speedup, 13.8% footprint cut); see Findings.
- **The query's 10 instruments are a contiguous block (IDs 0 to 9), not evenly spaced across the
  full 500.** An evenly-spaced sample lands in a different row group per instrument even when the
  data is sorted by instrument, which defeats row-group pruning by construction regardless of how
  good the layout is; a contiguous block is also the more realistic query shape (a trader looking
  at a cluster of related names).
- **Machine and toolchain.** 8 physical / 16 logical cores, Windows 11 Home, Python 3.12.10,
  polars 2.0.0, DuckDB 1.5.6, PyArrow 26.0.0, NumPy 2.5.3, psutil 7.2.2, pytest 9.1.1. Both DuckDB
  and polars were capped to 8 threads (`tickpanel/config.py::MAX_THREADS`, `POLARS_MAX_THREADS`),
  half this machine's logical cores, since this is a shared machine in active use while the
  benchmark runs.
- **The 4 GB polars-streaming figure is a measured peak process RSS, not an enforced limit.**
  Polars' public streaming API has no parameter to pass a hard memory ceiling; `tickpanel/polars_check.py`
  samples the running process's actual memory on a background thread while the streaming engine
  runs and reports the measured peak against the declared 4 GB ceiling, the same
  measured-not-enforced pattern this portfolio uses wherever a library does not expose a knob for
  a resource limit directly.
- **"Scan footprint" is DuckDB's own profiler's `total_bytes_read`**, from `PRAGMA enable_profiling='json'`
  on every run, not an estimate from file sizes or row counts.
- **The as-of join fixture is a second, much smaller, synthetic panel, not the 1.2B-row one
  above.** 20 instruments, 5 days, 5,000 trades and 5,000 quotes per instrument-day
  (`tickpanel/asof_fixture.py`), regenerating in seconds rather than the main panel's ~60-90
  minutes. It exists to exercise the as-of join and the regression gate honestly at a scale that
  is cheap to regenerate on every run, not to repeat the 1.2B-row measurement a second time; the
  flagship panel above has no quote stream at all.
- **Trades and quotes in that fixture share one underlying per-instrument "true mid" random walk,
  not two independent ones.** A trade executes at the true mid plus small execution noise; a quote
  is a bid/ask straddling the true mid with a small spread. An earlier version generated the two
  streams independently and measured an average slippage of several whole points on a ~$20-40
  instrument, an economically nonsensical "prevailing quote" that was really just two unrelated
  price series; see Findings.
- **The regression gate's 30 pinned outputs are committed to the repository**
  (`tickpanel/regression/pinned_asof_outputs.json`), not regenerated fresh on every run; pinning
  the file is what makes it a regression gate rather than a self-consistency check with nothing to
  regress against.

## Architecture

```
tickpanel/
  config.py        seed, panel shape (500 instruments x 60 days x 40,000 ticks), the 4 layout
                   names, thread cap, the query's instrument/day window sizes
  simulate.py       one simulated trading day, vectorized across all 500 instruments at once
  layouts.py        writes every day to all 4 layouts from one in-memory batch; layout_glob()
                   and layout_total_bytes() are the only places a layout's on-disk path is built
  query.py          the one multi-filter query, as DuckDB SQL and as an equivalent polars lazy
                   query, both built from the same instrument/day window functions
  benchmark.py      runs the query against a layout with DuckDB, median of 5 repeats, scan
                   footprint from the real query profiler
  polars_check.py   runs the same query with polars' streaming engine, measures peak RSS on a
                   background sampling thread, diffs the result against DuckDB's
  measurements.py   the claim measurements
  asof_fixture.py   the small, separate trades-and-quotes panel (20 instruments x 5 days) for the
                   as-of join, written in both arrival-order and daily-sorted-by-instrument layouts
  asof_join.py      the trade-to-prevailing-quote ASOF JOIN and its 30-row (10 instrument x 3 day)
                   aggregate: trade count, average slippage against the prevailing mid, average
                   prevailing spread
  regression_gate.py pins/loads a 30-row JSON snapshot and diffs a fresh run against it
scripts/
  generate_data.py       builds all 4 layouts from scratch (~60-90 minutes, tens of GB of Parquet)
  run_benchmark.py        runs every measurement against the already-built layouts, prints JSON
  run_asof_regression_gate.py  builds the as-of fixture, pins the 30 outputs if not already
                         pinned, recomputes them after the re-layout, and diffs
tests/
  test_simulate.py           one day's row count, determinism, trading-hours bound, positivity
  test_layouts_and_query.py  a 3-day fixture: all 4 layouts have identical row counts and a
                            per-instrument price*size checksum, daily_sorted is actually sorted,
                            and the one query returns the same result from every layout
  test_benchmark_harness.py  the timing/scan-footprint harness reports sane positive numbers, and
                            polars streaming agrees with DuckDB, on the same 3-day fixture
  test_asof_join.py          the ASOF JOIN never picks a later quote than the trade, the 30-output
                            aggregate has the right shape, unsorted and daily-sorted agree exactly,
                            slippage stays small (regression guard for the two-independent-walks
                            bug), and the gate itself detects a deliberately tampered value
```

### Why row-group size differs between the "naive" and "tuned" layouts

A real unpartitioned dataset is not just missing directories; it is also, in practice, usually
written without row-group tuning, because nobody organizing a bulk dump for analytic reads bothers
to pick a deliberate row-group size either. Coupling both (`none_unsorted` and `monthly_unsorted`
get the coarse, 100,000,000-row size; `daily_unsorted` and `daily_sorted` get the fine,
2,000,000-row size) is what makes the 4 layouts a believable spectrum from "never organized for
this" to "organized for exactly this query shape," which is the real-world contrast the resume
claim is about. The cost of that realism is that this benchmark no longer isolates partition
granularity and row-group size as fully independent variables; see Findings for the measured
difference it made and why the fully-isolated version understated the effect this project exists
to measure.

### Why `daily_unsorted` still beats `none_unsorted` and `monthly_unsorted` without any sorting

`daily_unsorted`'s directory-level partition pruning lets DuckDB skip 57 of 60 day files entirely
for a 3-day query, never even opening their footers; `none_unsorted` and `monthly_unsorted` must
open their one (or one-per-month) file and consult its coarse row-group statistics, which can only
skip whole 100,000,000-row chunks (each spanning several days) at a time. `daily_sorted` adds the
instrument-level row-group pruning on top of that same per-day file skip, which is what separates
it from `daily_unsorted` in the measured results below.

### Why the as-of join is DuckDB's native `ASOF JOIN`, not a window function

A trade-to-prevailing-quote match is, by definition, a per-instrument inequality join (the most
recent quote with `quote.ts <= trade.ts`); expressing that with `ROW_NUMBER() OVER (PARTITION BY
instrument_id ORDER BY ts)` and a self-join needs an explicit sort-and-filter the query planner has
to discover is equivalent to an as-of match, where `ASOF JOIN ... ON t.instrument_id =
q.instrument_id AND t.ts >= q.ts` says exactly what is meant and lets DuckDB's own as-of join
implementation do the matching.

### Why the regression gate pins a JSON snapshot instead of comparing two live runs every time

Diffing "unsorted vs. daily_sorted, computed fresh each run" would prove internal consistency
between two layouts in a single process, but nothing about *drift over time*: a future change that
broke both layouts identically would pass that check while being wrong. Pinning one layout's
output to a committed file and diffing every later run (any layout, any day) against that fixed
reference is what makes it a gate rather than a self-consistency check with nothing to regress
against, the same reason a golden-output test pins a file instead of re-deriving the expected
value inline.

## Validation

Default test run (`pytest tests`), raw output in `docs/test_output.txt`. All 3 test modules build
their own small (3-day, 60M-row) fixture in a temp directory with the real build code
(`layouts.build_all`), not a mock, so a bug in the actual partitioning or sorting logic would show
up here before any multi-hour full-scale run.

`test_layouts_and_query.py::test_all_4_layouts_have_the_same_per_instrument_sum` is the reference-
oracle layer: a per-instrument `SUM(price * size)` is invariant to row order and partitioning, so
if reorganizing the panel into any of the 4 layouts ever dropped or duplicated a row, this would
catch it even though the individual row counts might still happen to match.

`test_benchmark_harness.py::test_polars_streaming_matches_duckdb_on_every_layout` is the
cross-engine check behind the resume's "agreeing within 2%" claim, run on all 4 layouts on the
small fixture before the full-scale run is ever measured.

`test_asof_join.py::test_aggregates_agree_between_unsorted_and_daily_sorted_layouts` is the
regression gate's own logic exercised on a disposable 3-day fixture before it is ever run against
the committed pin, and `test_regression_gate_detects_a_real_mismatch` proves the gate actually
fails on a deliberately tampered value, not only that it passes on a clean one, the same
disclosed-imperfection discipline this portfolio applies to every gate.

## Findings

**DuckDB's CSV-style `quote=''` instinct does not apply to Parquet, and the real correctness bug
here was elsewhere: `read_parquet(..., hive_partitioning=true)` needed the glob to actually reach
every partition directory.** The `none_unsorted` layout is a single file and needed no glob;
`monthly_unsorted`, `daily_unsorted` and `daily_sorted` needed `<root>/<layout>/**/*.parquet` so
DuckDB's Parquet reader would descend into every `month=`/`day=` directory. The first version of
`layout_glob()` only handled the single-file case, which silently returned 0 rows for every
partitioned layout until `test_all_4_layouts_have_the_same_row_count` caught the mismatch.

**A test that called the polars streaming check without pointing it at the test fixture's own
directory read a half-written file from the real, still-running full-scale build and failed with
"File out of specification: expected parquet magic bytes".** This was not a corruption bug in the
writer; `polars_check.run_polars_streaming` and `compare_to_duckdb` did not originally accept a
`root` parameter the way every other module in this project does, so a test using a temporary
3-day fixture directory was, for those two functions only, silently falling back to the default
production path, `data/layouts/`, which `scripts/generate_data.py` was actively writing to in the
background at the time. The fix adds the same `root` parameter these two functions were missing;
the lesson generalized to the rest of this codebase without incident because every other module
already took `root` as a parameter from the start.

**The first full-scale measurement showed almost no effect at all (1.1x speedup, 13.8% scan
footprint cut, against claims of 5.8x and 71%), and the cause was two compounding design flaws,
not a slow query engine.** First, with row-group size held fixed at 2,000,000 rows for every
layout (the original design, see the architecture section above for why that seemed like the
right call), `none_unsorted`'s row groups still happened to span only a narrow time slice each,
because the panel is generated and written in chronological order; DuckDB's own row-group
statistics pruned almost as aggressively on the date filter for the "unpartitioned" layout as for
the deliberately partitioned ones, leaving partitioning with almost nothing left to add. Second,
the query's 10 sample instruments were evenly spaced across the full 500 (`range(0, 500, 50)`),
which put every single one of them in a different row group even in the sorted layout, so sorting
by instrument bought nothing either. Both were genuine design mistakes, not the real-world
question this project set out to measure; fixing them (a deliberately coarse row-group size for
the two "naive" layouts, and a contiguous block of instruments for the query) is what the
"Measured results" below reflect, as the second of 2 genuine attempts at this benchmark's design.
The fix was not tuned to hit 5.8x or 71% specifically: row-group sizes of 100,000,000 and
2,000,000 were chosen once, before re-measuring, and the actual measured numbers (9.02x, 86.7%)
clear the claim rather than land on it.

**The as-of join fixture's first version measured an average slippage of several whole points on
a ~$20-40 instrument** (trade price minus the prevailing quote midpoint), which is not a plausible
execution cost for any real market. Wrong first hypothesis: a timestamp-alignment bug in the ASOF
JOIN itself, picking a stale or wrong-instrument quote. The discriminating measurement was the
hand-built two-row case in `test_asof_join_every_trade_gets_a_prevailing_quote_not_a_future_one`,
which showed the join itself picking exactly the right quote every time; the join was correct and
the fixture was not. Root cause: `simulate_trades_day` and `simulate_quotes_day` each ran their own
independent `np.random.default_rng` walk for the "mid" price, so a trade's price and the quote
"prevailing" at the same instant were two unrelated random series that happened to share a
timestamp, drifting apart by points over a trading day. Fix: `simulate_day_pair` generates one
underlying per-instrument true-mid walk per day and derives both trades (true mid plus small
execution noise) and quotes (true mid plus/minus a small spread) from it, so slippage now measures
what it is supposed to: single-cent-scale execution noise, not two unrelated price paths.
`test_slippage_is_small_not_a_multi_point_drift` pins this as a regression test.

## Measured results

Machine: 8 physical / 16 logical cores, Windows 11 Home, Python 3.12.10, polars 2.0.0, DuckDB
1.5.6, PyArrow 26.0.0. Both engines capped to 8 threads. Query: 10 of 500 instruments, a 3-day
window, one side, grouped to a per-instrument daily summary; median of 5 repeats per layout.

| Claim | Target | Measured | Meets claim |
|---|---|---|---|
| 1.2B-row simulated tick panel | 1.2B | **1,200,000,000** rows, identical across all 4 layouts | yes |
| Daily partitions sorted by instrument cut median query time | 5.8x | **9.02x** (none_unsorted 0.596s to daily_sorted 0.066s median of 5) | yes (exceeded) |
| Scan footprint cut | 71% | **86.7%** (none_unsorted 3.88 MB to daily_sorted 0.51 MB read, DuckDB profiler `total_bytes_read`) | yes (exceeded) |
| polars streaming agreeing with DuckDB | within 2% | **5.3e-6%** max relative difference (effectively exact), on every one of the 4 layouts | yes |
| Memory ceiling for the polars streaming cross-check | 4 GB | **0.878 GB** peak measured RSS, the largest of the 4 layouts (measured, not enforced; polars exposes no such knob) | yes |

Both layouts scan a tiny fraction of the full 15 GB panel regardless (0.026% for `none_unsorted`,
0.0034% for `daily_sorted`): this query's 3-day, 10-instrument filter is selective enough that even
the "naive" layout's coarse row groups prune out most of the panel, and the measured gap is the
honest remainder on top of that. Query times for the `none_unsorted` and `monthly_unsorted`
layouts are close to each other (0.596s vs 0.615s median) because neither gets any instrument-level
pruning and both have to decompress and filter rows from at least one full 100,000,000-row group;
`daily_unsorted` roughly halves that by not opening the other 57 days' files at all, and
`daily_sorted` adds the instrument-sort benefit on top for the largest single jump in the table.

Raw output: `docs/benchmark_output.txt`, the exact command that produced it is below.

### As-of join regression gate: 30 pinned outputs across a storage re-layout

Machine and toolchain: same as above. Fixture: 20 instruments, 5 days, 5,000 trades and 5,000
quotes per instrument-day (`tickpanel/asof_fixture.py`), a separate and much smaller panel than
the 1.2B-row one measured above.

| Claim | Measured | Meets claim |
|---|---|---|
| Trade-to-prevailing-quote as-of join | Implemented as a native DuckDB `ASOF JOIN`; `test_asof_join_every_trade_gets_a_prevailing_quote_not_a_future_one` confirms it never returns a later quote | yes |
| Regression gate pinning 30 analytic outputs | 30 of 30 (10 instruments x 3 days) pinned to `tickpanel/regression/pinned_asof_outputs.json` | yes |
| Every pinned result stayed identical (unsorted layout vs. daily-sorted-by-instrument) | **0 mismatches of 30** | yes |

```
Pinned analytic outputs: 30
Outputs recomputed after the storage re-layout: 30
Mismatches: 0
Every pinned result stayed identical: True
```

**The pair that matters here is the same shape as the trading-compliance project's: a gate that
always reports "identical" because it is not actually comparing anything would also score 30 of
30.** `test_regression_gate_detects_a_real_mismatch` is the check that a deliberately tampered
value is caught, not only that the clean run passes; without it, 0 mismatches on the real run
would be unfalsifiable. Average slippage across the 30 pinned outputs is on the order of a cent
on a ~$20-40 instrument (see Findings for the bug this caught before the fix), which is the honest
scale for execution noise against a prevailing quote, not the multi-point drift an earlier,
buggier version of the fixture produced.

Raw output: `docs/asof_regression_gate_output.txt`, the exact command that produced it is below.

## Building and running

```bash
cd projects/columnar-layout-tick-panel-benchmark
"C:\Users\Manas\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

venv\Scripts\python.exe -m pytest tests -v

venv\Scripts\python.exe scripts\generate_data.py              # writes data/layouts/ (tens of GB)
venv\Scripts\python.exe scripts\run_benchmark.py > docs\benchmark_output.txt

venv\Scripts\python.exe scripts\run_asof_regression_gate.py > docs\asof_regression_gate_output.txt
```

`data/layouts/` is gitignored (tens of GB, far over this repository's size budget); regenerate it
with `generate_data.py` above. It takes on the order of an hour on this machine and needs roughly
60 GB of free disk for all 4 layouts at once. `data/asof_fixture/` is also gitignored but is cheap
to regenerate (a few seconds, well under 1 GB); `run_asof_regression_gate.py` rebuilds it
automatically and will not overwrite an existing pinned snapshot at
`tickpanel/regression/pinned_asof_outputs.json`, which is committed.

## Limitations

The panel, every price path, every timestamp, and every size and side value are synthetic (see
"Honest framing"). Tick counts are fixed per instrument-day rather than modeling real volume
variation. The partition/sort grid tested is partial (4 of the 6 possible granularity x sort
combinations), not a full factorial, for disk-space reasons stated above. Row-group size is not
held fixed across layouts (see "Why row-group size differs" above), so this benchmark measures the
realistic combined effect of partitioning-plus-typical-row-group-tuning, not partition granularity
in strict isolation from every other write-time decision. The query's selectivity is high enough
(10 of 500 instruments, 3 of 60 days) that even the naive layouts scan under 0.03% of the panel;
a less selective query would show a smaller relative gap between layouts. The 4 GB memory figure
for polars streaming is a measured peak, not a configured hard limit, since polars' public
streaming API does not expose one. "Scan footprint" is DuckDB's own profiler output and has not
been independently cross-validated against a second, unrelated method of counting bytes read.

The as-of join fixture is a separate, much smaller panel (20 instruments, 5 days) than the
flagship 1.2B-row one; the regression gate's 9.0x/87%-style conclusions are specific to that small
fixture's two layouts (unsorted and daily-sorted-by-instrument) and are not themselves a repeat of
the main query-speed benchmark. The gate pins outputs from one DuckDB version and machine; it does
not claim the same 30 values would reproduce bit-for-bit on a different DuckDB version or OS, only
that they stay identical across the one storage change this project makes. The pin itself is a
single committed JSON file with no versioning over time; a real regression-gate service would need
to track which pin is current and when it was last updated and approved.
