# Training pipeline

This doc explains what gets trained, when, where the data lives, and how to
inspect freshness. Everything here runs locally on your PC — there is no
cloud dependency for the core pipeline.

## TL;DR

0. **Sanity check:** `make doctor` — reports which data sources are enabled,
   what's in the parquet cache, and whether champion params have been
   trained. Prints the exact next command.
1. **First install:** `make first-run` — one shot that runs `doctor → pretrain
   → retune`. Or run each step on its own:
   - `make pretrain` pulls 20 years of daily bars, 90 days of 5-minute
     intraday bars (if Alpaca keys set), and FRED macro series (if key set)
     into `data/parquet/`. ~5–10 minutes on a clean install. Idempotent.
   - `make retune` runs walk-forward and writes `data/params/champion.json`.
2. **Nightly @ 03:00 UTC:** the worker refreshes yesterday's bars and pulls a
   fresh sentiment snapshot.
3. **Weekly @ Sunday 05:00 UTC:** walk-forward retune refits a small set of
   strategy parameters and promotes the challenger only if it clears the
   same promotion gate the live champion/challenger flow uses. **NaN/inf
   metrics are rejected outright** — the bot will never "succeed" on broken
   data.

You don't need to start the workflows by hand once `make schedules` has been
run against your local Temporal cluster.

## Data sources

| Source | Adapter | Cost | What it powers |
|---|---|---|---|
| Alpaca market data (IEX feed) | `packages/data/adapters/alpaca_data.py` | Free with paper keys | Daily + 5-min bars |
| Yahoo Finance (chart API) | `packages/data/adapters/yfinance.py` | Free, no key | Daily-bar fallback |
| FRED | `packages/data/adapters/fred.py` | Free with key | VIX, unemployment, CPI, yield curve |
| Reddit + RSS | `packages/data/adapters/sentiment.py` | Free, no key | Sentiment overlay agent |

Paid adapters (Polygon, Alpha Vantage, Finnhub, SEC EDGAR) are wired but
inactive unless you set their API keys. The agent layer reads through the
registry (`packages/data/registry.py`) so swapping in a paid feed never
requires editing agent code.

## What gets written to disk

```
data/
  parquet/
    daily/{symbol}.parquet      # 20yr OHLCV from Alpaca → yfinance fallback
    intraday/{symbol}.parquet   # 90d 5-min OHLCV from Alpaca
    macro/{series_id}.parquet   # FRED observations
    sentiment/latest.json       # Rolling 24h sentiment snapshot, per symbol
  params/
    champion.json               # Current live parameter set
    retune_log.jsonl            # One line per weekly retune (audit trail)
```

Universe defaults to 4 index ETFs + 11 sector ETFs + 18 megacaps. Override
with `PRETRAIN_UNIVERSE=AAPL,MSFT,NVDA`.

Locations are overridable via env vars: `DATA_PARQUET_ROOT` and
`DATA_PARAMS_ROOT`. Tests rely on this — please don't hardcode paths.

## The cron jobs

Both jobs are Temporal workflows. The worker entrypoint registers them as
activities; `make schedules` installs the cron specs.

### Nightly refresh (`data.nightly_refresh`, daily @ 03:00 UTC)

```
packages/data/jobs/nightly_refresh.py
```

Calls `pretrain.run(refresh_after_days=0.5)` so only files older than 12h
get refetched, plus a fresh aggregated sentiment snapshot. Idempotent —
re-running within the freshness window is a no-op.

### Weekly retune (`data.weekly_retune`, Sun @ 05:00 UTC)

```
packages/data/jobs/weekly_retune.py
packages/backtests/walk_forward.py
```

Per-symbol it:

1. Loads close-price history from `data/parquet/daily/{symbol}.parquet`.
2. Refits a small parameter grid (`fast_window`, `slow_window`,
   `zscore_threshold`) on a rolling 2-year window.
3. Evaluates the best in-sample candidate on the next 60 days OOS.
4. Calls the existing `promotion_gate` from
   `packages/backtests/champion_challenger.py` — same rules as the live
   champion-vs-challenger flow.
5. Promotes only if the challenger beats Sharpe by ≥ 10% with no max-DD
   regression. Writes the new champion to `data/params/champion.json`
   atomically; appends an audit entry to `data/params/retune_log.jsonl`.

The same 60-day promotion gate and 5%→100% canary schedule still apply to
the live broker — walk-forward only changes the *paper* champion params.

## Inspecting freshness

- **Cockpit:** the **Data Sources** panel polls `/data/sources` every 15s and
  shows last-update time + file count for each source.
- **CLI:** `ls -la data/parquet/daily/ | head` works too.
- **Logs:** the nightly + weekly jobs emit a `... summary: {...}` log line at
  the end of each run. Tail with `make logs`.

## Local commands

```sh
make pretrain    # first-install bootstrap
make retune      # one-off walk-forward retune (without Temporal)
make schedules   # install nightly + weekly cron schedules
make test        # unit tests for adapters, walk-forward, jobs
```

## Failure modes & fallbacks

- Alpaca down → daily falls back to yfinance (intraday silently skipped).
- yfinance down → daily fails for that symbol; partial summary returned.
- FRED key missing → macro is skipped, the rest still runs.
- Reddit / RSS errors → sentiment job swallows per-source failures and
  returns whatever it managed to collect.
- Champion file missing or corrupt → defaults to `ParamSet()` (the v3.1
  starter knobs), so the system never falls into an inconsistent state.

## Where to look in the code

| Concern | File |
|---|---|
| Bootstrap | `packages/data/pretrain.py` |
| Adapters | `packages/data/adapters/{yfinance,alpaca_data,sentiment,fred}.py` |
| Registry | `packages/data/registry.py` |
| Nightly job | `packages/data/jobs/nightly_refresh.py` |
| Weekly job | `packages/data/jobs/weekly_retune.py` |
| Walk-forward math | `packages/backtests/walk_forward.py` |
| Temporal wiring | `packages/agents/temporal_workflow.py` |
| Schedule installer | `packages/data/jobs/scheduler.py` |
| Cockpit panel | `apps/cockpit/app/components/DataSourcesPanel.tsx` |
| Cockpit API endpoint | `apps/api/main.py` → `/data/sources` |
