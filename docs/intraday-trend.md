# Intraday Trend-Following Strategy

5-minute opening-range breakout with VWAP trailing exit.

## Spec compliance

Per [master spec §17](../README.md), this is **intraday-aware swing trading**,
not autonomous scalping. Holding periods range from one bar to a full
session. Entries are blocked in the first 45 minutes (opening range + 15 min
buffer) and in the last 15 minutes of the cash session. The strategy emits
long-only weights summing to ≤ 1.0 — same contract as the daily strategies,
so it plugs into the existing backtester and paper-trade runner without
changes.

## Signal

Per symbol, per 5-min bar:

| Condition | Action |
|---|---|
| `close > opening_range_high` AND `close > vwap` AND `45 min <= minutes_since_open <= 375` | Enter long |
| `close <= vwap` while long | Exit |
| Peak-to-trough drawdown >= `stop_loss` (default 1%) | Exit, no re-entry this session |
| `minutes_since_open >= 375` | Force flat |

Across symbols, weight is split equally among active longs (each capped at
`max_gross / n_active`).

## Configuration

```python
IntradayTrendFollowing(
    opening_range_minutes=30,      # 09:30 – 10:00 ET defines the range
    entry_block_minutes=15,        # 10:00 – 10:15 ET: still no entries
    exit_block_minutes=15,         # 15:45 – 16:00 ET: force flat
    stop_loss=0.01,                # 1% peak-to-trough kills the trade
    max_gross=1.0,                 # cap gross exposure across symbols
)
```

## Real-data smoke test

Tested on 35 days of 5-min Alpaca bars across {SPY, QQQ, IWM, AAPL, MSFT,
NVDA}:

- 2,625 bars in the union index
- 962 bars with at least one long active (~37% time-in-market)
- Rough Sharpe (5-min, no slippage): -2.25 (sample too short to be
  meaningful)
- Win/loss bar ratio: 455 / 504

**Note**: 30 days is far too short to validate any intraday strategy. The
mechanics are confirmed (entries trigger, exits flatten, session-close
guard works). Longer history (90d+ rolling) and a proper walk-forward tune
are required before this strategy moves out of the dev tier.

## Backtest contract

`IntradayTrendFollowing.generate_signals(prices)` raises
`NotImplementedError` — the daily backtester contract doesn't carry the
OHLCV+volume needed for VWAP. Use:

```python
strat.generate_weights_for_panel({"SPY": ohlcv_df, "QQQ": ohlcv_df, ...})
```

where each value is a DatetimeIndex'd DataFrame with columns `open, high,
low, close, volume`.

## Files

- `packages/strategies/intraday_trend.py` — strategy implementation
- `packages/strategies/tests/test_intraday_trend.py` — 9 behaviour tests
- `packages/features/intraday.py` — VWAP, opening-range, session features
