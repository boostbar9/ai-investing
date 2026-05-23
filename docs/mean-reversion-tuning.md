# Mean-Reversion Walk-Forward Tuning

Generated: 2026-05-23T23:17:27.284539+00:00

## Setup

- Panel: SPY + QQQ + IWM, 5032 bars (2006-05-23 -> 2026-05-22)
- Grid: 18 combos (rsi_entry x rsi_exit x sma)
- Walk-forward: 4 folds, expanding window
- Costs: 3.0 bps/side (6.0 bps round-trip)

## Headline numbers

| Variant | Sharpe | CAGR | Max DD | Turnover/yr |
|---|---|---|---|---|
| Baseline (default params) | 0.43 | 4.2% | 24.4% | 28.64 |
| Naive in-sample winner | 0.54 | 4.9% | 19.6% | 34.94 |
| Walk-forward (OOS avg) | 0.53 | n/a | n/a | n/a |

**Overfit gap (in-sample minus OOS): 0.01**

**Verdict:** OK: walk-forward Sharpe (0.53) beats baseline (0.43) by a meaningful margin. The tuned params look reasonably robust.

## Per-fold detail

| Fold | Fit window | Test window | Best params | IS Sharpe | OOS Sharpe |
|---|---|---|---|---|---|
| 1 | 2006-05-23..2010-05-20 | 2010-05-21..2014-05-20 | entry=15 exit=60 sma=100 | 0.46 | 0.36 |
| 2 | 2006-05-23..2014-05-20 | 2014-05-21..2018-05-17 | entry=5 exit=80 sma=100 | 0.47 | 0.27 |
| 3 | 2006-05-23..2018-05-17 | 2018-05-18..2022-05-16 | entry=15 exit=60 sma=200 | 0.47 | 0.51 |
| 4 | 2006-05-23..2022-05-16 | 2022-05-17..2026-05-20 | entry=15 exit=60 sma=200 | 0.45 | 0.99 |

## Caveats

- This is daily-bar mean-reversion; intraday signals not exercised here.
- The grid is intentionally small (18 combos). A larger grid will look
  better in-sample but does not actually find more real edge.
- Walk-forward addresses parameter overfitting, not survivorship bias
  (universe is today's liquid ETFs, not the same ETFs that existed in 2006).
- Tier 1/2/3 gates still apply: a 'winning' param set must clear them
  before any promotion. See ``tools/validate_real_data.py``.