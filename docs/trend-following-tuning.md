# Trend-Following Walk-Forward Tuning

Generated: 2026-05-24T01:07:03.646865+00:00

## Setup

- Panel: SPY + QQQ + IWM, 5032 bars (2006-05-23 -> 2026-05-22)
- Grid: 15 combos (fast x slow x vol_target)
- Walk-forward: 4 folds, expanding window
- Costs: 3.0 bps/side (6.0 bps round-trip)

## Headline numbers

| Variant | Sharpe | CAGR | Max DD | Turnover/yr |
|---|---|---|---|---|
| Baseline (default params) | 0.63 | 6.4% | 20.3% | 5.04 |
| Naive in-sample winner | 0.73 | 8.0% | 15.9% | 6.46 |
| Walk-forward (OOS avg) | 0.74 | n/a | n/a | n/a |

**Overfit gap (in-sample minus OOS): -0.01**

**Verdict:** OK: walk-forward Sharpe (0.74) beats baseline (0.63) by a meaningful margin. The tuned params look reasonably robust.

## Per-fold detail

| Fold | Fit window | Test window | Best params | IS Sharpe | OOS Sharpe |
|---|---|---|---|---|---|
| 1 | 2006-05-23..2010-05-20 | 2010-05-21..2014-05-20 | fast=50 slow=100 vol_target=0.10 | 0.60 | 0.70 |
| 2 | 2006-05-23..2014-05-20 | 2014-05-21..2018-05-17 | fast=50 slow=100 vol_target=0.10 | 0.67 | 0.76 |
| 3 | 2006-05-23..2018-05-17 | 2018-05-18..2022-05-16 | fast=50 slow=100 vol_target=0.10 | 0.75 | 0.76 |
| 4 | 2006-05-23..2022-05-16 | 2022-05-17..2026-05-20 | fast=50 slow=100 vol_target=0.10 | 0.72 | 0.76 |

## Caveats

- Daily-bar trend; intraday momentum is handled by the IntradayTrend
  strategy with its own opening-range + VWAP-trail logic.
- Trend-following has long, structural flat regimes (chop): expect
  the OOS Sharpe to look modest even with good parameters.
- The grid is intentionally small (18 combos). Larger grids will
  look better in-sample without finding more real edge.
- Tier 1/2/3 gates still apply before any param promotion.