# Tier-2 Stress Backtest

Generated: 2026-05-24T01:09:05.926006+00:00

## Setup

- Panel: 13 ETFs, 2006-05-23 -> 2026-05-22
- Costs: 3.0 bps/side
- Verdict rule: max DD across all windows must be ≥ -25%, and median Sharpe ≥ 0.

## Windows

| Window | Range | Description |
|---|---|---|
| 2008-gfc | 2008-01-02 .. 2009-06-30 | Global Financial Crisis: Lehman, TARP, March 2009 low |
| 2015-china | 2015-06-01 .. 2016-02-29 | China devaluation, Aug 2015 flash, Q1 2016 oil bottom |
| 2018-q4 | 2018-01-01 .. 2018-12-31 | Vol-mageddon (Feb), Q4 rate-hike sell-off |
| 2020-covid | 2020-01-02 .. 2020-12-31 | COVID crash (Feb-Mar) + record rebound |
| 2022-rates | 2022-01-03 .. 2023-06-30 | Fed hiking cycle: bond + tech bear |

## Verdicts

| Strategy | Pass | Reason |
|---|---|---|
| trend-following | FAIL | FAIL — worst DD -25.8% < -25%; median Sharpe -0.95 < 0 |
| mean-reversion | FAIL | FAIL — median Sharpe -0.46 < 0 |
| sector-rotation | FAIL | FAIL — worst DD -48.8% < -25%; median Sharpe -0.21 < 0 |

## Per-strategy detail

### trend-following

| Window | Sharpe | Max DD | CAGR | Hit-rate | Worst day |
|---|---|---|---|---|---|
| 2008-gfc | -0.95 | -14.7% | -9.1% | 13% | -2.83% |
| 2015-china | -1.66 | -19.8% | -20.9% | 38% | -3.96% |
| 2018-q4 | -0.46 | -18.4% | -7.2% | 56% | -4.13% |
| 2020-covid | 0.84 | -13.5% | 13.0% | 47% | -5.40% |
| 2022-rates | -1.04 | -25.8% | -13.1% | 31% | -3.85% |

### mean-reversion

| Window | Sharpe | Max DD | CAGR | Hit-rate | Worst day |
|---|---|---|---|---|---|
| 2008-gfc | -0.74 | -6.2% | -3.6% | 20% | -1.51% |
| 2015-china | -1.15 | -8.5% | -8.8% | 43% | -3.07% |
| 2018-q4 | -0.46 | -7.2% | -4.2% | 52% | -3.83% |
| 2020-covid | -0.37 | -22.1% | -7.4% | 54% | -7.09% |
| 2022-rates | -0.24 | -6.7% | -1.3% | 49% | -1.53% |

### sector-rotation

| Window | Sharpe | Max DD | CAGR | Hit-rate | Worst day |
|---|---|---|---|---|---|
| 2008-gfc | -0.89 | -48.8% | -26.3% | 49% | -8.28% |
| 2015-china | -0.28 | -13.0% | -5.7% | 52% | -4.17% |
| 2018-q4 | -0.21 | -16.0% | -4.2% | 54% | -4.20% |
| 2020-covid | 0.69 | -33.6% | 21.4% | 56% | -12.46% |
| 2022-rates | 0.25 | -15.2% | 3.7% | 51% | -3.99% |

## Caveats

- Universe is today's liquid ETFs. Some (XLRE 2015, XLC 2018) are
  excluded automatically by the dropna() filter when a window
  pre-dates inception.
- The stress gate is behavioural (survival + non-negative median
  Sharpe), not a profit gate. Even a perfect risk system can lose
  modestly through a crisis -- the goal is not to blow up.
- Costs assume DEFAULT_COST_MODEL bps/side and ignore slippage
  outsize, which is conservative for ETFs and unrealistic for
  small-caps. Don't extend these results to single names without
  refitting the cost model.