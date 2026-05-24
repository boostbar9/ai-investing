# Regime Weight Calibration

Generated: 2026-05-24T01:45:30.622056+00:00

## Source data

- Panel: 15 ETFs, 2006-05-23 -> 2026-05-22
- Costs: 3.0 bps/side
- Regime detection: heuristic over 5012 bars

## Regime share (overall panel)

| Regime | Share |
|---|---|
| bull | 68.2% |
| chop | 5.7% |
| bear | 19.0% |
| crisis | 7.1% |

## Per-strategy Sharpe by regime

| Strategy | Bull | Chop | Bear | Crisis |
|---|---|---|---|---|
| trend-following | +1.75 | +0.18 | -1.65 | -4.08 |
| mean-reversion | +1.47 | +1.86 | -0.95 | -1.72 |
| sector-rotation | +1.55 | +0.69 | +0.13 | -2.24 |

## Calibrated multipliers (this run)

| Strategy | Bull | Chop | Bear | Crisis |
|---|---|---|---|---|
| trend-following | 1.00 | 0.12 | 0.00 | 0.00 |
| mean-reversion | 0.98 | 1.00 | 0.00 | 0.00 |
| sector-rotation | 1.00 | 0.46 | 0.09 | 0.00 |
| intraday-trend | 0.40 | 0.40 | 0.20 | 0.00 |

## Defaults (for comparison)

| Strategy | Bull | Chop | Bear | Crisis |
|---|---|---|---|---|
| trend-following | 1.00 | 0.30 | 0.00 | 0.00 |
| mean-reversion | 0.50 | 1.00 | 0.50 | 0.00 |
| sector-rotation | 0.80 | 0.40 | 0.20 | 0.00 |
| intraday-trend | 0.40 | 0.40 | 0.20 | 0.00 |

## Mapping

Sharpe -> multiplier (clamped linear):

- Sharpe <= 0.0 -> 0.00 (strategy off in regimes where it's a coin flip)
- Sharpe == 0.75 -> 0.50
- Sharpe >= 1.5 -> 1.00 (full size)
- Crisis always forced to 0.00

## How to use

The ensemble loader prefers `data/params/regime_weights.json` over
the in-code defaults. Re-run this tool whenever the strategy code
changes or the regime detector is recalibrated.