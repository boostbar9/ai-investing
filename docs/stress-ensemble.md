# Tier-2 Stress — Regime-Gated Ensemble

Generated: 2026-05-24T01:45:41.669042+00:00

## Setup

- Panel: 15 ETFs, 2006-05-23 -> 2026-05-22
- Costs: 3.0 bps/side
- Strategies: trend-following + mean-reversion + sector-rotation
- Gating: ``packages.regime.ensemble.DEFAULT_REGIME_WEIGHTS``
- Regime detection: heuristic (SPY 20d return, realised vol proxy, breadth)

## Regime share (overall panel)

| Regime | Share |
|---|---|
| bull | 68.2% |
| chop | 5.7% |
| bear | 19.0% |
| crisis | 7.1% |

## Per-window results

| Window | Sharpe | Max DD | CAGR | Avg Gross | Hit-rate | Worst day | Regime share (bull/chop/bear/crisis) |
|---|---|---|---|---|---|---|---|
| 2008-gfc | -0.47 | -14.1% | -4.5% | 0.17 | 27% | -3.61% | 13/1/43/43% |
| 2015-china | -1.10 | -10.5% | -11.9% | 0.66 | 46% | -3.10% | 55/9/32/5% |
| 2018-q4 | -0.83 | -15.2% | -8.3% | 0.68 | 51% | -4.14% | 59/6/31/4% |
| 2020-covid | +0.51 | -9.1% | +6.9% | 0.58 | 47% | -5.56% | 56/1/24/19% |
| 2022-rates | -0.34 | -12.8% | -3.2% | 0.48 | 44% | -2.57% | 42/3/42/13% |

## Verdicts

- **Survival gate**: FAIL — worst DD -15.2%, median Sharpe -0.47
- **§16 v1.0 acceptance**: v1.0 DD gate FAIL (-15.2% > -15%); v1.0 Sharpe gate FAIL (-0.47 < 1.0)

## Reading this report

- The point of the regime gate is to compress drawdowns, not to
  always beat individual strategies on raw Sharpe.
- Compare directly with ``docs/stress-backtest.md`` — that is the
  baseline of each strategy run alone. If the ensemble has smaller
  worst-DD numbers, the gate is working as designed even if the
  median Sharpe is similar.
- The `Avg Gross` column shows what fraction of capital the system
  was actually risking during the window. A 0.30 average gross in
  2008 means the regime gate was throttling exposure aggressively.

## Caveats

- Regime labels here come from a deterministic heuristic. The full
  HMM is in `packages/regime/hmm.py`; once hmmlearn is installed,
  swap the call in `detect_regime_series` for the full HMM.
- The multiplier table is hand-set from the per-strategy stress
  results; we have not (yet) calibrated it through a proper
  regime-conditional walk-forward. That is the obvious next
  upgrade once the ensemble is wired into paper trading.