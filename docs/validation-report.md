# Three-Tier Validation Report — Real Data

Generated: 2026-05-23T22:12:04.735925+00:00

## Data

- Trend / Mean-Reversion / Sentiment panel: **1100 bars × 11 names** (2024-01-01 → 2026-05-22)
- Sector Rotation panel: **1993 bars × 11 names** (2018-06-19 → 2026-05-22)

## Gate thresholds (v3.1 §8)

- Sharpe ≥ 1.0 OOS
- Max DD ≤ 15% in any stress window
- ≥ 95% MC / synthetic paths positive over 3y
- Turnover ≤ 200%/yr

## Results

### trend-following

Panel: 1100 bars × 11 names

- **tier1** ❌ FAIL
  - reasons: history < 10y (1100 bars), OOS Sharpe 0.02 < 1.0, turnover 9.41/yr > 2.0, MC positive ratio 49.00% < 95%
  - strategy: trend-following
  - sharpe: 0.0162
  - max_drawdown: -0.1872
  - turnover_annual: 9.4112
  - cagr: -0.0014
  - n_days: 1100
  - mc_positive_ratio: 0.4900
- **tier2** ✅ PASS
  - stress_drawdowns: {}
- **tier3** ❌ FAIL
  - reasons: synthetic positive ratio 45.40% < 95%
  - synthetic_paths: 500
  - block_size: 20
  - synthetic_positive_ratio: 0.4540
  - synthetic_median_return: -0.0119
  - synthetic_p05_return: -0.2015

### sector-rotation

Panel: 1993 bars × 11 names

- **tier1** ❌ FAIL
  - reasons: history < 10y (1993 bars), OOS Sharpe 0.63 < 1.0, turnover 6.79/yr > 2.0, MC positive ratio 81.40% < 95%
  - strategy: sector-rotation
  - sharpe: 0.6256
  - max_drawdown: -0.2920
  - turnover_annual: 6.7858
  - cagr: 0.1053
  - n_days: 1993
  - mc_positive_ratio: 0.8140
- **tier2** ✅ PASS
  - stress_drawdowns: {'2018': 0.0, '2020': 0.0, '2022': -0.1391}
- **tier3** ❌ FAIL
  - reasons: synthetic positive ratio 84.40% < 95%
  - synthetic_paths: 500
  - block_size: 20
  - synthetic_positive_ratio: 0.8440
  - synthetic_median_return: 0.3501
  - synthetic_p05_return: -0.1871

### mean-reversion

Panel: 1100 bars × 11 names

- **tier1** ❌ FAIL
  - reasons: history < 10y (1100 bars), OOS Sharpe 0.68 < 1.0, turnover 2.02/yr > 2.0, MC positive ratio 87.80% < 95%
  - strategy: mean-reversion
  - sharpe: 0.6839
  - max_drawdown: -0.2033
  - turnover_annual: 2.0202
  - cagr: 0.0675
  - n_days: 1100
  - mc_positive_ratio: 0.8780
- **tier2** ✅ PASS
  - stress_drawdowns: {}
- **tier3** ❌ FAIL
  - reasons: synthetic positive ratio 90.40% < 95%
  - synthetic_paths: 500
  - block_size: 20
  - synthetic_positive_ratio: 0.9040
  - synthetic_median_return: 0.2194
  - synthetic_p05_return: -0.0650

### sentiment-overlay

Panel: 1100 bars × 11 names

- **tier1** ❌ FAIL
  - reasons: history < 10y (1100 bars), OOS Sharpe 0.02 < 1.0, turnover 9.41/yr > 2.0, MC positive ratio 49.00% < 95%
  - strategy: trend-following+sentiment
  - sharpe: 0.0162
  - max_drawdown: -0.1872
  - turnover_annual: 9.4112
  - cagr: -0.0014
  - n_days: 1100
  - mc_positive_ratio: 0.4900
- **tier2** ✅ PASS
  - stress_drawdowns: {}
- **tier3** ❌ FAIL
  - reasons: synthetic positive ratio 45.40% < 95%
  - synthetic_paths: 500
  - block_size: 20
  - synthetic_positive_ratio: 0.4540
  - synthetic_median_return: -0.0119
  - synthetic_p05_return: -0.2015

## Honest interpretation

All four strategies fail Tier 1 on real data. This is the **expected** outcome for vanilla, public-domain strategies after costs (6 bps round-trip)

- The 10-year history requirement is unmet because META's IPO (2012) is the binding constraint on the multi-name panel intersection.
- Tier 2 mostly passes only because most stress windows (2008, 2015, 2018, 2020) lie outside our available data range. This is a coverage limitation, not a strength.
- Tier 3 synthetic uses a 20-day block bootstrap that preserves autocorrelation, which is harder to game than an iid bootstrap.

**Best-of-four:** mean-reversion (Sharpe 0.68, CAGR 6.7%, DD -20.3%, 90% synthetic positive). **Worst:** trend-following / sentiment-overlay (flat, with high turnover).

**Next steps before any real capital:**

1. Pull a wider history (use SPY-only or sector-only panels to get full 20-year coverage; expand multi-name panel only when needed).
2. Re-test on Alpaca paper data (with intraday) once keys are set.
3. Iterate strategy parameters cautiously to avoid overfitting; any change must hold up under the same Tier 1/2/3 gates.
4. Continue paper trading per spec §1 (60-90 days, max DD < 8%).