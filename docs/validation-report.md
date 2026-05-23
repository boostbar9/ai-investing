# Three-Tier Validation Report — Real Data

Generated: 2026-05-23T23:18:23.102573+00:00

## Data

- Trend / Sentiment panel: **4000 bars × 10 names** (2010-06-29 → 2026-05-22)
- Sector Rotation panel: **5032 bars × 9 names** (2006-05-23 → 2026-05-22)
- Mean-Reversion panel: SPY + QQQ + IWM, ~5000 bars (2006 → 2026)
- Mean-reversion uses walk-forward tuned params (entry=15, exit=60, sma=200; see ``docs/mean-reversion-tuning.md``)

## Gate thresholds (v3.1 §8)

- Sharpe ≥ 1.0 OOS
- Max DD ≤ 15% in any stress window
- ≥ 95% MC / synthetic paths positive over 3y
- Turnover ≤ 200%/yr

## Results

### trend-following

Panel: 4000 bars × 10 names

- **tier1** ❌ FAIL
  - reasons: OOS Sharpe 0.73 < 1.0, turnover 8.70/yr > 2.0, MC positive ratio 86.20% < 95%
  - strategy: trend-following
  - sharpe: 0.7328
  - max_drawdown: -0.1499
  - turnover_annual: 8.7032
  - cagr: 0.0853
  - n_days: 4000
  - mc_positive_ratio: 0.8620
- **tier2** ✅ PASS
  - stress_drawdowns: {'2015': 0.0, '2018': 0.0, '2020': 0.0, '2022': 0.0}
- **tier3** ❌ FAIL
  - reasons: synthetic positive ratio 89.20% < 95%
  - synthetic_paths: 500
  - block_size: 20
  - synthetic_positive_ratio: 0.8920
  - synthetic_median_return: 0.2874
  - synthetic_p05_return: -0.0905

### sector-rotation

Panel: 5032 bars × 9 names

- **tier1** ❌ FAIL
  - reasons: OOS Sharpe 0.44 < 1.0, turnover 6.66/yr > 2.0, MC positive ratio 71.00% < 95%
  - strategy: sector-rotation
  - sharpe: 0.4399
  - max_drawdown: -0.4889
  - turnover_annual: 6.6606
  - cagr: 0.0661
  - n_days: 5032
  - mc_positive_ratio: 0.7100
- **tier2** ❌ FAIL
  - reasons: 2008 max DD -40.20% > 15%
  - stress_drawdowns: {'2008': -0.402, '2015': -0.0006, '2018': 0.0, '2020': 0.0, '2022': -0.1391}
- **tier3** ❌ FAIL
  - reasons: synthetic positive ratio 77.00% < 95%
  - synthetic_paths: 500
  - block_size: 20
  - synthetic_positive_ratio: 0.7700
  - synthetic_median_return: 0.2222
  - synthetic_p05_return: -0.2850

### mean-reversion

Panel: 5032 bars × 3 names

- **tier1** ❌ FAIL
  - reasons: OOS Sharpe 0.44 < 1.0, turnover 34.99/yr > 2.0, MC positive ratio 76.00% < 95%
  - strategy: mean-reversion
  - sharpe: 0.4357
  - max_drawdown: -0.2002
  - turnover_annual: 34.9889
  - cagr: 0.0380
  - n_days: 5032
  - mc_positive_ratio: 0.7600
- **tier2** ✅ PASS
  - stress_drawdowns: {'2008': -0.0371, '2015': 0.0, '2018': 0.0, '2020': 0.0, '2022': -0.0115}
- **tier3** ❌ FAIL
  - reasons: synthetic positive ratio 80.60% < 95%
  - synthetic_paths: 500
  - block_size: 20
  - synthetic_positive_ratio: 0.8060
  - synthetic_median_return: 0.1250
  - synthetic_p05_return: -0.1359

### sentiment-overlay

Panel: 4000 bars × 10 names

- **tier1** ❌ FAIL
  - reasons: OOS Sharpe 0.73 < 1.0, turnover 8.70/yr > 2.0, MC positive ratio 86.20% < 95%
  - strategy: trend-following+sentiment
  - sharpe: 0.7328
  - max_drawdown: -0.1499
  - turnover_annual: 8.7032
  - cagr: 0.0853
  - n_days: 4000
  - mc_positive_ratio: 0.8620
- **tier2** ✅ PASS
  - stress_drawdowns: {'2015': 0.0, '2018': 0.0, '2020': 0.0, '2022': 0.0}
- **tier3** ❌ FAIL
  - reasons: synthetic positive ratio 89.20% < 95%
  - synthetic_paths: 500
  - block_size: 20
  - synthetic_positive_ratio: 0.8920
  - synthetic_median_return: 0.2874
  - synthetic_p05_return: -0.0905

## Honest interpretation

All four strategies still fail Tier 1's Sharpe ≥1.0 bar on real data, but with the longer panels we now satisfy the 10-year history check and the numbers are real OOS estimates rather than artifacts of a short 2024-2026 window.

- Mean-reversion now runs on a 20-year SPY/QQQ/IWM panel with walk-forward-tuned params. Honest OOS Sharpe ~0.53 (see tuning report).
- Sector rotation runs on 9 long-history sector ETFs (no XLC/XLRE) so the panel reaches 2006 and includes 2008 + 2020 stress windows.
- Trend-following / sentiment-overlay still struggle: vanilla 50/200 SMA crossover does not earn its cost after 6 bps round-trip.
- Sentiment-overlay is mathematically identical to base trend until a real sentiment dict feeds it (currently all-ones placeholder).

**Outstanding gaps before any real capital:**

1. Survivorship bias: universe is today's liquid ETFs, not the point-in-time S&P constituents. Hard to fix without paid data.
2. Real sentiment signal: wire the LLM news agent into the dict so sentiment-overlay has something to actually overlay.
3. Trend-following needs better filters (vol-targeting, regime detection) or it should be retired in favor of mean-reversion.
4. Continue paper trading per spec §1 (60-90 days, max DD < 8%).