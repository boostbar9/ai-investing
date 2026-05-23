"""VectorBT Pro + Nautilus backtest harnesses.

Three-tier validation gate (§8):
- Tier 1 Standard: ≥10y history, walk-forward, ≥1,000 MC paths
- Tier 2 Stress: 2008, 2015, 2018, 2020, 2022 shocks
- Tier 3 Synthetic: GAN + bootstrap paths

Gate thresholds: Sharpe ≥ 1.0 OOS, max DD ≤ 15% in stress, ≥95% MC paths
positive over 3y, turnover ≤ 200%/yr, capacity ≥ $5M.
"""
