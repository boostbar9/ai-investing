# Glossary

| Term | Meaning |
|---|---|
| **HMM** | Hidden Markov Model — used for 4-state regime detection (Bull / Bear / Chop / Crisis). |
| **OOS** | Out-of-sample — backtest data not seen during strategy fit. |
| **MC paths** | Monte Carlo simulated price paths used in Tier-1 validation. |
| **Tier-1 / 2 / 3** | Validation gates: Standard / Stress / Synthetic (see §8). |
| **Champion/Challenger** | A live strategy (champion) is only replaced after a challenger beats it OOS for 30 consecutive trading days. |
| **Decision ID** | A UUID attached to every agent decision; flows through every span, order, and audit row. |
| **HITL** | Human-In-The-Loop — Telegram/Discord approval gate before any order is sent. |
| **PWA** | Progressive Web App — installable cockpit on iOS/Android. |
| **Sharpe ratio** | Risk-adjusted return: (return − risk-free) / stdev. Gate threshold ≥ 1.0 OOS. |
| **Max DD** | Maximum peak-to-trough drawdown. Gate threshold ≤ 15% in stress, ≤ 8% in paper. |
| **Kelly** | Position sizing formula based on edge / odds. Sized down by regime + vol target. |
| **MTTR** | Mean Time To Recovery — target ≤ 10 min for any single-service crash. |
| **p95** | 95th percentile latency. |
