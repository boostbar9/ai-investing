# Risk Register (v3.1 §18)

Living document. Every newly-discovered risk must be added with an owner and a
mitigation before merge.

| Risk | Likelihood | Impact | Mitigation | Owner | Verified |
|---|---|---|---|---|---|
| Hardware too weak (LLM OOM) | M | M | Quantized GGUF + auto-fallback router (`packages/agents/llm_router.py`) | Devin | ✅ tests |
| Broker outage | M | H | Broker abstraction + failover router (`packages/execution/broker.py`); `TRADING_PAUSED=true` if all down | Devin | ✅ tests |
| Strategy overfitting | H | H | 3-Tier Validation Gate (`packages/backtests/validation.py`); champion/challenger 30d gate | Devin | ✅ Tier-1, ⏳ Tier-2/3 |
| Model drift | M | M | Nightly OOS in `.github/workflows/nightly-backtests.yml`; auto-rollback on Sharpe drop > 10% | Devin | ⏳ Phase 2.5 |
| Locked out on phone | L | H | Telegram + Tailscale Funnel + paper recovery codes in 1Password | Devin | ⏳ Phase 4 |
| Catastrophic loss | L | Critical | Hard DD halt at 8% (`risk.engine.drawdown_halt`); 5% cash floor; human gate | Devin | ✅ tests |
| Vendor schema break | M | M | Pydantic adapters at every boundary; CI canary `make backtest` runs full ingestion smoke daily | Devin | ⏳ |
| LLM provider outage | L | M | Local-first via Ollama; cloud fallback only if explicitly enabled | Devin | ✅ design |
| Regulatory change | L | H | Quarterly compliance checklist; n8n reminder | Devin | ⏳ |
| Audit-log tampering | L | Critical | Append-only table with UPDATE/DELETE trigger (migration `0001_audit_log.py`) | Devin | ✅ |
| Secret leak in git | L | Critical | `.gitignore` whitelists only `.env.example`; SECURITY.md; key-rotation runbook | Devin | ✅ |

Likelihood / Impact scale: L low, M medium, H high.
