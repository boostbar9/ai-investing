# Local AI Investing Platform

**Owner:** Devin (Watsonville, CA) · **Spec:** v3.1 Master Spec · **Status:** Phase 0 — Foundation

A hybrid AI-assisted quant platform. **Not** a fully autonomous HFT bot. Survival first. Smaller drawdowns over higher returns. Human approval layer required during early deployment. Paper trading minimum 60–90 trading days before any real capital.

## Design Mandate

Every feature is judged against seven hard requirements:

- **Modern** · **Friendly** · **Mobile** · **Long-Term**
- **Observable** — every action emits a structured event (OTel span)
- **Reversible** — every automated change has a one-click rollback
- **Cheap-to-run** — idle cost ≤ $20/mo

## The 2026 Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph + Temporal |
| Local LLM | Ollama + vLLM |
| Reasoning LLM | DeepSeek R1 |
| Code/finance LLM | Qwen 2.5 72B |
| Backtesting | VectorBT Pro + Nautilus |
| Feature store | Feast |
| DB | Postgres 16 + TimescaleDB |
| Cache | DragonflyDB |
| Frontend | Next.js 16 + React 19 + Tailwind |
| Approvals | Telegram + Discord |
| CI/CD | GitHub Actions + Renovate |
| Secrets | Doppler / 1Password CLI |

## Repository Layout

```
ai-investing/
├── apps/
│   ├── cockpit/          # Next.js 16 PWA
│   ├── api/              # FastAPI backend
│   └── telegram-bot/     # Approvals bot
├── packages/
│   ├── agents/           # Research, Strategy, Risk, Execution agents
│   ├── strategies/       # Trend, Sector Rotation, Mean Reversion, Sentiment
│   ├── regime/           # 4-state Gaussian HMM
│   ├── risk/             # Dynamic risk engine, halt logic
│   ├── execution/        # Order routing
│   ├── data/             # Ingestion adapters
│   ├── features/         # Feast feature definitions
│   ├── backtests/        # VectorBT + Nautilus harnesses
│   └── shared/           # Schemas, OTel helpers, JWT signing
├── infra/                # docker-compose, Grafana, Temporal, MLflow
├── docs/                 # ADRs, runbooks, glossary
└── .github/workflows/    # Nightly backtests, CI
```

## Toolchain

- **Node 22 + pnpm 9** (cockpit, telegram-bot)
- **Python 3.12 + uv** (api, packages/*)
- **Docker + docker-compose** (Postgres/Timescale, DragonflyDB, Temporal, Grafana, MLflow, Ollama)
- **Make** (one-command setup)

## Quickstart (one command)

The full stack (Docker, LLMs, Postgres) is overkill for paper trading. The fastest path to a working paper-trading runner:

**Windows (PowerShell):**

```powershell
iwr https://raw.githubusercontent.com/boostbar9/ai-investing/main/scripts/install.ps1 -UseBasicParsing | iex
```

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/boostbar9/ai-investing/main/scripts/install.sh | bash
```

The installer:

1. Verifies Python 3.12+ and Git
2. Clones the repo to `~/ai-investing`
3. Creates a `.venv` and installs Python deps
4. Creates `.env` from `.env.example`
5. Runs the doctor smoke test

Then edit `.env` with your [Alpaca paper keys](https://app.alpaca.markets/paper/dashboard/overview) and run:

```bash
source .venv/bin/activate                                 # Windows: .\.venv\Scripts\Activate.ps1
PYTHONPATH=. python -m packages.data.pretrain             # download 20yr daily + 90d intraday
PYTHONPATH=. python tools/paper_trade.py --strategy ensemble --dry-run
PYTHONPATH=. python tools/cockpit.py                      # local web GUI at http://127.0.0.1:8765
```

The **cockpit** is a Python-only FastAPI dashboard that runs at
`http://127.0.0.1:8765` and auto-opens in your browser. It shows live account
equity, target positions, regime classification, the §16 promotion-gate
streak, and recent trades. It also exposes controls: pause/resume the bot,
run a trade cycle on demand, override the regime detector, and an emergency
flatten button that cancels orders and submits closing market orders for
every open position.

For the legacy static dashboard (no server required):

```bash
PYTHONPATH=. python tools/paper_dashboard.py && open docs/paper-dashboard.html
```

## Full stack (advanced)

For the full Docker + Ollama + Postgres + LangGraph stack:

```bash
make setup        # install all deps, pull LLM models, init DB
make dev          # start full stack locally
make backtest     # run nightly matrix locally
make test         # all tests
```

**Goal:** `clone → first backtest in < 30 min` (acceptance criterion §16).

## Roadmap (12 Weeks to Paper-Live)

| Phase | Weeks | Deliverables |
|---|---|---|
| 0 — Foundation | 1 | Repo, monorepo, docker-compose, Makefile, CI |
| 1 — LLMs + Data | 2–3 | Ollama + DeepSeek R1 + Qwen 2.5; ingestion; Alpaca paper |
| 2 — Backtests + Signals | 4–6 | VectorBT + Nautilus; 4 strategies; Tier-1 |
| 3 — Agents + Risk + Regime | 7–9 | LangGraph on Temporal; HMM; Tier-2/3 |
| 4 — Cockpit + Mobile + Bot | 10–12 | Next.js PWA; Telegram bot; 60-day paper |
| 5 — Small Live + Scale | 13+ | 5–10% capital; champion/challenger; alt data |

## Out of Scope

Pure RL bots · Meme-stock chasing · Autonomous scalping · Overfit NN preds · Cloud-only · Crypto day one · Native app day one · Options/futures · Leverage > 1.0×

## Security

- **No secrets in git.** Real keys live in Doppler / 1Password CLI. Only `.env.example` is committed.
- Read-only broker keys in cockpit; trading keys only in Execution Agent.
- TLS via Caddy or Tailscale Funnel.
- Daily encrypted Postgres + MLflow backup.
- Immutable audit log.
- Inter-service calls signed with 5-min JWTs.

See [`SECURITY.md`](./SECURITY.md) and [`docs/runbooks/on-call.md`](./docs/runbooks/on-call.md).

## License

MIT (see [`LICENSE`](./LICENSE)).
