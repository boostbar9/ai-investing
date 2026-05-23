# Install ai-investing on Windows — the easy path

Goal: get from a fresh Windows 11 PC to a running cockpit with a paper-trading
account in under 30 minutes, with one command for setup and one click for updates.

> Already familiar with WSL or want GPU acceleration on your RX 7900 XT?
> See [`local-setup-windows.md`](./local-setup-windows.md) for Path B.

---

## What you'll end up with

* A **desktop shortcut** that opens a tray icon (green = running, red = stopped).
* The tray menu has **Open cockpit**, **Start / Stop stack**, and **Update from GitHub**.
* The tray auto-checks GitHub for new commits every 15 minutes; updates are one click away.
* A paper-trading account on Alpaca (free, $100k fake cash) wired into the cockpit.
* A **Paper / Shadow / Live** toggle per strategy in the cockpit.

---

## Step 1 — One-shot install

Open PowerShell **as your normal user** (not Administrator) and run:

```powershell
irm https://raw.githubusercontent.com/boostbar9/ai-investing/main/scripts/install.ps1 | iex
```

This script:

1. Checks for `winget` and installs git, Node, Python, Docker Desktop, Ollama, and make if any are missing.
2. Clones the repo to `%USERPROFILE%\ai-investing`.
3. Copies `.env.example` → `.env` and pins `HARDWARE_PROFILE=rx_7900_xt`.
4. Runs `make setup-windows` (npm + python + Postgres + Dragonfly).
5. Pulls the 5 LLM models for your hardware tier (~30 GB — go grab coffee).
6. Installs `pystray` + `Pillow` for the tray app.
7. Drops `ai-investing.lnk` on your Desktop.

**Re-running the script is safe.** It's idempotent — it'll skip anything already installed.

---

## Step 2 — Paste your Alpaca paper keys

1. Open [app.alpaca.markets/paper/dashboard/overview](https://app.alpaca.markets/paper/dashboard/overview) and sign up (free).
2. Click **View API Keys** in the right rail. Generate a key pair.
3. Open `%USERPROFILE%\ai-investing\.env` in Notepad and set:

```ini
ALPACA_PAPER_KEY_ID=PK...
ALPACA_PAPER_SECRET=...
```

Save the file. You're done — no restart needed for env changes; the tray's
**Stop stack** + **Start stack** will pick them up.

---

## Step 3 — Launch it

Double-click `ai-investing` on your Desktop. A tray icon appears.

* **Right-click → Open cockpit** → opens `http://localhost:3000` in your browser.
* **Right-click → Start stack** → boots api, cockpit, worker, postgres, dragonfly.
* **Left-click** → same as Open cockpit (default action).

The icon turns green once everything is healthy.

---

## Step 4 — Train the agents on the paper account

In the cockpit, scroll to **Strategies & training mode**. Each registered strategy
has a row with three buttons:

| Mode    | What it does                                                                 |
|---------|------------------------------------------------------------------------------|
| **paper**  | Submits orders to Alpaca paper. Builds your 60-day promotion history. *(default)* |
| **shadow** | Generates signals + logs them. No orders sent anywhere. Use for brand-new strategies. |
| **live**   | Real money. Will be **silently downgraded to paper** until both `ENABLE_LIVE_TRADING=true` is set AND the 60-day gate clears (max DD < 8 %, Sharpe > 0.8). This is by design — see `packages/backtests/live_promotion.py`. |

You can mix-and-match: TrendFollowing on paper while a new SentimentOverlay sits
in shadow. The paper account stats (equity / cash / buying power) update every
30 seconds at the top of the panel.

---

## Step 5 — Keeping the local install up to date

You have three options, pick whichever feels right:

### A. Tray menu (recommended)

Right-click the tray → **Update from GitHub**. Pulls main, rebuilds containers,
restarts. Done. Title bar shows "(update available)" when there's something new.

### B. Double-click `scripts\update.bat`

Same effect, no tray needed. Useful if you're not running the tray.

### C. Manually

```powershell
cd $env:USERPROFILE\ai-investing
git pull
docker compose -f infra\docker\docker-compose.yml up -d --build
```

---

## Going live with real money

Until you flip these two switches together, the platform **cannot** place a real-money order — even if you click "live" in the cockpit:

1. **The live-promotion gate must pass.** That's a deterministic check in
   `packages/backtests/live_promotion.py`: 60 consecutive paper days, max
   drawdown < 8 %, annualized Sharpe > 0.8. The cockpit's **Live Promotion**
   panel shows current status.
2. **You must set `ENABLE_LIVE_TRADING=true` in `.env`** and restart the stack.
3. **You must add `ALPACA_LIVE_KEY_ID` / `ALPACA_LIVE_SECRET`** for real-money
   keys (separate from your paper keys).

Even then, the **canary capital schedule** ramps from 5 % → 10 % → 25 % → 100 %
with a 30-day dwell at each tier — so the system will only ever risk a tiny
slice on day one. This is intentional.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Tray icon stays red | Docker Desktop not running. Open Docker Desktop and wait for the whale icon to settle. |
| "Paper account unreachable" in cockpit | Your Alpaca keys are missing or wrong. Check `.env`. |
| Update menu greyed out / no effect | Local repo has uncommitted changes. Open a terminal and either `git stash` them or `git reset --hard origin/main` (destroys local edits). |
| `make` not found | Re-run `install.ps1` — it installs GnuWin32 Make via winget. |
| Models won't pull (Ollama errors) | Check Ollama Desktop is running. From PowerShell: `ollama list` should respond. |
| Cockpit loads but no strategies | The worker hasn't seeded the strategy catalogue yet. Tail logs: tray → Stop, then Start. |

---

## What's running under the hood

Six containers:

* **postgres + Timescale** — market data, audit log, paper history.
* **dragonfly** — feature cache (Redis-compatible).
* **api** — FastAPI backend, port 8000.
* **cockpit** — Next.js UI, port 3000.
* **worker** — Temporal workers running the agent + strategy + risk pipeline.
* **ollama** — local LLM inference (only if you use the Docker path; the Windows install uses the host's Ollama on port 11434 instead).

Stop everything with the tray's **Stop stack**. Disk usage with all 5 models is
about 35 GB; clear with `docker system prune` if needed.
