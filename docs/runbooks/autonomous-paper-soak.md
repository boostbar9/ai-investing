# Autonomous Paper Soak Runbook

The 60-day paper soak is now fully autonomous. This runbook describes the
moving parts, what the operator is expected to do (almost nothing), and
what to check when something looks off.

## TL;DR — the one-button experience

1. Start the cockpit (tray app, or `start_cockpit.ps1`).
2. Open `http://127.0.0.1:8765/autopilot`.
3. Flip the **Master switch** to On. Leave `Strategy = ensemble` and
   `Dry-run = off`.
4. Walk away. Check `/promote` once a day (or wait for the 6 PM Pacific
   digest email).

That's it. The remaining 60 days happen on their own.

## Moving parts

### Paper autopilot (`/autopilot`)

- Runs a 30-second background loop inside the cockpit process.
- Fires the paper loop **twice per US trading day**:
  - **9:35 ET** — 5 minutes after open so the data feed has warmed up.
  - **10 minutes before close** — handles regular and early-close days.
- Honors the **global pause switch** and the **watchdog halt file**.
  If either is active, the autopilot logs a skip rather than spawning.
- Records the last 200 fires in memory; the page shows the most recent
  20 with timestamp + pid + ok/failed status.

### Drawdown watchdog (`/api/watchdog`)

- Evaluates the equity curve against the §16 **8 % hard ceiling**.
- On breach: writes `data/cockpit/halt.json` and stops the paper loop.
- The halt is **operator-cleared only** — recovering to within 8 % of
  peak does not auto-release the halt. Investigate first.
- Clear from the cockpit (`POST /api/watchdog/clear`) once the cause is
  understood.

### `/promote` gate

- Five gates must all be green for live capital to unlock:
  1. **Paper days ≥ 60.**
  2. **Max drawdown ≤ 8.00 %.**
  3. **Sharpe ≥ 0.80.**
  4. **Telegram approval bot connected** (`TELEGRAM_BOT_TOKEN` +
     `TELEGRAM_CHAT_ID` in `.env`).
  5. **`ENABLE_LIVE_TRADING=true`** in `.env` (informational on the
     page; still required by the trader to arm real capital).
- When all five pass, the **canary tier card** appears showing the
  5 % → 10 % → 25 % → 100 % ramp progress.

### Daily digest

- Scheduled task fires at **6 PM Pacific** every day.
- Pulls `/api/health/full`, `/api/equity-summary`, `/api/watchdog`,
  `/api/promote`, `/api/autopilot`.
- Sends one combined notification:
  - **In-app** card with the headline metric.
  - **Email** to `devfarinsky@gmail.com` using the `finance_digest`
    template — overview tiles for equity, day change, drawdown,
    soak progress; sections for promotion gate, today's trades,
    watchdog state, system health.
- Quiet-day titles ("Markets quiet — portfolio +0.10%") are intentional
  — they confirm the soak is running even on flat days.
- If the cockpit is unreachable when the digest runs, only an in-app
  warning is sent (no email). Common cause: laptop asleep or cockpit
  crashed — see `/health` next time you open it.

## What the operator does

**Daily (30 seconds):**

- Glance at the 6 PM digest. If the title says anything other than
  "Soak day N/60 — portfolio +/-X%" or "Markets quiet — …", open
  `/promote` or `/autopilot` to investigate.

**On a watchdog halt:**

1. Open `/health` and `/promote`; read the watchdog reason.
2. Pull the paper log for the run that breached:
   `data/paper_log/runs.jsonl` (last entry around the breach time).
3. If the breach was a legit risk-management trigger, leave the halt
   in place and treat it as a §16 incident.
4. If it was a data hiccup (bad close print, vendor outage), clear
   the halt from the cockpit and re-enable the autopilot.

**On day 60:**

- The `/promote` page goes green.
- Telegram + `ENABLE_LIVE_TRADING=true` must already be configured.
- The canary tier card shows the current capital fraction. Live
  trades will request Telegram approval per order.

## What the operator does NOT do

- Manually start the paper loop each morning.
- Run nightly attribution or retrain — those are already on their
  own schedulers.
- Watch drawdown by eyeball. The watchdog enforces it.
- Manually promote to live. The `/promote` page gates it; toggling
  the env flag without all five gates green still won't arm capital.

## Failure modes & fixes

| Symptom                                  | First check                          | Likely fix                                    |
| ---------------------------------------- | ------------------------------------ | --------------------------------------------- |
| No fires today, autopilot says "running" | Was today a US market holiday?       | Expected — calendar in `paper_autopilot.py`.  |
| Autopilot stays off after restart        | `/api/autopilot` `enabled=false`?    | Flip the master switch on once; the startup hook persists it across restarts. |
| Watchdog halt fires on a recovered curve | `data/cockpit/halt.json` not cleared | Operator action — clear from cockpit.         |
| Digest email never arrives               | Cron list shows the task active?     | `pplx-tool schedule_cron list` — recreate if missing. |
| `/promote` shows wrong Sharpe / DD       | `data/paper_log/runs.jsonl` recent   | Curve is computed from this file; if empty, autopilot hasn't fired yet. |

## Sources

- `packages/cockpit/paper_autopilot.py` — schedule + market calendar.
- `packages/cockpit/watchdog.py` — drawdown math + halt persistence.
- `packages/backtests/live_promotion.py` — §16 thresholds + canary ramp.
- `packages/cockpit/web/server.py` — endpoint glue + startup hook.
- Daily digest cron — managed via `pplx-tool schedule_cron`.
