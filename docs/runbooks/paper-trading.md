# Runbook — Nightly Paper Trading

How to run the bot against the Alpaca paper account, daily, with full kill
switches and an audit log.

## Prerequisites

1. Alpaca paper keys in `.env` (see [alpaca-paper-keys.md](alpaca-paper-keys.md)).
2. Data parquet files exist in `data/parquet/daily/` (run `tools/doctor.py` if
   unsure, or `python3 -c "from packages.data.pretrain import run; import asyncio; asyncio.run(run())"`).
3. `ENABLE_PAPER_TRADING=true` in `.env` (this is the master kill switch).

## Daily run

After market close, from the repo root:

```bash
export $(grep -v '^#' .env | xargs)
PYTHONPATH=. python3 tools/paper_trade.py --strategy mean-reversion
```

Strategies: `mean-reversion` (walk-forward tuned), `trend-following`, `sector-rotation`.

Add `--use-sentiment` to apply the Reddit/RSS sentiment overlay.

Always **dry-run first** when changing strategy or params:

```bash
PYTHONPATH=. python3 tools/paper_trade.py --strategy trend-following --dry-run
```

Dry-run prints the planned order list without submitting.

## Kill switches

The runner halts (no orders sent) if **any** of these fire:

| Switch | Default | Override env var |
|---|---|---|
| `ENABLE_PAPER_TRADING != "true"` | required | `ENABLE_PAPER_TRADING` |
| Account status != ACTIVE | — | n/a |
| `trading_blocked` or `account_blocked` | — | n/a |
| Drawdown from session peak | 8% | `MAX_DD_PCT=0.08` |
| Margin utilization | 95% | `MARGIN_HALT_PCT=0.95` |

The session peak persists across runs at `data/paper_log/session_peak.json`.
Delete that file to reset the drawdown baseline (e.g. on a fresh paper account).

The runner also skips orders smaller than `MIN_REBALANCE_BPS` (default 25 bps of
equity) to avoid churn.

## Audit log

Every run appends one JSON line to `data/paper_log/runs.jsonl` with:

- timestamp, strategy, dry_run flag
- whether the run halted, and why
- account equity, buying power
- target weights, planned orders, submitted orders + broker IDs
- any errors

Tail it:

```bash
tail -5 data/paper_log/runs.jsonl | jq .
```

## Scheduling

A daily cron at 4:30 PM ET (after the close):

```cron
30 16 * * 1-5  cd /path/to/ai-investing && PYTHONPATH=. python3 tools/paper_trade.py --strategy mean-reversion >> data/paper_log/cron.out 2>&1
```

For the 60-90 day promotion clock per spec §1, run this every trading day
and watch `data/paper_log/runs.jsonl` for the cumulative P&L and DD.

## Promotion to live

**Not yet supported by this script** — `tools/paper_trade.py` only talks to
`paper-api.alpaca.markets`. To go live:

1. 60-90 consecutive trading days with paper DD < 8% and positive P&L
2. All four strategies still pass nightly_gate (see `packages/backtests/nightly_gate.py`)
3. Explicit human approval via the live-promotion CLI (see `packages/backtests/live_promotion_cli.py`)
4. Live keys in the execution-agent secret store (NOT in `.env`)
5. Start at 1-2% of intended capital, ramp slowly

## Rolling back a bad day

If a run submitted orders you don't want:

```bash
# Cancel everything that's still open
PYTHONPATH=. python3 -c "
import asyncio, httpx, os
h = {'APCA-API-KEY-ID': os.environ['ALPACA_PAPER_KEY_ID'],
     'APCA-API-SECRET-KEY': os.environ['ALPACA_PAPER_SECRET']}
async def main():
    async with httpx.AsyncClient(headers=h) as c:
        r = await c.delete('https://paper-api.alpaca.markets/v2/orders')
        print(r.status_code)
asyncio.run(main())
"
```

To flatten all positions:

```bash
PYTHONPATH=. python3 -c "
import asyncio, httpx, os
h = {'APCA-API-KEY-ID': os.environ['ALPACA_PAPER_KEY_ID'],
     'APCA-API-SECRET-KEY': os.environ['ALPACA_PAPER_SECRET']}
async def main():
    async with httpx.AsyncClient(headers=h) as c:
        r = await c.delete('https://paper-api.alpaca.markets/v2/positions')
        print(r.status_code, r.text[:200])
asyncio.run(main())
"
```
