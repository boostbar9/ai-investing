# Runbook — Phase 5 live promotion + canary capital

> Source spec: §15 Phase 5, §16 Acceptance Thresholds (v3.1)
> Issue: [#8](https://github.com/boostbar9/ai-investing/issues/8)

## TL;DR

Real capital is gated by two layers, both of which must pass on every call.
The execution layer multiplies its intended notional by `capital_fraction`
from `/live/promotion`; if `live_enabled=false`, the fraction is `0.0` and
no live orders are routed.

## Layer 1 — Live readiness gate

All of the following must be true before live trading is permitted **at all**:

| Check                  | Threshold (locked)  | Source |
|------------------------|---------------------|--------|
| Paper trading days     | ≥ 60 consecutive    | §16    |
| Max drawdown (60d)     | < 8%                | §16    |
| Annualized Sharpe (60d)| > 0.8               | §16    |
| `ENABLE_LIVE_TRADING`  | `=true` in env      | §15    |

If any fails, `decide_live_capital()` returns
`live_enabled=False, capital_fraction=0.0` and the API exposes the reasons
on `/live/promotion`. There is no override path in code — the flag is the
override path, and it MUST be combined with passing metrics.

## Layer 2 — Canary capital schedule

Once the readiness gate passes, capital ratchets up only on sustained live
performance:

| Tier | Fraction | Dwell to advance | Live thresholds        |
|------|----------|------------------|------------------------|
| 0    | 5%       | 30 days          | DD<8%, Sharpe>0.8      |
| 1    | 10%      | 30 days          | DD<8%, Sharpe>0.8      |
| 2    | 25%      | 30 days          | DD<8%, Sharpe>0.8      |
| 3    | 100%     | —                | —                      |

Advancement rule: to step from tier *i* to tier *i+1* the live equity curve
must contain at least `dwell_days` for tier *i* AND the metrics over that
window must still satisfy §16. Otherwise the canary stays put. The walk is
deterministic — the curve is the source of truth.

If live performance regresses inside a tier, the canary does NOT
auto-rollback. Manual action: pause `ENABLE_LIVE_TRADING`, investigate,
truncate `LIVE_EQUITY_PATH` to the last known-good day, restart.

## Day-of operations

```bash
# Check status from the command line (returns exit 0 only if live is enabled):
python -m packages.backtests.live_promotion_cli check \
    --paper artifacts/paper_equity.json \
    --live  artifacts/live_equity.json \
    --json
```

```bash
# Or via the API:
curl -s http://localhost:8000/live/promotion | jq
```

The cockpit's "Live promotion" panel polls `/live/promotion` every 30s and
shows tier, dwell progress bar, and any blocking reasons.

## Promotion checklist (must complete IN ORDER)

1. Paper equity JSON exists at `PAPER_EQUITY_PATH` with ≥ 60 days
2. `/live/promotion` shows `readiness.ready=true` with no blocking reasons
3. Doppler `ENABLE_LIVE_TRADING=true` rotated into the API container
4. Restart API; confirm `/live/promotion` shows `live_enabled=true`
5. Confirm `capital_fraction=0.05` (tier 0) on first live day
6. Watch Grafana drawdown + Sharpe panels for the first session
7. After 30 live days, panel advances to 10% automatically — no code change

## Rollback ("pull the cord")

```bash
# Fast: disable the flag in Doppler and roll the API
doppler secrets set ENABLE_LIVE_TRADING=false
docker compose -f infra/docker/docker-compose.yml restart api
```

Within ≤ 60 seconds, `/live/promotion` returns `live_enabled=false` and the
execution layer's sizing multiplier becomes 0. Open positions are NOT
auto-flattened — that is a separate operator decision tracked under §17.

## Why no auto-flatten?

§1 mandates "smaller drawdowns > higher returns. Survival first." but also
"every automated change has a one-click rollback." A forced liquidation on
flag-flip would be a one-way door — the inverse of reversible. Operator
opens the cockpit, flattens manually if warranted.
