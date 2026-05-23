# On-Call Runbook

## Pager Triggers

Operator is paged within **60 seconds** when any of the following occur:

1. Any agent fails closed (Pydantic validation error or timeout).
2. Risk Engine halt fires.
3. Broker connection error > 30s.
4. Postgres / Timescale unavailable > 30s.
5. CI nightly red 2 days in a row.
6. Cockpit health endpoint returns non-200 for 2 consecutive minutes.

## First 5 Minutes

1. Acknowledge in Telegram.
2. Check Grafana `ai-investing / overview` dashboard.
3. Check `make ps` — which service is down?
4. If trading is live, flip `ENABLE_LIVE_TRADING=false` in Doppler and redeploy execution agent.
5. Open the run in Temporal UI (`http://localhost:8233`) — find the failing workflow and copy the `decision_id`.

## Common Incidents

### Agent timeout / fail-closed
- Check Ollama health: `curl http://localhost:11434/api/tags`
- Roll back to backup model per §5 table (Qwen 2.5 → Llama 3.3, etc.)
- File ADR if rollback needed > once/week.

### Broker error
- Confirm Alpaca status page.
- Read-only keys are isolated — cockpit will still show last known positions.

### Postgres / Timescale outage
- `docker compose restart postgres`
- Restore from last encrypted backup if data corruption suspected: see `backup-restore.md`.

## MTTR Target
≤ 10 minutes for any single-service crash (§16).

## Postmortem
Every page → blameless postmortem within 48h. Template in `docs/runbooks/postmortem-template.md`.
