# n8n workflows

Visual automation, source of truth for our cron-driven side-effects (§10).
Workflows live as JSON in `workflows/` and import into n8n via the UI or the
n8n CLI:

```bash
# Local dev (n8n service in docker-compose):
docker compose -f infra/docker/docker-compose.yml exec n8n \
  n8n import:workflow --input=/data/workflows/quarterly-key-rotation.json
```

## Required environment variables

The workflows reference these directly via `{{$env.VAR}}`. Set them in
Doppler (n8n config) and let the Doppler operator inject them:

| Variable                  | Used by                            |
|---------------------------|------------------------------------|
| `TELEGRAM_BOT_TOKEN`      | quarterly-key-rotation             |
| `TELEGRAM_CHAT_ID`        | quarterly-key-rotation             |
| `DISCORD_WEBHOOK_URL`     | quarterly-key-rotation             |
| `AI_INVESTING_API_BASE`   | audit log POST (in-cluster URL)    |

## Current workflows

### quarterly-key-rotation.json

Fires at **14:00 UTC on the 1st of January / April / July / October**
(06:00 PT, matching the daily-briefing window so the operator is already
expected to be in-app).

Pipeline: cron → build payload → fan-out to Telegram + Discord → POST
immutable audit log to the API. Scope is everything in the §13 quarterly
tier (broker, market-data, Telegram/Discord, OneSignal, JWT secret).

If the operator does not reply `/rotated` within 48h, the Telegram bot
re-pings (handled by the API's `/telegram/poll`, NOT by this workflow).

Runbook: [docs/runbooks/key-rotation.md](../../../docs/runbooks/key-rotation.md).

## Adding a workflow

1. Build it in n8n UI, then **File → Download** as JSON.
2. Drop the JSON in `workflows/`, name it kebab-case.
3. Reference any new env vars from the table above.
4. Re-import via the CLI snippet above. Workflow execution history is local
   to the n8n instance — commit the JSON only.
