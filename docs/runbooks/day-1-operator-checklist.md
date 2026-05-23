# Day-1 Operator Checklist

A one-page checklist for the very first day running the platform.

## Before market open (06:00 PT)

- [ ] Pull latest `main` and re-run `make setup` if there were merges.
- [ ] `make ps` — all 7 services green (postgres, dragonfly, temporal, ollama, mlflow, grafana, otel-collector).
- [ ] Grafana `Overview` dashboard — all panels green for last 60 min.
- [ ] Doppler keys rotated within last 90 days.
- [ ] Telegram bot responds to `/ping`.
- [ ] Regime badge matches expected state (sanity check against SPY/VIX).
- [ ] No active alerts in `#ai-investing-alerts`.

## During session

- [ ] Approvals queue cleared within SLA (≤ 5s p95 to approve/deny).
- [ ] Risk Engine NOT in halt state (unless intended).
- [ ] No silent failures — every agent emits an OTel span.

## After close

- [ ] Daily briefing PDF generated and pushed.
- [ ] Nightly CI matrix green.
- [ ] Backup completed (Postgres + MLflow).
- [ ] One-line journal entry in `docs/runbooks/journal.md`.
