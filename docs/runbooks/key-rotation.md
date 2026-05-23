# Key Rotation Runbook

Quarterly reminder fires via n8n. Rotation can also be triggered ad-hoc on suspected leak.

## Scope

- Broker (Alpaca paper + live)
- Polygon, Alpha Vantage, Finnhub, FRED
- Telegram bot, Discord webhook
- OneSignal, Firebase
- `INTERNAL_JWT_SECRET`
- Postgres password
- OAuth client secrets

## Steps

1. Generate new secret in source platform (broker / Polygon / etc.).
2. Set new value in Doppler under correct config (`dev` / `stg` / `prd`).
3. Re-deploy affected service. Wait for healthcheck.
4. Revoke old key.
5. Tag rotation in `docs/runbooks/rotation-log.md`.

## On suspected leak

Immediately:
- Set `ENABLE_LIVE_TRADING=false`.
- Revoke broker keys at the broker — do not wait for rotation.
- Page on-call.
- File security incident.
