# Key Rotation Runbook

> Source spec: §13 Secrets (v3.1) · Issue [#10](https://github.com/boostbar9/ai-investing/issues/10)

Quarterly reminder fires via n8n (see
[`infra/n8n/workflows/quarterly-key-rotation.json`](../../infra/n8n/workflows/quarterly-key-rotation.json))
at 14:00 UTC on the 1st of January / April / July / October. Rotation can
also be triggered ad-hoc on suspected leak — see §3 below.

## 1. Scope (what gets rotated)

| Tier        | Cadence    | Secrets                                                                                                                               |
|-------------|------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Broker      | quarterly  | `ALPACA_PAPER_KEY_ID`, `ALPACA_PAPER_SECRET`, `ALPACA_LIVE_KEY_ID`, `ALPACA_LIVE_SECRET`                                              |
| Market data | quarterly  | `POLYGON_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `FRED_API_KEY`                                                         |
| Comms       | quarterly  | `TELEGRAM_BOT_TOKEN`, `DISCORD_WEBHOOK_URL`, `ONESIGNAL_API_KEY`                                                                      |
| Internal    | quarterly  | `INTERNAL_JWT_SECRET`                                                                                                                 |
| Database    | annual     | `POSTGRES_PASSWORD`                                                                                                                   |
| On event    | immediate  | any suspected leak — see §3                                                                                                           |

The authoritative list lives in
[`infra/doppler/doppler.yaml`](../../infra/doppler/doppler.yaml) under
`rotation_cadence`. CI verifies the docs/yaml stay in sync.

## 2. Standard quarterly procedure

Pre-flight (≤ 5 min):
- [ ] Acknowledge the Telegram reminder (`/rotated start` to extend snooze 4h)
- [ ] Open this runbook in cockpit on phone (PWA → bookmarks)
- [ ] Confirm no live trading session in flight: cockpit shows `PAPER ONLY`
      OR `ENABLE_LIVE_TRADING=false` is acceptable for the rotation window

For each secret in the quarterly tier:

1. **Generate** new value in the source platform (broker dashboard,
   Polygon UI, etc.). Do NOT delete the old key yet.
2. **Set** the new value in Doppler under each affected config:
   ```bash
   doppler secrets set --config dev ALPACA_PAPER_KEY_ID=<new>
   doppler secrets set --config stg ALPACA_PAPER_KEY_ID=<new>
   doppler secrets set --config prd ALPACA_PAPER_KEY_ID=<new>
   ```
3. **Roll** affected services and wait for healthcheck:
   ```bash
   docker compose -f infra/docker/docker-compose.yml restart api worker
   curl -sf http://localhost:8000/health/detail | jq
   ```
4. **Smoke** the integration: cockpit Health panel green, one paper order
   round-trip succeeds.
5. **Revoke** the OLD key at the source platform.
6. **Log** the rotation in [`rotation-log.md`](./rotation-log.md) — date,
   secret, operator, ticket link.

After all secrets in scope:
- [ ] Reply `/rotated` in Telegram to clear the reminder
- [ ] Confirm `/security/audit` shows the n8n rotation-reminder event AND
      the matching `/rotated` reply (immutable pair)

## 3. On suspected leak — immediate response

> Target MTTR (§16): ≤ 10 minutes from suspicion to revoked key.

```bash
# 1. Kill live trading first — this is the only one-way door we automate.
doppler secrets set --config prd ENABLE_LIVE_TRADING=false
docker compose -f infra/docker/docker-compose.yml restart api

# 2. Revoke broker keys AT THE BROKER (do not wait for rotation pipeline).
#    Alpaca: https://app.alpaca.markets/paper/dashboard/overview → API keys
#    IB:     Account Mgmt → Settings → API Settings → Revoke

# 3. Page on-call.
curl -X POST https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage \
  -d chat_id=$TELEGRAM_CHAT_ID \
  -d text="\ud83d\udea8 LEAK SUSPECTED — broker keys revoked, ENABLE_LIVE_TRADING=false"

# 4. File security incident.
gh issue create -R boostbar9/ai-investing \
  --title "SECURITY: suspected key leak — $(date -u +%Y-%m-%d)" \
  --label security,incident \
  --body "Triggered manual rotation. Affected: <list>. See on-call.md."
```

Then run the full §2 procedure on EVERY secret in scope, not just the
suspected one. Assume blast radius is the entire process boundary.

## 4. Rotation log

A short append-only log lives at
[`docs/runbooks/rotation-log.md`](./rotation-log.md). One entry per secret
per rotation event. Fields: `date_utc`, `secret_name`, `config`, `operator`,
`reason` (`quarterly` / `leak` / `staff-change`), `audit_id`.

## 5. What MUST NOT happen

- Do not commit any secret to git — pre-commit hook + GitHub secret
  scanning catch most, but assume neither is perfect.
- Do not share a Doppler service token across configs. One token = one
  config (dev OR stg OR prd).
- Do not bypass the cockpit health smoke after a rotation; a 401 on the
  next live order is a 10× worse outage than a 5-min delay.
- Do not rotate live broker keys while the canary is above tier 0 without
  flipping `ENABLE_LIVE_TRADING=false` first. The §15 gate must briefly
  fail closed during the window.
