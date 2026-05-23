# Security Policy

## No Secrets in Git

This repo contains **only** `.env.example` files. Real secrets live in **Doppler** or **1Password CLI**.

If you find a leaked key, follow `docs/runbooks/key-rotation.md` immediately.

## Broker Key Separation

- **Cockpit (read-only):** account balance, positions, P&L — read-only API keys only.
- **Execution Agent (trading):** order placement keys, scoped to a single paper or live account.
- These keys **must not** share a process or environment.

## Inter-service Auth

All internal HTTP calls between `apps/api`, `apps/telegram-bot`, and agent workers MUST be signed with a JWT whose `exp` ≤ 5 minutes from `iat`. Shared signing secret loaded from Doppler at boot, never logged.

## Reporting

Email Devin or open a private security advisory on GitHub. Do not file a public issue.

## Disclaimer

Software provided for educational and research use. Not financial advice. Trading involves substantial risk of loss.
