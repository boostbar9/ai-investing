# Scope-Creep Tripwires

Tripwires that automatically trigger an explicit "do we really want this?" review. Lifted directly from §17 (Out of Scope) plus operational guardrails.

## Hard No (do not build)

- Pure RL bots
- Meme-stock chasing strategies
- Autonomous scalping (sub-minute)
- Overfit neural-net price predictors
- Cloud-only deployments (must run locally too)
- Crypto on day one
- Native mobile apps on day one
- Options or futures
- Leverage > 1.0×

## Trip on PR if...

- New dependency adds > 50 MB to install size.
- A strategy file exceeds 600 lines.
- Any code path bypasses the 3-tier validation gate.
- Any code path bypasses the Telegram approval signal in live mode.
- Idle infra cost projection > $20/mo.
- New external API without a rate-limit entry in §14.

Reviewer must explicitly acknowledge tripwire and link to ADR.
