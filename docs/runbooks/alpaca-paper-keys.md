# Runbook — Wiring Alpaca Paper Keys

Five-minute setup. Free, no compliance review, no OAuth dance.

## What you need

Paper API keys from your own Alpaca account. **Not** an Alpaca Connect OAuth
application — that's for public-facing products and takes 30 business days of
compliance review. We do not need it.

## Steps

1. Sign in at [alpaca.markets](https://alpaca.markets).
2. Open the paper dashboard:
   [app.alpaca.markets/paper/dashboard/overview](https://app.alpaca.markets/paper/dashboard/overview).
3. Right sidebar → "Your API Keys" → "Generate New Key".
4. Copy both values immediately (the secret is only shown once):
   - Key ID: starts with `PK...`
   - Secret: longer alphanumeric string
5. Paste into `.env` at the repo root:
   ```bash
   ALPACA_PAPER_KEY_ID=PKXXXXXXXXXXXXXXXXXX
   ALPACA_PAPER_SECRET=...paste secret here...
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```
6. Verify:
   ```bash
   export $(grep -v '^#' .env | xargs)
   PYTHONPATH=. python3 tools/smoke_alpaca.py
   ```

Expected output: two `PASS` lines (`/v2/account` and `/v2/stocks/SPY/bars`)
plus your paper equity / buying power.

## What this unlocks

- 90 days of 5-minute IEX-adjusted intraday bars (vs. 60 days on yfinance)
- Cleaner Tier 2 stress-window coverage in `tools/validate_real_data.py`
- The execution agent's paper-trading path (still gated by
  `ENABLE_LIVE_TRADING=false`)

## What it does NOT do

- Place real-money orders. The base URL is `paper-api.alpaca.markets`. The
  live URL is `api.alpaca.markets` and is intentionally never set in
  `.env.example`.
- Replace yfinance. If Alpaca ever 401s or rate-limits, the pretrain pipeline
  falls back to yfinance automatically (see `packages/data/pretrain.py`
  `_fetch_intraday`).

## Rotating keys

If a key leaks (e.g. accidentally pasted in a chat):

1. Same dashboard → revoke the old key.
2. Generate a new one.
3. Update `.env` and re-run the smoke test.
4. Nothing else to do — the bot reads fresh values on each pretrain run.
