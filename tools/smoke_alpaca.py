"""Smoke test for Alpaca paper credentials.

Run locally after pasting your paper keys into ``.env``::

    export $(grep -v '^#' .env | xargs)  # or: source ./scripts/load-env.sh
    PYTHONPATH=. python3 tools/smoke_alpaca.py

What it does
------------
1. Verifies ``ALPACA_PAPER_KEY_ID`` + ``ALPACA_PAPER_SECRET`` are set.
2. Hits the *trading* endpoint (``/v2/account``) to confirm broker auth works
   and prints the paper buying-power / equity so you know it is paper.
3. Hits the *data* endpoint (``/v2/stocks/SPY/bars``) to confirm the free IEX
   feed works and returns at least one bar.

Both endpoints use the same credentials. If both succeed, the bot can run
nightly pretrain with Alpaca as the primary intraday source (yfinance falls
back automatically if either call ever fails).
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta

import httpx

from packages.data.adapters.alpaca_data import AlpacaDataAdapter

TRADING_BASE = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")


async def _check_account() -> dict:
    """Hit the trading API /v2/account to confirm broker auth + paper status."""
    key = os.getenv("ALPACA_PAPER_KEY_ID", "")
    secret = os.getenv("ALPACA_PAPER_SECRET", "")
    async with httpx.AsyncClient(
        timeout=15,
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
    ) as c:
        r = await c.get(f"{TRADING_BASE}/v2/account")
        r.raise_for_status()
        return r.json()


async def _check_data() -> int:
    """Hit the market-data API for one day of SPY 5-min bars."""
    adapter = AlpacaDataAdapter()
    end = datetime.now(UTC)
    start = end - timedelta(days=2)
    try:
        bars = await adapter.get_bars(
            "SPY",
            start.isoformat(),
            end.isoformat(),
            timeframe="5Min",
            feed="iex",
        )
        return len(bars)
    finally:
        await adapter.aclose()


async def main() -> int:
    key = os.getenv("ALPACA_PAPER_KEY_ID", "")
    secret = os.getenv("ALPACA_PAPER_SECRET", "")
    if not key or not secret:
        print("FAIL: ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET not set in environment.")
        print("Fix: paste the values from https://app.alpaca.markets/paper/dashboard/overview")
        print("     into .env, then `export $(grep -v '^#' .env | xargs)` and retry.")
        return 1

    print(f"key id:    {key[:4]}...{key[-4:]}  (len={len(key)})")
    print(f"secret:    set (len={len(secret)})")
    print(f"trading:   {TRADING_BASE}")
    print()

    # 1) Trading endpoint
    try:
        acct = await _check_account()
    except httpx.HTTPStatusError as e:
        print(f"FAIL trading /v2/account: HTTP {e.response.status_code} {e.response.text[:200]}")
        return 2
    except Exception as e:
        print(f"FAIL trading /v2/account: {e}")
        return 2

    status = acct.get("status", "?")
    is_paper = bool(acct.get("trading_blocked") is False and "paper" in TRADING_BASE)
    print(f"PASS /v2/account  status={status}  paper={is_paper}")
    print(f"     equity        = ${float(acct.get('equity', 0)):,.2f}")
    print(f"     buying_power  = ${float(acct.get('buying_power', 0)):,.2f}")
    print(f"     cash          = ${float(acct.get('cash', 0)):,.2f}")
    print()

    # 2) Market data endpoint
    try:
        n = await _check_data()
    except Exception as e:
        print(f"FAIL data /v2/stocks/SPY/bars: {e}")
        return 3
    print(f"PASS /v2/stocks/SPY/bars  bars_returned={n}  feed=iex  timeframe=5Min")
    print()

    if not is_paper:
        print("WARNING: ALPACA_BASE_URL does not contain 'paper'. Double-check before going live.")
        return 4

    print("All checks green. The bot can use Alpaca for nightly pretrain.")
    print("Next: re-run `PYTHONPATH=. python3 tools/validate_real_data.py` to refresh Tier 2.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
