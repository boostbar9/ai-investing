"""Per-provider connection tests for the cockpit Settings page.

Each test returns ``(ok, message)``. Tests time out at 8s so the UI stays
responsive even with bad credentials hitting slow endpoints.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import httpx

TIMEOUT = 8.0


def _alpaca_paper() -> tuple[bool, str]:
    kid = os.environ.get("ALPACA_PAPER_KEY_ID", "")
    sec = os.environ.get("ALPACA_PAPER_SECRET", "")
    if not (kid and sec):
        return False, "missing key id or secret"
    try:
        r = httpx.get(
            "https://paper-api.alpaca.markets/v2/account",
            headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            return True, f"connected (equity ${float(data.get('equity', 0)):,.2f})"
        return False, f"HTTP {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _alpaca_live() -> tuple[bool, str]:
    kid = os.environ.get("ALPACA_LIVE_KEY_ID", "")
    sec = os.environ.get("ALPACA_LIVE_SECRET", "")
    if not (kid and sec):
        return False, "missing key id or secret"
    try:
        r = httpx.get(
            "https://api.alpaca.markets/v2/account",
            headers={"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return True, "connected (LIVE - be careful)"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _fred() -> tuple[bool, str]:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        return False, "missing api key"
    try:
        r = httpx.get(
            "https://api.stlouisfed.org/fred/series",
            params={"series_id": "GDP", "api_key": key, "file_type": "json"},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return True, "connected"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _polygon() -> tuple[bool, str]:
    key = os.environ.get("POLYGON_API_KEY", "")
    if not key:
        return False, "missing api key"
    try:
        r = httpx.get(
            "https://api.polygon.io/v3/reference/tickers/AAPL",
            params={"apiKey": key},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return True, "connected"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _alphavantage() -> tuple[bool, str]:
    key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not key:
        return False, "missing api key"
    try:
        r = httpx.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": "AAPL", "apikey": key},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("Note") or data.get("Information"):
                return False, "rate-limited (key works, free tier exhausted)"
            return True, "connected"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _finnhub() -> tuple[bool, str]:
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        return False, "missing api key"
    try:
        r = httpx.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": "AAPL", "token": key},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return True, "connected"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


TESTS: dict[str, Callable[[], tuple[bool, str]]] = {
    "alpaca_paper": _alpaca_paper,
    "alpaca_live": _alpaca_live,
    "fred": _fred,
    "polygon": _polygon,
    "alphavantage": _alphavantage,
    "finnhub": _finnhub,
}


def check_provider(provider_id: str) -> tuple[bool, str]:
    """Run the configured connectivity check for ``provider_id``.

    Named ``check_*`` rather than ``test_*`` so pytest does not collect it.
    """
    fn = TESTS.get(provider_id)
    if fn is None:
        return False, f"unknown provider: {provider_id}"
    return fn()
