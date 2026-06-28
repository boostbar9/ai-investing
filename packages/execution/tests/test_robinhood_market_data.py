"""Tests for the read-only market-data methods on RobinhoodAgenticBroker.

These exercise the new research/scoring/regime read tools (historicals,
fundamentals, earnings, indexes, scans, realized P&L) against a MOCKED MCP
client (NO live network). The mock returns the real two-layer Robinhood
envelope -- MCP content blocks wrapping a ``{"data": {...}}`` presentation
object -- so we prove the existing unwrap helper handles the new shapes.

Every method must be READ-ONLY and fail safe: a transport/MCP error yields
``[]`` / ``None`` (never an exception, never a fabricated value).
"""
from __future__ import annotations

import json

import pytest

from packages.execution.modes import ExecutionMode
from packages.execution.robinhood import RobinhoodAgenticBroker
from packages.execution.robinhood_mcp import McpCallResult, McpError


def _envelope(payload: dict) -> list[dict]:
    """Wrap a domain payload the way the agentic server does: a single MCP
    text content block whose text is ``{"data": <payload>, "guide": "..."}``."""
    return [{"type": "text", "text": json.dumps({"data": payload, "guide": "x"})}]


class FakeMcp:
    def __init__(self, responses=None, *, error: Exception | None = None):
        self._responses = responses or {}
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._error is not None:
            raise self._error
        content = self._responses.get(name, [])
        return McpCallResult(tool=name, content=content)


def _broker(fake: FakeMcp) -> RobinhoodAgenticBroker:
    return RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        mcp_client=fake,
        token_loader=lambda: object(),  # presence of "a token" is enough
        account_number="668863863",
    )


# ---------------------------------------------------------------------------
# historicals
# ---------------------------------------------------------------------------
async def test_equity_historicals_parses_rows():
    fake = FakeMcp(
        {
            "get_equity_historicals": _envelope(
                {"historicals": [{"begins_at": "t1", "close_price": "10.0"}]}
            )
        }
    )
    rows = await _broker(fake).equity_historicals("AAPL", start_time="2026-01-01")
    assert rows == [{"begins_at": "t1", "close_price": "10.0"}]
    # required + supplied optional args forwarded; account number attached.
    name, args = fake.calls[0]
    assert name == "get_equity_historicals"
    assert args["symbols"] == ["AAPL"]
    assert args["start_time"] == "2026-01-01"
    assert args["account_number"] == "668863863"


async def test_equity_historicals_nested_per_symbol():
    fake = FakeMcp(
        {
            "get_equity_historicals": _envelope(
                {"results": [{"symbol": "AAPL", "historicals": [{"close_price": 1.0}]}]}
            )
        }
    )
    rows = await _broker(fake).equity_historicals("AAPL")
    assert rows == [{"close_price": 1.0}]


async def test_equity_historicals_optional_args_omitted_when_none():
    fake = FakeMcp({"get_equity_historicals": _envelope({"historicals": []})})
    await _broker(fake).equity_historicals("AAPL")
    _, args = fake.calls[0]
    assert "interval" not in args and "span" not in args and "start_time" not in args


async def test_equity_historicals_failsafe_on_error():
    fake = FakeMcp(error=McpError("boom"))
    assert await _broker(fake).equity_historicals("AAPL") == []


# ---------------------------------------------------------------------------
# fundamentals
# ---------------------------------------------------------------------------
async def test_equity_fundamentals_matches_symbol():
    fake = FakeMcp(
        {
            "get_equity_fundamentals": _envelope(
                {"fundamentals": [{"symbol": "AAPL", "pe_ratio": 30.0}]}
            )
        }
    )
    row = await _broker(fake).equity_fundamentals("AAPL")
    assert row["pe_ratio"] == 30.0


async def test_equity_fundamentals_failsafe_none():
    fake = FakeMcp(error=McpError("boom"))
    assert await _broker(fake).equity_fundamentals("AAPL") is None


# ---------------------------------------------------------------------------
# earnings
# ---------------------------------------------------------------------------
async def test_earnings_calendar_no_symbol():
    fake = FakeMcp(
        {"get_earnings_calendar": _envelope({"earnings": [{"symbol": "MSFT"}]})}
    )
    rows = await _broker(fake).earnings_calendar()
    assert rows == [{"symbol": "MSFT"}]
    _, args = fake.calls[0]
    assert "symbols" not in args  # no-arg form


async def test_earnings_results_per_symbol():
    fake = FakeMcp(
        {"get_earnings_results": _envelope({"results": [{"eps_actual": 1.2}]})}
    )
    rows = await _broker(fake).earnings_results("AAPL")
    assert rows == [{"eps_actual": 1.2}]


# ---------------------------------------------------------------------------
# indexes + index quotes
# ---------------------------------------------------------------------------
async def test_indexes_and_index_quotes():
    fake = FakeMcp(
        {
            "get_indexes": _envelope({"indexes": [{"symbol": "VIX", "id": "v1"}]}),
            "get_index_quotes": _envelope({"quotes": [{"last_price": 18.0}]}),
        }
    )
    b = _broker(fake)
    idx = await b.indexes()
    assert idx[0]["symbol"] == "VIX"
    quotes = await b.index_quotes(["v1"])
    assert quotes[0]["last_price"] == 18.0


async def test_index_quotes_empty_ids_skips_call():
    fake = FakeMcp({})
    assert await _broker(fake).index_quotes([]) == []
    assert fake.calls == []  # never hit the server


# ---------------------------------------------------------------------------
# scans
# ---------------------------------------------------------------------------
async def test_scans_empty_list():
    fake = FakeMcp({"get_scans": _envelope({"scans": []})})
    assert await _broker(fake).scans() == []


async def test_run_scan_results():
    fake = FakeMcp(
        {"run_scan": _envelope({"results": [{"symbol": "TSLA"}, {"symbol": "NVDA"}]})}
    )
    rows = await _broker(fake).run_scan("scan-1")
    assert [r["symbol"] for r in rows] == ["TSLA", "NVDA"]
    _, args = fake.calls[0]
    assert args["scan_id"] == "scan-1"


async def test_run_scan_empty_id_skips():
    fake = FakeMcp({})
    assert await _broker(fake).run_scan("") == []
    assert fake.calls == []


# ---------------------------------------------------------------------------
# realized P&L
# ---------------------------------------------------------------------------
async def test_realized_pnl_uses_account_number():
    fake = FakeMcp({"get_realized_pnl": _envelope({"realized_pnl": 1234.5})})
    res = await _broker(fake).realized_pnl()
    assert res == {"realized_pnl": 1234.5}
    name, args = fake.calls[0]
    assert name == "get_realized_pnl"
    assert args == {"account_number": "668863863"}  # no extra acct args wrapper


async def test_realized_pnl_failsafe_none():
    fake = FakeMcp(error=McpError("boom"))
    assert await _broker(fake).realized_pnl() is None
