"""Tests for Robinhood agentic-account targeting + the gated live path.

Covers:
  * ``select_agentic_account`` picks the single agentic_allowed=true
    account among non-agentic ones, and refuses (None) when none qualify.
  * ``discover_agentic_account_number`` returns the agentic account from a
    mocked ``get_accounts`` (never hits the network).
  * A live-mode buy threads ``account_number`` into ``place_equity_order``
    AND flows through the float cap + daily-notional ledger + ref_id.
  * The live order path refuses to submit without a resolved account.
"""

from __future__ import annotations

import pytest

from packages.execution import daily_notional as dn
from packages.execution.broker import BrokerError, OrderRequest
from packages.execution.modes import ExecutionMode
from packages.execution.robinhood import (
    RobinhoodAgenticBroker,
    discover_agentic_account_number,
    select_agentic_account,
)
from packages.execution.robinhood_mcp import McpCallResult
from packages.execution.robinhood_token import TokenSet

# Real account numbers from a confirmed live session: only AGENTIC_ACCT is
# agentic_allowed. Used as fixture values; never hardcoded in source.
AGENTIC_ACCT = "668863863"
MARGIN_ACCT = "5SA87845"
MANAGED_ACCT = "181701389106"


def _accounts_payload():
    """A realistic get_accounts list: one agentic, two non-agentic."""
    return [
        {"account_number": MARGIN_ACCT, "type": "margin", "agentic_allowed": False},
        {"account_number": AGENTIC_ACCT, "type": "cash", "agentic_allowed": True},
        {"account_number": MANAGED_ACCT, "type": "managed", "agentic_allowed": False},
    ]


def _good_tokens() -> TokenSet:
    import time

    return TokenSet(
        access_token="acc", refresh_token="ref", expires_at=time.time() + 3600
    )


class _FakeMcpClient:
    """Records every call_tool so we can assert args (esp. account_number)."""

    def __init__(self, *, results=None):
        self.calls: list[tuple[str, dict]] = []
        # mapping of tool name -> McpCallResult
        self._results = results or {}

    async def initialize(self):
        return {}

    async def list_tools(self):
        return []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self._results.get(
            name, McpCallResult(tool=name, content={}, is_error=False)
        )

    async def aclose(self):
        pass


@pytest.fixture(autouse=True)
def _isolate_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(dn, "DAILY_NOTIONAL_PATH", tmp_path / "daily_notional.jsonl")


# ---------------------------------------------------------------------------
# select_agentic_account
# ---------------------------------------------------------------------------


def test_select_picks_agentic_among_non_agentic():
    assert select_agentic_account(_accounts_payload()) == AGENTIC_ACCT


def test_select_refuses_when_none_agentic():
    accounts = [
        {"account_number": MARGIN_ACCT, "agentic_allowed": False},
        {"account_number": MANAGED_ACCT, "agentic_allowed": False},
    ]
    assert select_agentic_account(accounts) is None


def test_select_skips_deactivated_agentic():
    accounts = [
        {
            "account_number": AGENTIC_ACCT,
            "agentic_allowed": True,
            "status": "deactivated",
        },
    ]
    assert select_agentic_account(accounts) is None


def test_select_accepts_camelcase_alias():
    accounts = [{"accountNumber": AGENTIC_ACCT, "agenticAllowed": True}]
    assert select_agentic_account(accounts) == AGENTIC_ACCT


@pytest.mark.asyncio
async def test_discover_returns_agentic_account_no_network():
    fake = _FakeMcpClient(
        results={
            "get_accounts": McpCallResult(
                tool="get_accounts",
                content={"accounts": _accounts_payload()},
                is_error=False,
            )
        }
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW, mcp_client=fake, token_loader=_good_tokens
    )
    acct = await discover_agentic_account_number(broker)
    assert acct == AGENTIC_ACCT
    assert [c[0] for c in fake.calls] == ["get_accounts"]


# ---------------------------------------------------------------------------
# Live path: account_number threaded + safety stack (cap + ledger + ref_id)
# ---------------------------------------------------------------------------


@pytest.fixture
def _open_live_gate(monkeypatch):
    """Open the live gate the way test_robinhood.py does so an explicit
    LIVE broker actually reaches the wire."""
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    monkeypatch.setenv("ROBINHOOD_FORCE_LIVE_GATE", "true")


@pytest.fixture
def isolated_onboarding(monkeypatch, tmp_path):
    from packages.cockpit import onboarding as ob

    path = tmp_path / "onboarding.json"
    monkeypatch.setattr(ob, "ONBOARDING_PATH", path)
    import json

    path.write_text(
        json.dumps({"live_float_cap_usd": 300.0, "rh_mode": "live"})
    )
    return path


@pytest.mark.asyncio
async def test_live_buy_threads_account_and_flows_through_safety_stack(
    _open_live_gate, isolated_onboarding
):
    fake = _FakeMcpClient(
        results={
            "place_equity_order": McpCallResult(
                tool="place_equity_order",
                content={"order_id": "abc123", "status": "accepted"},
                is_error=False,
            )
        }
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
        account_number=AGENTIC_ACCT,
    )
    # $50 notional buy, well under the $300 cap.
    ack = await broker.submit(
        OrderRequest(
            symbol="SPY",
            side="buy",
            qty=1,
            limit_price=50.0,
            decision_id="d1",
            bar_ts="2026-06-16T15:00:00Z",
        )
    )
    assert ack.status == "accepted"
    # The order carried the agentic account_number + a UUID ref_id.
    name, args = fake.calls[-1]
    assert name == "place_equity_order"
    assert args["account_number"] == AGENTIC_ACCT
    assert "ref_id" in args and len(args["ref_id"]) == 36  # UUID format
    # The buy was recorded in the daily-notional ledger (live).
    assert dn.deployed_today() == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_live_buy_over_cap_rejected_before_network(
    _open_live_gate, isolated_onboarding
):
    fake = _FakeMcpClient()
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
        account_number=AGENTIC_ACCT,
    )
    with pytest.raises(BrokerError, match="exceeds float cap"):
        await broker.submit(
            OrderRequest(symbol="SPY", side="buy", qty=10, limit_price=100.0)
        )
    assert fake.calls == []  # never reached the wire


@pytest.mark.asyncio
async def test_live_buy_daily_ledger_blocks_cumulative_breach(
    _open_live_gate, isolated_onboarding
):
    """Two $200 buys = $400 deployed > $300 cap; the second is blocked by
    the cumulative daily-notional ledger."""
    fake = _FakeMcpClient(
        results={
            "place_equity_order": McpCallResult(
                tool="place_equity_order",
                content={"order_id": "x", "status": "accepted"},
                is_error=False,
            )
        }
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
        account_number=AGENTIC_ACCT,
    )
    await broker.submit(
        OrderRequest(symbol="AAA", side="buy", qty=1, limit_price=200.0)
    )
    with pytest.raises(BrokerError, match="daily float cap"):
        await broker.submit(
            OrderRequest(symbol="BBB", side="buy", qty=1, limit_price=200.0)
        )


@pytest.mark.asyncio
async def test_live_buy_refuses_without_account_number(
    _open_live_gate, isolated_onboarding
):
    """Fail safe: no resolved agentic account -> refuse the live order
    rather than letting Robinhood reject it at the gateway."""
    fake = _FakeMcpClient()
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
        account_number=None,
    )
    with pytest.raises(BrokerError, match="no agentic account"):
        await broker.submit(
            OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=50.0)
        )
    assert fake.calls == []
