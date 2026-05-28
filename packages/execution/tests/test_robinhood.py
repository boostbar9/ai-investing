"""Tests for ``RobinhoodAgenticBroker`` (Phase 2).

The broker is the user's last line of defense against an LLM ordering
$10k of GME at 3am. These tests pin down:

  * Shadow mode never touches the network.
  * Float cap rejects oversize *buys* before the MCP call.
  * Sells bypass the cap (deployment ceiling, not exposure ceiling).
  * Market orders with no limit price are treated conservatively.
  * Positions parsing accepts a flexible Robinhood payload.
  * The ``build_broker_from_settings`` factory honors ``rh_mode``.
"""

from __future__ import annotations

import json

import pytest

from packages.execution import robinhood as rh_mod
from packages.execution.broker import BrokerError, OrderRequest
from packages.execution.modes import ExecutionMode
from packages.execution.robinhood import (
    ABSOLUTE_MAX_FLOAT_USD,
    RobinhoodAgenticBroker,
    build_broker_from_settings,
    load_shadow_trades,
    resolve_float_cap,
)
from packages.execution.robinhood_mcp import McpCallResult, McpError
from packages.execution.robinhood_token import TokenSet

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_shadow_log(monkeypatch, tmp_path):
    """Redirect the shadow trades JSONL to a tmp file so tests don't
    pollute ``data/cockpit/`` (which is gitignored but still shared
    across the local dev workspace)."""
    path = tmp_path / "shadow_trades.jsonl"
    monkeypatch.setattr(rh_mod, "SHADOW_TRADES_PATH", path)
    return path


@pytest.fixture
def isolated_onboarding(monkeypatch, tmp_path):
    """Point onboarding state at a tmp file so float-cap tests can set a
    deterministic cap without touching the real user state."""
    from packages.cockpit import onboarding as ob

    path = tmp_path / "onboarding.json"
    monkeypatch.setattr(ob, "ONBOARDING_PATH", path)
    return path


def _write_onboarding(path, *, cap=300.0, mode="shadow"):
    path.write_text(
        json.dumps(
            {
                "completed": True,
                "robinhood_status": "granted",
                "live_float_cap_usd": cap,
                "rh_mode": mode,
                "accepted_disclaimer_at": "2026-05-27T00:00:00+00:00",
            }
        )
    )


class _FakeMcpClient:
    """Stand-in for ``RobinhoodMcpClient``. Records every ``call_tool``
    so tests can assert (or assert *absence*) of network calls."""

    def __init__(self, *, call_tool_result=None, raises=None):
        self.calls: list[tuple[str, dict]] = []
        self._result = call_tool_result
        self._raises = raises
        self.list_tools_calls = 0

    async def initialize(self):
        return {}

    async def list_tools(self):
        self.list_tools_calls += 1
        return []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self._raises:
            raise self._raises
        return self._result or McpCallResult(tool=name, content={}, is_error=False)

    async def aclose(self):
        pass


def _good_tokens() -> TokenSet:
    """A fresh, non-stale token set."""
    import time

    return TokenSet(
        access_token="acc",
        refresh_token="ref",
        expires_at=time.time() + 3600,
    )


# ---------------------------------------------------------------------------
# Shadow mode: never touches the network
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_submit_does_not_call_mcp(
    isolated_shadow_log, isolated_onboarding
):
    """The headline safety property: in SHADOW, every submit() bypasses
    Robinhood entirely. Even with a hostile MCP client that would raise,
    we still get a clean fake ack."""
    _write_onboarding(isolated_onboarding, cap=300.0)
    boom = _FakeMcpClient(raises=McpError("network exploded"))
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        mcp_client=boom,
        token_loader=_good_tokens,
    )

    ack = await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=100.0)
    )

    assert ack.broker == "robinhood_agentic"
    assert ack.status == "accepted_shadow"
    assert ack.broker_order_id.startswith("shadow-")
    assert boom.calls == []  # critical: no network


@pytest.mark.asyncio
async def test_shadow_submit_writes_audit_log(
    isolated_shadow_log, isolated_onboarding
):
    _write_onboarding(isolated_onboarding, cap=300.0)
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        mcp_client=_FakeMcpClient(),
        token_loader=_good_tokens,
    )

    await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=2, limit_price=50.0)
    )
    await broker.submit(
        OrderRequest(symbol="QQQ", side="sell", qty=1, limit_price=400.0)
    )

    rows = load_shadow_trades()
    assert len(rows) == 2
    assert rows[0]["symbol"] == "SPY"
    assert rows[0]["mode"] == "shadow"
    assert rows[0]["side"] == "buy"
    assert rows[0]["notional_estimate"] == pytest.approx(100.0)
    assert rows[1]["side"] == "sell"
    # Sells don't get a notional estimate (cap bypassed).
    assert rows[1]["notional_estimate"] is None


@pytest.mark.asyncio
async def test_paper_mode_is_treated_as_shadow(
    isolated_shadow_log, isolated_onboarding
):
    """There is no 'paper Robinhood' -- PAPER must behave like SHADOW
    so anything non-LIVE stays off the wire."""
    _write_onboarding(isolated_onboarding, cap=300.0)
    fake = _FakeMcpClient()
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.PAPER,
        mcp_client=fake,
        token_loader=_good_tokens,
    )
    ack = await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=100.0)
    )
    assert ack.status == "accepted_shadow"
    assert fake.calls == []


# ---------------------------------------------------------------------------
# Float cap
# ---------------------------------------------------------------------------


def test_resolve_float_cap_reads_onboarding(isolated_onboarding):
    _write_onboarding(isolated_onboarding, cap=750.0)
    assert resolve_float_cap() == 750.0


def test_resolve_float_cap_clamps_to_absolute_max(isolated_onboarding):
    _write_onboarding(isolated_onboarding, cap=1_000_000.0)
    assert resolve_float_cap() == ABSOLUTE_MAX_FLOAT_USD


def test_resolve_float_cap_clamps_negative_via_onboarding(isolated_onboarding):
    """Negative cap on disk is repaired by the onboarding loader to the
    default ($300). Defense-in-depth: ``resolve_float_cap`` also clamps
    at min 0 in case the loader contract ever changes."""
    _write_onboarding(isolated_onboarding, cap=-50.0)
    # Onboarding loader resets to DEFAULT_FLOAT_CAP_USD on negative input.
    assert resolve_float_cap() == 300.0


@pytest.mark.asyncio
async def test_buy_exceeding_cap_raises_before_mcp(
    isolated_shadow_log, isolated_onboarding
):
    """Even in LIVE mode, the cap is enforced *before* we contact
    Robinhood -- this is the user's 'I can't accidentally yolo my
    paycheck' guarantee."""
    _write_onboarding(isolated_onboarding, cap=300.0)
    fake = _FakeMcpClient()
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )

    with pytest.raises(BrokerError) as excinfo:
        await broker.submit(
            OrderRequest(symbol="SPY", side="buy", qty=10, limit_price=100.0)
        )

    assert "cap" in str(excinfo.value).lower()
    assert fake.calls == []  # never hit the wire


@pytest.mark.asyncio
async def test_buy_within_cap_passes_through_to_mcp(
    isolated_shadow_log, isolated_onboarding
):
    _write_onboarding(isolated_onboarding, cap=300.0)
    fake = _FakeMcpClient(
        call_tool_result=McpCallResult(
            tool="submit_order",
            content={"order_id": "rh-1", "status": "accepted"},
            is_error=False,
        )
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )

    ack = await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=2, limit_price=100.0)
    )

    assert ack.broker_order_id == "rh-1"
    assert ack.status == "accepted"
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "submit_order"
    assert fake.calls[0][1]["symbol"] == "SPY"


@pytest.mark.asyncio
async def test_sell_bypasses_float_cap(
    isolated_shadow_log, isolated_onboarding
):
    """Sells reduce exposure -- refusing them would lock the user out of
    risk reduction. The cap is a *deployment* ceiling, not an exposure
    ceiling."""
    _write_onboarding(isolated_onboarding, cap=300.0)
    fake = _FakeMcpClient(
        call_tool_result=McpCallResult(
            tool="submit_order",
            content={"order_id": "rh-2", "status": "accepted"},
            is_error=False,
        )
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )

    # $10k sell -- 33x the cap -- must succeed.
    ack = await broker.submit(
        OrderRequest(
            symbol="SPY", side="sell", qty=100, limit_price=100.0
        )
    )
    assert ack.broker_order_id == "rh-2"
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_market_order_without_limit_price_uses_cap_as_ceiling(
    isolated_shadow_log, isolated_onboarding
):
    """A market buy with no price hint must be treated as if it spends
    the entire cap. Anything > cap implicitly: the order is rejected
    if even a notional-cap fill would breach the limit. With qty=1 and
    cap=$300, notional==cap so it just barely passes (<=)."""
    _write_onboarding(isolated_onboarding, cap=300.0)
    fake = _FakeMcpClient(
        call_tool_result=McpCallResult(
            tool="submit_order", content={"order_id": "rh-3"}
        )
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )

    # Right at the cap: passes (notional == cap is allowed).
    await broker.submit(
        OrderRequest(symbol="SPY", side="buy", qty=1, type="market")
    )
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# Order shape validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bad_qty_rejected(
    isolated_shadow_log, isolated_onboarding
):
    _write_onboarding(isolated_onboarding)
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        mcp_client=_FakeMcpClient(),
        token_loader=_good_tokens,
    )
    with pytest.raises(BrokerError):
        await broker.submit(OrderRequest(symbol="SPY", side="buy", qty=0))


@pytest.mark.asyncio
async def test_bad_side_rejected(
    isolated_shadow_log, isolated_onboarding
):
    _write_onboarding(isolated_onboarding)
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        mcp_client=_FakeMcpClient(),
        token_loader=_good_tokens,
    )
    with pytest.raises(BrokerError):
        await broker.submit(
            OrderRequest(symbol="SPY", side="short", qty=1, limit_price=10.0)
        )


# ---------------------------------------------------------------------------
# MCP errors get wrapped in BrokerError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_mcp_error_becomes_broker_error(
    isolated_shadow_log, isolated_onboarding
):
    _write_onboarding(isolated_onboarding, cap=10_000.0)
    boom = _FakeMcpClient(raises=McpError("server hiccup"))
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=boom,
        token_loader=_good_tokens,
    )
    with pytest.raises(BrokerError) as excinfo:
        await broker.submit(
            OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=100.0)
        )
    assert "server hiccup" in str(excinfo.value)


@pytest.mark.asyncio
async def test_live_is_error_response_raises(
    isolated_shadow_log, isolated_onboarding
):
    """An MCP response with ``isError=true`` is a structured rejection
    from Robinhood (e.g. PDT, insufficient buying power). Surface it
    as BrokerError so the strategy can react."""
    _write_onboarding(isolated_onboarding, cap=10_000.0)
    fake = _FakeMcpClient(
        call_tool_result=McpCallResult(
            tool="submit_order",
            content={"reason": "pdt restriction"},
            is_error=True,
        )
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )
    with pytest.raises(BrokerError) as excinfo:
        await broker.submit(
            OrderRequest(symbol="SPY", side="buy", qty=1, limit_price=100.0)
        )
    assert "pdt" in str(excinfo.value).lower()


# ---------------------------------------------------------------------------
# Token gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_in_shadow_reports_token_presence(
    isolated_shadow_log, isolated_onboarding
):
    """Shadow health is just 'do we have tokens?' -- no network."""
    broker_no_tokens = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        token_loader=lambda: None,
    )
    assert await broker_no_tokens.health() is False

    broker_with_tokens = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        token_loader=_good_tokens,
    )
    assert await broker_with_tokens.health() is True


@pytest.mark.asyncio
async def test_health_in_live_pings_mcp(
    isolated_shadow_log, isolated_onboarding
):
    fake = _FakeMcpClient()
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )
    assert await broker.health() is True
    assert fake.list_tools_calls == 1


@pytest.mark.asyncio
async def test_health_in_live_returns_false_on_mcp_error(
    isolated_shadow_log, isolated_onboarding
):
    class _BoomTools(_FakeMcpClient):
        async def list_tools(self):
            raise McpError("boom")

    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=_BoomTools(),
        token_loader=_good_tokens,
    )
    assert await broker.health() is False


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_positions_parses_flat_list(
    isolated_shadow_log, isolated_onboarding
):
    fake = _FakeMcpClient(
        call_tool_result=McpCallResult(
            tool="list_positions",
            content=[
                {
                    "symbol": "SPY",
                    "qty": 10,
                    "avg_price": 500.0,
                    "last_price": 510.0,
                    "pnl_pct": 0.02,
                },
                {"symbol": "BROKEN", "qty": "nope"},  # skipped
            ],
        )
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )
    ps = await broker.positions()
    assert len(ps) == 1
    assert ps[0].symbol == "SPY"
    assert ps[0].qty == 10
    assert ps[0].pnl_pct == 0.02


@pytest.mark.asyncio
async def test_positions_accepts_wrapped_payload(
    isolated_shadow_log, isolated_onboarding
):
    """Robinhood may wrap the list under {'positions': [...]} or
    {'items': [...]}. Both shapes must parse."""
    fake = _FakeMcpClient(
        call_tool_result=McpCallResult(
            tool="list_positions",
            content={
                "positions": [
                    {"symbol": "QQQ", "qty": 5, "avg_price": 400.0}
                ]
            },
        )
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )
    ps = await broker.positions()
    assert len(ps) == 1
    assert ps[0].symbol == "QQQ"


@pytest.mark.asyncio
async def test_positions_returns_empty_on_mcp_error(
    isolated_shadow_log, isolated_onboarding
):
    """Position reads must NEVER raise -- the dashboard depends on a
    consistent shape even when Robinhood is down."""
    boom = _FakeMcpClient(raises=McpError("server down"))
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=boom,
        token_loader=_good_tokens,
    )
    assert await broker.positions() == []


@pytest.mark.asyncio
async def test_positions_returns_empty_when_no_tokens(
    isolated_shadow_log, isolated_onboarding
):
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        token_loader=lambda: None,
    )
    assert await broker.positions() == []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_broker_from_settings_defaults_to_shadow(isolated_onboarding):
    """No onboarding state on disk -> default SHADOW. The user has to
    actively flip rh_mode to live; we never assume."""
    broker = build_broker_from_settings()
    assert broker._mode is ExecutionMode.SHADOW


def test_build_broker_from_settings_honors_live_mode(isolated_onboarding):
    _write_onboarding(isolated_onboarding, mode="live")
    broker = build_broker_from_settings()
    assert broker._mode is ExecutionMode.LIVE


def test_build_broker_from_settings_honors_shadow_mode(isolated_onboarding):
    _write_onboarding(isolated_onboarding, mode="shadow")
    broker = build_broker_from_settings()
    assert broker._mode is ExecutionMode.SHADOW


# ---------------------------------------------------------------------------
# Shadow log resilience
# ---------------------------------------------------------------------------


def test_load_shadow_trades_skips_malformed_lines(isolated_shadow_log):
    """One bad line shouldn't lose the rest of the audit log."""
    isolated_shadow_log.parent.mkdir(parents=True, exist_ok=True)
    isolated_shadow_log.write_text(
        '{"symbol":"SPY"}\n'
        "not json at all\n"
        '\n'
        '{"symbol":"QQQ"}\n'
    )
    rows = load_shadow_trades()
    assert [r["symbol"] for r in rows] == ["SPY", "QQQ"]


def test_load_shadow_trades_returns_empty_when_missing(isolated_shadow_log):
    assert load_shadow_trades() == []
