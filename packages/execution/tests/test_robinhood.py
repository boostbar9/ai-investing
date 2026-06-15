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
            tool="place_equity_order",
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
    # Real Robinhood agentic-trading tool name (not the legacy guess).
    assert fake.calls[0][0] == "place_equity_order"
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
            tool="place_equity_order",
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
            tool="place_equity_order", content={"order_id": "rh-3"}
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
            tool="get_equity_positions",
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
    # Real Robinhood agentic-trading tool name (not the legacy guess).
    assert fake.calls[0][0] == "get_equity_positions"


@pytest.mark.asyncio
async def test_positions_accepts_wrapped_payload(
    isolated_shadow_log, isolated_onboarding
):
    """Robinhood may wrap the list under {'positions': [...]} or
    {'items': [...]}. Both shapes must parse."""
    fake = _FakeMcpClient(
        call_tool_result=McpCallResult(
            tool="get_equity_positions",
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


# ---------------------------------------------------------------------------
# MCP payload normalization helpers
# ---------------------------------------------------------------------------


def test_unwrap_content_decodes_text_block_json():
    """MCP wraps results in content blocks; a text block carrying a JSON
    string must be decoded to the underlying object."""
    blocks = [{"type": "text", "text": '{"a": 1, "b": [2, 3]}'}]
    assert rh_mod._unwrap_content(blocks) == {"a": 1, "b": [2, 3]}


def test_unwrap_content_prefers_json_block():
    blocks = [{"type": "json", "json": {"x": 9}}]
    assert rh_mod._unwrap_content(blocks) == {"x": 9}


def test_unwrap_content_passes_through_plain_objects():
    assert rh_mod._unwrap_content({"k": "v"}) == {"k": "v"}
    assert rh_mod._unwrap_content([{"row": 1}]) == [{"row": 1}]
    assert rh_mod._unwrap_content(None) is None
    # A bare JSON string also decodes.
    assert rh_mod._unwrap_content('{"n": 5}') == {"n": 5}
    # Non-JSON string is returned verbatim.
    assert rh_mod._unwrap_content("hello") == "hello"


def test_normalize_rows_from_text_block_wrapper():
    blocks = [{"type": "text", "text": '{"positions": [{"symbol": "AAPL"}]}'}]
    rows = rh_mod._normalize_rows(blocks, keys=("positions", "items"))
    assert rows == [{"symbol": "AAPL"}]


def test_normalize_rows_returns_empty_when_no_list():
    assert rh_mod._normalize_rows({"nope": 1}, keys=("positions",)) == []
    assert rh_mod._normalize_rows(None, keys=("positions",)) == []


def test_first_float_picks_first_parseable_key():
    obj = {"a": None, "b": "not-a-number", "c": "42.5", "d": 1.0}
    assert rh_mod._first_float(obj, ("a", "b", "c", "d")) == 42.5
    assert rh_mod._first_float({}, ("x",)) is None


# ---------------------------------------------------------------------------
# account_snapshot() -- live read-only account context for the AI
# ---------------------------------------------------------------------------


class _MultiToolMcpClient:
    """MCP fake that returns a different payload per tool name so we can
    exercise ``account_snapshot`` (which calls get_accounts + portfolio +
    get_equity_positions). Records every call for assertions."""

    def __init__(self, *, responses=None, raises_for=None):
        self.calls: list[tuple[str, dict]] = []
        self._responses = responses or {}
        self._raises_for = raises_for or {}

    async def initialize(self):
        return {}

    async def list_tools(self):
        return []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name in self._raises_for:
            raise self._raises_for[name]
        content = self._responses.get(name, {})
        return McpCallResult(tool=name, content=content, is_error=False)

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_account_snapshot_not_connected_returns_stable_shape(
    monkeypatch, isolated_onboarding
):
    """No token -> connected False, empty sections, never raises, and
    never touches the network."""
    monkeypatch.setattr(rh_mod, "is_connected", lambda: False)
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        mcp_client=_MultiToolMcpClient(),
        token_loader=lambda: None,
    )
    snap = await broker.account_snapshot()
    assert snap["connected"] is False
    assert snap["accounts"] == []
    assert snap["positions"] == []
    assert snap["buying_power"] is None
    assert snap["total_equity"] is None


@pytest.mark.asyncio
async def test_account_snapshot_is_read_only_in_shadow_mode(
    monkeypatch, isolated_onboarding
):
    """The headline safety property: snapshot NEVER places/cancels an
    order, so it only ever calls read tools -- even in shadow mode."""
    monkeypatch.setattr(rh_mod, "is_connected", lambda: True)
    fake = _MultiToolMcpClient(
        responses={
            "get_accounts": {
                "accounts": [
                    {"buying_power": "500.00", "cash": "123.45"}
                ]
            },
            "portfolio": {"equity": "1750.00"},
            "get_equity_positions": {
                "positions": [
                    {"symbol": "NVDA", "qty": 2, "avg_price": 100.0}
                ]
            },
        }
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        mcp_client=fake,
        token_loader=_good_tokens,
    )
    snap = await broker.account_snapshot()

    called = {name for name, _ in fake.calls}
    # ONLY read tools -- nothing that mutates the account.
    assert called <= {"get_accounts", "portfolio", "get_equity_positions"}
    assert "place_equity_order" not in called
    assert "cancel_order" not in called
    # Parsed top-line numbers.
    assert snap["connected"] is True
    assert snap["mode"] == "shadow"
    assert snap["buying_power"] == 500.0
    assert snap["cash"] == 123.45
    assert snap["total_equity"] == 1750.0
    assert snap["positions"][0]["symbol"] == "NVDA"


@pytest.mark.asyncio
async def test_account_snapshot_parses_mcp_content_blocks(
    monkeypatch, isolated_onboarding
):
    """Robinhood returns results inside MCP content blocks (a list of
    ``{"type": "text", "text": "<json>"}``). The snapshot must unwrap
    them transparently."""
    monkeypatch.setattr(rh_mod, "is_connected", lambda: True)
    fake = _MultiToolMcpClient(
        responses={
            "get_accounts": [
                {"type": "text", "text": '{"accounts": [{"buying_power": 300}]}'}
            ],
            "portfolio": [
                {"type": "text", "text": '{"market_value": "2200.50"}'}
            ],
            "get_equity_positions": [
                {"type": "text", "text": '[{"symbol": "SPY", "qty": 1, "avg_price": 500}]'}
            ],
        }
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )
    snap = await broker.account_snapshot()
    assert snap["buying_power"] == 300.0
    assert snap["total_equity"] == 2200.50
    assert snap["positions"][0]["symbol"] == "SPY"
    assert snap["mode"] == "live"


@pytest.mark.asyncio
async def test_account_snapshot_degrades_on_partial_failure(
    monkeypatch, isolated_onboarding
):
    """If one read tool errors, the snapshot records it but still returns
    the sections that succeeded -- it must never raise."""
    monkeypatch.setattr(rh_mod, "is_connected", lambda: True)
    fake = _MultiToolMcpClient(
        responses={
            "portfolio": {"equity": "999.00"},
            "get_equity_positions": {"positions": []},
        },
        raises_for={"get_accounts": McpError("accounts endpoint down")},
    )
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.LIVE,
        mcp_client=fake,
        token_loader=_good_tokens,
    )
    snap = await broker.account_snapshot()
    assert snap["total_equity"] == 999.0
    assert any("get_accounts" in e for e in snap["errors"])
    assert snap["connected"] is True


@pytest.mark.asyncio
async def test_robinhood_account_snapshot_module_fn_when_disconnected(
    monkeypatch
):
    """The cockpit-facing wrapper returns a stable empty shape (no
    network) when the user hasn't connected Robinhood."""
    monkeypatch.setattr(rh_mod, "is_connected", lambda: False)
    snap = await rh_mod.robinhood_account_snapshot()
    assert snap["connected"] is False
    assert snap["positions"] == []


@pytest.mark.asyncio
async def test_broker_aclose_is_idempotent(isolated_onboarding):
    """aclose() must be safe when no client was ever built and on repeat
    calls -- the snapshot wrapper relies on this in its finally block."""
    broker = RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        token_loader=lambda: None,
    )
    await broker.aclose()  # no cached client -> no-op
    await broker.aclose()  # repeat -> still fine
