"""Phase 25.4 \u2014 FinnhubWebSocketClient contract tests.

Uses a controllable fake WebSocket transport (no real network) to
verify:

* No key \u2192 ``start()`` returns False, nothing else happens
* Connect \u2192 subscribe frames sent for desired symbols
* Trade frame \u2192 ``on_tick`` invoked with (symbol, price, ts)
* Multiple ticks in one frame \u2192 only the latest price per symbol fires
* ``set_symbols`` diffs and emits subscribe/unsubscribe correctly
* Symbol cap evicts overflow at the tail
* Index symbols filter is the caller's job, not the client's
* Server ``error`` messages are captured in status, not raised
* Server ``ping`` triggers a ``pong`` response
* Bad/non-JSON frames are tolerated
* Reconnect after disconnect, with subscriptions replayed
* ``stop()`` cancels the loop and closes the socket cleanly
* ``status()`` shape matches what the dashboard expects
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from packages.cockpit.web.finnhub_ws import (
    DEFAULT_MAX_SYMBOLS,
    FinnhubWebSocketClient,
)

# ---------------------------------------------------------------------------
# Fake WebSocket transport \u2014 every test drives the client through this.
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """In-memory bidirectional WS for tests.

    Tests push frames via :meth:`push_server_frame`; the client's
    ``recv()`` consumes them in order. Frames the client ``send``s
    land in :attr:`sent`. Closing the socket raises an error from
    ``recv`` so the client's reconnect path engages.
    """

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        self.close_calls = 0

    def push_server_frame(self, payload: Any) -> None:
        """Schedule a frame for the client to consume. ``None`` = EOF."""
        if isinstance(payload, dict):
            payload = json.dumps(payload)
        self._incoming.put_nowait(payload)

    def push_disconnect(self) -> None:
        """Trigger a connection-loss error on the next recv()."""
        self._incoming.put_nowait(_DisconnectSentinel)

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str | bytes:
        item = await self._incoming.get()
        if item is _DisconnectSentinel:
            raise ConnectionError("simulated drop")
        return item

    async def close(self) -> None:
        self.closed = True
        self.close_calls += 1


class _DisconnectSentinel:
    """Marker pushed via push_disconnect()."""


class FakeFactory:
    """Async-callable factory that returns successive FakeWebSocket instances."""

    def __init__(self) -> None:
        self.sockets: list[FakeWebSocket] = []
        self.connect_calls: list[str] = []
        # Future for synchronization: completed when each socket is ready.
        self.ready_events: list[asyncio.Event] = []

    async def __call__(self, url: str) -> FakeWebSocket:
        self.connect_calls.append(url)
        ws = FakeWebSocket()
        self.sockets.append(ws)
        evt = asyncio.Event()
        evt.set()
        self.ready_events.append(evt)
        return ws

    @property
    def latest(self) -> FakeWebSocket:
        return self.sockets[-1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_ticks() -> tuple[list[tuple[str, float, Any]], Any]:
    ticks: list[tuple[str, float, Any]] = []

    def on_tick(sym: str, price: float, ts: Any) -> None:
        ticks.append((sym, price, ts))

    return ticks, on_tick


async def _wait_for(predicate, timeout: float = 1.0, step: float = 0.01) -> None:
    """Poll until predicate() is truthy or timeout."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(step)
    raise AssertionError("predicate never became true")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_key_start_returns_false() -> None:
    _, on_tick = _collect_ticks()
    client = FinnhubWebSocketClient(api_key="", on_tick=on_tick)

    async def _go():
        return client.start()

    assert asyncio.run(_go()) is False
    assert client.running is False
    assert client.has_key is False


def test_connect_subscribe_and_tick_ingest() -> None:
    ticks, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        client.start()
        await client.set_symbols(["AAPL", "SPY"])
        await _wait_for(lambda: len(factory.sockets) >= 1)
        ws = factory.latest
        # The client should have already sent subscribe frames.
        await _wait_for(lambda: len(ws.sent) >= 2)
        # Now feed a trade frame.
        ws.push_server_frame({
            "type": "trade",
            "data": [
                {"s": "AAPL", "p": 199.99, "t": 1_700_000_000_000, "v": 100},
            ],
        })
        await _wait_for(lambda: len(ticks) >= 1)
        await client.stop()

    asyncio.run(_go())

    # Subscribed AAPL + SPY.
    payloads = [json.loads(s) for s in factory.latest.sent]
    subs = {p["symbol"] for p in payloads if p.get("type") == "subscribe"}
    assert subs == {"AAPL", "SPY"}
    # Tick captured with the latest price.
    assert ticks == [("AAPL", 199.99, ticks[0][2])]
    # Stats reflect at least one ingested tick.
    st = client.status()
    assert st["ticks_ingested"] == 1
    assert st["last_tick_symbol"] == "AAPL"


def test_trade_frame_coalesces_to_latest_price_per_symbol() -> None:
    ticks, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        client.start()
        await client.set_symbols(["AAPL"])
        await _wait_for(lambda: len(factory.sockets) >= 1)
        ws = factory.latest
        ws.push_server_frame({
            "type": "trade",
            "data": [
                {"s": "AAPL", "p": 100.0, "t": 1_700_000_000_000},
                {"s": "AAPL", "p": 100.5, "t": 1_700_000_000_500},
                {"s": "AAPL", "p": 101.0, "t": 1_700_000_001_000},
            ],
        })
        await _wait_for(lambda: len(ticks) >= 1)
        await client.stop()

    asyncio.run(_go())

    # Only the latest price for AAPL in that frame.
    assert len(ticks) == 1
    assert ticks[0][0] == "AAPL"
    assert ticks[0][1] == 101.0


def test_set_symbols_diffs_subscribe_and_unsubscribe() -> None:
    _, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        client.start()
        await client.set_symbols(["AAPL", "SPY"])
        await _wait_for(lambda: len(factory.sockets) >= 1)
        ws = factory.latest
        await _wait_for(lambda: len(ws.sent) >= 2)
        # Replace SPY with MSFT \u2014 expect unsub SPY + sub MSFT only.
        result = await client.set_symbols(["AAPL", "MSFT"])
        await client.stop()
        return result, ws.sent

    result, sent = asyncio.run(_go())

    assert result["subscribed_now"] == ["MSFT"]
    assert result["unsubscribed_now"] == ["SPY"]
    decoded = [json.loads(s) for s in sent]
    types = [(p.get("type"), p.get("symbol")) for p in decoded]
    assert ("subscribe", "MSFT") in types
    assert ("unsubscribe", "SPY") in types


def test_symbol_cap_drops_overflow() -> None:
    _, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST",
        on_tick=on_tick,
        max_symbols=3,
        connect_factory=factory,
    )

    async def _go():
        result = await client.set_symbols(["A", "B", "C", "D", "E"])
        await client.stop()
        return result

    result = asyncio.run(_go())
    assert result["capped"] is True
    assert sorted(result["desired"]) == ["A", "B", "C"]


def test_set_symbols_normalizes_and_dedupes() -> None:
    _, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        result = await client.set_symbols([" aapl ", "AAPL", "spy", "SPY"])
        await client.stop()
        return result

    result = asyncio.run(_go())
    assert sorted(result["desired"]) == ["AAPL", "SPY"]


def test_default_max_symbols_is_50() -> None:
    assert DEFAULT_MAX_SYMBOLS == 50


def test_server_error_message_captured_in_status() -> None:
    _, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        client.start()
        await client.set_symbols(["AAPL"])
        await _wait_for(lambda: len(factory.sockets) >= 1)
        ws = factory.latest
        ws.push_server_frame({"type": "error", "msg": "unsupported symbol"})
        await _wait_for(lambda: client.status()["last_error"] is not None)
        st = client.status()
        await client.stop()
        return st

    st = asyncio.run(_go())
    assert "unsupported symbol" in st["last_error"]


def test_server_ping_triggers_pong() -> None:
    _, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        client.start()
        await client.set_symbols(["AAPL"])
        await _wait_for(lambda: len(factory.sockets) >= 1)
        ws = factory.latest
        # Drain the initial subscribe frame.
        await _wait_for(lambda: len(ws.sent) >= 1)
        initial = len(ws.sent)
        ws.push_server_frame({"type": "ping"})
        await _wait_for(lambda: len(ws.sent) > initial)
        await client.stop()
        return ws.sent

    sent = asyncio.run(_go())
    decoded = [json.loads(s) for s in sent]
    assert any(p.get("type") == "pong" for p in decoded)


def test_bad_json_frame_is_tolerated() -> None:
    ticks, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        client.start()
        await client.set_symbols(["AAPL"])
        await _wait_for(lambda: len(factory.sockets) >= 1)
        ws = factory.latest
        ws.push_server_frame("not-json{{{")
        # Follow it with a valid tick to prove the loop survived.
        ws.push_server_frame({
            "type": "trade",
            "data": [{"s": "AAPL", "p": 50.0, "t": 1_700_000_000_000}],
        })
        await _wait_for(lambda: len(ticks) >= 1)
        await client.stop()

    asyncio.run(_go())
    assert ticks[0][1] == 50.0


def test_reconnect_replays_subscriptions() -> None:
    _, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )
    # Patch backoff start to be tiny so the test doesn't sleep a full second.
    import packages.cockpit.web.finnhub_ws as mod
    original_backoff = mod.BACKOFF_START_S
    mod.BACKOFF_START_S = 0.01

    async def _go():
        try:
            client.start()
            await client.set_symbols(["AAPL"])
            await _wait_for(lambda: len(factory.sockets) >= 1)
            ws = factory.latest
            await _wait_for(lambda: len(ws.sent) >= 1)
            # Force a disconnect and wait for reconnect.
            ws.push_disconnect()
            await _wait_for(lambda: len(factory.sockets) >= 2, timeout=2.0)
            ws2 = factory.latest
            # New socket should have received the resubscribe.
            await _wait_for(lambda: len(ws2.sent) >= 1, timeout=2.0)
            decoded = [json.loads(s) for s in ws2.sent]
            assert any(
                p.get("type") == "subscribe" and p.get("symbol") == "AAPL"
                for p in decoded
            )
            st = client.status()
            assert st["reconnect_count"] >= 1
        finally:
            mod.BACKOFF_START_S = original_backoff
            await client.stop()

    asyncio.run(_go())


def test_stop_is_idempotent_and_closes_socket() -> None:
    _, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        client.start()
        await client.set_symbols(["AAPL"])
        await _wait_for(lambda: len(factory.sockets) >= 1)
        ws = factory.latest
        await _wait_for(lambda: len(ws.sent) >= 1)
        await client.stop()
        await client.stop()  # second call must not raise
        assert client.running is False
        assert client.connected is False

    asyncio.run(_go())


def test_status_shape_when_idle() -> None:
    _, on_tick = _collect_ticks()
    client = FinnhubWebSocketClient(api_key="TEST", on_tick=on_tick)
    st = client.status()
    assert st["enabled"] is True
    assert st["running"] is False
    assert st["connected"] is False
    assert st["subscribed"] == []
    assert st["ticks_ingested"] == 0
    assert st["max_symbols"] == DEFAULT_MAX_SYMBOLS
    assert "url" in st
    assert st["last_tick_per_symbol"] == {}


def test_zero_timestamp_in_tick_substitutes_now() -> None:
    ticks, on_tick = _collect_ticks()
    factory = FakeFactory()
    client = FinnhubWebSocketClient(
        api_key="TEST", on_tick=on_tick, connect_factory=factory
    )

    async def _go():
        client.start()
        await client.set_symbols(["AAPL"])
        await _wait_for(lambda: len(factory.sockets) >= 1)
        ws = factory.latest
        ws.push_server_frame({
            "type": "trade",
            "data": [{"s": "AAPL", "p": 100.0, "t": 0}],
        })
        await _wait_for(lambda: len(ticks) >= 1)
        await client.stop()

    asyncio.run(_go())
    # ts is a real datetime, not epoch-0.
    from datetime import datetime
    assert isinstance(ticks[0][2], datetime)
    assert ticks[0][2].year >= 2024
