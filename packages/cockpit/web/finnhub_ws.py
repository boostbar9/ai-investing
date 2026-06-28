"""Phase 25.4 — Finnhub WebSocket live tick stream.

Connects to ``wss://ws.finnhub.io`` and pumps real-time trade ticks
into :class:`packages.cockpit.web.live_quotes.LiveQuoteCache`. The
free tier caps the connection at 50 simultaneously-subscribed symbols
so we maintain an LRU subscription set sized to that limit.

Design contract
---------------

* **Connection lifecycle** — ``start()`` spawns a background coroutine
  that opens the WS, subscribes to the current symbol set, and reads
  ticks. On disconnect the loop sleeps with exponential backoff and
  reconnects automatically. ``stop()`` cancels the task and closes
  the socket cleanly. The whole thing is idempotent.

* **Subscription diffing** — callers push the *desired* symbol set
  via ``set_symbols(symbols)``. The client computes the diff against
  the currently-subscribed set and only sends the necessary
  ``subscribe`` / ``unsubscribe`` frames. Symbols above the
  ``max_symbols`` cap are dropped (LRU on insertion order).

* **Tick handling** — Finnhub sends ``{"type": "trade", "data":
  [{"s": "AAPL", "p": 199.21, "t": 1700000000123, "v": 100}, ...]}``
  frames. We extract the last price per symbol per frame and call
  ``cache.ingest_ws_tick()`` for each.

* **Failure handling** — every exception inside the read loop is
  caught, counted, and triggers a reconnect. The REST cache from
  Phase 25.3 keeps the system working while the WS is down.

* **Telemetry** — ``status()`` returns connection state, subscribed
  symbols, messages received, ticks ingested, reconnect count, and
  the timestamp of the last tick per symbol. Surfaced under
  ``/api/data-feed -> websocket`` so the dashboard can show a "ws"
  badge.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

FINNHUB_WS_URL = "wss://ws.finnhub.io"

# Free-tier cap: 50 symbols per connection. Override via env for paid
# tier or for tests that want to exercise the cap with smaller numbers.
DEFAULT_MAX_SYMBOLS = int(os.getenv("FINNHUB_WS_MAX_SYMBOLS", "50"))

# Reconnect backoff: start short, cap at 60s. We never give up — the
# REST cache covers the gap.
BACKOFF_START_S = 1.0
BACKOFF_MAX_S = 60.0

# Cap the opening handshake. The live logs showed "timed out during opening
# handshake" hangs; bounding the connect attempt lets the backoff/reconnect
# loop (and the REST polling fallback) take over quickly instead of stalling.
CONNECT_TIMEOUT_S = float(os.getenv("FINNHUB_WS_CONNECT_TIMEOUT_S", "10.0"))


@dataclass
class WSStats:
    """Counters surfaced via ``status()`` for the data-feed endpoint."""

    connected_at: datetime | None = None
    last_tick_at: datetime | None = None
    last_tick_symbol: str | None = None
    messages_received: int = 0
    ticks_ingested: int = 0
    reconnect_count: int = 0
    last_error: str | None = None
    # Per-symbol last-tick timestamp for the status endpoint.
    last_tick_per_symbol: dict[str, datetime] = field(default_factory=dict)


# Type aliases.
TickHandler = Callable[[str, float, datetime], None]
WSConnectFactory = Callable[..., Awaitable[Any]]


class FinnhubWebSocketClient:
    """Live-tick WS client with auto-reconnect + LRU symbol cap.

    Parameters
    ----------
    api_key
        Finnhub API key. When empty the client refuses to start so we
        don't hammer the WS endpoint with a guaranteed-401 handshake.
    on_tick
        Callback invoked once per tick with ``(symbol, price, ts)``.
        In production this points at ``LiveQuoteCache.ingest_ws_tick``.
    max_symbols
        Hard subscription cap. Defaults to ``FINNHUB_WS_MAX_SYMBOLS``
        env var or 50.
    connect_factory
        Async callable that returns an open websocket connection. Tests
        inject a fake; production uses :func:`websockets.connect`.
    """

    def __init__(
        self,
        api_key: str,
        on_tick: TickHandler,
        *,
        max_symbols: int | None = None,
        connect_factory: WSConnectFactory | None = None,
        url: str = FINNHUB_WS_URL,
    ) -> None:
        self._api_key = api_key
        self._on_tick = on_tick
        self._max_symbols = max_symbols or DEFAULT_MAX_SYMBOLS
        self._url = url
        self._connect_factory = connect_factory or _default_connect

        # OrderedDict gives us LRU semantics: oldest insertion is the
        # first candidate for eviction when we hit the symbol cap.
        self._desired: OrderedDict[str, None] = OrderedDict()
        self._subscribed: set[str] = set()

        self._ws: Any | None = None
        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._stats = WSStats()
        self._connected = False
        # Last connection error we logged at WARNING, so a flapping/blocked
        # socket doesn't spam an identical warning on every retry.
        self._last_logged_error: str | None = None

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> bool:
        """Spawn the background read/reconnect loop.

        Returns ``False`` if no API key is configured (caller should
        log + skip — the REST path still works) or the loop is already
        running. Otherwise schedules the task and returns ``True``.
        """
        if not self._api_key:
            log.info("finnhub_ws: no API key, WS stream disabled")
            return False
        if self.running:
            return False
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run_forever(), name="finnhub_ws_client"
        )
        return True

    async def stop(self) -> None:
        """Cancel the background loop and close the socket."""
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        await self._close_ws()
        self._connected = False

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    async def set_symbols(self, symbols: list[str]) -> dict[str, Any]:
        """Replace the desired subscription set.

        Diffs against ``_subscribed`` and sends ``subscribe`` /
        ``unsubscribe`` frames for the delta. Symbols above the cap
        are dropped from the *tail* of the desired set (oldest-inserted
        first) so the most recent caller intent wins.
        """
        normalized = [str(s).upper().strip() for s in symbols if s]
        # Dedupe while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for s in normalized:
            if s not in seen:
                seen.add(s)
                ordered.append(s)

        # Enforce free-tier cap.
        if len(ordered) > self._max_symbols:
            ordered = ordered[: self._max_symbols]

        new_desired = OrderedDict.fromkeys(ordered)
        async with self._lock:
            self._desired = new_desired
            target = set(new_desired.keys())
            to_subscribe = target - self._subscribed
            to_unsubscribe = self._subscribed - target

            if self._ws is not None and self._connected:
                for sym in sorted(to_subscribe):
                    await self._send({"type": "subscribe", "symbol": sym})
                for sym in sorted(to_unsubscribe):
                    await self._send({"type": "unsubscribe", "symbol": sym})

            # Track regardless of socket state — when the connection
            # comes up we replay this set in ``_resubscribe_all``.
            self._subscribed = target
            return {
                "desired": sorted(target),
                "subscribed_now": sorted(to_subscribe),
                "unsubscribed_now": sorted(to_unsubscribe),
                "capped": len(symbols) > self._max_symbols,
            }

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _run_forever(self) -> None:
        """Connect → read → reconnect forever, until ``stop()``.

        Backoff doubles on each failure, capped at ``BACKOFF_MAX_S``.
        Successful connections reset the backoff.
        """
        backoff = BACKOFF_START_S
        while not self._stop_event.is_set():
            try:
                async with self._open() as ws:
                    self._ws = ws
                    self._connected = True
                    self._stats.connected_at = datetime.now(UTC)
                    backoff = BACKOFF_START_S
                    self._last_logged_error = None
                    log.info("finnhub_ws: connected to %s", self._url)
                    await self._resubscribe_all()
                    await self._read_loop(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._stats.reconnect_count += 1
                sig = f"{type(exc).__name__}: {exc}"
                self._stats.last_error = sig[:240]
                # Only warn when the failure signature changes; repeats of the
                # same handshake timeout drop to debug so we don't flood logs.
                if sig != self._last_logged_error:
                    log.warning(
                        "finnhub_ws: connection failed (%s); retrying with "
                        "backoff, REST polling covers the gap", exc,
                    )
                    self._last_logged_error = sig
                else:
                    log.debug("finnhub_ws: still failing to connect: %s", exc)
            finally:
                self._connected = False
                self._ws = None

            if self._stop_event.is_set():
                break
            # Sleep with backoff but wake immediately on stop().
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=backoff
                )
                break
            except TimeoutError:
                pass
            backoff = min(backoff * 2, BACKOFF_MAX_S)

    @contextlib.asynccontextmanager
    async def _open(self):
        url = f"{self._url}?token={self._api_key}"
        # Bound the handshake so a wedged connect can't stall the loop.
        ws = await asyncio.wait_for(
            self._connect_factory(url), timeout=CONNECT_TIMEOUT_S
        )
        try:
            yield ws
        finally:
            with contextlib.suppress(Exception):
                close = getattr(ws, "close", None)
                if close is not None:
                    result = close()
                    if asyncio.iscoroutine(result):
                        await result

    async def _resubscribe_all(self) -> None:
        for sym in sorted(self._subscribed):
            await self._send({"type": "subscribe", "symbol": sym})

    async def _read_loop(self, ws: Any) -> None:
        """Consume frames until the socket closes or stop() fires."""
        while not self._stop_event.is_set():
            recv = ws.recv()
            raw = await recv if asyncio.iscoroutine(recv) else recv
            if raw is None:
                # Some fakes signal EOF with None.
                return
            self._stats.messages_received += 1
            await self._handle_frame(raw)

    async def _handle_frame(self, raw: str | bytes) -> None:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            msg = json.loads(raw)
        except Exception as exc:
            log.debug("finnhub_ws: bad frame %r: %s", raw[:200], exc)
            return

        msg_type = msg.get("type")
        if msg_type == "trade":
            # Coalesce: only emit the latest price per symbol per frame.
            latest: dict[str, dict[str, Any]] = {}
            for tick in msg.get("data") or []:
                sym = tick.get("s")
                if not sym:
                    continue
                latest[str(sym).upper()] = tick
            for sym, tick in latest.items():
                price = tick.get("p")
                if price is None:
                    continue
                ts_ms = tick.get("t")
                ts = (
                    datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
                    if isinstance(ts_ms, (int, float)) and ts_ms > 0
                    else datetime.now(UTC)
                )
                try:
                    self._on_tick(sym, float(price), ts)
                    self._stats.ticks_ingested += 1
                    self._stats.last_tick_at = ts
                    self._stats.last_tick_symbol = sym
                    self._stats.last_tick_per_symbol[sym] = ts
                except Exception as exc:  # pragma: no cover — handler bug
                    log.debug("finnhub_ws: on_tick failed for %s: %s", sym, exc)
        elif msg_type == "ping":
            # Finnhub keepalive — respond with pong if the lib doesn't.
            with contextlib.suppress(Exception):
                await self._send({"type": "pong"})
        elif msg_type == "error":
            # E.g. unsupported symbol — log + continue.
            log.warning("finnhub_ws: server error: %s", msg.get("msg"))
            self._stats.last_error = f"server: {msg.get('msg')}"[:240]

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            return
        data = json.dumps(payload)
        send = getattr(self._ws, "send", None)
        if send is None:
            return
        result = send(data)
        if asyncio.iscoroutine(result):
            await result

    async def _close_ws(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is None:
            return
        with contextlib.suppress(Exception):
            close = getattr(ws, "close", None)
            if close is not None:
                r = close()
                if asyncio.iscoroutine(r):
                    await r

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        last_ticks: dict[str, str] = {
            sym: ts.isoformat(timespec="seconds")
            for sym, ts in sorted(self._stats.last_tick_per_symbol.items())
        }
        return {
            "enabled": self.has_key,
            "running": self.running,
            "connected": self._connected,
            "url": self._url,
            "max_symbols": self._max_symbols,
            "subscribed": sorted(self._subscribed),
            "subscribed_count": len(self._subscribed),
            "connected_at": (
                self._stats.connected_at.isoformat(timespec="seconds")
                if self._stats.connected_at
                else None
            ),
            "messages_received": self._stats.messages_received,
            "ticks_ingested": self._stats.ticks_ingested,
            "reconnect_count": self._stats.reconnect_count,
            "last_error": self._stats.last_error,
            "last_tick_at": (
                self._stats.last_tick_at.isoformat(timespec="seconds")
                if self._stats.last_tick_at
                else None
            ),
            "last_tick_symbol": self._stats.last_tick_symbol,
            "last_tick_per_symbol": last_ticks,
        }


# ---------------------------------------------------------------------------
# Default WebSocket connect (production path)
# ---------------------------------------------------------------------------


async def _default_connect(url: str) -> Any:  # pragma: no cover — network
    """Production WS connect using the ``websockets`` library."""
    import websockets

    return await websockets.connect(
        url,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
        max_size=2 ** 20,  # 1MB frames are plenty for trade ticks
    )


# ---------------------------------------------------------------------------
# Module-level singleton consumed by server.py
# ---------------------------------------------------------------------------


_default_client: FinnhubWebSocketClient | None = None


def get_default_client() -> FinnhubWebSocketClient | None:
    """Return the process-wide WS client (or ``None`` when not built)."""
    return _default_client


def build_default_client(on_tick: TickHandler) -> FinnhubWebSocketClient | None:
    """Construct the WS client when ``FINNHUB_API_KEY`` is set.

    Returns ``None`` when the key is missing so callers can no-op
    cleanly. The cache's REST path still works without a WS connection.
    """
    global _default_client
    api_key = os.getenv("FINNHUB_API_KEY", "")
    if not api_key:
        return None
    _default_client = FinnhubWebSocketClient(api_key=api_key, on_tick=on_tick)
    return _default_client


def reset_default_client_for_tests() -> None:
    global _default_client
    _default_client = None
