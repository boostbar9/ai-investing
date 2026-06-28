"""Robinhood read-only live market-data facade (primary source + fallback).

The Seer already uses Robinhood read-only to *price simulated fills*
(``packages/execution/robinhood_paper.py``). This module extends that same
in-process, READ-ONLY MCP call path to feed the AI's
research / scoring / regime DECISIONS with a more-reliable primary source
than some free feeds (Yahoo ``quoteSummary`` was 401ing, Reddit 403ing).

Design (per the task spec):

* **Priority + fallback.** Robinhood is preferred when fresh+available. On
  any RH failure / timeout / disabled-toggle / missing token we FAIL SAFE to
  the caller's existing source (yfinance / parquet) or to a cached/stale
  value (clearly labeled). We NEVER fabricate a value and NEVER turn a
  missing source into a bearish signal -- an absent RH input simply drops
  out and the existing feed takes over.
* **Routed through the shared infra.** Every RH fetch goes through the
  process-wide :class:`~packages.data.cache.TTLCache` (dedupe + freshness +
  serve-stale-on-error) and is recorded against the
  :class:`~packages.data.health.SourceRegistry` so the Data Sources health
  page gets status/latency/freshness for free. The per-source enable/disable
  TOGGLE is honoured: a disabled RH source short-circuits straight to
  fallback.
* **Provenance everywhere.** Each result is a :class:`Provenanced` carrying
  ``source`` (``rh_quotes`` / ``yfinance`` / ``parquet`` / ``cached`` /
  ``stale`` / ``none``), ``age_s`` and a ``stale`` flag so the decision layer
  can down-weight fallback/stale inputs and the UI can show
  "price: Robinhood live, 3s ago".

STRICTLY read-only: this module only ever calls Robinhood *read* tools via
the broker. It never enables live trading, never changes the execution mode,
and never places/cancels orders.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from packages.data.cache import get_cache
from packages.data.health import get_registry, is_enabled

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source names (these are the health-registry / toggle keys the UI shows) and
# per-source TTLs. Quotes are volatile so they expire fast; fundamentals
# barely move intraday so they live an hour.
# ---------------------------------------------------------------------------
SRC_QUOTES = "rh_quotes"
SRC_BARS = "rh_bars"
SRC_FUNDAMENTALS = "rh_fundamentals"
SRC_EARNINGS = "rh_earnings"
SRC_INDEXES = "rh_indexes"
SRC_SCANS = "rh_scans"

ALL_SOURCES = (
    SRC_QUOTES,
    SRC_BARS,
    SRC_FUNDAMENTALS,
    SRC_EARNINGS,
    SRC_INDEXES,
    SRC_SCANS,
)

TTL_QUOTES_S = 15.0
TTL_BARS_S = 300.0
TTL_FUNDAMENTALS_S = 3600.0
TTL_EARNINGS_S = 1800.0
TTL_INDEXES_S = 30.0
TTL_SCANS_S = 600.0


@dataclass(frozen=True)
class Provenanced:
    """A value plus where it came from, for the decision layer + UI.

    ``ok`` is ``False`` only when neither Robinhood nor the fallback could
    produce a usable value -- the caller must then treat the input as absent
    (NEVER bearish). ``source`` is a plain label: ``rh_quotes`` etc. for live
    Robinhood, the fallback's own label otherwise, and ``stale`` when an
    expired cached RH value was served as a last resort.
    """

    value: Any
    source: str
    age_s: float = 0.0
    stale: bool = False
    ok: bool = True

    @property
    def is_rh(self) -> bool:
        return self.source in ALL_SOURCES


# A fallback is an async callable returning ``(value, label)`` or ``None``.
Fallback = Callable[[], Awaitable[tuple[Any, str] | None]]


class _RHUnavailable(RuntimeError):
    """Internal: raised inside a cache ``fetch`` when Robinhood returned no
    usable data, so :meth:`TTLCache.get_or_fetch` serves a stale value if one
    exists (else propagates and we fall back)."""


# ---------------------------------------------------------------------------
# Shared in-process broker (SHADOW, read-only). Mirrors
# robinhood_paper._ensure_read_broker so we reuse the exact same MCP path.
# An injected broker (tests) bypasses the connection gate.
# ---------------------------------------------------------------------------
_broker: Any | None = None
_broker_lock = threading.Lock()
_injected = False

# When set (by the sync→async bridge running on the persistent worker loop),
# this broker is used instead of the shared ``_broker`` for the duration of the
# bridged coroutine. It isolates the bridge's httpx client to the worker loop so
# it is never shared with -- nor bound to -- the async-accessor caller loop.
_active_broker: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "rh_live_active_broker", default=None
)


def _build_broker() -> Any | None:
    """Build a fresh SHADOW read-only broker, or ``None`` when RH isn't
    connected. Import lazily to avoid an import cycle."""
    from packages.execution import robinhood as rh

    if not rh.is_connected():
        return None
    from packages.execution.modes import ExecutionMode

    return rh.RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW,
        account_number=rh.resolve_agentic_account_number(),
    )


def set_broker_for_test(broker: Any | None) -> None:
    """Inject (or clear) the shared read-only broker. Test-only seam; an
    injected broker is treated as connected so facade tests need no tokens."""
    global _broker, _injected
    with _broker_lock:
        _broker = broker
        _injected = broker is not None


def reset_for_test() -> None:
    """Drop the cached broker + injection flag + worker loop/broker (test
    hygiene)."""
    global _broker, _injected
    with _broker_lock:
        _broker = None
        _injected = False
    _reset_worker_for_test()


def _get_broker() -> Any | None:
    """Return the read-only broker to use, or ``None`` when Robinhood is not
    connected (no token). ``None`` means callers fall back -- never an error.

    A context-local override (set by the bridge on the worker loop) wins so the
    bridge's coroutines use the worker-bound broker."""
    override = _active_broker.get()
    if override is not None:
        return override
    global _broker
    with _broker_lock:
        if _broker is not None:
            return _broker
    broker = _build_broker()
    if broker is None:
        return None
    with _broker_lock:
        if _broker is None:
            _broker = broker
        return _broker


def _rh_active(source: str) -> bool:
    """Robinhood usable for ``source``: toggle on AND a broker available."""
    if not is_enabled(source):
        return False
    return _get_broker() is not None


# ---------------------------------------------------------------------------
# Core: fetch one RH-backed value through cache + health, with fallback.
# ---------------------------------------------------------------------------
async def _serve(
    source: str,
    query: str,
    ttl_s: float,
    rh_fetch: Callable[[Any], Awaitable[Any]],
    *,
    is_empty: Callable[[Any], bool],
    fallback: Fallback | None,
    empty_is_success: bool = False,
) -> Provenanced:
    """Fetch ``query`` from Robinhood (cached), else fall back.

    ``rh_fetch(broker)`` performs the actual read-tool call and returns the
    parsed value; ``is_empty(value)`` decides whether RH produced nothing
    usable (so we fall back rather than caching an empty success).

    ``empty_is_success`` distinguishes a *successful-but-empty* RH response
    (e.g. ``rh_bars`` on a closed market: the call worked, there are simply no
    bars) from a real call error. When set, an empty response records a
    SUCCESS on the health registry (the source stays ``ok``) and we fall back
    quietly -- it is never counted as a failure. A genuinely *raised* RH error
    still records a failure either way.
    """
    registry = get_registry()
    cache = get_cache()

    if _rh_active(source):
        broker = _get_broker()
        empty_seen = {"hit": False}

        async def _fetch() -> Any:
            value = await rh_fetch(broker)
            if is_empty(value):
                # Don't cache an empty RH response as a success; raise so the
                # cache serves a prior good value if it has one, else we drop
                # to the fallback below.
                empty_seen["hit"] = True
                raise _RHUnavailable(source)
            return value

        t0 = time.perf_counter()
        registry.record_attempt(source)
        try:
            res = await cache.get_or_fetch(
                source, query, _fetch, ttl_s=ttl_s, serve_stale_on_error=True
            )
        except Exception as exc:  # RH failed AND no cached value to serve
            if empty_is_success and empty_seen["hit"]:
                # Success-but-empty (e.g. market closed): NOT a failure. Record
                # a healthy (empty) success so the pill stays ``ok``/ok-empty,
                # then fall back quietly to yfinance/parquet below.
                registry.record_success(
                    source,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    stale_after_s=ttl_s,
                )
            else:
                registry.record_failure(source, exc, stale_after_s=ttl_s)
        else:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            registry.record_success(
                source,
                latency_ms=latency_ms,
                from_cache=res.hit,
                stale=res.stale,
                stale_after_s=ttl_s,
            )
            label = "stale" if res.stale else source
            return Provenanced(
                value=res.value,
                source=label,
                age_s=res.age_s,
                stale=res.stale,
                ok=True,
            )

    # ---- fallback (RH disabled / disconnected / failed with no cache) ----
    if fallback is not None:
        try:
            fb = await fallback()
        except Exception as exc:  # pragma: no cover — fallback is caller code
            log.debug("rh_live fallback for %s raised: %s", source, exc)
            fb = None
        if fb is not None:
            value, label = fb
            return Provenanced(value=value, source=label, age_s=0.0, ok=True)

    # Nothing available anywhere: absent input (NEVER bearish, NEVER faked).
    return Provenanced(value=None, source="none", age_s=0.0, ok=False)


# ---------------------------------------------------------------------------
# Public market-data accessors (each RH-primary with optional fallback).
# ---------------------------------------------------------------------------
async def get_quote(
    symbol: str, *, fallback: Fallback | None = None
) -> Provenanced:
    """Live quote dict (``bid/ask/last/mid``) for ``symbol``, RH-primary."""

    async def _fetch(broker: Any) -> Any:
        return await broker.equity_quote(symbol)

    return await _serve(
        SRC_QUOTES,
        symbol.strip().upper(),
        TTL_QUOTES_S,
        _fetch,
        is_empty=lambda v: not v,
        fallback=fallback,
    )


async def get_bars(
    symbol: str,
    *,
    start_time: str | None = None,
    interval: str | None = None,
    span: str | None = None,
    fallback: Fallback | None = None,
) -> Provenanced:
    """Historical/intraday bars for ``symbol`` via RH historicals, RH-primary.
    ``value`` is a list of raw bar-row dicts on success."""
    key = f"{symbol.strip().upper()}:{span or ''}:{interval or ''}:{start_time or ''}"

    async def _fetch(broker: Any) -> Any:
        return await broker.equity_historicals(
            symbol, start_time=start_time, interval=interval, span=span
        )

    return await _serve(
        SRC_BARS,
        key,
        TTL_BARS_S,
        _fetch,
        is_empty=lambda v: not v,
        fallback=fallback,
        # Empty bars (e.g. market closed -- today is Sunday) are a SUCCESSFUL
        # but empty response: fall back quietly, never a recorded health
        # failure. Only a raised RH error marks rh_bars down.
        empty_is_success=True,
    )


async def get_fundamentals(
    symbol: str, *, fallback: Fallback | None = None
) -> Provenanced:
    """Company fundamentals dict for ``symbol`` via RH, RH-primary."""

    async def _fetch(broker: Any) -> Any:
        return await broker.equity_fundamentals(symbol)

    return await _serve(
        SRC_FUNDAMENTALS,
        symbol.strip().upper(),
        TTL_FUNDAMENTALS_S,
        _fetch,
        is_empty=lambda v: not v,
        fallback=fallback,
    )


async def get_earnings(
    symbol: str | None = None, *, fallback: Fallback | None = None
) -> Provenanced:
    """Upcoming earnings rows (optionally filtered by ``symbol``) via RH.
    ``value`` is a list of earnings dicts on success."""

    async def _fetch(broker: Any) -> Any:
        return await broker.earnings_calendar(symbol)

    return await _serve(
        SRC_EARNINGS,
        (symbol or "*").strip().upper(),
        TTL_EARNINGS_S,
        _fetch,
        is_empty=lambda v: not v,
        fallback=fallback,
    )


# ---------------------------------------------------------------------------
# Regime inputs: live index levels (VIX) + daily closes (SPY etc.).
# These are RH-primary with NO fabricated fallback here -- the regime module
# already has its own yfinance default providers we wrap (see below).
# ---------------------------------------------------------------------------
def _bar_close(row: dict[str, Any]) -> float | None:
    for k in ("close_price", "close", "adj_close", "last_close_price", "price"):
        v = row.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return None


def _historicals_start_time(days: int) -> str:
    """ISO-8601 UTC ``start_time`` covering ``days`` of look-back (plus a small
    buffer for weekends/holidays). ``get_equity_historicals`` requires both
    ``symbols`` and ``start_time``; omitting it makes the tool reject the call.
    """
    from datetime import UTC, datetime, timedelta

    start = datetime.now(UTC) - timedelta(days=max(days, 1) + 5)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ")


async def rh_daily_closes(symbol: str, *, days: int = 90) -> list[float] | None:
    """Daily closes for ``symbol`` from RH historicals, or ``None`` to fall
    back. Returns oldest-first floats; ``None`` on any failure / disabled."""
    res = await get_bars(
        symbol,
        start_time=_historicals_start_time(days),
        interval="day",
        span="year",
    )
    if not res.ok or not isinstance(res.value, list):
        return None
    closes = [c for c in (_bar_close(r) for r in res.value if isinstance(r, dict)) if c]
    if not closes:
        return None
    return closes[-max(days, 1):]


_VIX_SYMBOLS = ("VIX", "^VIX", "INDEXCBOE:VIX", "CBOE VOLATILITY INDEX")


def _level_from_quote(row: dict[str, Any]) -> float | None:
    for k in ("last_trade_price", "last_price", "last", "price", "value", "level"):
        v = row.get(k)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return None


def _resolve_index_id(idx: list[Any], tags: tuple[str, ...]) -> str | None:
    """First instrument id in ``idx`` whose symbol/name matches any ``tag``."""
    for row in idx:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or row.get("name") or "").upper()
        if any(tag in sym for tag in tags):
            iid = (
                row.get("id")
                or row.get("instrument_id")
                or row.get("instrumentId")
            )
            if iid:
                return str(iid)
    return None


async def rh_vix_level() -> float | None:
    """Current VIX level via RH ``get_indexes`` + ``get_index_quotes``, or
    ``None`` to fall back. Resolves the VIX instrument id from the index
    catalog, then reads its live level.

    Health semantics: ``rh_indexes`` is OK whenever ``get_indexes`` is
    reachable (returns rows) -- that is the live-index API health signal. If
    Robinhood's catalog simply doesn't list VIX (or its quote is unavailable),
    this is a clean degrade: we record a SUCCESS and return ``None`` so the
    regime detector falls back to yfinance ^VIX (fallback intact). Only an
    empty/raised ``get_indexes`` records a failure (true RH outage)."""
    if not _rh_active(SRC_INDEXES):
        return None
    broker = _get_broker()
    if broker is None:
        return None
    registry = get_registry()
    registry.record_attempt(SRC_INDEXES)
    t0 = time.perf_counter()
    try:
        idx = await broker.indexes()
        if not idx:
            # get_indexes returned nothing -> RH index API unavailable.
            raise _RHUnavailable("indexes-empty")
        level: float | None = None
        vix_id = _resolve_index_id(idx, ("VIX", "VOLATILITY"))
        if vix_id:
            quotes = await broker.index_quotes([vix_id])
            for q in quotes:
                if isinstance(q, dict):
                    level = _level_from_quote(q)
                    if level:
                        break
    except Exception as exc:
        registry.record_failure(SRC_INDEXES, exc, stale_after_s=TTL_INDEXES_S)
        return None
    # get_indexes was reachable -> rh_indexes is healthy regardless of whether
    # VIX was found. A missing VIX just falls back to yfinance ^VIX.
    registry.record_success(
        SRC_INDEXES,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        stale_after_s=TTL_INDEXES_S,
    )
    return level


# ---------------------------------------------------------------------------
# Candidate sourcing: RH saved scans -> additional tickers (ADDITIVE).
# ---------------------------------------------------------------------------
def _symbol_of(row: Any) -> str | None:
    if isinstance(row, str):
        s = row.strip().upper()
        return s or None
    if isinstance(row, dict):
        for k in ("symbol", "ticker", "instrument_symbol"):
            v = row.get(k)
            if v:
                s = str(v).strip().upper()
                if s:
                    return s
    return None


async def get_scan_candidates(*, max_symbols: int = 25) -> Provenanced:
    """Tickers contributed by the user's saved Robinhood screeners.

    ADDITIVE only: the caller unions these into the existing universe and
    must NEVER let an empty/failed scan shrink it. Returns ``ok=True`` with an
    empty list when the user has no saved scans (the common case today) so the
    caller degrades cleanly. We do NOT auto-create scans.
    """
    if not _rh_active(SRC_SCANS):
        return Provenanced(value=[], source="none", ok=False)
    broker = _get_broker()
    if broker is None:
        return Provenanced(value=[], source="none", ok=False)
    registry = get_registry()
    registry.record_attempt(SRC_SCANS)
    t0 = time.perf_counter()
    try:
        scans = await broker.scans()
        symbols: list[str] = []
        seen: set[str] = set()
        for scan in scans:
            if not isinstance(scan, dict):
                continue
            sid = scan.get("id") or scan.get("scan_id") or scan.get("slug")
            if not sid:
                continue
            for row in await broker.run_scan(str(sid)):
                sym = _symbol_of(row)
                if sym and sym not in seen:
                    seen.add(sym)
                    symbols.append(sym)
                    if len(symbols) >= max_symbols:
                        break
            if len(symbols) >= max_symbols:
                break
    except Exception as exc:
        registry.record_failure(SRC_SCANS, exc, stale_after_s=TTL_SCANS_S)
        return Provenanced(value=[], source="none", ok=False)
    registry.record_success(
        SRC_SCANS,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        stale_after_s=TTL_SCANS_S,
    )
    # Empty (no saved scans) is a clean, successful degrade -- not bearish.
    return Provenanced(value=symbols, source=SRC_SCANS, ok=True)


# ---------------------------------------------------------------------------
# Sync bridges for the (synchronous) regime providers, which run inside the
# autonomy tick's event loop. We cannot use ``asyncio.run`` per call: it
# creates AND CLOSES a fresh event loop each time, and the cached
# ``httpx.AsyncClient`` binds to the first loop -- so the second call hits
# "Event loop is closed". Instead we run every bridged coroutine on a SINGLE
# PERSISTENT worker loop+thread (created once, never closed) and submit work
# with ``run_coroutine_threadsafe``. The bridge also uses its OWN dedicated
# broker (context-local) so its httpx client is created on, and only ever used
# on, that one persistent loop -- never shared with the async-accessor loop.
# ANY problem -> return None -> caller falls back to the existing yfinance
# default provider (so the protected yfinance-path behaviour is unchanged when
# RH is off / in tests).
# ---------------------------------------------------------------------------
_worker_lock = threading.Lock()
_worker_loop: asyncio.AbstractEventLoop | None = None
_worker_thread: threading.Thread | None = None
_worker_broker: Any | None = None


def _ensure_worker_loop() -> asyncio.AbstractEventLoop:
    """Return the long-lived worker event loop, starting it (once) if needed.
    The loop runs forever on a daemon thread and is never closed between
    calls, so an httpx client bound to it stays valid for the process."""
    global _worker_loop, _worker_thread
    with _worker_lock:
        if (
            _worker_loop is not None
            and not _worker_loop.is_closed()
            and _worker_thread is not None
            and _worker_thread.is_alive()
        ):
            return _worker_loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="rh-live-worker", daemon=True
        )
        thread.start()
        _worker_loop = loop
        _worker_thread = thread
        return loop


def _worker_broker_for() -> Any | None:
    """Broker used ONLY on the persistent worker loop by the sync bridges.

    Tests inject a single fake (``_injected``) that is used everywhere -- no
    real network/loops involved. In production the bridge gets its own
    dedicated broker instance, lazily built and reused across calls, so its
    httpx client binds to the worker loop and is never shared with the
    async-accessor ``_broker``."""
    with _broker_lock:
        if _injected:
            return _broker
    global _worker_broker
    with _worker_lock:
        if _worker_broker is not None:
            return _worker_broker
    broker = _build_broker()
    if broker is None:
        return None
    with _worker_lock:
        if _worker_broker is None:
            _worker_broker = broker
        return _worker_broker


def _reset_worker_for_test() -> None:
    """Stop the worker loop + drop the dedicated worker broker (test hygiene).
    Called from :func:`reset_for_test`."""
    global _worker_loop, _worker_thread, _worker_broker
    with _worker_lock:
        loop, _worker_loop = _worker_loop, None
        _worker_thread = None
        _worker_broker = None
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)


def _run_blocking(coro_factory: Callable[[], Awaitable[Any]], timeout_s: float) -> Any:
    """Run ``coro_factory()`` on the persistent worker loop with the dedicated
    worker broker bound, returning its result or ``None`` on timeout/error."""

    async def _wrapped() -> Any:
        token = _active_broker.set(_worker_broker_for())
        try:
            return await coro_factory()
        finally:
            _active_broker.reset(token)

    loop = _ensure_worker_loop()
    try:
        fut = asyncio.run_coroutine_threadsafe(_wrapped(), loop)
    except RuntimeError:  # pragma: no cover — loop went away mid-call
        return None
    try:
        return fut.result(timeout_s)
    except Exception:  # timeout / cancelled / RH error -> fall back
        fut.cancel()
        return None


def regime_price_provider(symbol: str, *, days: int = 90) -> list[float] | None:
    """RH-first daily closes for the regime detector; ``None`` falls back to
    the module's yfinance default. Safe to call from inside an event loop."""
    if not _rh_active(SRC_BARS):
        return None
    return _run_blocking(lambda: rh_daily_closes(symbol, days=days), timeout_s=8.0)


def regime_vix_provider() -> float | None:
    """RH-first VIX level for the regime detector; ``None`` falls back to the
    module's yfinance default."""
    if not _rh_active(SRC_INDEXES):
        return None
    return _run_blocking(rh_vix_level, timeout_s=8.0)
