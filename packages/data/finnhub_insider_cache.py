"""Phase 31: persistent disk cache for Finnhub insider-transaction signals.

Why this exists: the in-memory cache in :mod:`packages.data.finnhub_insider`
has two production issues that the live log surfaced as a burst of HTTP
429s during the morning sweep:

  1. **Lost on restart.** Every time the bot reboots, the cache is cold
     and the next research sweep re-fetches every symbol within a few
     hundred milliseconds. With ~9 ETFs in the universe and Finnhub's
     ~10 req/s sub-limit on ``/stock/insider-transactions``, this trips
     429s on the second half of the burst.

  2. **Empty results aren't cached.** Symbols like SPY, XLE, XLU
     legitimately have *no* insider activity (they're ETFs, not
     operating companies). The current code only caches payloads with
     at least one transaction, so these symbols are re-fetched on
     every single sweep — forever — and they're the most common 429
     offenders in the live log.

This module is a thin **JSON-on-disk** layer that the
:class:`FinnhubInsiderClient` can consult before hitting the network.
It is intentionally simple:

  * One file per symbol at ``data/cache/finnhub_insider/<SYM>.json``.
  * Atomic write via tmp + rename so a crash mid-write doesn't corrupt
    the cache.
  * Two TTLs: long (24h default) for symbols with non-zero activity,
    short (6h default) for empty results. Both ETF-friendly: an ETF's
    insider state changes maybe quarterly; 24h is a no-op floor.

This is a *complement* to the in-memory cache, not a replacement. The
in-memory cache still absorbs intra-sweep hits; the disk cache absorbs
cross-process and post-restart hits.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.data.finnhub_insider import InsiderSignal

log = logging.getLogger(__name__)


# --- Tunables ----------------------------------------------------------------

_FALLBACK_CACHE_DIR = Path("data") / "cache" / "finnhub_insider"


def _resolved_default_dir() -> Path:
    """Resolve the default cache dir at call time (not import time).

    Reading the env var lazily means tests that ``monkeypatch.setenv``
    *after* import still get isolation. Production callers that never
    set the env get the on-disk default.
    """
    override = os.getenv("FINNHUB_INSIDER_CACHE_DIR")
    return Path(override) if override else _FALLBACK_CACHE_DIR


# Kept for back-compat with imports that reference the constant. Note
# this is captured at *import* time — prefer ``_resolved_default_dir()``
# inside this module so env overrides flow through.
DEFAULT_CACHE_DIR = _resolved_default_dir()
"""Default cache directory at import time. Tests should override per-call."""

# Long TTL: applies when the signal carries at least one transaction.
# Insider filings (Form 4) post on a T+2 SEC deadline and don't change
# retroactively, so 24h is conservative even during earnings season.
DEFAULT_TTL_S = 24 * 60 * 60

# Short TTL: applies when the signal is empty (no insider activity).
# Empty payloads are common for ETFs and big-index proxies. We re-check
# every 6 hours so genuinely-new activity surfaces within a single
# trading day, but we don't burn 429 quota re-fetching SPY every sweep.
DEFAULT_EMPTY_TTL_S = 6 * 60 * 60

CACHE_VERSION = 1
"""On-disk format version. Bump if the JSON shape changes — older files
will be ignored (treated as miss) so we don't deserialize the wrong
schema into a NamedTuple."""


# --- Serialization helpers ---------------------------------------------------


def _signal_to_dict(signal: InsiderSignal) -> dict[str, Any]:
    """Serialize an :class:`InsiderSignal` for JSON storage.

    ``fresh_at`` is a naive datetime in the dataclass — we encode it as
    an ISO string and round-trip it on the way back. ``top_buyers`` is
    a tuple in the dataclass; JSON has no tuple so we list-ify here and
    re-tuple on load.
    """
    raw = asdict(signal)
    raw["fresh_at"] = signal.fresh_at.isoformat()
    raw["top_buyers"] = list(signal.top_buyers)
    return raw


def _signal_from_dict(payload: dict[str, Any]) -> InsiderSignal:
    """Deserialize a cached row back to an :class:`InsiderSignal`.

    Tolerates missing optional fields by falling back to dataclass
    defaults — this keeps backward compat if we ever add fields to
    InsiderSignal.
    """
    fa_raw = payload.get("fresh_at")
    if isinstance(fa_raw, str):
        # datetime.fromisoformat handles both naive ('2026-06-01T22:36:26')
        # and aware ('...+00:00') strings.
        fresh_at = datetime.fromisoformat(fa_raw)
        # Strip tz so we match the in-memory dataclass invariant.
        if fresh_at.tzinfo is not None:
            fresh_at = fresh_at.replace(tzinfo=None)
    else:
        fresh_at = datetime.now(UTC).replace(tzinfo=None)
    top_buyers_raw = payload.get("top_buyers") or []
    return InsiderSignal(
        symbol=str(payload.get("symbol", "")).upper(),
        score=float(payload.get("score", 0.0)),
        confidence=float(payload.get("confidence", 0.0)),
        label=str(payload.get("label", "neutral")),
        buy_count=int(payload.get("buy_count", 0)),
        sell_count=int(payload.get("sell_count", 0)),
        unique_buyers=int(payload.get("unique_buyers", 0)),
        net_shares=float(payload.get("net_shares", 0.0)),
        net_notional_usd=float(payload.get("net_notional_usd", 0.0)),
        cluster_buy=bool(payload.get("cluster_buy", False)),
        cluster_score=float(payload.get("cluster_score", 0.0)),
        fresh_at=fresh_at,
        top_buyers=tuple(str(x) for x in top_buyers_raw if x),
    )


# --- Cache I/O ---------------------------------------------------------------


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    """Map a symbol to its on-disk cache file."""
    return cache_dir / f"{symbol.upper()}.json"


def load_cached_signal(
    symbol: str,
    *,
    cache_dir: Path | None = None,
    ttl_s: float = DEFAULT_TTL_S,
    empty_ttl_s: float = DEFAULT_EMPTY_TTL_S,
    now_unix: float | None = None,
) -> InsiderSignal | None:
    """Read a cached signal if present and unexpired.

    Returns ``None`` on any of: missing file, unreadable file, version
    mismatch, expired TTL. Caller treats a ``None`` as a cache miss
    and proceeds to the network path.
    """
    cd = cache_dir if cache_dir is not None else _resolved_default_dir()
    path = _cache_path(cd, symbol)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("insider cache: unreadable %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("_version") or 0) != CACHE_VERSION:
        return None
    written_at = float(payload.get("_written_at") or 0.0)
    if written_at <= 0:
        return None
    signal_payload = payload.get("signal")
    if not isinstance(signal_payload, dict):
        return None

    # Empty payloads expire faster than non-empty ones.
    is_empty = (
        int(signal_payload.get("buy_count", 0))
        + int(signal_payload.get("sell_count", 0))
        == 0
    )
    effective_ttl = empty_ttl_s if is_empty else ttl_s
    now = now_unix if now_unix is not None else time.time()
    if now - written_at > effective_ttl:
        return None
    try:
        return _signal_from_dict(signal_payload)
    except (TypeError, ValueError, KeyError) as exc:
        log.warning("insider cache: %s deserialize failed: %s", symbol, exc)
        return None


def save_cached_signal(
    signal: InsiderSignal,
    *,
    cache_dir: Path | None = None,
    now_unix: float | None = None,
) -> None:
    """Atomically write a signal to disk for future cross-process reads.

    Writes to a temp file in the same directory and ``os.replace``s
    into place so a crash mid-write can never leave a half-written
    JSON file. Failures are logged at debug level and swallowed — the
    cache is best-effort, never required for correctness.
    """
    cd = cache_dir if cache_dir is not None else _resolved_default_dir()
    try:
        cd.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.debug("insider cache: mkdir %s failed: %s", cd, exc)
        return
    path = _cache_path(cd, signal.symbol)
    payload = {
        "_version": CACHE_VERSION,
        "_written_at": now_unix if now_unix is not None else time.time(),
        "signal": _signal_to_dict(signal),
    }
    try:
        # Write to a tmp file in the same dir, then atomically replace.
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{signal.symbol.upper()}.", suffix=".json.tmp", dir=str(cd)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
            os.replace(tmp_name, path)
        except Exception:
            # Best-effort cleanup of the tmp file on any failure.
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
    except OSError as exc:
        log.debug("insider cache: write %s failed: %s", path, exc)


def purge_expired(
    *,
    cache_dir: Path | None = None,
    ttl_s: float = DEFAULT_TTL_S,
    empty_ttl_s: float = DEFAULT_EMPTY_TTL_S,
    now_unix: float | None = None,
) -> int:
    """Delete every cache file whose timestamp + TTL is in the past.

    Returns the count of files removed. Safe to call from a janitor cron
    — never raises on permission errors, just logs and continues.
    """
    cd = cache_dir if cache_dir is not None else _resolved_default_dir()
    if not cd.exists():
        return 0
    now = now_unix if now_unix is not None else time.time()
    removed = 0
    for path in cd.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if int(payload.get("_version") or 0) != CACHE_VERSION:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
            continue
        written_at = float(payload.get("_written_at") or 0.0)
        signal_payload = payload.get("signal") or {}
        is_empty = (
            int(signal_payload.get("buy_count", 0))
            + int(signal_payload.get("sell_count", 0))
            == 0
        )
        effective_ttl = empty_ttl_s if is_empty else ttl_s
        if now - written_at > effective_ttl:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def cache_stats(*, cache_dir: Path | None = None) -> dict[str, int]:
    """Diagnostics for the cockpit / health endpoint."""
    cd = cache_dir if cache_dir is not None else _resolved_default_dir()
    if not cd.exists():
        return {"files": 0, "bytes": 0}
    total_bytes = 0
    files = 0
    for path in cd.glob("*.json"):
        try:
            total_bytes += path.stat().st_size
            files += 1
        except OSError:
            continue
    return {"files": files, "bytes": total_bytes}


def list_cached_symbols(*, cache_dir: Path | None = None) -> list[str]:
    """Return all symbols with a cache file present (no TTL check)."""
    cd = cache_dir if cache_dir is not None else _resolved_default_dir()
    if not cd.exists():
        return []
    out: list[str] = []
    for path in cd.glob("*.json"):
        # Strip ".json" — the filename IS the symbol.
        out.append(path.stem.upper())
    return sorted(out)


__all__ = [
    "CACHE_VERSION",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_EMPTY_TTL_S",
    "DEFAULT_TTL_S",
    "cache_stats",
    "list_cached_symbols",
    "load_cached_signal",
    "purge_expired",
    "save_cached_signal",
]
