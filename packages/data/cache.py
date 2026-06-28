"""Lightweight TTL cache for adapter responses.

Two jobs:

1. **Freshness / cost** — adapter responses are cached by ``(source,
   query)`` for a per-source TTL so we don't refetch the same Finnhub
   company-news URL every few seconds (the live logs show exactly that).
2. **Labeling** — every read reports whether the value was served fresh,
   from cache, or is *stale* (past TTL but still returned as a fallback
   when a live fetch failed) so the decision layer can down-weight it.

In-memory by default; an optional on-disk mirror under ``data/cache/``
(gitignored) survives restarts. Disk is best-effort — any IO error falls
back to memory-only and never raises.

The cache stores JSON-serializable values. Deduplication of concurrent
identical calls is handled by :meth:`TTLCache.get_or_fetch`, which holds a
per-key asyncio lock so N near-simultaneous callers trigger ONE upstream
fetch and share the result.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

# Repo-root/data/cache — matches the gitignored runtime dir.
_DEFAULT_DISK_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


@dataclass(frozen=True)
class CacheEntry:
    """A cached value plus the metadata the decision layer needs."""

    value: Any
    stored_at: float        # epoch seconds when written
    ttl_s: float            # how long it stays "fresh"

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.stored_at)

    @property
    def is_stale(self) -> bool:
        return self.age_s > self.ttl_s


@dataclass(frozen=True)
class CacheResult:
    """Outcome of a cache read, labeled for the decision layer."""

    value: Any
    hit: bool               # served from cache (fresh OR stale)
    stale: bool             # value is past its TTL
    age_s: float


def _key(source: str, query: str) -> str:
    return f"{source}:{query}"


def _disk_name(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32] + ".json"


class TTLCache:
    """Async-friendly TTL cache, memory-first with optional disk mirror."""

    def __init__(
        self,
        *,
        default_ttl_s: float = 300.0,
        disk_dir: Path | None = None,
        use_disk: bool = True,
    ) -> None:
        self.default_ttl_s = default_ttl_s
        self._mem: dict[str, CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._use_disk = use_disk
        self._disk_dir = disk_dir or _DEFAULT_DISK_DIR

    # ------------------------------------------------------------------
    # Core get / set
    # ------------------------------------------------------------------

    def get(self, source: str, query: str) -> CacheResult | None:
        """Return a :class:`CacheResult` if a value exists (even if stale),
        else ``None``. Stale entries are returned with ``stale=True`` so the
        caller can decide whether to use them as a fallback."""
        key = _key(source, query)
        entry = self._mem.get(key)
        if entry is None:
            entry = self._load_disk(key)
            if entry is not None:
                self._mem[key] = entry
        if entry is None:
            return None
        return CacheResult(
            value=entry.value,
            hit=True,
            stale=entry.is_stale,
            age_s=entry.age_s,
        )

    def get_fresh(self, source: str, query: str) -> CacheResult | None:
        """Like :meth:`get` but returns ``None`` for stale entries."""
        res = self.get(source, query)
        if res is None or res.stale:
            return None
        return res

    def set(
        self, source: str, query: str, value: Any, *, ttl_s: float | None = None
    ) -> None:
        key = _key(source, query)
        entry = CacheEntry(
            value=value,
            stored_at=time.time(),
            ttl_s=self.default_ttl_s if ttl_s is None else ttl_s,
        )
        self._mem[key] = entry
        self._store_disk(key, entry)

    def invalidate(self, source: str, query: str) -> None:
        key = _key(source, query)
        self._mem.pop(key, None)
        if self._use_disk:
            try:
                (self._disk_dir / _disk_name(key)).unlink(missing_ok=True)
            except OSError:
                pass

    def clear(self) -> None:
        self._mem.clear()

    # ------------------------------------------------------------------
    # Dedupe + fetch
    # ------------------------------------------------------------------

    async def get_or_fetch(
        self,
        source: str,
        query: str,
        fetch: Callable[[], Awaitable[Any]],
        *,
        ttl_s: float | None = None,
        serve_stale_on_error: bool = True,
    ) -> CacheResult:
        """Return a fresh cached value, or call ``fetch`` exactly once for
        concurrent callers of the same key.

        On a fetch error: if a stale value exists and
        ``serve_stale_on_error`` is set, the stale value is returned
        (``stale=True``); otherwise the exception propagates so the caller
        can degrade.
        """
        fresh = self.get_fresh(source, query)
        if fresh is not None:
            return fresh

        key = _key(source, query)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check inside the lock: a peer may have populated it.
            fresh = self.get_fresh(source, query)
            if fresh is not None:
                return fresh
            try:
                value = await fetch()
            except Exception:
                if serve_stale_on_error:
                    stale = self.get(source, query)
                    if stale is not None:
                        return stale
                raise
            self.set(source, query, value, ttl_s=ttl_s)
            return CacheResult(value=value, hit=False, stale=False, age_s=0.0)

    # ------------------------------------------------------------------
    # Disk mirror (best-effort)
    # ------------------------------------------------------------------

    def _load_disk(self, key: str) -> CacheEntry | None:
        if not self._use_disk:
            return None
        path = self._disk_dir / _disk_name(key)
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            return None
        try:
            d = json.loads(raw)
            return CacheEntry(
                value=d["value"],
                stored_at=float(d["stored_at"]),
                ttl_s=float(d["ttl_s"]),
            )
        except (ValueError, KeyError, TypeError):
            return None

    def _store_disk(self, key: str, entry: CacheEntry) -> None:
        if not self._use_disk:
            return
        try:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "value": entry.value,
                    "stored_at": entry.stored_at,
                    "ttl_s": entry.ttl_s,
                }
            )
            tmp = self._disk_dir / (_disk_name(key) + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._disk_dir / _disk_name(key))
        except (OSError, TypeError, ValueError) as exc:
            log.debug("cache: disk write skipped for %s: %s", key, exc)


# Process-wide default cache. Disk mirror enabled unless explicitly off via
# env (tests set DATA_CACHE_DISABLE_DISK=1 to stay hermetic).
_DISK_OFF = os.getenv("DATA_CACHE_DISABLE_DISK", "").lower() in ("1", "true", "yes")
_CACHE = TTLCache(use_disk=not _DISK_OFF)


def get_cache() -> TTLCache:
    """Return the process-wide default cache."""
    return _CACHE
