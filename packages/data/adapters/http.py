"""Shared resilient HTTP client for data adapters.

Every external feed that uses plain HTTP should fetch through
:class:`ResilientHTTPClient` instead of calling ``httpx`` directly. It
centralizes the behaviours the live logs proved we need:

* **Realistic headers** — a browser-like ``User-Agent`` (Reddit /
  StockTwits return 403 to generic clients) plus a sane ``Accept``.
* **Per-host rate limiting** — reuses the shared token buckets in
  :mod:`packages.shared.rate_limit` so we stop hammering Reddit into 429.
* **Exponential backoff + jitter** on 429 / 5xx, with a capped retry count
  and respect for ``Retry-After`` when present.
* **Short timeouts** so a hung feed can't stall the decision loop.
* **Graceful degradation** — failures NEVER raise into the decision path.
  :meth:`fetch` always returns a :class:`FetchResult`; callers inspect
  ``.ok`` and map a not-ok result to an empty/"unavailable" signal. A
  disabled source short-circuits to an ``unavailable`` result before any
  network call.

The client also records every attempt into the per-source health registry
(:mod:`packages.data.health`) so the cockpit can show what's working.

It deliberately does NOT change request *semantics* (URL, params, body) —
only how requests are retried, paced, and reported.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from packages.data import health as health_mod
from packages.data.redact import redact
from packages.shared.rate_limit import BUCKETS

log = logging.getLogger(__name__)

# A current, common desktop Chrome UA. Reddit/StockTwits block obvious bot
# UAs; presenting as a real browser materially improves the success rate
# from cloud IPs. Override per-adapter via the ``user_agent`` arg.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_RETRIES = 3
# Backoff is intentionally short: these are best-effort feeds and a stalled
# retry chain is worse than degrading to "unavailable" quickly. Override via
# env for noisier production tuning without touching call sites.
BACKOFF_BASE_S = float(os.getenv("DATA_HTTP_BACKOFF_BASE_S", "0.25"))
BACKOFF_CAP_S = float(os.getenv("DATA_HTTP_BACKOFF_CAP_S", "5.0"))
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a resilient fetch. Never an exception — inspect ``ok``.

    ``unavailable`` is True when the source was disabled or every attempt
    failed; callers must treat that as "no data" (NOT a negative signal).
    """

    ok: bool
    status: int | None
    response: httpx.Response | None
    error: str | None
    attempts: int
    unavailable: bool = False

    def json(self, default: Any = None) -> Any:
        if self.response is None:
            return default
        try:
            return self.response.json()
        except ValueError:
            return default

    @property
    def text(self) -> str:
        return "" if self.response is None else self.response.text


class ResilientHTTPClient:
    """Async HTTP client with retry/backoff, rate limiting and graceful
    failure, wired to the source health registry.

    Parameters
    ----------
    source
        Logical source name (e.g. ``"reddit"``). Used for the rate-limit
        bucket lookup and health-registry keys.
    bucket
        Override the rate-limit bucket name (defaults to ``source``). When
        no matching bucket exists rate limiting is skipped.
    client
        Inject an :class:`httpx.AsyncClient` (tests pass a MockTransport
        client). When omitted one is created with the resilient defaults.
    """

    def __init__(
        self,
        source: str,
        *,
        bucket: str | None = None,
        client: httpx.AsyncClient | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        headers: Mapping[str, str] | None = None,
        stale_after_s: float | None = None,
    ) -> None:
        self.source = source
        self._bucket = bucket or source
        self._max_retries = max_retries
        self._stale_after_s = stale_after_s
        self._own_client = client is None
        base_headers = {
            "User-Agent": user_agent,
            "Accept": "application/json, text/xml, text/html, */*",
        }
        if headers:
            base_headers.update(headers)
        self._client = client or httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers=base_headers,
        )

    async def _rate_limit(self, bucket_name: str) -> None:
        bucket = BUCKETS.get(bucket_name)
        if bucket is not None:
            await bucket.acquire()

    @staticmethod
    def _retry_after_seconds(resp: httpx.Response) -> float | None:
        raw = resp.headers.get("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    async def fetch(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        record_health: bool = True,
        bucket: str | None = None,
        health_key: str | None = None,
        **kwargs: Any,
    ) -> FetchResult:
        """Perform a request with retry/backoff. Always returns a
        :class:`FetchResult`; it never raises for transport/HTTP errors.

        A disabled source (per :func:`packages.data.health.is_enabled`)
        short-circuits to an ``unavailable`` result with no network call.

        ``bucket`` overrides the rate-limit bucket for this call;
        ``health_key`` overrides the source name used for health recording
        (lets one adapter track sub-feeds, e.g. reddit vs. rss).
        """
        src = health_key or self.source
        bucket_name = bucket or self._bucket
        if not health_mod.is_enabled(src):
            return FetchResult(
                ok=False,
                status=None,
                response=None,
                error="disabled",
                attempts=0,
                unavailable=True,
            )

        if record_health:
            health_mod.get_registry().record_attempt(src)

        attempts = 0
        last_error: str | None = None
        last_status: int | None = None
        t0 = time.monotonic()

        for attempt in range(self._max_retries + 1):
            attempts = attempt + 1
            await self._rate_limit(bucket_name)
            try:
                resp = await self._client.request(
                    method, url, params=params, **kwargs
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}"
                last_status = None
            else:
                last_status = resp.status_code
                if resp.status_code in RETRYABLE_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                    # Honor Retry-After on the last-but-one attempt too.
                    if attempt < self._max_retries:
                        await self._sleep_backoff(
                            attempt, self._retry_after_seconds(resp)
                        )
                        continue
                    self._on_failure(record_health, src, last_error)
                    return FetchResult(
                        ok=False,
                        status=resp.status_code,
                        response=resp,
                        error=last_error,
                        attempts=attempts,
                        unavailable=True,
                    )
                # Non-retryable status (2xx, 3xx, 4xx other than 429).
                ok = resp.is_success
                if ok:
                    if record_health:
                        latency = (time.monotonic() - t0) * 1000.0
                        health_mod.get_registry().record_success(
                            src,
                            latency_ms=latency,
                            stale_after_s=self._stale_after_s,
                        )
                else:
                    self._on_failure(
                        record_health, src, f"HTTP {resp.status_code}"
                    )
                return FetchResult(
                    ok=ok,
                    status=resp.status_code,
                    response=resp,
                    error=None if ok else f"HTTP {resp.status_code}",
                    attempts=attempts,
                    unavailable=not ok,
                )

            # Transport error path: retry if attempts remain.
            if attempt < self._max_retries:
                await self._sleep_backoff(attempt, None)
                continue

        self._on_failure(record_health, src, last_error or "request failed")
        return FetchResult(
            ok=False,
            status=last_status,
            response=None,
            error=last_error or "request failed",
            attempts=attempts,
            unavailable=True,
        )

    def _on_failure(self, record_health: bool, src: str, error: str) -> None:
        if record_health:
            health_mod.get_registry().record_failure(
                src, error, stale_after_s=self._stale_after_s,
            )

    async def _sleep_backoff(self, attempt: int, retry_after: float | None) -> None:
        """Exponential backoff with full jitter, capped. Respects an
        explicit ``Retry-After`` when the server supplied one."""
        if retry_after is not None:
            delay = min(retry_after, BACKOFF_CAP_S)
        else:
            ceiling = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** attempt))
            delay = random.uniform(0.0, ceiling)
        await asyncio.sleep(delay)

    async def get(self, url: str, **kwargs: Any) -> FetchResult:
        return await self.fetch("GET", url, **kwargs)

    async def aclose(self) -> None:
        if self._own_client:
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover - close is best-effort
                pass
