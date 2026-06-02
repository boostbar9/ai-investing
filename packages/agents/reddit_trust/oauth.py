"""Phase 25.5 — Reddit OAuth (application-only) token broker.

Reddit started 403-ing anonymous JSON requests from datacenter IPs in
2023. The right fix is to authenticate via Reddit's free "script" or
"installed app" OAuth flow, which raises the unauthenticated 60/min
cap to 100/min/per-OAuth-app for read-only "application-only" tokens.

This module produces a cached bearer token via the
``application/json`` grant (client_credentials variant Reddit
documents as "Application Only OAuth"). The token is short-lived
(~1h) so we refresh proactively before expiry and cache in memory.

Configuration (env-driven, no code changes per environment):
    REDDIT_CLIENT_ID       — required to enable OAuth
    REDDIT_CLIENT_SECRET   — for "script" or "web" apps; can be ""
                             for "installed" / public app types
    REDDIT_USERNAME        — optional: enables password grant for
                             "script" apps (higher per-user limit)
    REDDIT_PASSWORD        — optional, paired with REDDIT_USERNAME
    REDDIT_USER_AGENT      — recommended unique string per Reddit
                             API docs; falls back to fetcher's UA

When ``REDDIT_CLIENT_ID`` is unset, ``get_token()`` returns ``None``
and callers transparently fall back to the unauthenticated JSON /
RSS path. This preserves Phase-25.3-and-earlier behavior with zero
breakage when OAuth isn't configured.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

REDDIT_TOKEN_URL = os.getenv(
    "REDDIT_TOKEN_URL", "https://www.reddit.com/api/v1/access_token"
)

# Refresh ~2min before Reddit's stated expiry so requests in flight
# don't 401 right as the token rolls.
_EXPIRY_SLACK_S = 120.0


@dataclass
class _CachedToken:
    token: str
    expires_at: float  # monotonic seconds

    def expired(self, now: float, slack: float = _EXPIRY_SLACK_S) -> bool:
        return now + slack >= self.expires_at


class RedditOAuthClient:
    """Mints + caches application-only bearer tokens for Reddit.

    Thread-safe via a single :class:`asyncio.Lock`. One instance per
    process — exposed as the module-level singleton ``get_oauth()``.
    """

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        username: str | None = None,
        password: str | None = None,
        user_agent: str | None = None,
        http: httpx.AsyncClient | None = None,
        clock: Any = None,
    ) -> None:
        self._client_id = client_id if client_id is not None else os.getenv(
            "REDDIT_CLIENT_ID", ""
        )
        self._client_secret = (
            client_secret if client_secret is not None
            else os.getenv("REDDIT_CLIENT_SECRET", "")
        )
        self._username = username if username is not None else os.getenv(
            "REDDIT_USERNAME", ""
        )
        self._password = password if password is not None else os.getenv(
            "REDDIT_PASSWORD", ""
        )
        # Reddit's API docs explicitly recommend a unique, identifying UA.
        # When env doesn't supply one we fall back to the fetcher's UA.
        self._user_agent = user_agent or os.getenv(
            "REDDIT_USER_AGENT",
            os.getenv(
                "RICH_REDDIT_UA",
                "ai-investing/0.4 by /u/boostbar9",
            ),
        )
        self._http = http
        self._owned_http = http is None
        self._lock = asyncio.Lock()
        self._cached: _CachedToken | None = None
        self._clock = clock or time.monotonic
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self._client_id)

    @property
    def user_agent(self) -> str:
        return self._user_agent

    @property
    def last_error(self) -> str | None:
        return self._last_error

    async def get_token(self) -> str | None:
        """Return a fresh bearer token, or ``None`` if OAuth disabled.

        Cached in memory; refreshes on demand when expired. Returns
        ``None`` (instead of raising) on auth failure so callers can
        cleanly fall back to unauthenticated paths.
        """
        if not self.enabled:
            return None
        async with self._lock:
            now = self._clock()
            if self._cached is not None and not self._cached.expired(now):
                return self._cached.token
            tok = await self._mint_token()
            if tok is None:
                return None
            self._cached = tok
            return tok.token

    def invalidate(self) -> None:
        """Drop the cached token (test hook / 401-on-use recovery)."""
        self._cached = None

    async def aclose(self) -> None:
        if self._owned_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=10.0)
        return self._http

    async def _mint_token(self) -> _CachedToken | None:
        """POST to Reddit's token endpoint with the right grant type."""
        # Choose grant. Script apps (with username+password) get a
        # higher per-user limit; everything else uses the public
        # client_credentials grant.
        if self._username and self._password and self._client_secret:
            grant_data = {
                "grant_type": "password",
                "username": self._username,
                "password": self._password,
            }
        else:
            # Reddit's "installed" grant for public apps without secret;
            # falls back to client_credentials when a secret is given.
            if self._client_secret:
                grant_data = {"grant_type": "client_credentials"}
            else:
                # For installed apps Reddit requires a device_id (we
                # use the app id itself, which is what their docs show).
                grant_data = {
                    "grant_type": (
                        "https://oauth.reddit.com/grants/installed_client"
                    ),
                    "device_id": "ai-investing-cockpit",
                }

        client = await self._client()
        try:
            r = await client.post(
                REDDIT_TOKEN_URL,
                data=grant_data,
                auth=(self._client_id, self._client_secret or ""),
                headers={"User-Agent": self._user_agent},
            )
        except Exception as exc:
            self._last_error = f"transport: {type(exc).__name__}: {exc}"[:240]
            logger.warning("reddit oauth: token mint transport fail: %s", exc)
            return None

        if r.status_code != 200:
            self._last_error = f"HTTP {r.status_code}: {r.text[:160]}"
            logger.warning(
                "reddit oauth: token mint HTTP %s: %s",
                r.status_code, r.text[:160],
            )
            return None
        try:
            data = r.json()
        except Exception as exc:
            self._last_error = f"parse: {exc}"[:240]
            return None

        token = data.get("access_token")
        ttl = float(data.get("expires_in") or 0)
        if not token or ttl <= 0:
            self._last_error = "reddit returned no token / expiry"
            return None
        self._last_error = None
        logger.info(
            "reddit oauth: minted token (expires in %.0fs, grant=%s)",
            ttl, grant_data.get("grant_type"),
        )
        return _CachedToken(token=token, expires_at=self._clock() + ttl)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_default_client: RedditOAuthClient | None = None


def get_oauth() -> RedditOAuthClient:
    """Return the process-wide Reddit OAuth client."""
    global _default_client
    if _default_client is None:
        _default_client = RedditOAuthClient()
    return _default_client


def reset_oauth_for_tests() -> None:
    global _default_client
    _default_client = None
