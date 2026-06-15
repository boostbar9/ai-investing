"""Robinhood agentic-trading access detector.

Robinhood announced agentic trading on May 27, 2026 with a gradual
rollout: you sign up, then they email when your sub-account is enabled.
This module probes a small, safe surface to figure out which bucket the
user is in (unknown / waitlist / granted / declined) without ever
submitting a trade.

What it actually does:

1. Reads the *cached* status from ``data/cockpit/onboarding.json`` first
   so the cockpit can render instantly. Disk read only -- no network.
2. On explicit refresh (e.g. ``POST /api/onboarding/check-robinhood``)
   it does ONE short HTTP HEAD/GET against the public MCP discovery URL
   (``https://agent.robinhood.com/mcp/trading``) with an aggressive
   timeout. We're checking *reachability*, not authenticating.
3. If a Robinhood OAuth refresh token is present in the OS keychain
   (Phase 2 work, not yet implemented), the detector additionally calls
   the ``introspect`` endpoint to confirm 'granted'. For now this branch
   is a stub returning ``waitlist`` so the wizard still works end-to-end.

The detector is intentionally read-only and idempotent. It NEVER POSTS
or mutates anything on Robinhood's side.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import httpx

logger = logging.getLogger(__name__)

# Public discovery endpoint announced in the Robinhood agentic-trading
# overview. We only HEAD this -- it's a public URL, not a secret.
RH_MCP_DISCOVERY_URL = "https://agent.robinhood.com/mcp/trading"

# Generous-but-not-infinite timeout. Cockpit boot must NOT hang waiting
# for Robinhood, so we keep this short and fail open.
RH_PROBE_TIMEOUT_S = 4.0

ProbeOutcome = Literal[
    "granted",  # user has confirmed access (token present + introspect ok)
    "waitlist",  # endpoint reachable, but no token / introspect says no
    "unknown",  # network error or unexpected status; couldn't determine
    "offline",  # cockpit appears to be fully offline; skip silently
    "declined",  # user explicitly opted out; do not touch their choice
]


@dataclass
class ProbeResult:
    """Outcome of one detector run. Kept separate from OnboardingState so
    the caller can decide whether to persist it."""

    outcome: ProbeOutcome
    detail: str = ""
    # The HTTP status code we observed, if any. ``None`` means we never
    # got a response (DNS failure, timeout, etc.).
    http_status: int | None = None


def _probe_discovery(timeout_s: float = RH_PROBE_TIMEOUT_S) -> ProbeResult:
    """Make the actual network call. Isolated so tests can patch it."""
    try:
        with httpx.Client(timeout=timeout_s, follow_redirects=False) as client:
            r = client.head(RH_MCP_DISCOVERY_URL)
            return ProbeResult(
                outcome="waitlist",
                detail=f"discovery reachable (HEAD {r.status_code})",
                http_status=r.status_code,
            )
    except httpx.TimeoutException:
        return ProbeResult(
            outcome="unknown",
            detail=f"discovery probe timed out after {timeout_s}s",
        )
    except httpx.ConnectError as exc:
        # No internet, DNS broken, etc. Treat as offline so the wizard
        # doesn't show a scary red banner when the laptop just isn't
        # online yet.
        return ProbeResult(
            outcome="offline", detail=f"connect error: {exc.__class__.__name__}"
        )
    except httpx.HTTPError as exc:
        return ProbeResult(
            outcome="unknown",
            detail=f"http error: {exc.__class__.__name__}",
        )


def _check_granted_via_token() -> ProbeResult | None:
    """Confirm 'granted' by exercising the stored token against the MCP
    server.

    Logic:
      1. No token in keychain -> return ``None`` (fall through to the
         public discovery probe; the user is still on the waitlist).
      2. Token present (refreshing first if stale) -> run an authenticated
         MCP ``initialize`` + ``tools/list`` handshake. If the server
         answers with a tool catalog, the sub-account is live -> ``granted``.
      3. Auth rejected (401/403/refresh failure) -> ``waitlist`` with a
         detail explaining the token isn't accepted yet.
      4. Any other transport hiccup -> ``None`` so we fall through to the
         reachability probe rather than mislabeling the user.

    This is read-only: ``initialize`` + ``tools/list`` never submit a
    trade. Imports live inside the function so a fresh install without the
    keyring backend never breaks cockpit boot at import time.
    """
    import asyncio

    try:
        from packages.execution.broker import BrokerError
        from packages.execution.robinhood import is_connected
        from packages.execution.robinhood_mcp import (
            McpError,
            RobinhoodMcpClient,
        )
    except Exception as exc:  # pragma: no cover - import-time safety net
        logger.debug("rh token check unavailable: %s", exc.__class__.__name__)
        return None

    if not is_connected():
        return None  # no usable token -> still waitlist

    async def _handshake() -> ProbeResult:
        # Reuse the broker's token-resolution + refresh logic so a stale
        # access token is silently refreshed before we probe.
        from packages.execution.modes import ExecutionMode
        from packages.execution.robinhood import RobinhoodAgenticBroker

        broker = RobinhoodAgenticBroker(mode=ExecutionMode.SHADOW)
        try:
            tokens = broker._require_token()  # refreshes if stale
        except BrokerError as exc:
            return ProbeResult(
                outcome="waitlist",
                detail=f"token present but not usable: {exc}",
            )
        client = RobinhoodMcpClient(
            bearer_token=tokens.access_token,
            timeout_s=RH_PROBE_TIMEOUT_S,
        )
        try:
            await client.initialize()
            tools = await client.list_tools()
            return ProbeResult(
                outcome="granted",
                detail=f"authenticated MCP handshake ok ({len(tools)} tools)",
                http_status=200,
            )
        except McpError as exc:
            msg = str(exc)
            # 401/403 means the token isn't authorized for the agentic
            # account yet -- the user is approved for OAuth but the
            # sub-account isn't live. Treat as waitlist, not granted.
            if "401" in msg or "403" in msg:
                return ProbeResult(
                    outcome="waitlist",
                    detail=f"MCP rejected token (not yet provisioned): {msg}",
                )
            # Other MCP errors are ambiguous -> fall through.
            return ProbeResult(outcome="unknown", detail=f"mcp error: {msg}")
        finally:
            await client.aclose()

    try:
        result = asyncio.run(_handshake())
    except RuntimeError:
        # Already inside an event loop (e.g. called from async cockpit
        # context). Run in a dedicated loop on a thread to stay safe.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(lambda: asyncio.run(_handshake())).result()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("rh token handshake failed: %s", exc.__class__.__name__)
        return None

    # 'unknown' from the handshake means fall through to discovery probe.
    return None if result.outcome == "unknown" else result


def detect_access(
    *,
    timeout_s: float = RH_PROBE_TIMEOUT_S,
    declined_already: bool = False,
) -> ProbeResult:
    """Top-level detector. Honors a previous ``declined`` choice so we
    don't hassle the user if they explicitly opted out.

    Returns a ``ProbeResult``; the caller is responsible for persisting
    the outcome into ``OnboardingState.robinhood_status``.
    """
    if declined_already:
        return ProbeResult(
            outcome="declined",
            detail="user previously declined; not re-probing",
        )

    granted = _check_granted_via_token()
    if granted is not None:
        return granted

    return _probe_discovery(timeout_s=timeout_s)
