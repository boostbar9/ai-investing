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
    """Phase 2 stub: when a Robinhood refresh token is present in the OS
    keychain we'll introspect it to confirm the sub-account is active.

    Returns ``None`` until Phase 2 lands. Wizard treats ``None`` as
    'fall through to the discovery probe'.
    """
    # Intentional: Phase 2 (RobinhoodAgenticBroker) will add the keyring
    # lookup + introspection call here. Stubbed so callers don't crash
    # before that work lands.
    return None


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
