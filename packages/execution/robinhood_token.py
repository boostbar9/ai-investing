"""Robinhood OAuth token storage + PKCE helpers.

Robinhood's agentic-trading platform uses OAuth 2.1 with PKCE (browser-
based, no client secret). Tokens are short-lived access + long-lived
refresh; both live in the OS keychain via the ``keyring`` library so
they don't end up on disk in plaintext.

Why keyring and not a chmod'd file:
  * Cross-platform: Windows Credential Manager on the user's box,
    Keychain on macOS, SecretService on Linux.
  * Survives a workspace wipe (the tokens live in the OS, not the repo).
  * One less ``did I get the permissions right'' failure mode.

Public surface intentionally small:
  * save_tokens(access, refresh, expires_at)
  * load_tokens() -> TokenSet | None
  * clear_tokens()
  * new_pkce_pair() -> (verifier, challenge)

The actual browser flow is owned by ``packages/execution/robinhood.py``
because it needs to drive the broker's authorize endpoint + redirect
loopback. This module only handles the *storage* side.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

# Single service identifier so anyone hunting through the OS keychain
# knows exactly what these secrets are for.
KEYRING_SERVICE = "ai-investing.robinhood-agentic"

# We store one JSON blob under this username; simpler than juggling three
# separate keyring entries for access/refresh/expiry.
KEYRING_USERNAME = "default"

# When the access token is within this many seconds of expiry we treat
# it as stale and force a refresh on the next call. Robinhood tokens
# are typically 1h; 90s of slack lets us avoid in-flight expiry races.
EXPIRY_SLACK_S = 90


@dataclass
class TokenSet:
    """Everything we need to authenticate against Robinhood's MCP server."""

    access_token: str
    refresh_token: str
    # Unix epoch seconds; ``time.time() >= expires_at - EXPIRY_SLACK_S``
    # means the access token is stale and we must refresh before use.
    expires_at: float
    # Optional metadata (scope, token_type) preserved verbatim so a
    # future Robinhood server-change doesn't lose us information.
    scope: str = ""
    token_type: str = "Bearer"

    def is_stale(self, now: float | None = None) -> bool:
        ts = time.time() if now is None else now
        return ts >= (self.expires_at - EXPIRY_SLACK_S)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _keyring():
    """Lazy import so a test env without keyring backends doesn't blow up
    at module load. The real broker only calls into this when it actually
    needs to read or write a token."""
    import keyring

    return keyring


def save_tokens(tokens: TokenSet) -> None:
    """Persist the token set to the OS keychain. Atomic on the keyring
    side -- a partial write can't happen because we serialize the whole
    blob in one keyring set_password call.
    """
    payload = json.dumps(tokens.to_dict(), separators=(",", ":"))
    _keyring().set_password(KEYRING_SERVICE, KEYRING_USERNAME, payload)


def load_tokens() -> TokenSet | None:
    """Return the persisted ``TokenSet`` or ``None`` if no tokens are
    stored (fresh install, user revoked, etc.). NEVER raises on a
    missing/corrupt entry -- we return ``None`` so the broker can
    gracefully degrade to 'not yet connected to Robinhood'.
    """
    try:
        raw = _keyring().get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception as exc:  # pragma: no cover - keyring backend issues
        logger.warning("keyring read failed: %s", exc.__class__.__name__)
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return TokenSet(
            access_token=str(data["access_token"]),
            refresh_token=str(data["refresh_token"]),
            expires_at=float(data["expires_at"]),
            scope=str(data.get("scope", "")),
            token_type=str(data.get("token_type", "Bearer")),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("token blob malformed: %s", exc.__class__.__name__)
        return None


def clear_tokens() -> None:
    """Delete the stored tokens. Used by the 'Disconnect Robinhood' button
    in settings and by the reset-onboarding affordance."""
    try:
        _keyring().delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception as exc:  # pragma: no cover - backend variance
        # Idempotent: 'delete nonexistent' is fine.
        logger.debug("keyring delete no-op: %s", exc.__class__.__name__)


# ---------------------------------------------------------------------------
# PKCE helpers (RFC 7636)
# ---------------------------------------------------------------------------


def _b64url(b: bytes) -> str:
    """Base64-url-without-padding. The OAuth 2.1 spec requires this exact
    encoding for the code_challenge."""
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def new_pkce_pair() -> tuple[str, str]:
    """Generate a ``(code_verifier, code_challenge)`` pair.

    * The verifier is 64 hex chars (256 bits of entropy), well above the
      43-char minimum RFC 7636 mandates.
    * The challenge is ``base64url(sha256(verifier))``, the only method
      Robinhood supports per the agentic-trading docs.

    The caller passes ``code_challenge`` (and ``method=S256``) to the
    authorize URL, then exchanges the resulting ``code`` plus the
    original ``code_verifier`` at the token endpoint.
    """
    verifier = secrets.token_hex(32)  # 64 chars
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def new_state(nbytes: int = 16) -> str:
    """Opaque OAuth ``state`` parameter for CSRF protection. The caller
    stashes this in memory before opening the browser; the callback
    handler MUST verify the returned ``state`` matches before exchanging
    the code."""
    return _b64url(os.urandom(nbytes))
