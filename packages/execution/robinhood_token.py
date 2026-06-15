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
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Robinhood's MCP origin. OAuth endpoints are *discovered* from this base
# (RFC 8414 / RFC 9728 well-known metadata) rather than hardcoded, so a
# server-side path change doesn't require a code release. Override via env
# for testing against a mock.
RH_MCP_URL = os.getenv("ROBINHOOD_MCP_URL", "https://agent.robinhood.com/mcp/trading")
RH_OAUTH_BASE = os.getenv("ROBINHOOD_OAUTH_BASE", "https://agent.robinhood.com")

# Loopback redirect for the native-app authorization-code flow (RFC 8252).
# The cockpit runs a one-shot local listener on this port during auth.
RH_OAUTH_REDIRECT_PORT = int(os.getenv("ROBINHOOD_OAUTH_REDIRECT_PORT", "8788"))
RH_OAUTH_REDIRECT_URI = os.getenv(
    "ROBINHOOD_OAUTH_REDIRECT_URI",
    f"http://localhost:{RH_OAUTH_REDIRECT_PORT}/callback",
)

# Scopes the agent needs. Robinhood's authorization-server metadata
# advertises a single supported scope, "internal" (verified live against
# agent.robinhood.com's /.well-known/oauth-authorization-server). The
# server scopes the OAuth session to the dedicated Agentic sub-account at
# account-creation time; the granular per-tool trade/read permissions are
# approved by the user in Robinhood's own consent screen, not via OAuth
# scope strings. Overridable via env in case Robinhood widens the set.
RH_OAUTH_SCOPE = os.getenv("ROBINHOOD_OAUTH_SCOPE", "internal")

# Client identity for dynamic client registration (RFC 7591). Public
# native client -> no client secret.
RH_OAUTH_CLIENT_NAME = "ai-investing-cockpit"

# Bounded timeout for OAuth metadata + token calls.
OAUTH_HTTP_TIMEOUT_S = 10.0

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


# ---------------------------------------------------------------------------
# OAuth 2.1 authorization-server discovery + dynamic client registration
# ---------------------------------------------------------------------------
#
# The MCP auth spec (2025-06-18) layers RFC 9728 (protected-resource
# metadata) + RFC 8414 (authorization-server metadata) + RFC 7591 (dynamic
# client registration). Rather than hardcode Robinhood's endpoint paths --
# which the Cursor forum shows are still in flux -- we discover them at
# runtime and cache nothing across process restarts (cheap, and never
# stale). Every function here fails LOUDLY (raises) because auth is an
# explicit, user-initiated action; silent degradation would just confuse.


class OAuthError(RuntimeError):
    """Any failure in the OAuth discovery / exchange / refresh path."""


@dataclass
class OAuthEndpoints:
    """Resolved OAuth server endpoints for the Robinhood MCP resource."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _get_json(client: httpx.Client, url: str) -> dict[str, Any] | None:
    """GET a well-known doc; return parsed JSON or ``None`` on any miss.

    We tolerate 404s (a given well-known path may not exist on this
    server) by returning ``None`` so the caller can try the next
    candidate. Network/transport errors raise -- the user is mid-flow
    and deserves a real error, not a silent fallback.
    """
    try:
        r = client.get(url, headers={"Accept": "application/json"})
    except httpx.HTTPError as exc:
        raise OAuthError(f"discovery transport error for {url}: {exc!r}") from exc
    if r.status_code == 404:
        return None
    if r.status_code >= 400:
        return None
    try:
        data = r.json()
        return data if isinstance(data, dict) else None
    except ValueError:
        return None


def _well_known_candidates(as_url: str) -> list[str]:
    """Build authorization-server metadata URLs for an issuer/AS URL.

    Per RFC 8414 §3.1 the well-known segment is inserted *before* any path
    component of the issuer, i.e. ``https://host/.well-known/oauth-
    authorization-server/mcp/trading`` -- NOT after the path. We also try
    the OIDC convention (path appended after the well-known segment), the
    root form (no path), and the (non-standard but common) path-suffixed
    form for maximum compatibility. First responder wins.

    Robinhood (verified live) serves the metadata at the RFC 8414
    path-inserted form and the root form; the path-suffixed form 404s.
    """
    from urllib.parse import urlsplit

    parts = urlsplit(as_url.rstrip("/"))
    origin = f"{parts.scheme}://{parts.netloc}"
    path = parts.path.rstrip("/")  # e.g. "/mcp/trading" or ""
    candidates: list[str] = []
    if path:
        # RFC 8414: /.well-known/oauth-authorization-server<PATH>
        candidates.append(f"{origin}/.well-known/oauth-authorization-server{path}")
        # OIDC: /.well-known/openid-configuration<PATH>
        candidates.append(f"{origin}/.well-known/openid-configuration{path}")
    # Root forms (issuer has no path, or as a fallback)
    candidates.append(f"{origin}/.well-known/oauth-authorization-server")
    candidates.append(f"{origin}/.well-known/openid-configuration")
    if path:
        # Non-standard path-suffixed form some servers use as a last resort.
        candidates.append(f"{as_url.rstrip('/')}/.well-known/oauth-authorization-server")
        candidates.append(f"{as_url.rstrip('/')}/.well-known/openid-configuration")
    # De-dupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def discover_endpoints(
    *,
    mcp_url: str | None = None,
    oauth_base: str | None = None,
    client: httpx.Client | None = None,
) -> OAuthEndpoints:
    """Resolve Robinhood's OAuth endpoints via well-known metadata.

    Order of attempts (first hit wins):
      1. ``{mcp_origin}/.well-known/oauth-protected-resource`` (RFC 9728)
         -> follow its ``authorization_servers[0]`` and probe that server's
         metadata using RFC 8414 path-inserted + OIDC + root candidates.
      2. ``{oauth_base}`` metadata candidates directly.

    Raises ``OAuthError`` if none of the candidates yield the required
    ``authorization_endpoint`` + ``token_endpoint`` pair.
    """
    mcp_url = mcp_url or RH_MCP_URL
    oauth_base = (oauth_base or RH_OAUTH_BASE).rstrip("/")
    # MCP origin = scheme://host (drop the /mcp/trading path for well-known).
    from urllib.parse import urlsplit

    parts = urlsplit(mcp_url)
    mcp_origin = f"{parts.scheme}://{parts.netloc}"

    owns = client is None
    client = client or httpx.Client(
        timeout=OAUTH_HTTP_TIMEOUT_S, follow_redirects=True
    )

    def _probe(as_url: str) -> dict[str, Any] | None:
        for url in _well_known_candidates(as_url):
            meta = _get_json(client, url)
            if meta and meta.get("authorization_endpoint") and meta.get(
                "token_endpoint"
            ):
                return meta
        return None

    try:
        as_meta: dict[str, Any] | None = None

        # (1) protected-resource metadata -> authorization server
        prm = _get_json(
            client, f"{mcp_origin}/.well-known/oauth-protected-resource"
        )
        if prm:
            servers = prm.get("authorization_servers")
            if isinstance(servers, list) and servers:
                as_meta = _probe(str(servers[0]))

        # (2) direct metadata on the oauth base / mcp url
        if not as_meta:
            as_meta = _probe(mcp_url) or _probe(oauth_base)

        if not as_meta:
            raise OAuthError(
                "could not discover Robinhood OAuth endpoints from any "
                "well-known metadata document"
            )

        auth_ep = as_meta.get("authorization_endpoint")
        token_ep = as_meta.get("token_endpoint")
        if not auth_ep or not token_ep:
            raise OAuthError(
                "OAuth metadata missing authorization_endpoint or "
                "token_endpoint"
            )

        return OAuthEndpoints(
            issuer=str(as_meta.get("issuer", oauth_base)),
            authorization_endpoint=str(auth_ep),
            token_endpoint=str(token_ep),
            registration_endpoint=str(as_meta.get("registration_endpoint", "")),
        )
    finally:
        if owns:
            client.close()


def register_client(
    endpoints: OAuthEndpoints,
    *,
    redirect_uri: str | None = None,
    client: httpx.Client | None = None,
) -> str:
    """Dynamically register a public native client (RFC 7591).

    Returns the issued ``client_id``. If the server doesn't advertise a
    registration endpoint we fall back to the env-provided
    ``ROBINHOOD_OAUTH_CLIENT_ID`` (some servers pre-issue one); if neither
    is available we raise so the user gets a clear message.
    """
    redirect_uri = redirect_uri or RH_OAUTH_REDIRECT_URI

    if not endpoints.registration_endpoint:
        preset = os.getenv("ROBINHOOD_OAUTH_CLIENT_ID", "")
        if preset:
            return preset
        raise OAuthError(
            "server has no dynamic-registration endpoint and no "
            "ROBINHOOD_OAUTH_CLIENT_ID is set"
        )

    owns = client is None
    client = client or httpx.Client(timeout=OAUTH_HTTP_TIMEOUT_S)
    try:
        body = {
            "client_name": RH_OAUTH_CLIENT_NAME,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",  # public client
            "application_type": "native",
            "scope": RH_OAUTH_SCOPE,
        }
        try:
            r = client.post(endpoints.registration_endpoint, json=body)
        except httpx.HTTPError as exc:
            raise OAuthError(f"client registration transport error: {exc!r}") from exc
        if r.status_code >= 400:
            raise OAuthError(
                f"client registration failed: HTTP {r.status_code} "
                f"{r.text[:200]}"
            )
        data = r.json()
        cid = data.get("client_id")
        if not cid:
            raise OAuthError("registration response missing client_id")
        return str(cid)
    finally:
        if owns:
            client.close()


def build_authorize_url(
    endpoints: OAuthEndpoints,
    *,
    client_id: str,
    code_challenge: str,
    state: str,
    redirect_uri: str | None = None,
    scope: str | None = None,
) -> str:
    """Construct the authorization-code-with-PKCE authorize URL the user
    opens in their browser."""
    from urllib.parse import urlencode

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri or RH_OAUTH_REDIRECT_URI,
        "scope": scope or RH_OAUTH_SCOPE,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        # RFC 8707 resource indicator: bind the token to the MCP resource.
        "resource": RH_MCP_URL,
    }
    sep = "&" if "?" in endpoints.authorization_endpoint else "?"
    return f"{endpoints.authorization_endpoint}{sep}{urlencode(params)}"


def _post_token(
    token_endpoint: str, form: dict[str, str], client: httpx.Client | None
) -> TokenSet:
    """POST to the token endpoint and parse the response into a TokenSet."""
    owns = client is None
    client = client or httpx.Client(timeout=OAUTH_HTTP_TIMEOUT_S)
    try:
        try:
            r = client.post(
                token_endpoint,
                data=form,
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise OAuthError(f"token transport error: {exc!r}") from exc
        if r.status_code >= 400:
            raise OAuthError(
                f"token endpoint returned HTTP {r.status_code}: "
                f"{r.text[:200]}"
            )
        try:
            data = r.json()
        except ValueError as exc:
            raise OAuthError("token endpoint returned non-JSON body") from exc

        access = data.get("access_token")
        if not access:
            raise OAuthError("token response missing access_token")
        # Refresh token may be absent on refresh responses that keep the
        # same refresh token; caller decides whether to preserve the old.
        refresh = str(data.get("refresh_token", ""))
        expires_in = float(data.get("expires_in", 3600))
        return TokenSet(
            access_token=str(access),
            refresh_token=refresh,
            expires_at=time.time() + expires_in,
            scope=str(data.get("scope", "")),
            token_type=str(data.get("token_type", "Bearer")),
        )
    finally:
        if owns:
            client.close()


def exchange_code(
    endpoints: OAuthEndpoints,
    *,
    code: str,
    code_verifier: str,
    client_id: str,
    redirect_uri: str | None = None,
    client: httpx.Client | None = None,
) -> TokenSet:
    """Exchange an authorization ``code`` + PKCE verifier for tokens.

    The returned ``TokenSet`` is NOT persisted here -- the caller saves it
    only after verifying ``state`` matched, so a CSRF'd callback can't
    poison the keychain.
    """
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri or RH_OAUTH_REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    return _post_token(endpoints.token_endpoint, form, client)


def refresh_access_token(
    refresh_token: str,
    *,
    client_id: str,
    endpoints: OAuthEndpoints | None = None,
    client: httpx.Client | None = None,
) -> TokenSet:
    """Use a refresh token to mint a fresh access token.

    OAuth 2.1 mandates refresh-token rotation, so the response usually
    carries a NEW refresh token; if it doesn't, we preserve the one we
    were given so the next refresh still works.
    """
    endpoints = endpoints or discover_endpoints()
    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    tokens = _post_token(endpoints.token_endpoint, form, client)
    if not tokens.refresh_token:
        # Server kept the old refresh token; carry it forward.
        tokens.refresh_token = refresh_token
    return tokens


# ---------------------------------------------------------------------------
# client_id persistence (alongside tokens, separate keyring slot)
# ---------------------------------------------------------------------------

KEYRING_CLIENT_ID_USERNAME = "client_id"


def save_client_id(client_id: str) -> None:
    """Persist the dynamically-registered client_id so refreshes after a
    restart don't need to re-register."""
    _keyring().set_password(
        KEYRING_SERVICE, KEYRING_CLIENT_ID_USERNAME, client_id
    )


def load_client_id() -> str | None:
    """Load the persisted client_id, or env override, or None."""
    env = os.getenv("ROBINHOOD_OAUTH_CLIENT_ID", "")
    if env:
        return env
    try:
        return _keyring().get_password(
            KEYRING_SERVICE, KEYRING_CLIENT_ID_USERNAME
        )
    except Exception as exc:  # pragma: no cover - backend variance
        logger.warning("client_id read failed: %s", exc.__class__.__name__)
        return None


def clear_client_id() -> None:
    """Remove the stored client_id (part of full disconnect)."""
    try:
        _keyring().delete_password(
            KEYRING_SERVICE, KEYRING_CLIENT_ID_USERNAME
        )
    except Exception as exc:  # pragma: no cover - backend variance
        logger.debug("client_id delete no-op: %s", exc.__class__.__name__)
