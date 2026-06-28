"""Robinhood OAuth token storage + PKCE helpers.

Robinhood's agentic-trading platform uses OAuth 2.1 with PKCE (browser-
based, no client secret). Tokens are short-lived access + long-lived
refresh.

Storage design (the why):
  * The OAuth token blob (access + refresh + metadata) routinely exceeds
    2.5 KB. Windows Credential Manager caps a single credential's secret
    at ~2560 bytes, so writing the whole blob via ``keyring`` fails on
    Windows with ``CredWrite (1703, 'The stub received bad data')``.
  * So the *big* ciphertext lives in an encrypted file under
    ``data/cockpit/`` (not size-limited), while only the *small* 44-byte
    Fernet key lives in the OS keychain via ``keyring`` (well under the
    2.5 KB limit, so that write always succeeds). Small key in keyring,
    big ciphertext in file.
  * Encryption at rest uses ``cryptography.fernet`` when available. If
    that import is missing we degrade to a permission-restricted (0600)
    obfuscated file and warn -- the connect flow must never crash.
  * Every keyring call is wrapped in try/except: a keyring failure falls
    back to a restricted local key file so connect succeeds end-to-end.
  * Legacy: older installs stored the whole JSON blob directly in keyring.
    ``load_*`` transparently reads that on miss and migrates to the new
    encrypted-file store; ``clear_*`` removes both old and new locations.

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
import contextlib
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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
# Robinhood redirects the browser back here after the user approves; the
# cockpit's own FastAPI web server serves the ``/callback`` route and
# finishes the token exchange there (see packages/cockpit/web/server.py).
# There is NO separate one-shot listener -- the redirect MUST therefore
# point at the port the cockpit web server actually runs on (8000 by
# default; see tools/start_cockpit.ps1). We use 127.0.0.1 (not
# ``localhost``) to match $CockpitUrl and sidestep localhost-resolution
# edge cases. ROBINHOOD_OAUTH_REDIRECT_PORT is kept for back-compat but
# the cockpit port (COCKPIT_PORT) is what the callback is actually served
# on; full URI override stays available via ROBINHOOD_OAUTH_REDIRECT_URI
# (e.g. for the cloudflare-tunnel / phone case where loopback is unreachable).
RH_OAUTH_REDIRECT_PORT = int(
    os.getenv("ROBINHOOD_OAUTH_REDIRECT_PORT")
    or os.getenv("COCKPIT_PORT", "8000")
)
RH_OAUTH_REDIRECT_URI = os.getenv(
    "ROBINHOOD_OAUTH_REDIRECT_URI",
    f"http://127.0.0.1:{RH_OAUTH_REDIRECT_PORT}/callback",
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

# Legacy username: older installs stored the whole token JSON blob here.
# We still read it for backward-compatible migration, but never write it.
KEYRING_USERNAME = "default"

# Keyring username for the small (44-byte) Fernet key that encrypts the
# on-disk token/client_id files. Tiny, so the Windows 2.5 KB cap is a
# non-issue for *this* entry even though it broke the full token blob.
KEYRING_ENC_KEY_USERNAME = "enc_key"

# Directory + filenames for the encrypted on-disk stores. ``data/cockpit``
# is already gitignored. Overridable via env so tests stay hermetic.
REPO_ROOT = Path(__file__).resolve().parents[2]
_TOKEN_FILE_NAME = ".rh_tokens.enc"
_CLIENT_ID_FILE_NAME = ".rh_client_id.enc"
# Short-lived, single-use, encrypted store for the in-flight OAuth
# pending-auth blob (state + PKCE verifier + endpoints). Persisted so the
# flow survives a cockpit server auto-restart between begin_auth and the
# /callback redirect; see ``robinhood.py`` for the why + TTL.
_PENDING_AUTH_FILE_NAME = ".rh_pending_auth.enc"
_KEY_FILE_NAME = ".rh_key"  # last-resort key store when keyring is unusable


def _storage_dir() -> Path:
    """Resolve (and create) the directory holding the encrypted stores.

    Honors ``ROBINHOOD_TOKEN_DIR`` so tests can redirect to a tmp path
    without touching the user's real ``data/cockpit``.
    """
    override = os.getenv("ROBINHOOD_TOKEN_DIR")
    base = Path(override) if override else (REPO_ROOT / "data" / "cockpit")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _token_file() -> Path:
    return _storage_dir() / _TOKEN_FILE_NAME


def _client_id_file() -> Path:
    return _storage_dir() / _CLIENT_ID_FILE_NAME


def _pending_auth_file() -> Path:
    return _storage_dir() / _PENDING_AUTH_FILE_NAME


def _key_file() -> Path:
    return _storage_dir() / _KEY_FILE_NAME

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
    needs to read or write a key."""
    import keyring

    return keyring


# ---------------------------------------------------------------------------
# Encrypted-file storage primitives
# ---------------------------------------------------------------------------
#
# Big ciphertext -> file (no size cap). Small Fernet key -> keyring (or a
# 0600 file if keyring is unusable). Every keyring touch is wrapped so a
# CredWrite/CredRead failure on Windows can never crash the connect flow.

# Scheme markers prefixing each on-disk secret so reads can self-describe.
_SCHEME_FERNET = b"FERNET"
_SCHEME_PLAIN = b"PLAIN"


def _fernet():
    """Return the ``Fernet`` class, or ``None`` if ``cryptography`` isn't
    importable (we then degrade to a permission-restricted plaintext file)."""
    try:
        from cryptography.fernet import Fernet

        return Fernet
    except Exception:  # pragma: no cover - only when dep is absent
        return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes with restrictive (0600) perms, replacing atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    # chmod is a no-op-ish on Windows but harmless; suppress fs variance.
    with contextlib.suppress(OSError):  # pragma: no cover - platform variance
        os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    with contextlib.suppress(OSError):  # pragma: no cover - platform variance
        os.chmod(path, 0o600)


def _read_key_material() -> bytes | None:
    """Return the existing Fernet key from keyring, then the 0600 key file,
    or ``None`` if no key has been provisioned yet."""
    try:
        v = _keyring().get_password(KEYRING_SERVICE, KEYRING_ENC_KEY_USERNAME)
        if v:
            return v.encode("ascii")
    except Exception as exc:  # pragma: no cover - backend variance
        logger.warning("keyring key read failed: %s", exc.__class__.__name__)
    kf = _key_file()
    if kf.exists():
        try:
            raw = kf.read_bytes().strip()
            if raw:
                return raw
        except OSError as exc:  # pragma: no cover - fs variance
            logger.warning("key file read failed: %s", exc.__class__.__name__)
    return None


def _get_or_create_key() -> bytes:
    """Return the encryption key, generating + persisting one on first use.

    Tries keyring first (44 bytes -> well under the Windows 2.5 KB cap). If
    keyring is unusable, falls back to a 0600 local key file so connect
    still succeeds end-to-end on a broken-keyring box.
    """
    existing = _read_key_material()
    if existing:
        return existing
    fernet = _fernet()
    key = fernet.generate_key() if fernet else base64.urlsafe_b64encode(os.urandom(32))
    stored = False
    try:
        _keyring().set_password(
            KEYRING_SERVICE, KEYRING_ENC_KEY_USERNAME, key.decode("ascii")
        )
        stored = True
    except Exception as exc:
        logger.warning(
            "keyring key write failed (%s); falling back to 0600 key file",
            exc.__class__.__name__,
        )
    if not stored:
        _atomic_write_bytes(_key_file(), key)
    return key


def _write_secret(path: Path, plaintext: str) -> None:
    """Encrypt (if possible) and atomically persist ``plaintext`` to ``path``."""
    fernet = _fernet()
    if fernet is not None:
        key = _get_or_create_key()
        token = fernet(key).encrypt(plaintext.encode("utf-8"))
        data = _SCHEME_FERNET + b"\n" + token
    else:
        logger.warning(
            "cryptography unavailable; storing Robinhood secret obfuscated "
            "in a 0600 file (no real encryption at rest)"
        )
        data = _SCHEME_PLAIN + b"\n" + base64.b64encode(plaintext.encode("utf-8"))
    _atomic_write_bytes(path, data)


def _read_secret(path: Path) -> str | None:
    """Return the decrypted plaintext stored at ``path``, or ``None`` if the
    file is absent / unreadable / undecryptable."""
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:  # pragma: no cover - fs variance
        logger.warning("secret file read failed: %s", exc.__class__.__name__)
        return None
    scheme, _, body = data.partition(b"\n")
    try:
        if scheme == _SCHEME_FERNET:
            fernet = _fernet()
            if fernet is None:
                return None
            key = _read_key_material()
            if key is None:
                return None
            return fernet(key).decrypt(body).decode("utf-8")
        if scheme == _SCHEME_PLAIN:
            return base64.b64decode(body).decode("utf-8")
    except Exception as exc:
        logger.warning("secret decrypt failed: %s", exc.__class__.__name__)
    return None


def _read_old_keyring(username: str) -> str | None:
    """Best-effort read of the legacy keyring blob (pre-encrypted-file)."""
    try:
        return _keyring().get_password(KEYRING_SERVICE, username)
    except Exception as exc:  # pragma: no cover - backend variance
        logger.warning("legacy keyring read failed: %s", exc.__class__.__name__)
        return None


def _delete_keyring(username: str) -> None:
    """Best-effort delete of a keyring entry; missing entry is fine."""
    try:
        _keyring().delete_password(KEYRING_SERVICE, username)
    except Exception as exc:  # pragma: no cover - backend variance
        logger.debug("keyring delete no-op: %s", exc.__class__.__name__)


def _maybe_remove_key() -> None:
    """Drop the shared Fernet key once no encrypted store references it, so a
    full disconnect leaves nothing behind."""
    if (
        _token_file().exists()
        or _client_id_file().exists()
        or _pending_auth_file().exists()
    ):
        return
    _delete_keyring(KEYRING_ENC_KEY_USERNAME)
    kf = _key_file()
    try:
        kf.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - fs variance
        logger.debug("key file unlink no-op: %s", exc.__class__.__name__)


def _parse_token_blob(raw: str) -> TokenSet | None:
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


# ---------------------------------------------------------------------------
# Public token API (signatures unchanged)
# ---------------------------------------------------------------------------


def save_tokens(tokens: TokenSet) -> None:
    """Persist the token set to the encrypted on-disk store.

    The big OAuth blob goes to ``data/cockpit/.rh_tokens.enc`` (no size
    cap); only the tiny Fernet key touches the OS keychain. This sidesteps
    the Windows Credential Manager 2.5 KB limit that made the old
    single-keyring-blob approach fail with CredWrite 1703.
    """
    payload = json.dumps(tokens.to_dict(), separators=(",", ":"))
    _write_secret(_token_file(), payload)


def load_tokens() -> TokenSet | None:
    """Return the persisted ``TokenSet`` or ``None`` if none is stored.

    Reads the new encrypted file first; on miss, falls back to the legacy
    keyring blob and transparently migrates it to the new store. NEVER
    raises on a missing/corrupt entry -- returns ``None`` so the broker can
    degrade to 'not yet connected to Robinhood'.
    """
    raw = _read_secret(_token_file())
    if raw is not None:
        return _parse_token_blob(raw)

    legacy = _read_old_keyring(KEYRING_USERNAME)
    if not legacy:
        return None
    tokens = _parse_token_blob(legacy)
    if tokens is not None:
        # Transparent migration: write to the new store, then drop the old.
        try:
            save_tokens(tokens)
            _delete_keyring(KEYRING_USERNAME)
        except Exception as exc:  # pragma: no cover - best-effort migration
            logger.warning("token migration failed: %s", exc.__class__.__name__)
    return tokens


def clear_tokens() -> None:
    """Delete the stored tokens. Removes BOTH the encrypted file and the
    legacy keyring blob so 'Disconnect Robinhood' fully resets. Idempotent."""
    try:
        _token_file().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - fs variance
        logger.debug("token file unlink no-op: %s", exc.__class__.__name__)
    _delete_keyring(KEYRING_USERNAME)
    _maybe_remove_key()


# ---------------------------------------------------------------------------
# Pending-auth persistence (in-flight OAuth state, short-lived + single-use)
# ---------------------------------------------------------------------------
#
# The in-flight pending-auth blob (state + PKCE verifier + endpoints) was
# previously held ONLY in a module-level global. The cockpit launcher now
# runs the web server under an auto-restart loop, so the process can restart
# between begin_auth (which stores the global) and the /callback redirect
# (which reads it) -- wiping the global and failing every callback with an
# "OAuth state mismatch". We persist the blob encrypted on disk so it
# survives that restart. This reuses the SAME encrypted-file store as the
# tokens (big ciphertext in file, tiny Fernet key in keyring). It is a
# deliberate, documented tradeoff vs. the OAuth 2.1 "keep the verifier in
# memory only" guidance: the blob is encrypted at rest, 0600, single-use
# (deleted on consumption), and TTL-bounded by the caller (see
# ``robinhood.complete_auth``).


def save_pending_auth(payload: str) -> None:
    """Persist the in-flight OAuth pending-auth blob, encrypted on disk.

    ``payload`` is an opaque JSON string built by ``robinhood.begin_auth``
    (state, code_verifier, client_id, redirect_uri, serialized endpoints,
    created_at). Encrypted at rest with the shared Fernet key, same as the
    token store; never raises on a storage failure path the caller can't
    recover from -- the caller wraps this so connect never crashes."""
    _write_secret(_pending_auth_file(), payload)


def load_pending_auth() -> str | None:
    """Return the persisted pending-auth JSON string, or ``None`` if absent /
    unreadable. NEVER raises -- a missing/corrupt file degrades to ``None`` so
    the callback handler can report 'no auth flow' instead of crashing."""
    return _read_secret(_pending_auth_file())


def clear_pending_auth() -> None:
    """Delete the persisted pending-auth file (single-use consumption + full
    disconnect). Idempotent; drops the shared key if nothing else needs it."""
    try:
        _pending_auth_file().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - fs variance
        logger.debug("pending-auth file unlink no-op: %s", exc.__class__.__name__)
    _maybe_remove_key()


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
    restart don't need to re-register. Uses the same encrypted-file store as
    the tokens (consistent + resilient), not a raw keyring write."""
    _write_secret(_client_id_file(), client_id)


def load_client_id() -> str | None:
    """Load the persisted client_id, or env override, or None.

    Reads the new encrypted file first; on miss, falls back to the legacy
    keyring entry and migrates it forward."""
    env = os.getenv("ROBINHOOD_OAUTH_CLIENT_ID", "")
    if env:
        return env
    raw = _read_secret(_client_id_file())
    if raw:
        return raw
    legacy = _read_old_keyring(KEYRING_CLIENT_ID_USERNAME)
    if not legacy:
        return None
    try:
        save_client_id(legacy)
        _delete_keyring(KEYRING_CLIENT_ID_USERNAME)
    except Exception as exc:  # pragma: no cover - best-effort migration
        logger.warning("client_id migration failed: %s", exc.__class__.__name__)
    return legacy


def clear_client_id() -> None:
    """Remove the stored client_id (part of full disconnect). Removes both
    the encrypted file and the legacy keyring entry. Idempotent."""
    try:
        _client_id_file().unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:  # pragma: no cover - fs variance
        logger.debug("client_id file unlink no-op: %s", exc.__class__.__name__)
    _delete_keyring(KEYRING_CLIENT_ID_USERNAME)
    _maybe_remove_key()
