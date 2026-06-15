"""Tests for the Robinhood OAuth 2.1 + PKCE wiring.

Covers the pieces that turn "Connect your agent" from a stub into a
working flow:

  * Endpoint discovery via RFC 9728 / RFC 8414 well-known metadata
    (including the protected-resource -> authorization-server hop and the
    OIDC fallback).
  * Dynamic client registration (RFC 7591) + the preset/env fallback.
  * Authorization-URL construction (PKCE S256 + resource indicator).
  * Authorization-code exchange and refresh-token rotation.
  * The begin_auth -> complete_auth happy path + the CSRF state guard.
  * Broker auto-refresh of a stale access token.

Everything mocks httpx + keyring so the suite is hermetic -- no network,
no OS keychain.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from packages.execution import robinhood as rh
from packages.execution import robinhood_token as rt
from packages.execution.broker import BrokerError
from packages.execution.modes import ExecutionMode
from packages.execution.robinhood_token import OAuthEndpoints, TokenSet

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeKeyring:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, s: str, u: str, v: str) -> None:
        self.store[(s, u)] = v

    def get_password(self, s: str, u: str) -> str | None:
        return self.store.get((s, u))

    def delete_password(self, s: str, u: str) -> None:
        if (s, u) not in self.store:
            raise RuntimeError("not found")
        del self.store[(s, u)]


@pytest.fixture
def fake_kr(monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setattr(rt, "_keyring", lambda: fake)
    return fake


def _json_response(status: int, payload):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = "" if payload is None else str(payload)
    return r


def _endpoints() -> OAuthEndpoints:
    return OAuthEndpoints(
        issuer="https://agent.robinhood.com",
        authorization_endpoint="https://agent.robinhood.com/oauth/authorize",
        token_endpoint="https://agent.robinhood.com/oauth/token",
        registration_endpoint="https://agent.robinhood.com/oauth/register",
    )


# ---------------------------------------------------------------------------
# discover_endpoints
# ---------------------------------------------------------------------------


def test_discover_via_protected_resource_metadata():
    """PRM doc points at an authorization server; we follow it to AS
    metadata and pull the endpoints out."""
    prm = {"authorization_servers": ["https://auth.robinhood.com"]}
    as_meta = {
        "issuer": "https://auth.robinhood.com",
        "authorization_endpoint": "https://auth.robinhood.com/authorize",
        "token_endpoint": "https://auth.robinhood.com/token",
        "registration_endpoint": "https://auth.robinhood.com/register",
    }

    def fake_get(url, headers=None):
        if url.endswith("/.well-known/oauth-protected-resource"):
            return _json_response(200, prm)
        if url.endswith("/.well-known/oauth-authorization-server"):
            # First hit is on the AS base, second (the direct attempt)
            # would also match -- both fine, return AS meta.
            return _json_response(200, as_meta)
        return _json_response(404, None)

    client = MagicMock()
    client.get.side_effect = fake_get

    eps = rt.discover_endpoints(client=client)
    assert eps.authorization_endpoint == "https://auth.robinhood.com/authorize"
    assert eps.token_endpoint == "https://auth.robinhood.com/token"
    assert eps.registration_endpoint == "https://auth.robinhood.com/register"


def test_discover_via_direct_authorization_server_metadata():
    """No PRM doc; fall back to AS metadata directly on the oauth base."""
    as_meta = {
        "issuer": "https://agent.robinhood.com",
        "authorization_endpoint": "https://agent.robinhood.com/oauth/authorize",
        "token_endpoint": "https://agent.robinhood.com/oauth/token",
    }

    def fake_get(url, headers=None):
        if url.endswith("/oauth-protected-resource"):
            return _json_response(404, None)
        if url.endswith("/oauth-authorization-server"):
            return _json_response(200, as_meta)
        return _json_response(404, None)

    client = MagicMock()
    client.get.side_effect = fake_get

    eps = rt.discover_endpoints(client=client)
    assert eps.token_endpoint.endswith("/oauth/token")
    assert eps.registration_endpoint == ""  # absent in this server


def test_discover_via_openid_configuration_fallback():
    oidc = {
        "issuer": "https://agent.robinhood.com",
        "authorization_endpoint": "https://agent.robinhood.com/authorize",
        "token_endpoint": "https://agent.robinhood.com/token",
    }

    def fake_get(url, headers=None):
        if url.endswith("/openid-configuration"):
            return _json_response(200, oidc)
        return _json_response(404, None)

    client = MagicMock()
    client.get.side_effect = fake_get
    eps = rt.discover_endpoints(client=client)
    assert eps.authorization_endpoint.endswith("/authorize")


def test_discover_raises_when_nothing_found():
    client = MagicMock()
    client.get.return_value = _json_response(404, None)
    with pytest.raises(rt.OAuthError):
        rt.discover_endpoints(client=client)


def test_discover_matches_live_robinhood_layout():
    """Regression for the real agent.robinhood.com contract (verified live).

    PRM advertises the AS as the MCP URL itself
    (``https://agent.robinhood.com/mcp/trading``). The AS metadata is served
    ONLY at the RFC 8414 *path-inserted* well-known URL
    (``/.well-known/oauth-authorization-server/mcp/trading``); the
    path-suffixed form (``/mcp/trading/.well-known/...``) returns 404.
    Discovery must follow the RFC 8414 layout, not the path-suffixed one.
    """
    mcp = "https://agent.robinhood.com/mcp/trading"
    prm = {
        "authorization_servers": [mcp],
        "resource": mcp,
        "scopes_supported": ["internal"],
    }
    as_meta = {
        "issuer": mcp,
        "authorization_endpoint": "https://robinhood.com/oauth",
        "token_endpoint": "https://api.robinhood.com/oauth2/token/",
        "registration_endpoint": "https://agent.robinhood.com/oauth/trading/register",
        "scopes_supported": ["internal"],
        "code_challenge_methods_supported": ["S256"],
    }
    rfc8414_url = (
        "https://agent.robinhood.com/.well-known/"
        "oauth-authorization-server/mcp/trading"
    )
    path_suffixed_url = (
        "https://agent.robinhood.com/mcp/trading/.well-known/"
        "oauth-authorization-server"
    )
    seen: list[str] = []

    def fake_get(url, headers=None):
        seen.append(url)
        if url.endswith("/.well-known/oauth-protected-resource"):
            return _json_response(200, prm)
        if url == rfc8414_url:
            return _json_response(200, as_meta)
        # Everything else (incl. the path-suffixed form) 404s, mirroring prod.
        return _json_response(404, None)

    client = MagicMock()
    client.get.side_effect = fake_get

    eps = rt.discover_endpoints(client=client)
    assert eps.authorization_endpoint == "https://robinhood.com/oauth"
    assert eps.token_endpoint == "https://api.robinhood.com/oauth2/token/"
    assert (
        eps.registration_endpoint
        == "https://agent.robinhood.com/oauth/trading/register"
    )
    # We must have probed the RFC 8414 path-inserted URL.
    assert rfc8414_url in seen
    # The path-suffixed form must NOT be the one that resolved (it 404s in
    # prod); discovery should succeed without depending on it.
    assert eps.token_endpoint != path_suffixed_url


# ---------------------------------------------------------------------------
# register_client
# ---------------------------------------------------------------------------


def test_register_client_dynamic():
    client = MagicMock()
    client.post.return_value = _json_response(201, {"client_id": "cid-123"})
    cid = rt.register_client(_endpoints(), client=client)
    assert cid == "cid-123"
    # Posted a native public-client registration with our redirect.
    _args, kwargs = client.post.call_args
    body = kwargs["json"]
    assert body["token_endpoint_auth_method"] == "none"
    assert body["redirect_uris"] == [rt.RH_OAUTH_REDIRECT_URI]


def test_register_client_uses_env_preset_when_no_endpoint(monkeypatch):
    monkeypatch.setenv("ROBINHOOD_OAUTH_CLIENT_ID", "preset-cid")
    eps = OAuthEndpoints(
        issuer="x",
        authorization_endpoint="x",
        token_endpoint="x",
        registration_endpoint="",
    )
    assert rt.register_client(eps) == "preset-cid"


def test_register_client_raises_without_endpoint_or_preset(monkeypatch):
    monkeypatch.delenv("ROBINHOOD_OAUTH_CLIENT_ID", raising=False)
    eps = OAuthEndpoints(
        issuer="x",
        authorization_endpoint="x",
        token_endpoint="x",
        registration_endpoint="",
    )
    with pytest.raises(rt.OAuthError):
        rt.register_client(eps)


# ---------------------------------------------------------------------------
# build_authorize_url
# ---------------------------------------------------------------------------


def test_build_authorize_url_has_pkce_and_resource():
    url = rt.build_authorize_url(
        _endpoints(),
        client_id="cid",
        code_challenge="chal",
        state="st",
    )
    assert "response_type=code" in url
    assert "code_challenge=chal" in url
    assert "code_challenge_method=S256" in url
    assert "client_id=cid" in url
    assert "state=st" in url
    assert "resource=" in url


def test_build_authorize_url_uses_internal_scope():
    """Robinhood's AS advertises only the ``internal`` scope (verified live).
    Requesting ``trade read`` would be rejected."""
    assert rt.RH_OAUTH_SCOPE == "internal"
    url = rt.build_authorize_url(
        _endpoints(),
        client_id="cid",
        code_challenge="chal",
        state="st",
    )
    assert "scope=internal" in url
    assert "trade" not in url


# ---------------------------------------------------------------------------
# exchange_code / refresh_access_token
# ---------------------------------------------------------------------------


def test_exchange_code_returns_tokenset():
    client = MagicMock()
    client.post.return_value = _json_response(
        200,
        {
            "access_token": "acc",
            "refresh_token": "ref",
            "expires_in": 3600,
            "scope": "trade read",
            "token_type": "Bearer",
        },
    )
    tokens = rt.exchange_code(
        _endpoints(),
        code="thecode",
        code_verifier="ver",
        client_id="cid",
        client=client,
    )
    assert tokens.access_token == "acc"
    assert tokens.refresh_token == "ref"
    assert tokens.expires_at > time.time()
    # Sent the auth-code grant with the verifier (PKCE).
    _, kwargs = client.post.call_args
    form = kwargs["data"]
    assert form["grant_type"] == "authorization_code"
    assert form["code_verifier"] == "ver"


def test_exchange_code_raises_on_http_error():
    client = MagicMock()
    client.post.return_value = _json_response(400, {"error": "invalid_grant"})
    with pytest.raises(rt.OAuthError):
        rt.exchange_code(
            _endpoints(),
            code="x",
            code_verifier="v",
            client_id="c",
            client=client,
        )


def test_refresh_preserves_old_refresh_token_when_absent():
    """OAuth 2.1 usually rotates the refresh token, but if the server
    omits it we must carry the old one forward so the next refresh works."""
    client = MagicMock()
    client.post.return_value = _json_response(
        200, {"access_token": "new-acc", "expires_in": 3600}
    )
    tokens = rt.refresh_access_token(
        "old-ref", client_id="cid", endpoints=_endpoints(), client=client
    )
    assert tokens.access_token == "new-acc"
    assert tokens.refresh_token == "old-ref"


def test_refresh_uses_rotated_refresh_token_when_present():
    client = MagicMock()
    client.post.return_value = _json_response(
        200,
        {
            "access_token": "new-acc",
            "refresh_token": "rotated-ref",
            "expires_in": 3600,
        },
    )
    tokens = rt.refresh_access_token(
        "old-ref", client_id="cid", endpoints=_endpoints(), client=client
    )
    assert tokens.refresh_token == "rotated-ref"


# ---------------------------------------------------------------------------
# client_id persistence
# ---------------------------------------------------------------------------


def test_client_id_round_trip(fake_kr, monkeypatch):
    monkeypatch.delenv("ROBINHOOD_OAUTH_CLIENT_ID", raising=False)
    assert rt.load_client_id() is None
    rt.save_client_id("cid-xyz")
    assert rt.load_client_id() == "cid-xyz"
    rt.clear_client_id()
    assert rt.load_client_id() is None


def test_client_id_env_overrides_keyring(fake_kr, monkeypatch):
    rt.save_client_id("keyring-cid")
    monkeypatch.setenv("ROBINHOOD_OAUTH_CLIENT_ID", "env-cid")
    assert rt.load_client_id() == "env-cid"


# ---------------------------------------------------------------------------
# begin_auth / complete_auth (broker layer)
# ---------------------------------------------------------------------------


def test_begin_and_complete_auth_happy_path(fake_kr, monkeypatch):
    monkeypatch.delenv("ROBINHOOD_OAUTH_CLIENT_ID", raising=False)
    eps = _endpoints()
    monkeypatch.setattr(rh, "discover_endpoints", lambda: eps)
    monkeypatch.setattr(
        rh, "register_client", lambda endpoints, redirect_uri=None: "cid"
    )

    pending = rh.begin_auth()
    assert pending.client_id == "cid"
    assert "code_challenge_method=S256" in pending.authorize_url
    assert rh.pending_auth() is pending

    # Exchange returns a fresh token set.
    fresh = TokenSet(
        access_token="acc",
        refresh_token="ref",
        expires_at=time.time() + 3600,
    )
    monkeypatch.setattr(
        rh,
        "exchange_code",
        lambda endpoints, *, code, code_verifier, client_id, redirect_uri=None: fresh,
    )

    tokens = rh.complete_auth(code="thecode", state=pending.state)
    assert tokens.access_token == "acc"
    # Tokens + client_id persisted; pending flow consumed.
    assert rt.load_tokens().access_token == "acc"
    assert rt.load_client_id() == "cid"
    assert rh.pending_auth() is None
    assert rh.is_connected() is True


def test_complete_auth_rejects_state_mismatch(fake_kr, monkeypatch):
    eps = _endpoints()
    monkeypatch.setattr(rh, "discover_endpoints", lambda: eps)
    monkeypatch.setattr(
        rh, "register_client", lambda endpoints, redirect_uri=None: "cid"
    )
    rh.begin_auth()
    with pytest.raises(BrokerError, match="state mismatch"):
        rh.complete_auth(code="c", state="WRONG")
    # Pending flow preserved on mismatch (could be a stray callback).
    assert rh.pending_auth() is not None


def test_complete_auth_without_pending_raises(monkeypatch):
    monkeypatch.setattr(rh, "_PENDING_AUTH", None)
    with pytest.raises(BrokerError, match="no auth flow"):
        rh.complete_auth(code="c", state="s")


def test_begin_auth_wraps_discovery_failure(monkeypatch):
    def boom():
        raise rt.OAuthError("no metadata")

    monkeypatch.setattr(rh, "discover_endpoints", boom)
    with pytest.raises(BrokerError, match="could not start auth flow"):
        rh.begin_auth()


# ---------------------------------------------------------------------------
# disconnect / is_connected
# ---------------------------------------------------------------------------


def test_disconnect_wipes_everything(fake_kr):
    rt.save_tokens(
        TokenSet(access_token="a", refresh_token="r", expires_at=time.time() + 99)
    )
    rt.save_client_id("cid")
    assert rh.is_connected() is True
    rh.disconnect()
    assert rt.load_tokens() is None
    assert rh.is_connected() is False


def test_is_connected_false_when_stale_and_no_refresh(fake_kr):
    rt.save_tokens(
        TokenSet(access_token="a", refresh_token="", expires_at=time.time() - 10)
    )
    assert rh.is_connected() is False


# ---------------------------------------------------------------------------
# Broker auto-refresh of a stale access token
# ---------------------------------------------------------------------------


def test_require_token_auto_refreshes_stale(fake_kr, monkeypatch):
    monkeypatch.delenv("ROBINHOOD_OAUTH_CLIENT_ID", raising=False)
    rt.save_client_id("cid")
    stale = TokenSet(
        access_token="old",
        refresh_token="ref",
        expires_at=time.time() - 10,  # already expired
    )
    fresh = TokenSet(
        access_token="new",
        refresh_token="ref2",
        expires_at=time.time() + 3600,
    )
    monkeypatch.setattr(
        rh,
        "refresh_access_token",
        lambda refresh_token, *, client_id: fresh,
    )
    broker = rh.RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW, token_loader=lambda: stale
    )
    got = broker._require_token()
    assert got.access_token == "new"
    # Persisted the rotated set.
    assert rt.load_tokens().access_token == "new"


def test_require_token_refresh_failure_raises_reconnect(fake_kr, monkeypatch):
    rt.save_client_id("cid")
    stale = TokenSet(
        access_token="old", refresh_token="ref", expires_at=time.time() - 10
    )

    def boom(refresh_token, *, client_id):
        raise rt.OAuthError("revoked")

    monkeypatch.setattr(rh, "refresh_access_token", boom)
    broker = rh.RobinhoodAgenticBroker(
        mode=ExecutionMode.SHADOW, token_loader=lambda: stale
    )
    with pytest.raises(BrokerError, match="reconnect"):
        broker._require_token()
