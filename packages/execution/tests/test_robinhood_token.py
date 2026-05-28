"""Tests for Robinhood OAuth token storage + PKCE helpers.

We use the in-memory keyring backend (keyrings.alt is bundled with the
``keyring`` package as ``keyring.backends.fail.Keyring`` for headless
envs) and just patch ``keyring`` directly to a dict-backed fake. That
keeps the test hermetic -- no OS keychain access, no env pollution.
"""

from __future__ import annotations

import time

import pytest

from packages.execution import robinhood_token as rt


class _FakeKeyring:
    """Minimal in-memory replacement for the ``keyring`` module surface
    we actually call. Behaves like Windows Credential Manager / macOS
    Keychain / SecretService from the caller's perspective."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, value: str) -> None:
        self.store[(service, username)] = value

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        # Real keyring raises PasswordDeleteError on missing entries;
        # robinhood_token.clear_tokens swallows that so we mirror the
        # error here to confirm the swallow works.
        if (service, username) not in self.store:
            raise RuntimeError("password not found")
        del self.store[(service, username)]


@pytest.fixture
def fake_kr(monkeypatch):
    fake = _FakeKeyring()
    monkeypatch.setattr(rt, "_keyring", lambda: fake)
    return fake


# ---------------------------------------------------------------------------
# TokenSet
# ---------------------------------------------------------------------------


def test_tokenset_is_stale_respects_slack():
    """is_stale() must return True within EXPIRY_SLACK_S of expiry so the
    broker proactively refreshes instead of racing a mid-flight expiry."""
    now = 1_000_000.0
    just_past_slack = rt.TokenSet(
        access_token="a",
        refresh_token="r",
        expires_at=now + rt.EXPIRY_SLACK_S + 10,
    )
    inside_slack = rt.TokenSet(
        access_token="a",
        refresh_token="r",
        expires_at=now + rt.EXPIRY_SLACK_S - 10,
    )
    expired = rt.TokenSet(
        access_token="a", refresh_token="r", expires_at=now - 60
    )

    assert just_past_slack.is_stale(now=now) is False
    assert inside_slack.is_stale(now=now) is True
    assert expired.is_stale(now=now) is True


def test_tokenset_is_stale_uses_real_clock_when_none():
    """Default ``now=None`` must consult ``time.time()`` -- regression
    check that we didn't accidentally bind the import time."""
    tokens = rt.TokenSet(
        access_token="a", refresh_token="r", expires_at=time.time() - 10
    )
    assert tokens.is_stale() is True


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(fake_kr):
    tokens = rt.TokenSet(
        access_token="access-123",
        refresh_token="refresh-xyz",
        expires_at=12345.0,
        scope="trade read",
        token_type="Bearer",
    )
    rt.save_tokens(tokens)

    loaded = rt.load_tokens()
    assert loaded is not None
    assert loaded.access_token == "access-123"
    assert loaded.refresh_token == "refresh-xyz"
    assert loaded.expires_at == 12345.0
    assert loaded.scope == "trade read"
    assert loaded.token_type == "Bearer"


def test_load_returns_none_when_missing(fake_kr):
    assert rt.load_tokens() is None


def test_load_returns_none_on_malformed_blob(fake_kr):
    """A corrupt entry must NOT crash the broker -- we return None so the
    user gets a 'reconnect Robinhood' prompt instead of a stack trace."""
    fake_kr.set_password(rt.KEYRING_SERVICE, rt.KEYRING_USERNAME, "not json")
    assert rt.load_tokens() is None


def test_load_returns_none_on_missing_required_field(fake_kr):
    """KeyError on a missing field must produce None, not propagate."""
    fake_kr.set_password(
        rt.KEYRING_SERVICE,
        rt.KEYRING_USERNAME,
        '{"access_token": "a"}',  # missing refresh + expires_at
    )
    assert rt.load_tokens() is None


def test_clear_tokens_removes_entry(fake_kr):
    rt.save_tokens(
        rt.TokenSet(access_token="a", refresh_token="r", expires_at=1.0)
    )
    rt.clear_tokens()
    assert rt.load_tokens() is None


def test_clear_tokens_is_idempotent(fake_kr):
    """Calling clear on an empty store must not raise -- the
    'Disconnect Robinhood' button should always succeed."""
    rt.clear_tokens()  # no entry stored
    rt.clear_tokens()  # second call also fine


# ---------------------------------------------------------------------------
# PKCE
# ---------------------------------------------------------------------------


def test_new_pkce_pair_shape():
    """RFC 7636: verifier is 43-128 unreserved chars; challenge is
    base64url-no-padding of sha256(verifier)."""
    verifier, challenge = rt.new_pkce_pair()
    assert 43 <= len(verifier) <= 128
    # Our verifier is hex.
    int(verifier, 16)
    # Challenge is base64url-no-padding: no '=' and no '+' or '/'.
    assert "=" not in challenge
    assert "+" not in challenge
    assert "/" not in challenge
    # sha256 -> 32 bytes -> base64url length 43 (no padding).
    assert len(challenge) == 43


def test_new_pkce_pair_unique_per_call():
    """Each authorize flow must use a fresh verifier -- reusing one
    voids the security guarantee."""
    a_v, a_c = rt.new_pkce_pair()
    b_v, b_c = rt.new_pkce_pair()
    assert a_v != b_v
    assert a_c != b_c


def test_new_state_returns_nonempty_unique():
    s1 = rt.new_state()
    s2 = rt.new_state()
    assert s1 and s2
    assert s1 != s2
    # Default 16 bytes -> base64url-no-padding length 22.
    assert len(s1) == 22
