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
def fake_kr(monkeypatch, tmp_path):
    fake = _FakeKeyring()
    monkeypatch.setattr(rt, "_keyring", lambda: fake)
    # Redirect the encrypted on-disk store to a tmp dir so tests stay
    # hermetic and never touch the user's real data/cockpit.
    monkeypatch.setenv("ROBINHOOD_TOKEN_DIR", str(tmp_path))
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
# Encrypted-file storage: the Windows CredWrite 2.5 KB regression
# ---------------------------------------------------------------------------


def test_save_load_round_trip_large_blob(fake_kr):
    """A >2.5 KB token set is the exact case that broke Windows Credential
    Manager (CredWrite 1703). It must now round-trip through the encrypted
    file store and clear cleanly."""
    big = "x" * 3000  # > 2560-byte Windows credential limit on its own
    tokens = rt.TokenSet(
        access_token=big,
        refresh_token="y" * 1500,
        expires_at=98765.0,
        scope="internal",
        token_type="Bearer",
    )
    payload_len = len(
        __import__("json").dumps(tokens.to_dict(), separators=(",", ":"))
    )
    assert payload_len > 2560, "fixture must exceed the Windows cap to be valid"

    rt.save_tokens(tokens)
    loaded = rt.load_tokens()
    assert loaded is not None
    assert loaded.access_token == big
    assert loaded.refresh_token == "y" * 1500
    assert loaded.expires_at == 98765.0

    rt.clear_tokens()
    assert rt.load_tokens() is None


def test_big_blob_is_not_written_to_keyring(fake_kr):
    """The big secret must live in the file; only the tiny key may hit the
    keyring (otherwise we'd reproduce the Windows size failure)."""
    rt.save_tokens(
        rt.TokenSet(access_token="a" * 4000, refresh_token="r", expires_at=1.0)
    )
    # No keyring value may exceed the Windows 2.5 KB credential cap.
    for value in fake_kr.store.values():
        assert len(value) <= 2560
    # The encrypted file must exist on disk.
    assert rt._token_file().exists()


def test_load_migrates_legacy_keyring_blob(fake_kr):
    """Existing mac/linux users stored the blob directly in keyring. We must
    read it, return it, and migrate it to the encrypted file."""
    blob = (
        '{"access_token":"old-a","refresh_token":"old-r",'
        '"expires_at":42.0,"scope":"internal","token_type":"Bearer"}'
    )
    fake_kr.set_password(rt.KEYRING_SERVICE, rt.KEYRING_USERNAME, blob)

    loaded = rt.load_tokens()
    assert loaded is not None
    assert loaded.access_token == "old-a"
    # Migration: legacy keyring entry gone, encrypted file now present.
    assert (
        fake_kr.store.get((rt.KEYRING_SERVICE, rt.KEYRING_USERNAME)) is None
    )
    assert rt._token_file().exists()
    # Subsequent loads come from the file and still match.
    again = rt.load_tokens()
    assert again is not None and again.access_token == "old-a"


def test_save_succeeds_when_keyring_key_write_fails(fake_kr, monkeypatch):
    """A keyring failure (the Windows CredWrite case) must NOT crash connect.
    The key falls back to a 0600 file and the round-trip still works."""

    class _BrokenKeyring(_FakeKeyring):
        def set_password(self, service, username, value):
            raise RuntimeError("(1703, 'CredWrite', 'The stub received bad data')")

        def get_password(self, service, username):
            return None

    monkeypatch.setattr(rt, "_keyring", lambda: _BrokenKeyring())

    tokens = rt.TokenSet(
        access_token="z" * 3000, refresh_token="r", expires_at=7.0
    )
    rt.save_tokens(tokens)  # must not raise
    assert rt._key_file().exists()  # key fell back to a local file
    loaded = rt.load_tokens()
    assert loaded is not None and loaded.access_token == "z" * 3000
    rt.clear_tokens()
    assert rt.load_tokens() is None


# ---------------------------------------------------------------------------
# client_id persistence
# ---------------------------------------------------------------------------


def test_client_id_round_trip(fake_kr):
    rt.save_client_id("client-abc")
    assert rt.load_client_id() == "client-abc"
    rt.clear_client_id()
    assert rt.load_client_id() is None


def test_client_id_env_override_wins(fake_kr, monkeypatch):
    rt.save_client_id("stored-id")
    monkeypatch.setenv("ROBINHOOD_OAUTH_CLIENT_ID", "env-id")
    assert rt.load_client_id() == "env-id"


def test_client_id_migrates_legacy_keyring(fake_kr):
    fake_kr.set_password(
        rt.KEYRING_SERVICE, rt.KEYRING_CLIENT_ID_USERNAME, "legacy-id"
    )
    assert rt.load_client_id() == "legacy-id"
    assert (
        fake_kr.store.get((rt.KEYRING_SERVICE, rt.KEYRING_CLIENT_ID_USERNAME))
        is None
    )
    assert rt._client_id_file().exists()


def test_disconnect_removes_shared_key(fake_kr):
    """After both tokens and client_id are cleared, the shared Fernet key
    must be gone too -- a full reset leaves nothing behind."""
    rt.save_tokens(
        rt.TokenSet(access_token="a", refresh_token="r", expires_at=1.0)
    )
    rt.save_client_id("cid")
    rt.clear_tokens()
    rt.clear_client_id()
    assert (
        fake_kr.store.get((rt.KEYRING_SERVICE, rt.KEYRING_ENC_KEY_USERNAME))
        is None
    )
    assert not rt._token_file().exists()
    assert not rt._client_id_file().exists()


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
