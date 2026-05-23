"""Tests for the passkey state machine (#7)."""
from __future__ import annotations

import base64
import json
import time

import pytest

from packages.shared.passkeys import (
    CHALLENGE_TTL_SECONDS,
    PasskeyStore,
    PasskeyVerificationError,
    build_authentication_options,
    build_registration_options,
    verify_authentication,
    verify_registration,
)

RP_ID = "cockpit.local"
ORIGIN = "https://cockpit.local"


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _client_data(ctype: str, challenge: str, origin: str = ORIGIN) -> str:
    return _b64url(
        json.dumps(
            {"type": ctype, "challenge": challenge, "origin": origin}
        ).encode()
    )


def test_registration_options_includes_challenge_and_user():
    store = PasskeyStore()
    opts = build_registration_options(
        "devin",
        rp_id=RP_ID,
        rp_name="ai-investing cockpit",
        user_display_name="Devin",
        existing_credentials=[],
        store=store,
    )
    assert opts["rp"]["id"] == RP_ID
    assert opts["user"]["name"] == "devin"
    assert opts["challenge"]
    assert opts["authenticatorSelection"]["userVerification"] == "required"


def test_full_register_then_authenticate_round_trip():
    store = PasskeyStore()

    # 1. Server issues registration options
    reg_opts = build_registration_options(
        "devin",
        rp_id=RP_ID,
        rp_name="ai-investing cockpit",
        user_display_name="Devin",
        existing_credentials=[],
        store=store,
    )

    # 2. Browser returns a registration response (we forge the response shape)
    reg_response = {
        "id": "cred-abc",
        "rawId": "cred-abc",
        "type": "public-key",
        "response": {
            "clientDataJSON": _client_data("webauthn.create", reg_opts["challenge"]),
            "publicKey": "pk-cose-blob-b64url",
            "transports": ["internal"],
        },
    }
    cred = verify_registration(
        response=reg_response,
        expected_origin=ORIGIN,
        expected_rp_id=RP_ID,
        store=store,
        label="iPhone 15 Pro",
    )
    assert cred.credential_id == "cred-abc"
    assert cred.label == "iPhone 15 Pro"
    assert store.credentials_for_user("devin") == [cred]

    # 3. Server issues authentication options
    auth_opts = build_authentication_options("devin", rp_id=RP_ID, store=store)
    assert any(c["id"] == "cred-abc" for c in auth_opts["allowCredentials"])

    # 4. Browser returns a get-assertion response
    auth_response = {
        "id": "cred-abc",
        "rawId": "cred-abc",
        "type": "public-key",
        "response": {
            "clientDataJSON": _client_data("webauthn.get", auth_opts["challenge"]),
        },
    }
    signed_in = verify_authentication(
        response=auth_response,
        expected_origin=ORIGIN,
        expected_rp_id=RP_ID,
        store=store,
    )
    assert signed_in.user_id == "devin"
    assert signed_in.sign_count == 1  # monotonic bump


def test_registration_rejects_wrong_origin():
    store = PasskeyStore()
    reg_opts = build_registration_options(
        "devin",
        rp_id=RP_ID,
        rp_name="x",
        user_display_name="Devin",
        existing_credentials=[],
        store=store,
    )
    response = {
        "id": "cred-1",
        "rawId": "cred-1",
        "type": "public-key",
        "response": {
            "clientDataJSON": _client_data(
                "webauthn.create", reg_opts["challenge"], origin="https://evil.test"
            ),
            "publicKey": "pk",
        },
    }
    with pytest.raises(PasskeyVerificationError, match="origin mismatch"):
        verify_registration(
            response=response,
            expected_origin=ORIGIN,
            expected_rp_id=RP_ID,
            store=store,
        )


def test_registration_rejects_unknown_challenge():
    store = PasskeyStore()
    response = {
        "id": "cred-1",
        "rawId": "cred-1",
        "type": "public-key",
        "response": {
            "clientDataJSON": _client_data("webauthn.create", "never-issued"),
            "publicKey": "pk",
        },
    }
    with pytest.raises(PasskeyVerificationError, match="stale challenge"):
        verify_registration(
            response=response,
            expected_origin=ORIGIN,
            expected_rp_id=RP_ID,
            store=store,
        )


def test_authentication_rejects_expired_challenge(monkeypatch):
    store = PasskeyStore()
    reg_opts = build_registration_options(
        "devin",
        rp_id=RP_ID,
        rp_name="x",
        user_display_name="Devin",
        existing_credentials=[],
        store=store,
    )
    verify_registration(
        response={
            "id": "cred-x",
            "rawId": "cred-x",
            "type": "public-key",
            "response": {
                "clientDataJSON": _client_data(
                    "webauthn.create", reg_opts["challenge"]
                ),
                "publicKey": "pk",
            },
        },
        expected_origin=ORIGIN,
        expected_rp_id=RP_ID,
        store=store,
    )

    auth_opts = build_authentication_options("devin", rp_id=RP_ID, store=store)

    # Skip past the TTL.
    real_time = time.time
    monkeypatch.setattr(
        "packages.shared.passkeys.time.time",
        lambda: real_time() + CHALLENGE_TTL_SECONDS + 1,
    )

    response = {
        "id": "cred-x",
        "rawId": "cred-x",
        "type": "public-key",
        "response": {
            "clientDataJSON": _client_data("webauthn.get", auth_opts["challenge"]),
        },
    }
    with pytest.raises(PasskeyVerificationError, match="challenge expired"):
        verify_authentication(
            response=response,
            expected_origin=ORIGIN,
            expected_rp_id=RP_ID,
            store=store,
        )


def test_authentication_rejects_unknown_credential():
    store = PasskeyStore()
    # Issue an auth challenge for a user with NO registered credentials.
    auth_opts = build_authentication_options("ghost", rp_id=RP_ID, store=store)
    response = {
        "id": "cred-nope",
        "rawId": "cred-nope",
        "type": "public-key",
        "response": {
            "clientDataJSON": _client_data("webauthn.get", auth_opts["challenge"]),
        },
    }
    with pytest.raises(PasskeyVerificationError, match="unknown credential"):
        verify_authentication(
            response=response,
            expected_origin=ORIGIN,
            expected_rp_id=RP_ID,
            store=store,
        )
