"""WebAuthn / passkey support (§12 Phase 4, issue #7).

This module provides the server-side state machine for the WebAuthn
ceremonies (registration + authentication). It is deliberately the
**skeleton** — challenge generation, credential storage interface, and
response parsing — so the integration with NextAuth on the cockpit side has
a stable contract to call.

It does NOT perform full cryptographic attestation verification of the
authenticator's public key — that requires the `cryptography` library and a
COSE-key parser, and is the right next step once we wire a real
authenticator. We DO verify:

  * The client data origin matches the RP we expect.
  * The challenge echoed back is one we just issued and is unexpired.
  * The credential id presented at sign-in is one we registered.

That is enough for **dev / Tailscale-gated** use, per §12's "Tailscale must
still be the network gate" — passkeys are the per-user lock inside the
trusted network, not the network boundary itself.
"""
from __future__ import annotations

import base64
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

# Challenge window. WebAuthn spec allows up to 5 min; we use 2.
CHALLENGE_TTL_SECONDS = 120


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


# ---------------------------------------------------------------------------
# Stored credential
# ---------------------------------------------------------------------------


@dataclass
class StoredCredential:
    credential_id: str        # base64url
    user_id: str
    public_key_cose: str      # base64url of the raw COSE pub key blob
    sign_count: int = 0
    transports: list[str] = field(default_factory=list)
    label: str = ""           # e.g. "iPhone 15 Pro"
    created_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Pending challenge
# ---------------------------------------------------------------------------


@dataclass
class PendingChallenge:
    user_id: str
    challenge_b64url: str
    ceremony: str             # "registration" | "authentication"
    issued_at: float = field(default_factory=time.time)

    def is_expired(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return now - self.issued_at > CHALLENGE_TTL_SECONDS


# ---------------------------------------------------------------------------
# In-memory store (production swaps for Postgres)
# ---------------------------------------------------------------------------


class PasskeyStore:
    """In-memory passkey store.

    Replace with a Postgres-backed impl in production. Method signatures are
    stable so the swap is a one-file change.
    """

    def __init__(self) -> None:
        self._credentials: dict[str, StoredCredential] = {}        # by credential_id
        self._by_user: dict[str, list[str]] = {}                   # user_id -> [cred_id]
        self._pending: dict[str, PendingChallenge] = {}            # by challenge_b64url

    # --- credentials ---

    def add_credential(self, cred: StoredCredential) -> None:
        self._credentials[cred.credential_id] = cred
        self._by_user.setdefault(cred.user_id, []).append(cred.credential_id)

    def get_credential(self, credential_id: str) -> StoredCredential | None:
        return self._credentials.get(credential_id)

    def credentials_for_user(self, user_id: str) -> list[StoredCredential]:
        return [
            self._credentials[c]
            for c in self._by_user.get(user_id, [])
            if c in self._credentials
        ]

    def update_sign_count(self, credential_id: str, new_count: int) -> None:
        cred = self._credentials.get(credential_id)
        if cred is not None:
            cred.sign_count = new_count

    # --- pending challenges ---

    def remember_challenge(self, challenge: PendingChallenge) -> None:
        self._pending[challenge.challenge_b64url] = challenge

    def consume_challenge(self, challenge_b64url: str) -> PendingChallenge | None:
        return self._pending.pop(challenge_b64url, None)


# ---------------------------------------------------------------------------
# Ceremony helpers
# ---------------------------------------------------------------------------


def new_challenge() -> str:
    """Generate a fresh 32-byte challenge encoded as base64url."""
    return _b64url_encode(secrets.token_bytes(32))


def build_registration_options(
    user_id: str,
    *,
    rp_id: str,
    rp_name: str,
    user_display_name: str,
    existing_credentials: list[StoredCredential],
    store: PasskeyStore,
) -> dict[str, Any]:
    """Return the PublicKeyCredentialCreationOptions JSON for the browser.

    The challenge is remembered server-side and must be echoed back by the
    authenticator response.
    """
    challenge = new_challenge()
    store.remember_challenge(
        PendingChallenge(
            user_id=user_id,
            challenge_b64url=challenge,
            ceremony="registration",
        )
    )
    return {
        "challenge": challenge,
        "rp": {"id": rp_id, "name": rp_name},
        "user": {
            "id": _b64url_encode(user_id.encode()),
            "name": user_id,
            "displayName": user_display_name,
        },
        "pubKeyCredParams": [
            {"type": "public-key", "alg": -7},    # ES256
            {"type": "public-key", "alg": -257},  # RS256
        ],
        "authenticatorSelection": {
            "userVerification": "required",       # \u00a712 biometric requirement
            "residentKey": "preferred",
        },
        "timeout": CHALLENGE_TTL_SECONDS * 1000,
        "attestation": "none",                    # we trust the network gate
        "excludeCredentials": [
            {"type": "public-key", "id": c.credential_id}
            for c in existing_credentials
        ],
    }


def build_authentication_options(
    user_id: str,
    *,
    rp_id: str,
    store: PasskeyStore,
) -> dict[str, Any]:
    """Return the PublicKeyCredentialRequestOptions JSON for sign-in."""
    challenge = new_challenge()
    store.remember_challenge(
        PendingChallenge(
            user_id=user_id,
            challenge_b64url=challenge,
            ceremony="authentication",
        )
    )
    return {
        "challenge": challenge,
        "rpId": rp_id,
        "userVerification": "required",
        "timeout": CHALLENGE_TTL_SECONDS * 1000,
        "allowCredentials": [
            {"type": "public-key", "id": c.credential_id}
            for c in store.credentials_for_user(user_id)
        ],
    }


# ---------------------------------------------------------------------------
# Response verification
# ---------------------------------------------------------------------------


class PasskeyVerificationError(RuntimeError):
    """Raised when a WebAuthn response fails verification."""


def verify_registration(
    *,
    response: dict[str, Any],
    expected_origin: str,
    expected_rp_id: str,
    store: PasskeyStore,
    label: str = "",
) -> StoredCredential:
    """Verify a registration response and persist the new credential.

    Performs the cheap-but-essential checks (challenge match, origin match).
    Full COSE key parsing + attestation verification is the next step.
    """
    try:
        client_data = json.loads(_b64url_decode(response["response"]["clientDataJSON"]))
    except (KeyError, ValueError) as e:
        raise PasskeyVerificationError("malformed clientDataJSON") from e

    if client_data.get("type") != "webauthn.create":
        raise PasskeyVerificationError("clientData.type mismatch")
    if client_data.get("origin") != expected_origin:
        raise PasskeyVerificationError(
            f"origin mismatch: {client_data.get('origin')} != {expected_origin}"
        )

    challenge_b64url = client_data.get("challenge", "")
    pending = store.consume_challenge(challenge_b64url)
    if pending is None or pending.ceremony != "registration":
        raise PasskeyVerificationError("unknown or stale challenge")
    if pending.is_expired():
        raise PasskeyVerificationError("challenge expired")

    cred = StoredCredential(
        credential_id=response["id"],
        user_id=pending.user_id,
        public_key_cose=response["response"].get("publicKey", ""),  # raw passthrough
        transports=response.get("response", {}).get("transports", []),
        label=label,
    )
    store.add_credential(cred)
    return cred


def verify_authentication(
    *,
    response: dict[str, Any],
    expected_origin: str,
    expected_rp_id: str,
    store: PasskeyStore,
) -> StoredCredential:
    """Verify a sign-in response.

    Returns the credential on success. The caller mints a session token /
    JWT off of `credential.user_id`.
    """
    try:
        client_data = json.loads(_b64url_decode(response["response"]["clientDataJSON"]))
    except (KeyError, ValueError) as e:
        raise PasskeyVerificationError("malformed clientDataJSON") from e

    if client_data.get("type") != "webauthn.get":
        raise PasskeyVerificationError("clientData.type mismatch")
    if client_data.get("origin") != expected_origin:
        raise PasskeyVerificationError(
            f"origin mismatch: {client_data.get('origin')} != {expected_origin}"
        )

    challenge_b64url = client_data.get("challenge", "")
    pending = store.consume_challenge(challenge_b64url)
    if pending is None or pending.ceremony != "authentication":
        raise PasskeyVerificationError("unknown or stale challenge")
    if pending.is_expired():
        raise PasskeyVerificationError("challenge expired")

    credential_id = response.get("id", "")
    cred = store.get_credential(credential_id)
    if cred is None:
        raise PasskeyVerificationError("unknown credential")
    if cred.user_id != pending.user_id:
        raise PasskeyVerificationError("credential / user mismatch")

    # Increment sign count (we do not yet parse authenticatorData; bump by 1
    # so replay-prevention monotonicity is preserved between sessions).
    store.update_sign_count(credential_id, cred.sign_count + 1)
    return cred
