"use client";

import { API_BASE } from "./api";

// --- base64url helpers --------------------------------------------------------

function b64urlToBuffer(s: string): ArrayBuffer {
  const pad = "=".repeat((4 - (s.length % 4)) % 4);
  const b64 = (s + pad).replace(/-/g, "+").replace(/_/g, "/");
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return buf;
}

function bytesToB64url(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

// --- API client (raw fetch so we don't ship a heavy webauthn lib) ------------

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    credentials: "include",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return (await r.json()) as T;
}

// --- WebAuthn option shapes --------------------------------------------------

type RegistrationOptions = {
  challenge: string;
  rp: { id: string; name: string };
  user: { id: string; name: string; displayName: string };
  pubKeyCredParams: PublicKeyCredentialParameters[];
  authenticatorSelection: AuthenticatorSelectionCriteria;
  timeout: number;
  attestation: AttestationConveyancePreference;
  excludeCredentials: { type: string; id: string }[];
};

type AuthenticationOptions = {
  challenge: string;
  rpId: string;
  userVerification: UserVerificationRequirement;
  timeout: number;
  allowCredentials: { type: string; id: string }[];
};

// --- public API --------------------------------------------------------------

/**
 * Register a new passkey. Pops Touch ID / Face ID / Windows Hello / Android
 * biometric. Throws on user cancel.
 */
export async function registerPasskey(label: string): Promise<{ credential_id: string }> {
  const opts = await postJSON<RegistrationOptions>(
    "/auth/passkey/register/options",
    { user_display_name: label || "Operator" },
  );

  const publicKey: PublicKeyCredentialCreationOptions = {
    ...opts,
    challenge: b64urlToBuffer(opts.challenge),
    user: {
      ...opts.user,
      id: b64urlToBuffer(opts.user.id),
    },
    excludeCredentials: opts.excludeCredentials.map((c) => ({
      type: c.type as PublicKeyCredentialType,
      id: b64urlToBuffer(c.id),
    })),
  };

  const cred = (await navigator.credentials.create({ publicKey })) as PublicKeyCredential | null;
  if (!cred) throw new Error("registration cancelled");

  const r = cred.response as AuthenticatorAttestationResponse;
  const response = {
    id: cred.id,
    rawId: bytesToB64url(cred.rawId),
    type: cred.type,
    response: {
      clientDataJSON: bytesToB64url(r.clientDataJSON),
      attestationObject: bytesToB64url(r.attestationObject),
      transports: r.getTransports ? r.getTransports() : [],
    },
  };

  return postJSON<{ credential_id: string }>(
    "/auth/passkey/register/verify",
    { response, label },
  );
}

/**
 * Authenticate with an existing passkey. Returns the session hint on success.
 */
export async function authenticateWithPasskey(): Promise<{ session_hint: string; user_id: string }> {
  const opts = await postJSON<AuthenticationOptions>(
    "/auth/passkey/authenticate/options",
    {},
  );

  const publicKey: PublicKeyCredentialRequestOptions = {
    ...opts,
    challenge: b64urlToBuffer(opts.challenge),
    allowCredentials: opts.allowCredentials.map((c) => ({
      type: c.type as PublicKeyCredentialType,
      id: b64urlToBuffer(c.id),
    })),
  };

  const assertion = (await navigator.credentials.get({ publicKey })) as PublicKeyCredential | null;
  if (!assertion) throw new Error("sign-in cancelled");

  const r = assertion.response as AuthenticatorAssertionResponse;
  const response = {
    id: assertion.id,
    rawId: bytesToB64url(assertion.rawId),
    type: assertion.type,
    response: {
      clientDataJSON: bytesToB64url(r.clientDataJSON),
      authenticatorData: bytesToB64url(r.authenticatorData),
      signature: bytesToB64url(r.signature),
      userHandle: r.userHandle ? bytesToB64url(r.userHandle) : null,
    },
  };

  return postJSON<{ session_hint: string; user_id: string }>(
    "/auth/passkey/authenticate/verify",
    { response },
  );
}

export function isPasskeySupported(): boolean {
  if (typeof window === "undefined") return false;
  return !!window.PublicKeyCredential;
}
