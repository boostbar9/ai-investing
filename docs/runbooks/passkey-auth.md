# Passkey biometric login — runbook

> Source spec: §12 Phase 4 · Issue [#7](https://github.com/boostbar9/iaa-investing/issues/7)

## Threat model

Tailscale is the network gate. The cockpit and API are not exposed to the
public internet. Passkeys are the **per-user lock inside** the trusted
network — they prevent a borrowed phone or shared laptop from impersonating
the operator. They are not, by themselves, the network boundary.

## Endpoints

| Verb | Path                                | Purpose                                  |
|------|-------------------------------------|------------------------------------------|
| POST | `/auth/passkey/register/options`    | Issue a registration challenge           |
| POST | `/auth/passkey/register/verify`     | Verify + persist the new credential      |
| POST | `/auth/passkey/authenticate/options`| Issue a sign-in challenge                |
| POST | `/auth/passkey/authenticate/verify` | Verify a sign-in assertion + mint session|

All four endpoints look up the operator identity from the `X-Tailscale-User`
header set by the tsnsrv reverse proxy. In dev the header is absent and the
server falls back to `WEBAUTHN_DEV_USER` (default `devin`).

## Environment

| Variable                 | Required for      | Default                  |
|--------------------------|-------------------|--------------------------|
| `WEBAUTHN_RP_ID`         | prd               | `localhost`              |
| `WEBAUTHN_ORIGIN`        | prd               | `http://localhost:3000`  |
| `WEBAUTHN_DEV_USER`      | dev only          | `devin`                  |

`WEBAUTHN_RP_ID` must equal the *registrable domain* of the cockpit URL —
e.g. `cockpit.ts.net` if the Tailscale hostname is `cockpit.tailnet.ts.net`.

`WEBAUTHN_ORIGIN` must be the **full scheme + host + port** the browser
sees — e.g. `https://cockpit.tailnet.ts.net`. Mismatch → 400 from the
verify endpoint.

## Register a new device (operator flow)

1. Visit `/signin` in the cockpit (the Tailscale-gated URL).
2. Type a device label (e.g. `iPhone 15 Pro`).
3. Tap **Register passkey**. Touch ID / Face ID / Windows Hello prompts.
4. On success the page shows the truncated credential id.
5. To verify the registration is sticky, tap **Sign in with passkey** —
   biometric prompts again and you're signed in.

## Lost device

The current skeleton stores credentials in-process. In production, with the
Postgres-backed `PasskeyStore`, the operator (you) can issue a recovery
flow:

```sql
DELETE FROM webauthn_credentials WHERE label = 'iPhone 15 Pro';
```

…then re-register on the replacement device. There is intentionally no
"email me a magic link" recovery — that would defeat the purpose of the
biometric lock.

## What this skeleton does NOT yet do

- Full cryptographic attestation verification (parse COSE pub key, verify
  signature on `authenticatorData`). The §12 ticket #7 calls this out as
  future work — the next iteration depends on the `cryptography` lib and a
  COSE-key parser.
- Session token minting. The verify endpoint returns a `session_hint` UUID
  that the cockpit's NextAuth handler can stash in an HttpOnly cookie. The
  real signed-session impl lands with NextAuth integration.
- Postgres-backed `PasskeyStore`. The interface is stable; swapping is a
  one-file change.

These are tracked as a follow-up; **what ships today** is the full ceremony
plumbing end-to-end so Touch ID actually pops on a real device behind
Tailscale.
