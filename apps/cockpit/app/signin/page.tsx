"use client";

import { useEffect, useState } from "react";
import {
  authenticateWithPasskey,
  isPasskeySupported,
  registerPasskey,
} from "../components/usePasskey";

type Status =
  | { kind: "idle" }
  | { kind: "working"; what: string }
  | { kind: "ok"; msg: string }
  | { kind: "err"; msg: string };

export default function SignInPage() {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [label, setLabel] = useState("");

  useEffect(() => {
    setSupported(isPasskeySupported());
  }, []);

  async function onRegister() {
    setStatus({ kind: "working", what: "Creating passkey…" });
    try {
      const out = await registerPasskey(label || "Default device");
      setStatus({
        kind: "ok",
        msg: `Registered ${out.credential_id.slice(0, 8)}…`,
      });
    } catch (e) {
      setStatus({ kind: "err", msg: (e as Error).message });
    }
  }

  async function onSignIn() {
    setStatus({ kind: "working", what: "Verifying biometric…" });
    try {
      const out = await authenticateWithPasskey();
      setStatus({ kind: "ok", msg: `Signed in as ${out.user_id}` });
    } catch (e) {
      setStatus({ kind: "err", msg: (e as Error).message });
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 flex items-center justify-center p-6">
      <div className="w-full max-w-md rounded-2xl bg-neutral-900 p-6 space-y-5">
        <div className="space-y-1">
          <h1 className="text-xl font-semibold tracking-tight">ai-investing</h1>
          <p className="text-xs text-neutral-400">
            Sign in with biometric. Tailscale gates the network; passkeys
            unlock the cockpit.
          </p>
        </div>

        {supported === false && (
          <div className="rounded-xl bg-red-900/30 border border-red-800 text-red-200 text-sm p-3">
            This browser does not support WebAuthn. Use Safari iOS, Chrome,
            Edge, or Firefox.
          </div>
        )}

        <button
          onClick={onSignIn}
          disabled={supported === false || status.kind === "working"}
          className="w-full rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-neutral-700 disabled:text-neutral-400 px-4 py-3 text-sm font-medium transition"
        >
          {status.kind === "working" && status.what.startsWith("Verifying")
            ? "Verifying…"
            : "Sign in with passkey"}
        </button>

        <div className="border-t border-neutral-800" />

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-wider text-neutral-400">
            Register a new device
          </div>
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="e.g. iPhone 15 Pro"
            className="w-full rounded-xl bg-neutral-800 px-3 py-2 text-sm placeholder:text-neutral-500"
          />
          <button
            onClick={onRegister}
            disabled={supported === false || status.kind === "working"}
            className="w-full rounded-xl bg-neutral-800 hover:bg-neutral-700 disabled:bg-neutral-800/40 disabled:text-neutral-500 px-4 py-3 text-sm font-medium transition"
          >
            {status.kind === "working" && status.what.startsWith("Creating")
              ? "Creating…"
              : "Register passkey"}
          </button>
        </div>

        {status.kind === "ok" && (
          <div className="rounded-xl bg-emerald-900/30 border border-emerald-800 text-emerald-200 text-sm p-3">
            {status.msg}
          </div>
        )}
        {status.kind === "err" && (
          <div className="rounded-xl bg-red-900/30 border border-red-800 text-red-200 text-sm p-3">
            {status.msg}
          </div>
        )}
      </div>
    </main>
  );
}
