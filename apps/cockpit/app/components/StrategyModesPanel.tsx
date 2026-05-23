"use client";

import { useEffect, useState } from "react";
import { getJSON, postJSON } from "./api";

type Mode = "paper" | "shadow" | "live";

type ModesPayload = {
  modes: Record<string, Mode>;
  available: Mode[];
};

type AccountPayload = {
  broker: string;
  equity: string | null;
  cash: string | null;
  buying_power: string | null;
  status: string | null;
};

const MODE_COLORS: Record<Mode, string> = {
  paper: "#3b82f6",   // blue
  shadow: "#9ca3af",  // gray
  live: "#22c55e",    // green
};

const MODE_HELP: Record<Mode, string> = {
  paper: "Trades on the fake account. Counts toward the 60-day promotion gate.",
  shadow: "Generates signals only. No orders sent anywhere.",
  live: "Real money. Gated: requires ENABLE_LIVE_TRADING + promotion clearance.",
};

export default function StrategyModesPanel() {
  const [data, setData] = useState<ModesPayload | null>(null);
  const [account, setAccount] = useState<AccountPayload | null>(null);
  const [accountErr, setAccountErr] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = async () => {
    try {
      const m = await getJSON<ModesPayload>("/strategies/modes");
      setData(m);
      setErr(null);
    } catch (e: unknown) {
      setErr((e as Error).message);
    }
    try {
      const a = await getJSON<AccountPayload>("/broker/account");
      setAccount(a);
      setAccountErr(null);
    } catch (e: unknown) {
      setAccountErr((e as Error).message);
    }
  };

  useEffect(() => {
    load();
    const t = setInterval(load, 30_000);
    return () => clearInterval(t);
  }, []);

  const setMode = async (name: string, mode: Mode) => {
    setBusy(name);
    try {
      await postJSON(`/strategies/${name}/mode`, { mode });
      await load();
    } catch (e: unknown) {
      setErr((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <section
      style={{
        padding: 16,
        borderRadius: 12,
        background: "#0b1220",
        color: "#e5e7eb",
        marginTop: 16,
      }}
    >
      <h2 style={{ margin: 0, fontSize: 18 }}>Strategies & training mode</h2>
      <p style={{ marginTop: 4, marginBottom: 12, color: "#9ca3af", fontSize: 13 }}>
        Paper trains on fake money. Shadow is observation-only. Live needs the promotion gate.
      </p>

      {account && (
        <div
          style={{
            display: "flex",
            gap: 24,
            padding: "8px 12px",
            background: "#111827",
            borderRadius: 8,
            marginBottom: 12,
            fontSize: 14,
          }}
        >
          <div>
            <div style={{ color: "#9ca3af", fontSize: 11 }}>Broker</div>
            <div>{account.broker}</div>
          </div>
          <div>
            <div style={{ color: "#9ca3af", fontSize: 11 }}>Equity</div>
            <div>${account.equity ?? "—"}</div>
          </div>
          <div>
            <div style={{ color: "#9ca3af", fontSize: 11 }}>Cash</div>
            <div>${account.cash ?? "—"}</div>
          </div>
          <div>
            <div style={{ color: "#9ca3af", fontSize: 11 }}>Buying power</div>
            <div>${account.buying_power ?? "—"}</div>
          </div>
        </div>
      )}
      {accountErr && (
        <div style={{ color: "#f87171", fontSize: 12, marginBottom: 8 }}>
          Paper account unreachable — set ALPACA_PAPER_KEY_ID / ALPACA_PAPER_SECRET in .env.
        </div>
      )}

      {err && <div style={{ color: "#f87171", fontSize: 12 }}>{err}</div>}
      {!data && !err && <div style={{ color: "#9ca3af" }}>Loading…</div>}

      {data && (
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "#9ca3af", fontSize: 12, textAlign: "left" }}>
              <th style={{ padding: "6px 0" }}>Strategy</th>
              <th style={{ padding: "6px 0" }}>Mode</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.modes).map(([name, mode]) => (
              <tr key={name} style={{ borderTop: "1px solid #1f2937" }}>
                <td style={{ padding: "8px 0", fontFamily: "monospace" }}>{name}</td>
                <td style={{ padding: "8px 0" }}>
                  <div style={{ display: "flex", gap: 6 }}>
                    {data.available.map((m) => {
                      const active = mode === m;
                      return (
                        <button
                          key={m}
                          disabled={busy === name}
                          onClick={() => setMode(name, m)}
                          title={MODE_HELP[m]}
                          style={{
                            padding: "4px 10px",
                            borderRadius: 6,
                            border: "1px solid " + (active ? MODE_COLORS[m] : "#374151"),
                            background: active ? MODE_COLORS[m] : "transparent",
                            color: active ? "#0b1220" : "#e5e7eb",
                            fontSize: 12,
                            cursor: busy === name ? "wait" : "pointer",
                          }}
                        >
                          {m}
                        </button>
                      );
                    })}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
