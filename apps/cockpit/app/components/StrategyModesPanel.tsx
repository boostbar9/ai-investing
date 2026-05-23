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

const MODE_HELP: Record<Mode, string> = {
  paper: "Trades on the fake account. Counts toward the 60-day promotion gate.",
  shadow: "Generates signals only. No orders sent anywhere.",
  live: "Real money. Gated: requires ENABLE_LIVE_TRADING + promotion clearance.",
};

const MODE_ACTIVE_CLASS: Record<Mode, string> = {
  paper: "bg-sky-500 text-neutral-950 border-sky-500",
  shadow: "bg-neutral-300 text-neutral-950 border-neutral-300",
  live: "bg-emerald-500 text-neutral-950 border-emerald-500",
};

function formatMoney(s: string | null | undefined): string {
  if (s == null) return "—";
  const n = Number(s);
  if (Number.isNaN(n)) return s;
  return n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
}

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
    <section className="rounded-2xl bg-neutral-900 p-5">
      <div className="flex items-baseline justify-between gap-4">
        <div className="text-xs uppercase tracking-wider text-neutral-400">
          Strategies & training mode
        </div>
        {account && (
          <div className="text-xs text-neutral-500 font-mono">
            {account.broker}
          </div>
        )}
      </div>
      <p className="mt-1 text-xs text-neutral-500">
        Paper trains on fake money. Shadow is observation-only. Live needs the promotion gate.
      </p>

      {account && (
        <div className="mt-4 grid grid-cols-3 gap-2">
          <div className="rounded-xl bg-neutral-800/60 p-3">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">
              Equity
            </div>
            <div className="mt-1 text-lg font-medium tabular-nums">
              {formatMoney(account.equity)}
            </div>
          </div>
          <div className="rounded-xl bg-neutral-800/60 p-3">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">
              Cash
            </div>
            <div className="mt-1 text-lg font-medium tabular-nums">
              {formatMoney(account.cash)}
            </div>
          </div>
          <div className="rounded-xl bg-neutral-800/60 p-3">
            <div className="text-[10px] uppercase tracking-wider text-neutral-500">
              Buying power
            </div>
            <div className="mt-1 text-lg font-medium tabular-nums">
              {formatMoney(account.buying_power)}
            </div>
          </div>
        </div>
      )}
      {accountErr && (
        <div className="mt-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
          Paper account unreachable — set <code className="font-mono">ALPACA_PAPER_KEY_ID</code> and{" "}
          <code className="font-mono">ALPACA_PAPER_SECRET</code> in your <code className="font-mono">.env</code>.
        </div>
      )}

      {err && <div className="mt-3 text-xs text-red-400">{err}</div>}
      {!data && !err && <div className="mt-3 text-sm text-neutral-500">Loading…</div>}

      {data && (
        <ul className="mt-4 divide-y divide-neutral-800">
          {Object.entries(data.modes).map(([name, mode]) => (
            <li
              key={name}
              className="flex items-center justify-between gap-3 py-2.5"
            >
              <span className="font-mono text-sm text-neutral-200 truncate">{name}</span>
              <div className="flex gap-1">
                {data.available.map((m) => {
                  const active = mode === m;
                  return (
                    <button
                      key={m}
                      type="button"
                      disabled={busy === name}
                      onClick={() => setMode(name, m)}
                      title={MODE_HELP[m]}
                      aria-pressed={active}
                      className={[
                        "px-3 py-1 rounded-lg text-xs font-medium border transition-colors",
                        active
                          ? MODE_ACTIVE_CLASS[m]
                          : "border-neutral-700 text-neutral-300 hover:border-neutral-500 hover:text-neutral-100",
                        busy === name ? "cursor-wait opacity-60" : "cursor-pointer",
                      ].join(" ")}
                    >
                      {m}
                    </button>
                  );
                })}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
