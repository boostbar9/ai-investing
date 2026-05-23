"use client";

import { useCallback, useEffect, useState } from "react";
import { getJSON, postJSON } from "./api";

type PendingItem = {
  decision_id: string;
  symbol: string;
  side: string;
  qty: number;
  thesis: string;
  ts: string;
};

type PendingResp = { pending: PendingItem[] };

export default function ApprovalsQueue() {
  const [items, setItems] = useState<PendingItem[]>([]);
  const [busy, setBusy] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const j = await getJSON<PendingResp>("/approvals/pending");
      setItems(j.pending);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 4000);
    return () => clearInterval(id);
  }, [refresh]);

  async function decide(id: string, approve: boolean) {
    setBusy(id);
    try {
      await postJSON(`/approvals/${id}`, { approve, note: approve ? "ok" : "denied" });
      await refresh();
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="rounded-2xl bg-neutral-900 p-5">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider text-neutral-400">
          Approvals
        </div>
        <span className="text-xs text-neutral-500">{items.length} pending</span>
      </div>
      {items.length === 0 ? (
        <div className="mt-3 text-sm text-neutral-500">queue clear</div>
      ) : (
        <ul className="mt-3 divide-y divide-neutral-800">
          {items.map((it) => (
            <li key={it.decision_id} className="py-3 flex items-center gap-3">
              <div className="flex-1">
                <div className="text-sm font-medium">
                  {it.side.toUpperCase()} {it.qty} {it.symbol}
                </div>
                <div className="text-xs text-neutral-500 truncate">
                  {it.thesis}
                </div>
              </div>
              <button
                onClick={() => decide(it.decision_id, true)}
                disabled={busy === it.decision_id}
                className="rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-3 py-1 text-xs"
              >
                Approve
              </button>
              <button
                onClick={() => decide(it.decision_id, false)}
                disabled={busy === it.decision_id}
                className="rounded-lg bg-neutral-700 hover:bg-neutral-600 disabled:opacity-50 px-3 py-1 text-xs"
              >
                Deny
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
