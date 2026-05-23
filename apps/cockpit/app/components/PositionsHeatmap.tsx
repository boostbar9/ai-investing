"use client";

import { useEffect, useState } from "react";
import { getJSON } from "./api";

type Position = {
  symbol: string;
  qty: number;
  avg_price: number;
  last_price?: number;
  pnl_pct?: number;
};

type PositionsResp = { positions: Position[]; as_of: string };

function colorFor(pnl: number | undefined): string {
  if (pnl === undefined || pnl === null) return "bg-neutral-800";
  if (pnl > 0.02) return "bg-emerald-600";
  if (pnl > 0) return "bg-emerald-700/60";
  if (pnl > -0.02) return "bg-red-700/60";
  return "bg-red-600";
}

export default function PositionsHeatmap() {
  const [data, setData] = useState<PositionsResp | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    getJSON<PositionsResp>("/positions", ctrl.signal)
      .then(setData)
      .catch(() => {});
    return () => ctrl.abort();
  }, []);

  return (
    <div className="rounded-2xl bg-neutral-900 p-5">
      <div className="text-xs uppercase tracking-wider text-neutral-400">
        Positions
      </div>
      {!data ? (
        <div className="mt-3 text-sm text-neutral-500">loading…</div>
      ) : data.positions.length === 0 ? (
        <div className="mt-3 text-sm text-neutral-500">
          no open positions (paper account)
        </div>
      ) : (
        <div className="mt-3 grid grid-cols-3 sm:grid-cols-6 gap-2">
          {data.positions.map((p) => (
            <div
              key={p.symbol}
              className={`rounded-lg p-3 ${colorFor(p.pnl_pct)}`}
              title={`${p.symbol} qty ${p.qty}`}
            >
              <div className="text-sm font-medium tracking-tight">
                {p.symbol}
              </div>
              <div className="text-xs opacity-80 tabular-nums">
                {p.pnl_pct !== undefined
                  ? `${(p.pnl_pct * 100).toFixed(2)}%`
                  : "—"}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
