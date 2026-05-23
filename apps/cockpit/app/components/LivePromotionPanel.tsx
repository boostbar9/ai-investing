"use client";

import { useEffect, useState } from "react";
import { getJSON } from "./api";

type Readiness = {
  ready: boolean;
  reasons: string[];
  metrics: Record<string, number>;
};

type Canary = {
  tier_index: number;
  fraction: number;
  days_in_tier: number;
  dwell_required: number;
  next_fraction: number | null;
  reasons: string[];
} | null;

type Promotion = {
  live_enabled: boolean;
  capital_fraction: number;
  readiness: Readiness;
  canary: Canary;
};

function pct(x: number): string {
  return `${(x * 100).toFixed(1)}%`;
}

export default function LivePromotionPanel() {
  const [data, setData] = useState<Promotion | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const tick = () =>
      getJSON<Promotion>("/live/promotion")
        .then((d) => {
          setData(d);
          setErr(null);
        })
        .catch((e: Error) => setErr(e.message));
    tick();
    const id = setInterval(tick, 30_000);
    return () => clearInterval(id);
  }, []);

  if (err) {
    return (
      <div className="rounded-2xl bg-neutral-900 p-5">
        <div className="text-xs uppercase tracking-wider text-neutral-400">
          Live promotion
        </div>
        <div className="mt-3 text-sm text-red-400">{err}</div>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="rounded-2xl bg-neutral-900 p-5">
        <div className="text-xs uppercase tracking-wider text-neutral-400">
          Live promotion
        </div>
        <div className="mt-3 text-sm text-neutral-500">loading…</div>
      </div>
    );
  }

  const { live_enabled, capital_fraction, readiness, canary } = data;

  return (
    <div className="rounded-2xl bg-neutral-900 p-5">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider text-neutral-400">
          Live promotion
        </div>
        <span
          className={`text-xs px-2 py-0.5 rounded-full ${
            live_enabled
              ? "bg-emerald-600/30 text-emerald-300"
              : "bg-neutral-800 text-neutral-400"
          }`}
        >
          {live_enabled ? "LIVE" : "PAPER ONLY"}
        </span>
      </div>

      <div className="mt-4 flex items-end gap-3">
        <div className="text-3xl font-semibold tabular-nums">
          {pct(capital_fraction)}
        </div>
        <div className="text-xs text-neutral-400 mb-1">capital ceiling</div>
      </div>

      {canary && (
        <div className="mt-3 space-y-2">
          <div className="flex items-center gap-2 text-xs text-neutral-400">
            <span>Canary tier {canary.tier_index}</span>
            {canary.next_fraction !== null && (
              <span>· next {pct(canary.next_fraction)}</span>
            )}
          </div>
          {canary.dwell_required > 0 && (
            <div className="h-1.5 w-full rounded-full bg-neutral-800 overflow-hidden">
              <div
                className="h-full bg-emerald-500"
                style={{
                  width: `${Math.min(
                    100,
                    (canary.days_in_tier / canary.dwell_required) * 100,
                  )}%`,
                }}
              />
            </div>
          )}
          <div className="text-xs text-neutral-500">
            {canary.days_in_tier}/{canary.dwell_required} dwell days
          </div>
        </div>
      )}

      {!readiness.ready && readiness.reasons.length > 0 && (
        <ul className="mt-4 space-y-1 text-xs text-neutral-400 list-disc list-inside">
          {readiness.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}

      {readiness.metrics && Object.keys(readiness.metrics).length > 0 && (
        <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
          {readiness.metrics.paper_days !== undefined && (
            <Metric label="paper days" value={readiness.metrics.paper_days.toFixed(0)} />
          )}
          {readiness.metrics.sharpe !== undefined && (
            <Metric label="Sharpe" value={readiness.metrics.sharpe.toFixed(2)} />
          )}
          {readiness.metrics.max_drawdown !== undefined && (
            <Metric
              label="max DD"
              value={pct(readiness.metrics.max_drawdown)}
            />
          )}
        </div>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-neutral-800/60 p-2">
      <div className="text-[10px] uppercase tracking-wider text-neutral-500">
        {label}
      </div>
      <div className="text-sm tabular-nums">{value}</div>
    </div>
  );
}
