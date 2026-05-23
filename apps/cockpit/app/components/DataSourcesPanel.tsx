"use client";

import { useEffect, useState } from "react";
import { getJSON } from "./api";

type Source = {
  ok: boolean;
  files: number;
  last_update: string | null;
  path: string;
  adapter: string;
};

type DataSources = {
  sources: Record<string, Source>;
  free_tier: string[];
};

function ago(iso: string | null): string {
  if (!iso) return "never";
  const then = new Date(iso).getTime();
  const now = Date.now();
  const sec = Math.max(0, Math.floor((now - then) / 1000));
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}

function freshnessDot(iso: string | null): string {
  if (!iso) return "bg-red-500";
  const ageH = (Date.now() - new Date(iso).getTime()) / 3_600_000;
  if (ageH < 30) return "bg-emerald-500";
  if (ageH < 72) return "bg-amber-500";
  return "bg-red-500";
}

const LABELS: Record<string, string> = {
  daily_bars: "Daily bars",
  intraday_bars: "Intraday (5m)",
  macro: "FRED macro",
  sentiment: "Sentiment",
};

export default function DataSourcesPanel() {
  const [data, setData] = useState<DataSources | null>(null);

  useEffect(() => {
    const tick = () =>
      getJSON<DataSources>("/data/sources")
        .then(setData)
        .catch(() => setData(null));
    tick();
    const id = setInterval(tick, 15_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-2xl bg-neutral-900 p-5">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider text-neutral-400">
          Data Sources
        </div>
        <div className="text-[10px] uppercase tracking-wider text-neutral-500">
          {data?.free_tier?.join(" · ") ?? "free tier"}
        </div>
      </div>

      {!data ? (
        <div className="mt-3 text-sm text-neutral-500">api unreachable</div>
      ) : (
        <ul className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2">
          {Object.entries(data.sources).map(([key, src]) => (
            <li
              key={key}
              className="rounded-xl bg-neutral-800/60 p-3 flex items-center justify-between"
            >
              <div className="flex items-center gap-3 min-w-0">
                <span
                  className={`inline-block h-2 w-2 rounded-full ${freshnessDot(
                    src.last_update,
                  )}`}
                />
                <div className="min-w-0">
                  <div className="text-sm truncate">{LABELS[key] ?? key}</div>
                  <div className="text-[11px] text-neutral-500 truncate">
                    {src.adapter}
                  </div>
                </div>
              </div>
              <div className="text-right shrink-0">
                <div className="text-sm tabular-nums">{src.files}</div>
                <div className="text-[11px] text-neutral-500">
                  {ago(src.last_update)}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      <div className="mt-3 text-[11px] text-neutral-500">
        Nightly refresh @ 03:00 UTC · Weekly retune Sun 05:00 UTC
      </div>
    </div>
  );
}
