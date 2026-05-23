"use client";

import { useEffect, useState } from "react";
import { getJSON } from "./api";

type Subsystem = { ok: boolean; [k: string]: unknown };
type Health = Record<string, Subsystem>;

export default function HealthPanel() {
  const [data, setData] = useState<Health | null>(null);

  useEffect(() => {
    const tick = () =>
      getJSON<Health>("/health/detail")
        .then(setData)
        .catch(() => setData(null));
    tick();
    const id = setInterval(tick, 10_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="rounded-2xl bg-neutral-900 p-5">
      <div className="text-xs uppercase tracking-wider text-neutral-400">
        Health
      </div>
      {!data ? (
        <div className="mt-3 text-sm text-neutral-500">api unreachable</div>
      ) : (
        <ul className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-2">
          {Object.entries(data).map(([k, v]) => (
            <li
              key={k}
              className="rounded-xl bg-neutral-800/60 p-3 flex items-center gap-2"
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  v.ok ? "bg-emerald-500" : "bg-red-500"
                }`}
              />
              <span className="text-sm capitalize">{k}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
