"use client";

import { useEffect, useState } from "react";
import { getJSON } from "./api";

type Event = {
  decision_id: string;
  actor: string;
  event_type: string;
  ts?: string;
  [k: string]: unknown;
};

type ActivityResp = { events: Event[] };

const ACTOR_COLOR: Record<string, string> = {
  research: "bg-sky-500",
  strategy: "bg-indigo-500",
  risk: "bg-amber-500",
  execution: "bg-emerald-500",
  operator: "bg-fuchsia-500",
  system: "bg-neutral-500",
};

function ago(ts?: string): string {
  if (!ts) return "—";
  const d = new Date(ts).getTime();
  const s = Math.max(0, Math.floor((Date.now() - d) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

export default function ActivityFeed() {
  const [events, setEvents] = useState<Event[]>([]);

  useEffect(() => {
    let alive = true;
    const tick = () =>
      getJSON<ActivityResp>("/activity?limit=20")
        .then((j) => alive && setEvents(j.events))
        .catch(() => {});
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="rounded-2xl bg-neutral-900 p-5">
      <div className="text-xs uppercase tracking-wider text-neutral-400">
        Activity
      </div>
      {events.length === 0 ? (
        <div className="mt-3 text-sm text-neutral-500">no recent activity</div>
      ) : (
        <ul className="mt-3 space-y-2 max-h-80 overflow-y-auto pr-1">
          {events.map((e, i) => (
            <li key={i} className="flex items-center gap-3">
              <span
                className={`h-2 w-2 rounded-full ${
                  ACTOR_COLOR[e.actor] ?? "bg-neutral-500"
                }`}
              />
              <span className="text-xs text-neutral-300 flex-1 truncate">
                <span className="capitalize">{e.actor}</span>{" "}
                <span className="text-neutral-500">{e.event_type}</span>
              </span>
              <span className="text-xs text-neutral-600 tabular-nums">
                {ago(e.ts)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
