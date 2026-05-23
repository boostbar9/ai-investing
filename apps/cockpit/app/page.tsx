"use client";

import { useEffect, useState } from "react";
import ActivityFeed from "./components/ActivityFeed";
import ApprovalsQueue from "./components/ApprovalsQueue";
import DataSourcesPanel from "./components/DataSourcesPanel";
import HealthPanel from "./components/HealthPanel";
import LivePromotionPanel from "./components/LivePromotionPanel";
import PositionsHeatmap from "./components/PositionsHeatmap";
import StrategyModesPanel from "./components/StrategyModesPanel";
import { getJSON } from "./components/api";

type Regime = { regime: string; confidence: number; as_of: string };
type Agents = Record<string, { ok: boolean; model: string; last_run: string | null }>;

const REGIME_COLOR: Record<string, string> = {
  bull: "bg-emerald-500",
  chop: "bg-amber-500",
  bear: "bg-orange-500",
  crisis: "bg-red-600",
};

export default function Home() {
  const [regime, setRegime] = useState<Regime | null>(null);
  const [agents, setAgents] = useState<Agents | null>(null);

  useEffect(() => {
    getJSON<Regime>("/regime").then(setRegime).catch(() => {});
    getJSON<Agents>("/agents/status").then(setAgents).catch(() => {});
  }, []);

  return (
    <main className="min-h-screen bg-neutral-950 text-neutral-100 p-6">
      <header className="max-w-6xl mx-auto flex items-center justify-between">
        <h1 className="text-xl font-semibold tracking-tight">ai-investing</h1>
        <span className="text-xs text-neutral-500">v3.1 · phase 5</span>
      </header>

      <section className="max-w-6xl mx-auto mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="rounded-2xl bg-neutral-900 p-5">
          <div className="text-xs uppercase tracking-wider text-neutral-400">Regime</div>
          <div className="mt-3 flex items-center gap-3">
            <span
              className={`inline-block h-3 w-3 rounded-full ${
                REGIME_COLOR[regime?.regime ?? "chop"] ?? "bg-neutral-500"
              }`}
            />
            <span className="text-2xl font-medium capitalize">
              {regime?.regime ?? "—"}
            </span>
          </div>
          <div className="mt-1 text-xs text-neutral-500">
            confidence {regime ? Math.round(regime.confidence * 100) : "—"}%
          </div>
        </div>

        <div className="rounded-2xl bg-neutral-900 p-5 md:col-span-2">
          <div className="text-xs uppercase tracking-wider text-neutral-400">Agents</div>
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
            {agents
              ? Object.entries(agents).map(([name, a]) => (
                  <div key={name} className="rounded-xl bg-neutral-800/60 p-3">
                    <div className="flex items-center gap-2">
                      <span
                        className={`inline-block h-2 w-2 rounded-full ${
                          a.ok ? "bg-emerald-500" : "bg-red-500"
                        }`}
                      />
                      <span className="text-sm capitalize">{name}</span>
                    </div>
                    <div className="text-xs text-neutral-500 mt-1 truncate">
                      {a.model}
                    </div>
                  </div>
                ))
              : (
                <div className="text-sm text-neutral-500">loading…</div>
              )}
          </div>
        </div>

        <div className="md:col-span-2">
          <PositionsHeatmap />
        </div>
        <ApprovalsQueue />

        <LivePromotionPanel />
        <div className="md:col-span-2">
          <ActivityFeed />
        </div>
        <HealthPanel />
        <div className="md:col-span-2">
          <DataSourcesPanel />
        </div>
        <div className="md:col-span-3">
          <StrategyModesPanel />
        </div>
      </section>

      <footer className="max-w-6xl mx-auto mt-8 text-xs text-neutral-600">
        Hybrid AI-assisted quant. Survival first. Every trade has an Explain button.
      </footer>
    </main>
  );
}
