# Phase 25 — Automation & Autonomy Gap Analysis

**Question:** What automation and autonomous features are we missing for The Seer to be a success?

**Method:** Inventoried every shipped automation feature (Phases 1–24), then researched what successful platforms — QuantConnect's Algorithm Framework, Composer, Trade Ideas Holly, Tickeron, professional quant funds, and modern self-improving-agent systems (CoreWeave/W&B Weave, AutoAgent) — automate end-to-end. Mapped each missing capability to **importance**, **effort**, and **where it slots in**.

---

## What The Seer already automates (baseline)

| Layer | Shipped feature | Phase |
|---|---|---|
| Research / signal | Multi-source sweep (Yahoo news, analysts, insider, Reddit, Stocktwits) + corroboration trust | 3–14 |
| Brain / scoring | Self-improving bandit with 7 signal weights + Laplace-smoothed knowledge base + reflection | 21–22 |
| Memory | Atomic KV/AppendLog/FeatureIndex with rotation + reflections.jsonl + brain_memory.json | 22 |
| Sizing | 4 presets (Off / Conservative / Balanced / Aggressive) + fractional-Kelly + DD taper + audit log | 15 |
| Shadow trading | 14-day non-negative round-trip soak, auto-greenlight to live | 12 |
| Pre-flight | Readiness gates (session peak, calibration, decision log, disk, Alpaca keys, account, Telegram, sizing, paper loop, paper days) | 16 |
| Autonomy loop | `/api/autonomy/tick` runs the brain, candidates → judge → reflect → store | 20–21 |
| Promote | `/api/promote` + `/api/arm-live` flow with audit trail | 13 |
| Watchdog | `/api/watchdog/tick` + clear | 18 |
| Health / errors | `/api/health/full`, error log API, dashboard error pane | 16, 18 |
| Brand / UX | Topbar wordmark, hero, dividers, glass shimmer, status pills | 23–24 |

**This is already strong** — particularly the bandit + memory + shadow soak combo. The gaps below are what separates a hobbyist autopilot from a system you'd trust with $300 in two weeks and $30k a year from now.

---

## Gap analysis — what's missing

Each gap is rated:
- **Impact** — how much risk / how much upside (★★★ = makes-or-breaks live trading; ★ = nice polish)
- **Effort** — rough size (S = a session; M = a phase; L = multi-phase)
- **Where it slots in** — pointer to the existing code surface

### Tier 1 — Must-have before live capital is at risk

#### G1. Walk-forward validation harness ★★★ · Effort M
**What:** Replay any new bandit-weight change or sizing tweak across rolling in-sample / out-of-sample windows on historical data **before** it goes into production. [Walk-forward optimization](https://blog.quantinsti.com/walk-forward-optimization-introduction/) is the industry-standard guard against parameter overfitting that consistently catches strategies that look great in a single backtest but collapse on fresh data ([arxiv 2025 framework](https://arxiv.org/html/2512.12924v1)).
**Why we need it:** Today the bandit updates weights on **live** outcomes only. Any drift in the underlying signal mix (e.g., Reddit fatigue, analyst rating regime change) silently rots the model. Walk-forward gives us a "is this change actually better, or did we just overfit to the last 30 days?" answer in seconds.
**Where it slots in:** New `packages/cockpit/web/walkforward.py` + `/api/walkforward/run` route + Promote-page widget that blocks live-arm if last WF Sharpe < threshold.

#### G2. Multi-horizon drawdown circuit breakers ★★★ · Effort S
**What:** Hard auto-disable triggers at **multiple time horizons**, not just session peak. Daily / weekly / monthly DD limits, with the system auto-flipping to PAPER (not just halting the cycle) when any tier trips. [Nurp's 7-pillar framework](https://nurp.com/algorithmic-trading-blog/7-risk-management-strategies-for-algorithmic-trading/) lists this as #2 after position sizing.
**Why we need it:** Current preflight has a session-peak check, but a strategy can grind sideways losing 0.5%/day for three weeks (-10% rolling) without any single session tripping. The user asked for $300 first float — losing all of it in 3 weeks is the failure mode this prevents.
**Where it slots in:** Extend `packages/cockpit/web/preflight.py::_check_session_peak` into `_check_drawdown_ladder` with 4 horizons (session / day / week / month). Add `/api/risk/breakers` for the dashboard.

#### G3. Live-vs-shadow performance drift detector ★★★ · Effort M
**What:** Continuously compare live execution P&L against the shadow-trading simulation. If the gap exceeds a band (e.g., live is >15% worse than shadow over a 5-day rolling window — the [MEXC AI bot guide](https://www.mexc.com/news/1018983) cites 15% as the deploy-ready threshold), auto-disable live trading and require human re-arm. This catches slippage, broker queueing, and the silent-death failure modes documented in [The Silent Death of a DeFi Agent](https://www.moltbook.com/post/1b6d0818-61a9-4f61-9c35-d0fae46a1f91).
**Why we need it:** Today shadow and live are two universes that never talk. The whole point of the 14-day soak is to predict live performance — but once we cross the line, nothing checks that the prediction held.
**Where it slots in:** New `packages/cockpit/web/live_shadow_drift.py` + cron-tick check + dashboard "Live divergence" card.

#### G4. Kill-switch hierarchy ★★★ · Effort S
**What:** Three-tier emergency stop:
  1. **Pause cycle** (already have)
  2. **Disable autonomy + flatten new orders, keep positions** (NEW)
  3. **Liquidate-all + revoke API keys + Telegram blast** (NEW)
The [LinkedIn post on AI agents in hedge funds](https://www.linkedin.com/posts/silahian_hft-aiagents-systemarchitecture-activity-7428530777346322432-pXY5) describes a $500k kill-switch experiment as core infrastructure. Today we have one button; we need a panic ladder.
**Where it slots in:** `/api/control/kill` with `level=1|2|3` parameter; surfaced as red button cluster on `/health` page. Each level writes an audit row and (level 3) requires a typed confirmation.

### Tier 2 — Should-have to actually be autonomous

#### G5. Universe selection automation ★★ · Effort M
**What:** A dedicated universe-selection module — by liquidity, volatility, market-cap, sector cap — that auto-rebuilds the candidate pool nightly. [QuantConnect's Algorithm Framework](https://www.quantconnect.com/docs/v1/algorithm-framework/overview) treats Universe Selection as the **first** of five core modules; we currently rely on whatever the research sweep happens to surface, which is reactive and inconsistent.
**Why we need it:** Without it, the bandit is judging an effectively-random candidate set every night. With it, the bandit is judging the **same 200 most liquid eligible names**, which makes its trust scores actually mean something across cycles.
**Where it slots in:** New `packages/agents/universe.py` + `/api/universe` route + Settings widget for caps. Feeds into research sweep upstream.

#### G6. Auto-retraining / drift-triggered weight reset ★★ · Effort M
**What:** Monitor the bandit's prediction-vs-outcome calibration. When [PSI > 0.25 or rolling Sharpe drops > 3% over 4 weeks](https://smartdev.com/ai-model-drift-retraining-a-guide-for-ml-system-maintenance/), auto-reset that signal's trust weight to neutral (1.0×) and start a fresh learning window. Today weights only ever update, never reset — a regime change can leave them stuck in a bad basin forever.
**Why we need it:** The bandit currently has no "I no longer trust what I learned in Q1, start over for this signal" mechanism. Markets regime-shift quarterly; the model needs to forget on schedule, not just decay.
**Where it slots in:** Extend `packages/cockpit/web/bandit.py` with `detect_drift()` + `reset_signal()`. Cron job daily at market close.

#### G7. Champion / challenger sandbox ★★ · Effort L
**What:** Run two bandit configurations side-by-side in shadow — the live "champion" weights and a candidate "challenger" with proposed changes. After N days, if challenger Sharpe > champion Sharpe **and** challenger max-DD ≤ champion max-DD, auto-promote. This is the standard pattern in [CoreWeave / W&B Weave](https://letsdatascience.com/news/coreweave-launches-autonomous-agent-self-improvement-platfor-08689a3f) for autonomous agent improvement.
**Why we need it:** Today, "I want to try a higher reddit_trust weight" requires a human to flip a flag and hope. Champion/challenger turns it into a structured experiment with statistical exit criteria.
**Where it slots in:** New `packages/cockpit/web/challenger.py` that maintains two bandit instances; `/api/challenger/{propose,status,promote}` endpoints; dashboard widget on Agents page.

#### G8. Telegram / push state-change alerts ★★ · Effort S
**What:** Real-time push to phone for: arm/disarm, autonomy enable/disable, preflight state flip, drawdown breaker trip, kill-switch trigger, daily P&L summary, "your shadow soak just completed day X / 14." The [MEXC guide](https://www.mexc.com/news/1018983) lists this as one of the four monitoring necessities. The user already has Telegram in the preflight gates — but it's checked, not used.
**Why we need it:** The user's stated requirement is *"this will be as user-friendly as possible with me really needing only to click one or two buttons"*. That requires the system to **reach out** when something matters, not require checking the dashboard.
**Where it slots in:** New `packages/cockpit/notifications/telegram.py` + hook into existing arm_live, autonomy, preflight, watchdog modules. Wire to `send_notification` cron for digests.

### Tier 3 — Should-have to be best-in-class

#### G9. Auto trade journal with attribution ★ · Effort M
**What:** Every fill auto-logs **why** it happened (which signals triggered, with what weight, which agent approved), and a weekly post-mortem agent reads the journal and produces written learnings. [The MQL5 hybrid journal article](https://www.mql5.com/en/articles/21045) describes the "ALGO vs HUMAN" attribution pattern; we'd extend it to "which agent / which signal".
**Why we need it:** When we lose money, we currently can't answer "was it the analyst signal, the corroborator, or the sizing?" The reflection.py already writes some of this — we need to make it a structured journal a human (or another agent) can query.
**Where it slots in:** Extend `packages/cockpit/web/reflection.py` to write `data/cockpit/trade_journal.jsonl` with full attribution. New `/api/journal` + weekly post-mortem agent on `/agents`.

#### G10. Slippage & cost realism layer ★★ · Effort S
**What:** Inject realistic slippage + commission + spread into shadow trading. The 15%-variance threshold from MEXC only catches the gap if shadow assumes zero costs. Today shadow uses Alpaca paper, which is realistic enough — but we don't audit it.
**Why we need it:** First $300 is small enough that 5 bps slippage on a half-share order materially changes the answer. Better to know our shadow is honest.
**Where it slots in:** Add slippage-audit table to `/shadow` page comparing simulated fills to actual fills + extend pipeline tracking.

#### G11. LLM critic on reflections ★ · Effort S
**What:** Once a week, an LLM reads the last 7 days of reflections.jsonl + agent chatter + journal entries and writes a one-page diagnosis of what's working and what isn't, surfaced on the dashboard. AutoAgent and similar frameworks use this as their core self-improvement primitive.
**Why we need it:** The brain currently learns numerically (bandit weights). A critic agent learns linguistically — it can say "we keep buying right before earnings, and earnings are killing us." That's a structural insight the bandit can't surface.
**Where it slots in:** New `packages/agents/critic.py` runs weekly cron; output to `data/cockpit/critic_reports/` + new dashboard tile.

#### G12. Regime-conditional bandit weights ★★ · Effort M
**What:** Maintain separate weight vectors per regime (e.g., trend / chop / vol-spike from the existing `/api/regime`). Today one bandit serves all regimes — but Reddit sentiment matters in chop and analysts matter in trend; lumping them blurs both signals.
**Why we need it:** Hands-down the biggest **alpha** opportunity in the list — the existing infrastructure (bandit + regime detector) is already there, just unconnected.
**Where it slots in:** Extend `packages/cockpit/web/bandit.py` to key weights on regime. Reset behavior from G6 still applies per-regime.

---

## Recommended sequencing

If you ship one phase at a time:

1. **Phase 25: Risk safety net** — G2 (DD breakers) + G4 (kill-switch ladder) + G8 (Telegram alerts). Effort S+S+S = one session. Unlocks safe live deployment.
2. **Phase 26: Validation discipline** — G1 (walk-forward) + G3 (live-shadow drift) + G10 (slippage audit). Two sessions. Catches the silent-death failure mode before it costs money.
3. **Phase 27: Alpha lift** — G12 (regime-conditional bandit) + G5 (universe selection) + G9 (journal). Two sessions. Probably 10–30% Sharpe improvement.
4. **Phase 28: Continuous improvement** — G7 (champion/challenger) + G6 (drift-triggered reset) + G11 (LLM critic). Three sessions. This is the "self-improving" loop closing.

## What we're already strong on (don't rebuild)

- **Memory infrastructure** (Phase 22) — already best-in-class for the scale we're at
- **Bandit + reflection loop** (Phase 21) — solid foundation, just needs the additions above
- **Shadow soak gate** (Phase 12) — most retail platforms don't even have this
- **Pre-flight readiness** (Phase 16) — caught most of the "deploy without keys" foot-guns already
- **Brand / UX** (Phase 23–24) — done; ship it

---

## Bottom line

The Seer has roughly the **right skeleton**. What it's missing is the **risk safety net** (drawdown ladder + kill-switch hierarchy + live-shadow drift detector + Telegram state-change alerts) and the **continuous-improvement loop** (walk-forward validation + champion/challenger + drift-triggered weight reset + regime-conditional weights). These are not "make it work" features — the system works today. They're "make it safe to leave running unattended for a year" features, which is the actual product goal.

If we ship Phase 25 (the four items in Tier 1), live-arming $300 stops being "I hope this is OK" and becomes "the system will hard-stop and tell me if anything goes wrong." That's the bar.
