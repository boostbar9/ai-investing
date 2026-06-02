# Intraday Pivot — Gap Analysis & Rebuild Plan

**Date:** 2026-06-01  
**Trigger:** User clarified the bot must be a **pure day-trader** — every position opens & closes the same session, no overnight risk, target moderate turnover (1–2 round-trips/day), exit on profit-target hit. Phases 28/29/30 were originally designed around multi-day horizons (+1d / +5d / +20d). That's wrong.

---

## TL;DR

The codebase already has **~70% of the intraday infrastructure**, just unwired:

| Piece | State | Action |
|---|---|---|
| `intraday_trend.py` (5-min ORB + VWAP) | exists but NOT in `tools/paper_trade.py` | wire it in |
| `exit_rules.py` (TP / trail / hard-stop) | active in fast-tick, runs every 60s | tune for intraday |
| `run_fast_tick()` (60s exit loop) | active during market hours | keep |
| `run_one_tick()` (slow research sweep) | emits daily portfolio weights | replace with morning setup-finder |
| EOD flattener | does NOT exist (only manual button) | **build** |
| `predictions.py` (5d expected return) | wrong horizon | swap to intraday |
| `outcome_labeler.py` (1/5/20d horizons) | wrong horizons | rewrite (Phase 28-R) |
| Walk-forward backtest on intraday bars | does NOT exist | **build (Phase 29)** |
| Bandit reward signal | uses 24h judgment | **rewire to intraday outcomes (Phase 30)** |

The phases that already shipped this segment (26 news sentiment, 27 insider clusters) **stay as-is** — they're useful as morning-tilt inputs.

---

## What the bot does TODAY

### Slow loop — `run_one_tick()` (every ~15 min, market hours)
1. Research sweep → candidate symbols with corroboration scores
2. Apply LangGraph risk pass → approved set
3. Strategy emits **target portfolio weights** (continuous holdings, multi-day)
4. Compute delta vs current positions, plan orders ≥ 25 bps
5. Submit market orders via `AlpacaPaperBroker` with `time_in_force="day"`
6. Log to `decisions.jsonl` + `predictions.jsonl`

### Fast loop — `run_fast_tick()` (every 60s, market hours)
1. Refresh live quote cache (Finnhub WS + REST fallback)
2. **`exit_rules.run_tick()`** — for each open position:
   - take-profit at +3% (Balanced preset)
   - trailing-stop: arm at +2%, exit on 1.2% giveback from peak
   - hard-stop at −5%
3. **`dip_watch.run_tick()`** — re-buy positions that just sold on take-profit (mean-reverts the exit)

### Strategies wired (`tools/paper_trade.py`)
- `trend-following`, `mean-reversion`, `sector-rotation`, `ensemble`, `policy`
- **All daily-bar strategies, all multi-day holders.**
- `IntradayTrendFollowing` (5-min ORB + VWAP, long-only, 1% stop, 09:30–15:45 ET) **exists but is NOT in the choices list**.

---

## What pure-intraday day-trading REQUIRES

1. **Setup finder** that runs ONCE per morning (~6:30 AM PT, 30 min after open) and ranks tradeable setups for the day from a watchlist.
2. **Entry routing** that buys those setups during the entry window (09:45–15:00 ET), respecting daily $-budget ($300 first float).
3. **Profit-target manager** that exits each open position the moment its profit-target hits OR trail gives back.
4. **EOD flattener** that *unconditionally* closes every position at 15:55 ET, no exceptions.
5. **Horizon-aware labeling**: every trade gets evaluated at +30m, +2h, and at session-close — never beyond.
6. **Walk-forward backtest** on 5-min bars with a strict `as_of` clock to prevent lookahead.
7. **Bandit reward signal** fed by intraday round-trip outcomes (winner/loser/flat over the day), not 24h price drift.

---

## Gap map (what's missing vs what just needs rewiring)

### ✅ Reusable as-is
- `exit_rules.py` — Balanced preset (3% TP / 1.2% giveback / 5% hard-stop) is **fine for intraday** with one tweak: ignore previous-day peaks (PeakStore should reset at session open).
- `run_fast_tick()` — already runs every 60s, already evaluates exits.
- `AlpacaPaperBroker.liquidate_all()` — bulk-close already implemented; just needs to be called by scheduler at 15:55 ET.
- News sentiment (Phase 26) — useful as a morning-tilt filter.
- Insider clusters (Phase 27) — useful as a morning-tilt filter (cluster_buy → favor long).
- Decision journal infra (`decisions.jsonl`, `predictions.jsonl`, learning page UI) — keep all UI + aggregator code, only horizon labels change.

### 🔧 Needs rewiring
- `tools/paper_trade.py` — add `intraday-trend` to `STRATEGIES` dict & `STRATEGY_CHOICES`; switch live runner default.
- `predictions.py` — replace `REGIME_EXPECTED_RETURN_5D` with `REGIME_EXPECTED_RETURN_INTRADAY` (smaller magnitudes, ~0.15% Bull / 0.03% Chop / −0.10% Bear).
- `exit_rules._PeakStore` — reset per-symbol peak when bar timestamp crosses session open.

### 🏗 Needs to be built
1. **`packages/execution/eod_flattener.py`** — async fn `flatten_eod(broker)` that:
   - checks if current time (ET) is ≥ 15:55 and < 16:05
   - calls `broker.liquidate_all(cancel_orders=True)`
   - logs to `data/paper_log/eod_flatten.jsonl`
   - Wired into `run_fast_tick()` as a third parallel branch (gated on time window so it only fires once per day).

2. **`packages/strategies/intraday_setup_finder.py`** — runs ONCE per morning (idempotent, guarded by date stamp). For each symbol in watchlist:
   - Pull last 1h of 5-min bars + today's session-so-far
   - Score: ORB breakout strength × VWAP alignment × news_sentiment_tilt × insider_cluster_tilt
   - Top-K (default K=3) → emit `IntradayEntryPlan` rows: `{symbol, side=long, entry_price_band, position_dollars}`
   - Position sizing: split `$DAILY_BUDGET` (default = `min($300, equity * 0.01)`) across the K setups.
   - Hooks into `run_one_tick()` BUT only fires during the entry window 09:45–10:30 ET.

3. **`packages/execution/intraday_router.py`** — converts `IntradayEntryPlan` rows into `OrderRequest`s with `time_in_force="day"`, idempotent on `(symbol, session_date)`.

4. **Phase 28-R: `outcome_labeler.py` rewrite** —
   - New constant `INTRADAY_HORIZONS_MINUTES = (30, 120, "EOD")` replaces `DEFAULT_HORIZONS = (1, 5, 20)` days.
   - New bar source: 5-min intraday bars from yfinance (`interval="5m"`, `period="60d"` max). Fallback to Alpaca historical bars when key is set.
   - `is_pick_labelable()` → "longest horizon = market_close" so any pick from today is labelable AFTER 16:00 ET same-day.
   - Keep the per-agent aggregator, journal page UI, `/api/learning/*` endpoints — they're horizon-agnostic.

5. **Phase 29: walk-forward backtest** —
   - New module `packages/backtests/intraday_walk_forward.py`.
   - Iterates trading days in chronological order with `as_of_clock`: at each bar timestamp, only data ≤ that timestamp is visible.
   - Audit hook: assert `feature_max_ts <= as_of` for every feature row; fail loudly on lookahead.
   - Outputs: per-day round-trip P&L, hit rate, expectancy, max intraday DD, Sharpe (annualized from intraday returns).
   - Test on the existing 5-min bars in `data/parquet/intraday/` (if present) or download via `packages.data.intraday.fetch_intraday_bars`.

6. **Phase 30: bandit on intraday outcomes** —
   - `agent_bandit.update_with_outcome(features, reward)` already exists.
   - Wire reward = outcome_labeler.session_return for the day's pick:
     - reward = `+1` if EOD return ≥ +0.5%
     - reward = `−1` if EOD return ≤ −0.5%
     - reward = `−0.25` if `|EOD return| < 0.5%` (penalize flat trades — wasted slot).
   - Triggered nightly at 16:30 ET (after EOD bars settle) by a new cron action `learning_apply_daily_outcomes`.

---

## Proposed rebuild order (smallest blast radius first)

1. **Phase 28-R** (outcome_labeler intraday) — pure refactor, no live-trading impact. New tests replace the 21 daily-horizon tests. ~1 commit.
2. **EOD flattener** — single new module + wire into `run_fast_tick`. Zero impact when market is open mid-session. ~1 commit.
3. **`exit_rules` peak reset** — 1-line change in `_PeakStore` + new test. ~1 commit.
4. **Intraday setup-finder + router** — new modules, gated behind `INTRADAY_MODE=1` env flag so old daily strategies still run in parallel during transition. ~1 commit.
5. **Wire `intraday-trend` strategy into `tools/paper_trade.py` STRATEGIES dict.** ~1 commit.
6. **Phase 29** (walk-forward backtest) — pure analytic, no live impact. Validates the setup-finder before live shadow. ~1 commit.
7. **Phase 30** (bandit on intraday outcomes) — pure analytic, feeds back into Phase 4 setup-finder scoring. ~1 commit.

Total: **~7 commits**. Each is independently testable. The bot keeps running through every commit because the new path is gated behind `INTRADAY_MODE=1` until step 5 flips it on.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Intraday data feed (yfinance 5-min) is rate-limited & sometimes stale | Add Alpaca historical-bars adapter as primary; yfinance fallback. Cache aggressively. |
| EOD flattener fires twice if `run_fast_tick` runs at 15:55 and 15:56 | Idempotency key `(session_date, "eod_flatten")` in a JSONL flag file. |
| Setup-finder picks a thin/illiquid stock and gets a bad fill | Liquidity filter: avg dollar volume ≥ $50M over last 20 days. |
| User's first float is only $300 — pattern-day-trader rule (≥$25k for 4+ day-trades/5d in margin) doesn't apply because Alpaca paper has no PDT enforcement, but real Robinhood does | Document this; for live, gate to cash account or warn at promotion. |
| Lookahead in backtest | Mandatory `as_of` clock + audit assertion on every feature row. |
| Bandit learns the wrong thing if outcomes are scored only EOD (could miss +30m winners that gave back) | Reward shape uses MAX of (+30m, +2h, EOD) returns — captures "could have taken profit" signal. |

---

## What stays untouched

- All Phase 21–25 self-improvement machinery (`brain_memory`, `agent_bandit`, `agent_knowledge`, `regime_module`).
- Phase 26 news sentiment & Phase 27 insider clusters — both become **morning-tilt features** in the setup-finder.
- Cockpit UI scaffolding (`/learning`, `/shadow`, `/policy`, brand colors).
- Risk engine (`MAX_DD_PCT`, `MARGIN_HALT_PCT`, kill-switches).
- Mode plumbing (paper / shadow / live + live-promotion gate). Intraday strategy starts in **shadow** mode, promoted to paper after Phase 29 backtest passes.

---

## Decision needed from user

Approve this plan? Specifically the **commit-by-commit order** (1 → 7 above) and the **gate-flag approach** (`INTRADAY_MODE=1` lets new path run alongside old until step 5). If yes, I'll start with Phase 28-R since it's the lowest-risk refactor.

Two alternative shapes if you'd rather:
- **(A) Big-bang**: rip out all daily strategies first, then rebuild. Faster but the bot is dead during the transition.
- **(B) Conservative**: keep daily strategies forever, just add intraday alongside as a parallel strategy. Slower to commit fully to day-trading but you can A/B them.

My recommendation: the gated approach above (between A and B) — old path keeps running, new path is built behind a flag, flip the flag once Phase 29 backtest is green.
