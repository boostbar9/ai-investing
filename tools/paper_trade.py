"""Nightly paper-trading runner.

Reads the latest daily Parquet data, runs the champion strategy, diffs the
target weights against current Alpaca paper positions, and submits the
minimum set of orders to close the gap. Logs everything to
``data/paper_log/runs.jsonl`` (append-only).

Kill switches (any one of these halts before any order is sent):

1. Equity drawdown from session peak > ``MAX_DD_PCT`` (default 8%)
2. Margin utilization > ``MARGIN_HALT_PCT`` (default 95%)
3. ``ENABLE_PAPER_TRADING`` env var not set to ``true``
4. Alpaca account status not ACTIVE

Designed to run from cron once per trading day after the close. Idempotent:
running it twice in the same session is a no-op (orders are only created
when the target weight changes by more than ``MIN_REBALANCE_BPS``).

Usage::

    PYTHONPATH=. python3 tools/paper_trade.py --strategy mean-reversion --dry-run
    PYTHONPATH=. python3 tools/paper_trade.py --strategy mean-reversion
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from packages.agents.paper_bridge import advise as agent_advise
from packages.cockpit.state import load_state as load_cockpit_state
from packages.execution.bracket_attach import attach_bracket_after_entry
from packages.execution.broker import (
    AlpacaPaperBroker,
    BrokerError,
    OrderRequest,
)
from packages.paper.streak import compute_paper_streak
from packages.persistence import connect as _db_connect
from packages.persistence import insert_cycle as _db_insert_cycle
from packages.persistence import write_snapshot as _write_snapshot
from packages.regime.ensemble import (
    RegimeGatedEnsemble,
    RegimeWeights,
    detect_regime_series,
)
from packages.shared.dotenv import load_dotenv
from packages.shared.schemas import Position
from packages.strategies import (
    IntradayTrendFollowing,
    MeanReversion,
    SectorRotation,
    SentimentOverlay,
    TrendFollowing,
)

# Auto-load .env from the current working directory so users don't have to
# manually `export` Alpaca keys before running. Existing shell exports win.
load_dotenv()

log = logging.getLogger("paper_trade")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DATA_ROOT = Path("data/parquet/daily")
PAPER_LOG_DIR = Path("data/paper_log")
PAPER_LOG_FILE = PAPER_LOG_DIR / "runs.jsonl"
EQUITY_PEAK_FILE = PAPER_LOG_DIR / "session_peak.json"

# Defaults; overridable via env.
MAX_DD_PCT = float(os.getenv("MAX_DD_PCT", "0.08"))
MARGIN_HALT_PCT = float(os.getenv("MARGIN_HALT_PCT", "0.95"))
MIN_REBALANCE_BPS = float(os.getenv("MIN_REBALANCE_BPS", "25"))  # 0.25% min weight change


STRATEGIES = {
    "trend-following": lambda: TrendFollowing(fast=50, slow=200),
    "sector-rotation": lambda: SectorRotation(top_n=3),
    # Walk-forward-tuned params (see docs/mean-reversion-tuning.md).
    "mean-reversion": lambda: MeanReversion(rsi_entry=15.0, rsi_exit=60.0, sma=200),
    # Phase 28-R step 5: 5-min opening-range breakout + VWAP trail. Flat by
    # 15:45 ET. Uses intraday OHLCV bars (yfinance 5m) rather than the
    # daily parquet panel, so :func:`compute_target_weights` routes it
    # through a dedicated branch below.
    "intraday-trend": lambda: IntradayTrendFollowing(),
}

STRATEGY_UNIVERSE = {
    "trend-following": ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA"],
    "sector-rotation": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU"],
    "mean-reversion": ["SPY", "QQQ", "IWM"],
    # Phase 28-R step 5: liquid intraday names (matches IntradayTrendFollowing.meta.universe).
    "intraday-trend": ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    # Ensemble = union of trend + sector + mean-reversion, gated by HMM regime.
    "ensemble": [
        "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    ],
    # Phase 13: confidence-gated policy. Same universe as ensemble so the
    # A/B comparison during shadow soak is apples-to-apples -- only the
    # decision *policy* differs, not the candidate set.
    "policy": [
        "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "TSLA",
        "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU",
    ],
}

STRATEGY_CHOICES = [*STRATEGIES, "ensemble", "policy"]


# Phase 32 ----------------------------------------------------------------
# Default strategy auto-selection.
#
# The 2026-06-02 live log showed every cycle logged ``strategy:
# mean-reversion`` with 0 orders submitted all day. Root cause: the
# argparse default was ``mean-reversion`` and the Windows service
# invokes ``paper_trade.py`` without ``--strategy``. The intraday
# rewrite (Phase 28-R) and all its downstream wiring is in place but
# never actually selected.
#
# Fix: during US regular trading hours we default to ``intraday-trend``;
# outside RTH we fall back to ``mean-reversion`` so the after-hours
# advisory loop still does something useful. Explicit ``--strategy``
# on the command line always wins.
from datetime import time as _dt_time  # noqa: E402  (after STRATEGIES block)
from zoneinfo import ZoneInfo  # noqa: E402

_RTH_OPEN = _dt_time(9, 30)
_RTH_CLOSE = _dt_time(16, 0)
_ET = ZoneInfo("America/New_York")


def _is_rth(now_utc: datetime | None = None) -> bool:
    """True if the supplied (or wall-clock) instant is inside RTH ET.

    Mon–Fri only. Holidays are intentionally not handled here — they
    cost us at most one wasted sweep and zero correctness because the
    broker rejects orders on holidays anyway.
    """
    now = now_utc or datetime.now(UTC)
    et = now.astimezone(_ET)
    if et.weekday() >= 5:  # Sat/Sun
        return False
    return _RTH_OPEN <= et.time() < _RTH_CLOSE


def _safe_pos_float(val: Any) -> float | None:
    """Coerce to a strictly-positive float, rejecting bools/None/garbage."""
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if f > 0 else None
    return None


def resolve_fill_provenance(
    fill_meta: Any,
    last_price: Any,
    requested_qty: Any,
) -> dict[str, Any]:
    """Determine ``fill_price`` / ``filled_qty`` / ``fill_source`` at record time.

    READ-ONLY and fail-safe. Priority (per the fill-capture spec):

    a. ``broker_fill`` — the broker reported an average execution price for this
       order. In paper mode that is whatever the paper broker put in
       ``last_fill_meta`` (the realistic Robinhood sim records the modeled
       fill); we treat that as the broker's own fill, never a guess.
    b. ``mark_estimate`` — no broker fill available, so fall back to the
       last-known quote/mark for the symbol at submit time, CLEARLY labeled an
       estimate.
    c. ``unknown`` — neither is available: ``fill_price`` is ``None`` and the
       leg is later excluded from realized P&L. We NEVER fabricate a price.

    This function places no orders and mutates nothing on the broker; it only
    reads values already produced by the (read-only) submit path.
    """
    if isinstance(fill_meta, dict):
        price = _safe_pos_float(fill_meta.get("fill_price"))
        if price is not None:
            qty = _safe_pos_float(fill_meta.get("filled_qty"))
            if qty is None:
                qty = _safe_pos_float(requested_qty)
            return {
                "fill_price": price,
                "filled_qty": qty,
                "fill_source": "broker_fill",
            }

    mark = _safe_pos_float(last_price)
    if mark is not None:
        return {
            "fill_price": mark,
            "filled_qty": _safe_pos_float(requested_qty),
            "fill_source": "mark_estimate",
        }

    return {"fill_price": None, "filled_qty": None, "fill_source": "unknown"}


def _auto_default_strategy(now_utc: datetime | None = None) -> str:
    """Pick the right default strategy for the current wall-clock instant.

    Inside RTH: ``intraday-trend`` (the Phase 28-R brain).
    Outside RTH: ``mean-reversion`` (steady-state advisory).
    """
    return "intraday-trend" if _is_rth(now_utc) else "mean-reversion"


# Phase 32 ----------------------------------------------------------------
# Regime vocabulary translation.
#
# The cockpit (packages/cockpit/web/regime.py) publishes the chip badge
# using the vocabulary ``{risk_on, neutral, risk_off, volatile}``. The
# trader's HMM ensemble (packages/regime/ensemble.py via
# _build_regime_series) speaks ``{bull, bear, chop, crisis}``. The two
# classifiers ran in parallel and silently disagreed: live log on
# 2026-06-02 showed the chip green at ``risk_on`` while every decision
# logged ``regime: chop`` or ``regime: unknown``. The trader was gating
# on a different signal than the operator was reading.
#
# Fix: define a one-way mapping so we can log *both* vocabularies on
# every cycle. The trader keeps using HMM regimes for its gating logic
# (that's what its strategies were tuned against), but we publish the
# cockpit-vocab translation alongside so the operator UI matches what
# the brain is actually thinking.
_HMM_TO_COCKPIT: dict[str, str] = {
    "bull": "risk_on",
    "bear": "risk_off",
    "chop": "neutral",
    "crisis": "volatile",
}


def _to_cockpit_regime(hmm_label: str) -> str:
    """Translate the HMM regime vocabulary to the cockpit chip vocabulary.

    Unknown / unmapped labels degrade to ``"neutral"`` so the chip stays
    informative even on degraded data — never raises.
    """
    return _HMM_TO_COCKPIT.get(hmm_label.lower(), "neutral")


# Phase 33 ----------------------------------------------------------------
# Per-agent narration emitter.
#
# Turns each sweep's outcome into one ``AgentStatus`` row per actor
# (research / strategy / risk / execution / reflection / curiosity /
# discovery). The cockpit reads these to render the "AGENT STATUS"
# panel, giving the operator a glance-able answer to "what is each
# lane doing and what's blocking it?".
#
# This function is intentionally a *pure translator* — it does no
# trading logic. It must be cheap and crash-resistant: a narration
# error must never poison a real cycle. The caller wraps it in try/
# except for that reason.

# Phase 33: persistent curiosity state lives here. Carries cumulative
# relaxation across sweeps (capped at MAX_CUMULATIVE_RELAXATION) and
# remembers the last watchlist signature so we can compute its age.
_CURIOSITY_STATE_PATH = Path("data") / "learning" / "curiosity_state.json"

# A small fallback wildcard pool for when the discovery layer hasn't
# populated one yet. Mid-cap names outside the typical research universe.
_FALLBACK_WILDCARD_POOL: tuple[str, ...] = (
    "NET", "DDOG", "SNOW", "CRWD", "PLTR", "SHOP", "COIN", "RBLX",
    "PATH", "MDB", "ZS", "OKTA", "PANW", "FTNT", "WDAY", "TEAM",
)


def _load_curiosity_state() -> dict[str, Any]:
    """Read persisted curiosity state. Returns sane defaults on miss."""
    try:
        if _CURIOSITY_STATE_PATH.exists():
            return json.loads(_CURIOSITY_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.debug("curiosity: state read failed: %s", exc)
    return {}


def _save_curiosity_state(state: dict[str, Any]) -> None:
    """Persist curiosity state atomically (best-effort)."""
    try:
        _CURIOSITY_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CURIOSITY_STATE_PATH.write_text(
            json.dumps(state, indent=2, default=str), encoding="utf-8"
        )
    except OSError as exc:
        log.debug("curiosity: state write failed: %s", exc)


def _compute_idle_streak(limit: int = 10) -> int:
    """Count consecutive recent sweeps with zero submitted orders.

    Reads ``data/paper_log/decisions.jsonl`` (the canonical sweep ledger)
    newest-first, stopping at the first sweep that submitted >0 orders.
    Halted sweeps count as idle (they couldn't submit anything).
    """
    p = Path("data") / "paper_log" / "decisions.jsonl"
    if not p.exists():
        return 0
    try:
        rows = p.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    streak = 0
    for line in reversed(rows[-limit:]):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        submitted = rec.get("orders_submitted") or []
        if isinstance(submitted, list) and len(submitted) > 0:
            break
        streak += 1
    return streak


def _dominant_rejection_from_audit(agent_audit: list[dict[str, Any]] | None) -> str:
    """Infer which gate rejected the most candidates this cycle.

    Looks at risk-agent events in the audit trail. Returns one of
    ``"atr"``, ``"vwap"``, ``"cluster"``, ``"sentiment"`` or ``""``.
    """
    if not agent_audit:
        return ""
    counts: dict[str, int] = {"atr": 0, "vwap": 0, "cluster": 0, "sentiment": 0}
    for ev in agent_audit:
        payload = ev.get("payload") or {}
        text = json.dumps(payload, default=str).lower()
        for key in counts:
            if key in text:
                counts[key] += 1
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else ""


def _run_curiosity_step(
    *,
    universe: tuple[str, ...],
    agent_audit: list[dict[str, Any]] | None,
    halted: bool,
    halt_reasons: list[str] | None,
) -> Any | None:
    """Build CuriosityInput from sweep state, call decide(), log the action.

    Returns the ``CuriosityAction`` (or None on import/IO error). Pure
    side effects are: appending to ``curiosity_actions.jsonl`` and
    updating ``curiosity_state.json``. Never mutates trading state.
    """
    try:
        from packages.agents.curiosity import (
            CuriosityInput,
            decide,
            log_action,
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("curiosity import failed: %s", exc)
        return None

    state = _load_curiosity_state()
    cumulative = float(state.get("cumulative_relaxation", 0.0))

    # Watchlist age: signature is the sorted universe tuple. If it
    # hasn't changed since the last snapshot we accumulate age.
    sig = ",".join(sorted(universe))
    last_sig = state.get("watchlist_sig", "")
    last_change_ts = float(state.get("watchlist_changed_at", 0.0))
    now = datetime.now(UTC).timestamp()
    if sig != last_sig or last_change_ts == 0.0:
        state["watchlist_sig"] = sig
        state["watchlist_changed_at"] = now
        watchlist_age_s = 0.0
    else:
        watchlist_age_s = max(0.0, now - last_change_ts)

    # Dominant rejection: from this cycle's audit, falling back to halt
    # reason keywords.
    dominant = _dominant_rejection_from_audit(agent_audit)
    if not dominant and halted and halt_reasons:
        hr = " ".join(halt_reasons).lower()
        for key in ("sentiment", "atr", "vwap", "cluster"):
            if key in hr:
                dominant = key
                break

    # Reflection age: best-effort read of reflections.jsonl mtime.
    refl_path = Path("data") / "learning" / "reflections.jsonl"
    try:
        last_reflection_age_s = (
            max(0.0, now - refl_path.stat().st_mtime) if refl_path.exists() else 0.0
        )
    except OSError:
        last_reflection_age_s = 0.0

    wildcard_pool = tuple(
        s for s in state.get("wildcard_pool") or _FALLBACK_WILDCARD_POOL
        if s not in universe
    )

    inp = CuriosityInput(
        idle_streak=_compute_idle_streak(),
        watchlist_age_s=watchlist_age_s,
        cumulative_relaxation=cumulative,
        dominant_rejection=dominant,
        universe=universe,
        wildcard_pool=wildcard_pool,
        last_reflection_age_s=last_reflection_age_s,
    )
    action = decide(inp)

    # Honor lower_threshold by bumping cumulative; honor wildcard_scan
    # by stashing the symbols for the next sweep to consume.
    if action.kind == "lower_threshold":
        state["cumulative_relaxation"] = float(
            action.payload.get("new_cumulative", cumulative + 0.10)
        )
        state["last_relaxation_filter"] = action.payload.get("filter", "")
    elif action.kind == "wildcard_scan":
        state["pending_wildcards"] = list(action.payload.get("symbols") or [])

    state["last_action_kind"] = action.kind
    state["last_action_ts"] = datetime.now(UTC).isoformat(timespec="seconds")
    _save_curiosity_state(state)

    try:
        log_action(action)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("curiosity log_action failed: %s", exc)
    return action


def _emit_phase33_narration(
    *,
    cycle_id: str,
    strategy_name: str,
    live_regime: str,
    cockpit_regime: str,
    sentiment: float,
    approved_n: int,
    planned_n: int,
    submitted_n: int,
    errors_n: int,
    halted: bool,
    halt_reasons: list[str] | None = None,
    curiosity_action: Any | None = None,
) -> None:
    """Build and emit per-actor narration rows for one sweep cycle.

    Status lines read like a human-readable status update; the cockpit
    surfaces them verbatim. Keep them short, concrete, and noun-led.
    """
    from packages.agents.narration import AgentStatus, emit_many

    halts_str = ", ".join(halt_reasons or []) if halted else ""

    # ---- Research ---------------------------------------------------
    research = AgentStatus(
        actor="research",
        working_on=(
            f"scanning {strategy_name} universe in regime={cockpit_regime}"
        ),
        waiting_on=(
            "" if approved_n > 0 else
            "a candidate to clear sentiment/momentum gates"
        ),
        last_action=(
            f"published sentiment={sentiment:+.2f}, "
            f"approved={approved_n}"
        ),
        last_result="halt" if halted else ("ok" if approved_n > 0 else "idle"),
        hints=(
            ("sentiment under floor; loosen for intraday",)
            if halted and "sentiment" in halts_str.lower() else ()
        ),
        cycle_id=cycle_id,
    )

    # ---- Strategy ---------------------------------------------------
    strategy = AgentStatus(
        actor="strategy",
        working_on=f"computing target weights for strategy={strategy_name}",
        waiting_on=(
            "" if planned_n > 0 else "approved candidates from risk lane"
        ),
        last_action=f"planned {planned_n} order(s)",
        last_result="halt" if halted else ("ok" if planned_n > 0 else "idle"),
        cycle_id=cycle_id,
    )

    # ---- Risk -------------------------------------------------------
    risk = AgentStatus(
        actor="risk",
        working_on="vetting candidates against position/sector/floor limits",
        waiting_on=(halts_str if halted else ""),
        last_action=(
            f"halted: {halts_str}" if halted else
            f"approved {approved_n}"
        ),
        last_result="halt" if halted else "ok",
        cycle_id=cycle_id,
    )

    # ---- Execution --------------------------------------------------
    execution = AgentStatus(
        actor="execution",
        working_on=(
            "submitting orders" if planned_n > 0 else "idle (no plan)"
        ),
        waiting_on=(
            "" if submitted_n > 0 else ("a plan to execute" if not halted else "halt to clear")
        ),
        last_action=(
            f"submitted {submitted_n}/{planned_n}"
            f"{', ' + str(errors_n) + ' error(s)' if errors_n else ''}"
        ),
        last_result=(
            "halt" if halted else
            ("warn" if errors_n else ("ok" if submitted_n > 0 else "idle"))
        ),
        cycle_id=cycle_id,
    )

    # ---- Reflection / Curiosity / Discovery -------------------------
    # These run on their own cadence; we publish placeholder rows so
    # the cockpit panel is never missing a lane. The respective
    # subsystems overwrite when they actually fire.
    reflection = AgentStatus(
        actor="reflection",
        working_on="watching judged outcomes; will speak after 2+ picks",
        waiting_on="more EOD-settled picks",
        last_action="",
        last_result="idle",
        cycle_id=cycle_id,
    )
    # Curiosity: if the orchestrator ran the meta-agent this cycle,
    # surface its real action; otherwise show a watching-for-stall row.
    if curiosity_action is not None:
        _ck = getattr(curiosity_action, "kind", "noop")
        _crat = getattr(curiosity_action, "rationale", "")
        _cpayload = getattr(curiosity_action, "payload", {}) or {}
        if _ck == "wildcard_scan":
            _syms = _cpayload.get("symbols") or []
            _working = f"injecting {len(_syms)} wildcard symbols into next sweep"
            _waiting = "next sweep to consume wildcards"
            _last = f"wildcard_scan: {', '.join(_syms[:5])}"
            _result = "ok"
        elif _ck == "lower_threshold":
            _flt = _cpayload.get("filter", "?")
            _cum = _cpayload.get("new_cumulative", 0.0)
            _working = f"proposing 10% relaxation of {_flt} filter"
            _waiting = "sweep to honour proposal"
            _last = f"lower_threshold filter={_flt} cum={_cum:.0%}"
            _result = "ok"
        elif _ck == "narrate_blockers":
            _working = "publishing structured why-no-trades note"
            _waiting = ""
            _last = "narrate_blockers"
            _result = "ok"
        else:
            _working = "watching for idle streaks / stale watchlist"
            _waiting = "a stall to unstick"
            _last = "noop (bot correctly idle)"
            _result = "idle"
        curiosity = AgentStatus(
            actor="curiosity",
            working_on=_working,
            waiting_on=_waiting,
            last_action=_last,
            last_result=_result,
            hints=(_crat,) if _crat else (),
            cycle_id=cycle_id,
        )
    else:
        curiosity = AgentStatus(
            actor="curiosity",
            working_on="watching for idle streaks / stale watchlist",
            waiting_on="a stall to unstick",
            last_action="",
            last_result="idle",
            cycle_id=cycle_id,
        )
    discovery = AgentStatus(
        actor="discovery",
        working_on="deriving stub from research sentiment + regime",
        waiting_on="",
        last_action=f"regime={live_regime} ({cockpit_regime})",
        last_result="ok",
        cycle_id=cycle_id,
    )

    emit_many([research, strategy, risk, execution, reflection, curiosity, discovery])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_panel(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for sym in symbols:
        p = DATA_ROOT / f"{sym}.parquet"
        if not p.exists():
            log.warning("missing parquet for %s; skipping", sym)
            continue
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(None)
        df = df.set_index("ts").sort_index()
        frames.append(df["close"].rename(sym))
    panel = pd.concat(frames, axis=1).ffill().dropna(how="any")
    return panel


# ---------------------------------------------------------------------------
# Kill switches
# ---------------------------------------------------------------------------


@dataclass
class KillSwitchResult:
    halt: bool
    reasons: list[str]


def update_session_peak(equity: float) -> float:
    from packages.shared.atomic_io import write_json_atomic

    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    peak = equity
    if EQUITY_PEAK_FILE.exists():
        try:
            data = json.loads(EQUITY_PEAK_FILE.read_text())
            peak = max(float(data.get("peak", 0.0)), equity)
        except (json.JSONDecodeError, ValueError, OSError):
            pass
    write_json_atomic(
        EQUITY_PEAK_FILE,
        {"peak": peak, "updated_at": datetime.now(UTC).isoformat()},
    )
    return peak


def check_kill_switches(account: dict[str, Any]) -> KillSwitchResult:
    reasons: list[str] = []
    if os.getenv("ENABLE_PAPER_TRADING", "false").lower() != "true":
        reasons.append("ENABLE_PAPER_TRADING != true")
    if account.get("status") != "ACTIVE":
        reasons.append(f"account status={account.get('status')!r}")
    if account.get("trading_blocked"):
        reasons.append("trading_blocked=true")
    if account.get("account_blocked"):
        reasons.append("account_blocked=true")

    equity = float(account.get("equity", 0))
    last_equity = float(account.get("last_equity", equity))
    peak = update_session_peak(equity)
    if peak > 0:
        dd = (peak - equity) / peak
        if dd > MAX_DD_PCT:
            reasons.append(f"DD {dd*100:.2f}% > {MAX_DD_PCT*100:.0f}% (peak ${peak:,.0f}, now ${equity:,.0f})")

    buying_power = float(account.get("buying_power", 0))
    long_market_value = float(account.get("long_market_value", 0))
    if buying_power > 0:
        util = long_market_value / (long_market_value + buying_power)
        if util > MARGIN_HALT_PCT:
            reasons.append(f"margin util {util*100:.1f}% > {MARGIN_HALT_PCT*100:.0f}%")

    # daily P&L info (purely advisory; logged not enforced)
    log.info(
        "account equity=$%s last_equity=$%s peak=$%s dd_today=%.2f%%",
        f"{equity:,.0f}",
        f"{last_equity:,.0f}",
        f"{peak:,.0f}",
        (peak - equity) / peak * 100 if peak > 0 else 0.0,
    )
    return KillSwitchResult(halt=bool(reasons), reasons=reasons)


# ---------------------------------------------------------------------------
# Order planning
# ---------------------------------------------------------------------------


# Phase 13: holder for the policy's per-symbol decisions from the last
# compute_target_weights() call. Read by _log_decision_record so the
# calibration log captures the confidences that drove the trade.
# Cleared at the start of each cycle so stale decisions never bleed
# across cycles. Module-level so callers don't have to thread it
# through every signature.
_LAST_POLICY_DECISIONS: list[dict[str, Any]] = []

# Phase 15: holder for the most recent risk-adaptive sizing result
# from compute_policy_weights(). Read by _log_decision_record so the
# /shadow/sizing endpoint can render per-symbol size + DD taper +
# Kelly cap diagnostics. Reset at the top of each cycle alongside
# _LAST_POLICY_DECISIONS. Stored as a dict (SizingResult.to_dict())
# rather than the dataclass so the value is JSON-serialisable.
_LAST_SIZING_RESULT: dict[str, Any] = {}


def compute_target_weights(
    strategy_name: str, *, equity: float = 0.0
) -> dict[str, float]:
    """Run the strategy on real data; return last-bar weights as a dict.

    ``equity`` (Phase 15) is the account's current equity, used by the
    policy branch's risk-adaptive sizer to compute drawdown taper +
    fractional-Kelly position sizes. Defaults to 0.0; when 0 the sizer
    is constructed in equal-weight mode (Phase 13 behaviour).
    """
    # Reset policy + sizing instrumentation; only the 'policy' branch repopulates.
    global _LAST_POLICY_DECISIONS, _LAST_SIZING_RESULT
    _LAST_POLICY_DECISIONS = []
    _LAST_SIZING_RESULT = {}
    if strategy_name == "ensemble":
        return compute_ensemble_weights()
    if strategy_name == "policy":
        return compute_policy_weights(equity=equity)
    if strategy_name == "intraday-trend":
        return compute_intraday_trend_weights()
    symbols = STRATEGY_UNIVERSE[strategy_name]
    panel = load_panel(symbols)
    if panel.empty:
        raise RuntimeError(f"no price panel for {strategy_name}")
    strategy = STRATEGIES[strategy_name]()
    # Sentiment overlay handled below if requested by caller.
    weights = strategy.generate_signals(panel)
    last_row = weights.iloc[-1].to_dict()
    return {k: float(v) for k, v in last_row.items() if not pd.isna(v)}


def compute_intraday_trend_weights() -> dict[str, float]:
    """Phase 28-R step 5: 5-minute opening-range breakout, VWAP trail.

    Fetches recent 5m bars via the yfinance adapter for each symbol in
    the ``intraday-trend`` universe, converts them to OHLCV frames, and
    delegates to :meth:`IntradayTrendFollowing.generate_weights_for_panel`
    for the per-bar long weights. Returns the last-bar weights (the
    current target allocation) as a dict for the rebalancer.

    Symbols whose intraday fetch fails (network/data error) are silently
    skipped so a single bad ticker can't take down the whole cycle.
    """
    import asyncio

    from packages.data.adapters.yfinance import YFinanceAdapter

    symbols = STRATEGY_UNIVERSE["intraday-trend"]
    strategy = STRATEGIES["intraday-trend"]()

    async def _fetch_all() -> dict[str, pd.DataFrame]:
        adapter = YFinanceAdapter()
        try:
            panel: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                try:
                    bars = await adapter.get_intraday_bars(
                        sym, interval="5m", range_="5d"
                    )
                except Exception as exc:
                    log.warning("intraday-trend: %s fetch failed: %s", sym, exc)
                    continue
                if not bars:
                    continue
                df = pd.DataFrame(
                    [
                        {
                            "open": b.open,
                            "high": b.high,
                            "low": b.low,
                            "close": b.close,
                            "volume": b.volume,
                        }
                        for b in bars
                    ],
                    index=pd.DatetimeIndex([b.ts for b in bars], name="ts"),
                )
                panel[sym] = df
            return panel
        finally:
            await adapter.aclose()

    panel = asyncio.run(_fetch_all())
    if not panel:
        raise RuntimeError("no intraday bars available for intraday-trend")
    weights = strategy.generate_weights_for_panel(panel)
    if weights.empty:
        return {}
    last_row = weights.iloc[-1].to_dict()
    return {k: float(v) for k, v in last_row.items() if not pd.isna(v) and float(v) > 0}


def _build_regime_series(panel: pd.DataFrame) -> pd.Series:
    """Daily regime labels using realised-vol VIX proxy + cross-section breadth.

    Mirrors the construction used in ``tools/stress_ensemble.py`` so paper
    behaviour matches the Tier-2 stress results.
    """
    # Fallback to first column as broad-market proxy when SPY is absent.
    spy = panel["SPY"] if "SPY" in panel.columns else panel.iloc[:, 0]
    realised_vol = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vix_proxy = realised_vol.fillna(15.0)
    rets_5d = panel.pct_change(5)
    breadth = (rets_5d > 0).mean(axis=1).fillna(0.5)
    return detect_regime_series(spy, vix_proxy, breadth)


def compute_ensemble_weights() -> dict[str, float]:
    """Run the regime-gated ensemble and return last-bar target weights."""
    symbols = STRATEGY_UNIVERSE["ensemble"]
    panel = load_panel(symbols)
    if panel.empty:
        raise RuntimeError("no price panel for ensemble")
    regimes = _build_regime_series(panel)
    ensemble = RegimeGatedEnsemble(
        strategies={
            "trend-following": TrendFollowing(fast=50, slow=200),
            "mean-reversion": MeanReversion(rsi_entry=15.0, rsi_exit=60.0, sma=200),
            "sector-rotation": SectorRotation(top_n=3),
        },
        regime_weights=RegimeWeights.from_calibrated(),
    )
    weights = ensemble.generate_signals(panel, regimes)
    last_row = weights.iloc[-1].to_dict()
    return {k: float(v) for k, v in last_row.items() if not pd.isna(v) and float(v) > 0}


def compute_policy_weights(*, equity: float = 0.0) -> dict[str, float]:
    """Phase 13/15: confidence-gated policy with risk-adaptive sizing.

    The policy reads the *same* inputs as the ensemble (regime, panel,
    research sweep) so the two strategies can be A/B-compared during the
    shadow soak. The key difference: this returns weights only for
    symbols whose composite confidence cleared the BUY threshold, plus
    explicit zeros for symbols whose confidence dropped below the SELL
    threshold (so the rebalancer flattens them).

    Side effects:
      * Populates ``_LAST_POLICY_DECISIONS`` with the per-symbol
        decision detail so the cycle's decision log can record
        calibration pairs (confidence -> action -> realised outcome).
      * Populates ``_LAST_SIZING_RESULT`` with per-symbol diagnostics
        (raw weight, vol scalar, final weight, DD multiplier, Kelly
        cap) so /shadow/sizing can render the sizing panel.

    Phase 15: ``equity`` enables the risk-adaptive sizer's drawdown
    taper + fractional-Kelly modes. When 0.0 (default), sizing falls
    back to equal-weight (Phase 13 behaviour).
    """
    global _LAST_POLICY_DECISIONS, _LAST_SIZING_RESULT
    from packages.agents.policy import ConfidenceGatedPolicy
    from packages.agents.sizing import RiskSizer, RiskSizerConfig, load_peak_equity
    from packages.regime.hmm import detect_regime

    symbols = STRATEGY_UNIVERSE["policy"]
    panel = load_panel(symbols)
    if panel.empty:
        raise RuntimeError("no price panel for policy")

    # Regime + posterior confidence from the HMM (or its heuristic fallback).
    spy = panel["SPY"] if "SPY" in panel.columns else panel.iloc[:, 0]
    realised_vol = spy.pct_change().rolling(20).std() * np.sqrt(252) * 100
    vix_proxy = realised_vol.fillna(15.0)
    rets_5d = panel.pct_change(5)
    breadth = (rets_5d > 0).mean(axis=1).fillna(0.5)
    reading = detect_regime(spy, vix_proxy, breadth)
    regime_name = str(reading.regime)
    regime_conf = float(reading.confidence)

    # Pull the latest research sweep (per-symbol confidence + trust).
    sweep_cands = _load_latest_sweep_candidates()

    # Use the ensemble's signal as the "alignment" input -- if the
    # ensemble also wants a name, that's an independent vote.
    try:
        ensemble_weights = compute_ensemble_weights()
    except Exception as exc:  # pragma: no cover - defensive only
        log.warning("policy: ensemble alignment unavailable: %s", exc)
        ensemble_weights = {}

    # Current holdings: empty here because compute_target_weights is
    # called BEFORE we fetch positions in run(). The policy still emits
    # correct BUY decisions; SELLs for names we hold get handled the
    # next cycle once positions are visible. (Acceptable: the existing
    # rebalancer flattens dropped names anyway via weight=0 absence.)
    #
    # Phase 14: load the persisted isotonic calibrator if present. When
    # missing or unfitted the policy uses raw composite confidence
    # (identical to Phase 13 behaviour). The calibrator is rebuilt
    # offline by tools/fit_policy_calibrator.py from the decision log.
    try:
        from packages.agents.calibration import IsotonicCalibrator

        cal = IsotonicCalibrator.load()
        calibrator = cal if cal.is_fitted else None
    except Exception as exc:  # pragma: no cover - defensive only
        log.warning("policy: calibrator load failed: %s; using raw", exc)
        calibrator = None

    policy = ConfidenceGatedPolicy(calibrator=calibrator)
    decisions = policy.decide(
        sweep_candidates=sweep_cands,
        ensemble_weights=ensemble_weights,
        current_holdings=set(),
        regime=regime_name,
        regime_confidence=regime_conf,
    )
    _LAST_POLICY_DECISIONS = [d.to_dict() for d in decisions]

    # Phase 15: realised-vol lookup (per-symbol annualised stdev of
    # daily returns over the trailing 30 bars). Used by the sizer's
    # vol-target scalar so a name with twice the vol gets ~half the
    # raw weight. Computed from the same panel the policy already
    # loaded -- no extra I/O.
    realised_vols: dict[str, float] = {}
    try:
        rets = panel.pct_change().dropna(how="all")
        if not rets.empty:
            # min_periods=10 so we don't blow up on freshly-listed names;
            # missing entries fall through to the sizer's default scalar.
            vols = rets.rolling(30, min_periods=10).std().iloc[-1] * np.sqrt(252)
            for sym, v in vols.items():
                if pd.notna(v) and float(v) > 0:
                    realised_vols[str(sym)] = float(v)
    except Exception as exc:  # pragma: no cover - defensive only
        log.warning("policy: realised-vol lookup failed: %s", exc)

    # Phase 15: peak equity for the drawdown taper. Reads the same
    # JSON file that check_kill_switches updates each cycle so the
    # taper and the hard kill switch see consistent DD numbers.
    try:
        peak_equity = load_peak_equity(EQUITY_PEAK_FILE)
    except Exception as exc:  # pragma: no cover - defensive only
        log.warning("policy: peak-equity load failed: %s", exc)
        peak_equity = 0.0

    # Build the sizer from env (POLICY_SIZING_MODE etc). Default mode
    # is equal_weight so behaviour is identical to Phase 13 unless the
    # user opts in.
    try:
        sizer = RiskSizer(RiskSizerConfig())
    except Exception as exc:  # pragma: no cover - defensive only
        log.warning("policy: sizer construction failed: %s; equal-weight", exc)
        sizer = None

    weights = policy.to_target_weights(
        decisions,
        sizer=sizer,
        equity=float(equity or 0.0),
        peak_equity=float(peak_equity or 0.0),
        realised_vols=realised_vols,
    )

    # Snapshot the sizer's diagnostics for /shadow/sizing. Best-effort.
    if sizer is not None:
        try:
            buys = [d for d in decisions if str(d.action).upper() == "BUY"]
            sizing_result = sizer.size(
                buy_decisions=buys,
                max_positions=policy.max_positions,
                equity=float(equity or 0.0),
                peak_equity=float(peak_equity or 0.0),
                realised_vols=realised_vols,
            )
            _LAST_SIZING_RESULT = sizing_result.to_dict()
        except Exception as exc:  # pragma: no cover - defensive only
            log.warning("policy: sizing diagnostics capture failed: %s", exc)
            _LAST_SIZING_RESULT = {}
    else:
        _LAST_SIZING_RESULT = {}

    # Drop the explicit zeros from the return value -- the rebalancer
    # treats an absent symbol as "flatten if held" anyway, and zero
    # weights confuse downstream sentiment-overlay code that multiplies
    # weights. Zeros stay in _LAST_POLICY_DECISIONS for the calibration log.
    return {k: v for k, v in weights.items() if v > 0}


def compute_target_weights_with_sentiment(
    base_name: str,
    sentiment_scores: dict[str, float] | None,
) -> dict[str, float]:
    """Wrap the base strategy with SentimentOverlay using real scores."""
    if base_name == "ensemble":
        # Ensemble already aggregates per-strategy signals; sentiment overlay
        # is intentionally not stacked on top.
        return compute_ensemble_weights()
    if base_name == "intraday-trend":
        # Intraday strategy uses 5-min OHLCV bars, not the daily panel; the
        # SentimentOverlay assumes daily prices. Sentiment is already baked
        # into the intraday setup_finder's ranker (Phase 28-R step 4), so
        # we just delegate to the bare intraday weights here.
        return compute_intraday_trend_weights()
    symbols = STRATEGY_UNIVERSE[base_name]
    panel = load_panel(symbols)
    base = STRATEGIES[base_name]()
    # Convert sentiment scores in [-1, 1] to multipliers in [0.5, 1.25].
    # Negative sentiment dampens to 0.5x, neutral=1.0x, max bullish=1.25x.
    mults: dict[str, float] = {}
    for sym in panel.columns:
        s = (sentiment_scores or {}).get(sym, 0.0)
        # linear map -1 -> 0.5, 0 -> 1.0, +1 -> 1.25 (slightly asymmetric)
        if s >= 0:
            mults[sym] = 1.0 + 0.25 * s
        else:
            mults[sym] = 1.0 + 0.5 * s  # -1 -> 0.5
    overlay = SentimentOverlay(base=base, sentiment=mults)
    weights = overlay.generate_signals(panel)
    return {k: float(v) for k, v in weights.iloc[-1].to_dict().items() if not pd.isna(v)}


@dataclass
class PlannedOrder:
    symbol: str
    side: str
    qty: float
    target_weight: float
    current_weight: float
    delta_weight: float
    last_price: float


async def plan_orders(
    target_weights: dict[str, float],
    broker: AlpacaPaperBroker,
    equity: float,
    skipped: list[dict[str, Any]] | None = None,
) -> list[PlannedOrder]:
    """Diff target weights against current positions; size in shares.

    Phase 36g — pending-order guard. Before sizing, we fetch every
    open (un-filled) order on the account and SKIP planning for any
    symbol that already has an in-flight order on the same side. This
    prevents the 403 cascade we saw on 2026-06-04 where pre-market
    orders held buying power, the next cycle still saw zero positions,
    re-planned the same 10 buys, and Alpaca rejected every one with
    'insufficient buying power'.
    """
    positions = await broker.positions()
    pos_by_sym = {p.symbol: p for p in positions}

    # Phase 36g: pending-order guard. Best-effort — if Alpaca's
    # /v2/orders is down we'd rather plan than block, so we swallow
    # broker errors and log them. A working call returns a list of
    # raw order rows; we collapse to a set keyed by (symbol, side).
    pending_keys: set[tuple[str, str]] = set()
    try:
        for row in await broker.open_orders():
            sym = str(row.get("symbol", "")).upper()
            side = str(row.get("side", "")).lower()
            if sym and side in ("buy", "sell"):
                pending_keys.add((sym, side))
    except Exception as e:  # pragma: no cover — defensive
        log.warning("open_orders fetch failed; skipping pending-order guard: %s", e)

    # Current weights = position market value / equity
    current_weights: dict[str, float] = {}
    last_price: dict[str, float] = {}
    for p in positions:
        if p.last_price is None:
            continue
        mv = p.qty * p.last_price
        current_weights[p.symbol] = mv / equity if equity > 0 else 0.0
        last_price[p.symbol] = p.last_price

    # For symbols in target but not currently held, pull a last-price from
    # the most recent parquet bar.
    for sym in target_weights:
        if sym not in last_price:
            p = DATA_ROOT / f"{sym}.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                if not df.empty:
                    last_price[sym] = float(df["close"].iloc[-1])

    all_symbols = set(target_weights) | set(current_weights)
    planned: list[PlannedOrder] = []
    skipped_pending = 0
    for sym in sorted(all_symbols):
        tw = target_weights.get(sym, 0.0)
        cw = current_weights.get(sym, 0.0)
        delta = tw - cw
        if abs(delta) * 10000 < MIN_REBALANCE_BPS:
            continue
        side = "buy" if delta > 0 else "sell"
        # Phase 36g — if there's already an open order for this
        # (symbol, side), the broker has already reserved buying
        # power for it; submitting again would just get a 403. Check
        # this BEFORE pricing so we still log the skip even when
        # market data is unavailable.
        if (sym, side) in pending_keys:
            skipped_pending += 1
            log.info(
                "skipping %s %s: pending order already in flight",
                side,
                sym,
            )
            continue
        px = last_price.get(sym)
        if px is None or px <= 0:
            log.warning("no price for %s; cannot size order", sym)
            continue
        delta_dollars = delta * equity
        qty = abs(delta_dollars / px)
        # Round to 4 decimals -- Alpaca supports fractional shares.
        qty = round(qty, 4)
        if qty <= 0:
            continue
        # Don't sell more than we hold; cap to current qty.
        if side == "sell":
            pos = pos_by_sym.get(sym)
            current_qty = pos.qty if pos is not None else 0.0
            qty = min(qty, current_qty)
            if qty <= 0:
                continue
            # Held-qty guard. Shares locked by a working order (e.g. an OCO
            # bracket sell leg) are reported by the broker as unavailable:
            # ``qty_available`` < ``qty``. Submitting another sell for those
            # held shares is certain to be rejected by Alpaca with a 403
            # ``insufficient qty available ... held_for_orders`` — which used
            # to flood the exit ledger with executed:false broker-error rows.
            # When the broker doesn't report availability (``None``) we
            # fall open and treat the full position as sellable.
            available = pos.qty_available if pos is not None else None
            if available is not None and available < qty:
                if skipped is not None:
                    skipped.append(
                        {
                            "symbol": sym,
                            "side": "sell",
                            "requested_qty": qty,
                            "available_qty": max(available, 0.0),
                            "held_qty": max(current_qty - available, 0.0),
                            "reason": "skipped_qty_held",
                        }
                    )
                log.info(
                    "skipping sell %s: %.4f shares held for working orders "
                    "(requested %.4f, available %.4f)",
                    sym,
                    max(current_qty - available, 0.0),
                    qty,
                    max(available, 0.0),
                )
                continue
        planned.append(
            PlannedOrder(
                symbol=sym,
                side=side,
                qty=qty,
                target_weight=tw,
                current_weight=cw,
                delta_weight=delta,
                last_price=px,
            )
        )
    if skipped_pending:
        log.info(
            "pending-order guard: skipped %d symbol(s) with in-flight orders",
            skipped_pending,
        )
    return planned


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def _load_latest_sweep_candidates() -> list[dict[str, Any]]:
    """Best-effort: fetch the most recent research sweep's candidate list.

    Returned shape is the JSON-serialised ``Candidate`` rows the sweep
    persists. Any failure (no sweep yet, IO error, schema drift) returns
    an empty list -- decision logging stays best-effort.
    """
    try:
        from packages.agents.research_sweep import load_sweep

        snap = load_sweep()
        if not isinstance(snap, dict):
            return []
        cands = snap.get("candidates") or []
        return [c for c in cands if isinstance(c, dict)]
    except Exception as exc:  # pragma: no cover - defensive only
        log.debug("sweep load for decisions failed: %s", exc)
        return []


def _log_decision_record(record: dict[str, Any]) -> None:
    """Phase 11: write per-cycle decision trace to the shadow log.

    Best-effort: never raises. The decision log is purely observational;
    a failure here must not poison the actual trade loop. Pulls the
    sweep snapshot fresh on each cycle so the pipeline funnel reflects
    whatever signals were *available* at decision time, not stale state.
    """
    try:
        from packages.paper.decisions import append_decision, build_record

        sweep_candidates = _load_latest_sweep_candidates()
        agent_audit = record.get("agent_audit") or []
        approved_syms: list[str] = []
        # Recover approved symbols from the agent audit trail. The risk
        # agent emits one event with the approved list; if absent, we
        # fall back to whatever ended up in target_weights with weight>0.
        for evt in agent_audit:
            if not isinstance(evt, dict):
                continue
            payload = evt.get("payload") or {}
            if isinstance(payload, dict) and payload.get("approved_symbols"):
                approved_syms = list(payload["approved_symbols"])
                break
        if not approved_syms:
            approved_syms = [
                s for s, w in (record.get("target_weights") or {}).items()
                if abs(float(w)) >= 1e-6
            ]

        # Phase 13: capture the confidence-gated policy's per-symbol
        # decisions if this cycle ran the 'policy' strategy. List is
        # always present (defaults to []) so dashboards can rely on it.
        policy_decisions = list(_LAST_POLICY_DECISIONS or [])

        # Phase 15: capture the risk-adaptive sizer's diagnostics from
        # the same cycle. Empty dict when the strategy was not 'policy'
        # or the sizer ran in equal-weight mode (no DD taper, no Kelly).
        sizing_diag = dict(_LAST_SIZING_RESULT or {})

        rec = build_record(
            ts=str(record.get("ts") or datetime.now(UTC).isoformat()),
            strategy=str(record.get("strategy") or ""),
            dry_run=bool(record.get("dry_run", False)),
            halted=bool(record.get("halted", False)),
            halt_reasons=list(record.get("reasons") or []),
            sweep_candidates=sweep_candidates,
            agent_approved_symbols=approved_syms,
            target_weights=record.get("target_weights") or {},
            planned_orders=record.get("orders_planned") or [],
            submitted_orders=record.get("orders_submitted") or [],
            errors=record.get("errors") or [],
            account_equity=float(record.get("account_equity") or 0.0),
            regime=str(record.get("agent_regime") or ""),
            decision_id=str(record.get("agent_decision_id") or ""),
            policy_decisions=policy_decisions,
            sizing=sizing_diag,
        )
        append_decision(rec)
    except Exception as exc:  # pragma: no cover - defensive only
        log.warning("decision-log write failed: %s", exc)


def log_run(record: dict[str, Any]) -> None:
    PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with PAPER_LOG_FILE.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    # Phase 11: structured decision trace alongside the raw run record.
    # Best-effort; safe to call on every code path (halt + success).
    _log_decision_record(record)
    # Dual-write to SQLite (best-effort). JSONL remains the source of truth
    # during this transition; the SQL mirror unlocks queryable history.
    if os.getenv("COCKPIT_DB_DUAL_WRITE", "1") != "0":
        try:
            with _db_connect() as conn:
                _db_insert_cycle(conn, record)
        except Exception as e:  # pragma: no cover - never let SQL break the loop
            log.warning("sqlite cycle insert failed: %s", e)


async def run(
    strategy_name: str,
    *,
    dry_run: bool = False,
    use_sentiment: bool = False,
    sentiment_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    broker = AlpacaPaperBroker()
    try:
        if not broker.key_id or not broker.secret:
            return {"halted": True, "reasons": ["Alpaca paper keys not in environment"]}

        account = await broker.account()
        kill = check_kill_switches(account)
        if kill.halt:
            log.warning("HALT: %s", "; ".join(kill.reasons))
            record = {
                "ts": started.isoformat(),
                "strategy": strategy_name,
                "halted": True,
                "reasons": kill.reasons,
                "account_equity": float(account.get("equity", 0)),
                "orders_planned": 0,
                "orders_submitted": 0,
            }
            log_run(record)
            return record

        # Cockpit pause: respect a manual halt set from the web GUI.
        cockpit = load_cockpit_state()
        if cockpit.paused and not dry_run:
            log.warning("HALT: cockpit paused by user (%s)", cockpit.last_action or "no note")
            record = {
                "ts": started.isoformat(),
                "strategy": strategy_name,
                "halted": True,
                "reasons": ["cockpit-paused"],
                "cockpit_note": cockpit.last_action,
                "account_equity": float(account.get("equity", 0)),
                "orders_planned": 0,
                "orders_submitted": 0,
            }
            log_run(record)
            return record

        equity = float(account["equity"])
        if use_sentiment and strategy_name in STRATEGIES:
            target = compute_target_weights_with_sentiment(strategy_name, sentiment_scores)
        else:
            # Phase 15: thread account equity through so the policy
            # branch's risk-adaptive sizer can compute DD taper and
            # Kelly sizing against the actual portfolio size.
            target = compute_target_weights(strategy_name, equity=equity)
        log.info("target weights for %s: %s", strategy_name, target)

        # ------------------------------------------------------------------
        # LangGraph advisory pass (Research -> Strategy -> Risk -> Approval).
        # Runs in advisory mode: never submits orders itself, but can halt
        # the run via the risk agent or veto specific concentration breaches.
        # ------------------------------------------------------------------
        symbols = STRATEGY_UNIVERSE.get(strategy_name, list(target.keys()))
        agent_positions: list[Position] = []
        try:
            for p in await broker.positions():
                agent_positions.append(
                    Position(symbol=p.symbol, qty=float(p.qty), avg_price=float(p.avg_price))
                )
        except Exception as e:
            log.warning("could not fetch positions for agent advisory: %s", e)

        # Detect the live regime from the same panel the strategy uses so the
        # advisory chain gates on real conditions (spec §7).
        try:
            _panel = load_panel(symbols)
            if not _panel.empty:
                _regimes = _build_regime_series(_panel)
                _live_regime = (
                    str(_regimes.iloc[-1]).lower() if len(_regimes) else "chop"
                )
            else:
                _live_regime = "chop"
        except Exception as e:
            log.warning("regime detection failed (%s); defaulting to chop", e)
            _live_regime = "chop"
        if _live_regime not in ("bull", "bear", "chop", "crisis"):
            _live_regime = "chop"
        log.info("agent advisory regime: %s", _live_regime)

        # Phase 32: relax the sentiment floor for intraday strategies.
        # The default floor of -0.5 halted half of the 2026-06-02 live
        # sweeps because aggregate news sentiment dipped on a chop day.
        # For an intraday trend-follower that's the wrong gate — we
        # actively want to trade down-momentum on bearish sentiment, and
        # sentiment should scale position size, not halt the sweep.
        # Multi-day strategies still use the tight floor.
        _intraday_strategies = {"intraday-trend"}
        _min_sentiment = -1.0 if strategy_name in _intraday_strategies else -0.5

        agent_result = await agent_advise(
            symbols=symbols,
            regime=_live_regime,
            positions=agent_positions,
            target_weights=target,
            sentiment_scores=sentiment_scores,
            min_sentiment=_min_sentiment,
        )
        agent_audit = [
            {"actor": a.actor, "event_type": a.event_type, "payload": a.payload}
            for a in agent_result.audit
        ]
        if agent_result.halted:
            reason = agent_result.risk.halt_reason or "risk-agent halted"
            log.warning("AGENT HALT: %s", reason)
            record = {
                "ts": started.isoformat(),
                "strategy": strategy_name,
                "halted": True,
                "reasons": [f"agent_halt: {reason}"],
                "account_equity": equity,
                "agent_audit": agent_audit,
                "agent_decision_id": str(agent_result.decision_id),
                "agent_sentiment": agent_result.research.sentiment,
                # Phase 32: include regime on the halted-path record so the
                # cockpit can show *why* a sweep halted in the same vocab.
                "agent_regime": _live_regime,
                "cockpit_regime": _to_cockpit_regime(_live_regime),
            }
            log_run(record)
            # Phase 33: narrate the halted path so the cockpit can show
            # WHY each agent stopped instead of going silent. Also run
            # the curiosity meta-agent so the cockpit shows whether the
            # bot is going to take any unblocking action next sweep.
            halt_reasons_list = list(record.get("reasons") or [])
            curiosity_action = None
            try:
                curiosity_action = _run_curiosity_step(
                    universe=tuple(symbols),
                    agent_audit=agent_audit,
                    halted=True,
                    halt_reasons=halt_reasons_list,
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("phase33 curiosity step failed: %s", exc)
            try:
                _emit_phase33_narration(
                    cycle_id=str(agent_result.decision_id),
                    strategy_name=strategy_name,
                    live_regime=_live_regime,
                    cockpit_regime=_to_cockpit_regime(_live_regime),
                    sentiment=float(agent_result.research.sentiment or 0.0),
                    approved_n=0,
                    planned_n=0,
                    submitted_n=0,
                    errors_n=0,
                    halted=True,
                    halt_reasons=halt_reasons_list,
                    curiosity_action=curiosity_action,
                )
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("phase33 narration emit failed: %s", exc)
            return record

        # Filter target weights by the symbols the risk agent approved.
        approved_syms = {s.symbol for s in agent_result.risk.approved}

        # LLM-chosen target weights: if a strategy signal carries a
        # non-null ``target_weight``, prefer it over the rule-based value
        # (only for approved symbols). Caps below still clip the result.
        # Sign must agree with side; mismatched signs are dropped to
        # avoid incoherent intent (already validated by risk prompt).
        llm_weights: dict[str, float] = {}
        for s in agent_result.risk.approved:
            if s.target_weight is None:
                continue
            tw = float(s.target_weight)
            expected_sign = 1.0 if s.side == "buy" else -1.0
            if tw == 0.0 or (tw > 0) != (expected_sign > 0):
                continue  # incoherent; defer to rule-based path
            llm_weights[s.symbol] = tw
        if llm_weights:
            log.info("LLM target_weights override: %s", llm_weights)

        if approved_syms:
            target = {
                sym: (llm_weights.get(sym, w) if sym in approved_syms else 0.0)
                for sym, w in target.items()
            }
            log.info("agent-approved symbols: %s", sorted(approved_syms))

        skipped_held: list[dict[str, Any]] = []
        planned = await plan_orders(target, broker, equity, skipped=skipped_held)
        log.info("planned %d orders", len(planned))
        if skipped_held:
            log.info(
                "held-qty guard: skipped %d sell(s) whose shares are held "
                "for working orders",
                len(skipped_held),
            )

        submitted = []
        errors = []
        # Order-routing seam: orders go to the *active* broker (the user's
        # selected backend) while account/positions/risk reads above keep
        # using the Alpaca paper broker as the data source. Default / unset
        # / any error resolves to Alpaca paper, so existing behavior is
        # unchanged; only an explicit, connected, gated Robinhood selection
        # routes orders to Robinhood (still SHADOW unless the live gate
        # authorizes). Never raises — falls back to the Alpaca ``broker``.
        order_broker: Any = broker
        order_broker_backend = "alpaca_paper"
        try:
            from packages.execution.broker_factory import (
                resolve_broker_selection,
            )

            sel = resolve_broker_selection()
            order_broker = sel.broker
            order_broker_backend = sel.effective_backend
            if sel.fell_back:
                log.warning("active broker fell back to paper: %s", sel.reason)
            else:
                log.info("active order broker: %s (%s)", sel.effective_backend, sel.reason)
        except Exception as e:  # pragma: no cover - fail safe to paper
            log.warning("broker selection failed (%s) -- using alpaca paper", e)
            order_broker = broker
            order_broker_backend = "alpaca_paper"
        # Brackets are an Alpaca-paper-specific OCO affordance; only attach
        # them when orders actually route to the Alpaca paper broker.
        _alpaca_order_path = order_broker_backend == "alpaca_paper"
        # Phase 35 — collect bracket-attach results per entry for the
        # cycle audit record. Each row gets the outcome of
        # ``attach_bracket_after_entry`` so the operator can see whether
        # the broker-side OCO armed for every long.
        brackets_attached: list[dict[str, Any]] = []
        if not dry_run:
            # Resolve active exit thresholds ONCE per cycle so every
            # entry's bracket uses the same policy snapshot.
            try:
                from packages.cockpit.web.exit_rules import (
                    current_thresholds as _exit_thresholds,
                )
                _bracket_th = _exit_thresholds()
            except Exception:  # pragma: no cover — defensive
                _bracket_th = None
            for po in planned:
                try:
                    req = OrderRequest(
                        symbol=po.symbol,
                        side=po.side,
                        qty=po.qty,
                        type="market",
                        time_in_force="day",
                    )
                    ack = await order_broker.submit(req)
                    rec = {
                        "symbol": po.symbol,
                        "side": po.side,
                        "qty": po.qty,
                        "broker_order_id": ack.broker_order_id,
                        "status": ack.status,
                        "broker": order_broker_backend,
                    }
                    # Robinhood-realistic sim exposes per-fill provenance
                    # (pricing source, modeled spread/slippage, partial) so
                    # the learning layer/UI can show how realistic the data
                    # is. Other brokers don't set this -> key stays absent.
                    fill_meta = getattr(order_broker, "last_fill_meta", None)
                    if isinstance(fill_meta, dict):
                        for _k in (
                            "pricing_source",
                            "fill_price",
                            "spread_bps",
                            "slippage_bps",
                            "filled_qty",
                            "partial",
                        ):
                            if _k in fill_meta:
                                rec[_k] = fill_meta[_k]
                    # Stamp the authoritative fill provenance the performance
                    # engine consumes: an average fill_price + filled_qty +
                    # fill_source (broker_fill | mark_estimate | unknown).
                    # Read-only and fail-safe — a missing fill yields
                    # fill_source="unknown" with a null price, never a guess.
                    rec.update(
                        resolve_fill_provenance(
                            fill_meta=fill_meta,
                            last_price=po.last_price,
                            requested_qty=po.qty,
                        )
                    )
                    submitted.append(rec)
                    log.info("submitted %s %s %.4f -> %s", po.side, po.symbol, po.qty, ack.status)
                    # Phase 35 — attach OCO bracket on successful long
                    # entries so exits run at exchange speed even if
                    # our loop hangs. Best-effort: any failure here
                    # must never unwind the parent fill.
                    if (
                        _alpaca_order_path
                        and _bracket_th is not None
                        and not _bracket_th.is_off()
                        and po.side == "buy"
                        and po.last_price
                    ):
                        try:
                            br = await attach_bracket_after_entry(
                                broker=broker,
                                symbol=po.symbol,
                                qty=po.qty,
                                side=po.side,
                                entry_price=float(po.last_price),
                                take_profit_pct=_bracket_th.take_profit_pct,
                                hard_stop_pct=_bracket_th.hard_stop_pct,
                            )
                        except Exception as bx:  # pragma: no cover
                            log.warning(
                                "bracket attach unexpectedly raised for %s: %s",
                                po.symbol,
                                bx,
                            )
                            br = {
                                "attached": False,
                                "reason": "exception",
                                "error": str(bx)[:200],
                            }
                        brackets_attached.append({"symbol": po.symbol, **br})
                        if br.get("attached"):
                            log.info(
                                "bracket armed %s tp=%.2f sl=%.2f",
                                po.symbol,
                                br.get("take_profit_price", 0.0),
                                br.get("stop_loss_stop_price", 0.0),
                            )
                except BrokerError as e:
                    log.warning("order failed %s %s: %s", po.side, po.symbol, e)
                    errors.append({"symbol": po.symbol, "side": po.side, "error": str(e)})

        record = {
            "ts": started.isoformat(),
            "strategy": strategy_name,
            "dry_run": dry_run,
            "halted": False,
            "account_equity": equity,
            "account_buying_power": float(account.get("buying_power", 0)),
            "target_weights": target,
            "orders_planned": [
                {
                    "symbol": po.symbol, "side": po.side, "qty": po.qty,
                    "target_w": po.target_weight, "current_w": po.current_weight,
                    "delta_w": po.delta_weight, "last_price": po.last_price,
                }
                for po in planned
            ],
            "orders_submitted": submitted,
            "errors": errors,
            # Sells skipped because their shares are held for working orders.
            # A distinct, non-error outcome (reason=skipped_qty_held) so the
            # exit ledger isn't polluted with certain-to-reject broker 403s.
            "orders_skipped": skipped_held,
            # Phase 35 — per-entry OCO bracket attach outcomes.
            "brackets_attached": brackets_attached,
            "agent_decision_id": str(agent_result.decision_id),
            "agent_sentiment": agent_result.research.sentiment,
            "agent_regime": _live_regime,
            # Phase 32: log the cockpit-vocab translation alongside the
            # raw HMM label so the operator UI and the decision ledger
            # speak the same language. ``agent_regime`` stays the source
            # of truth for gating logic; ``cockpit_regime`` is for
            # display / cross-reference only.
            "cockpit_regime": _to_cockpit_regime(_live_regime),
            "agent_thesis": agent_result.research.thesis,
            "agent_audit": agent_audit,
            "duration_sec": (datetime.now(UTC) - started).total_seconds(),
        }
        # Phase 11: emit per-symbol predicted PnL so the /shadow page
        # has an ex-ante baseline to score actuals against. Best-effort.
        try:
            from packages.paper.predictions import append_predictions

            current_weights_for_pred = {
                po.symbol: float(po.current_weight) for po in planned
            }
            n_pred = append_predictions(
                target_weights=target,
                current_weights=current_weights_for_pred,
                equity=equity,
                strategy=strategy_name,
                regime=_live_regime,
                decision_id=str(agent_result.decision_id),
                ts=started.isoformat(),
            )
            if n_pred:
                log.info("wrote %d predictions for shadow reconciliation", n_pred)
        except Exception as exc:  # pragma: no cover - defensive only
            log.warning("prediction-log write failed: %s", exc)
        # Phase 34: persist the candidate feature vectors keyed by this
        # cycle's decision_id so the nightly LightGBM trainer has a
        # training table to join against outcomes. Best-effort — a
        # snapshot write failure must never poison a real cycle.
        try:
            from packages.learning.feature_snapshot import append_snapshots

            target_syms = {
                str(s).upper() for s, w in (target or {}).items()
                if abs(float(w or 0.0)) >= 1e-6
            }
            sweep_candidates = _load_latest_sweep_candidates()
            snap_rows = [
                c for c in (sweep_candidates or [])
                if isinstance(c, dict)
                and str(c.get("symbol") or "").upper() in target_syms
            ]
            if snap_rows:
                n_snap = append_snapshots(
                    decision_id=str(agent_result.decision_id),
                    regime=_live_regime,
                    rows=snap_rows,
                    ts=started.isoformat(),
                )
                if n_snap:
                    log.info(
                        "wrote %d feature snapshots for ranker training", n_snap
                    )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("feature snapshot write failed: %s", exc)
        log_run(record)
        # Phase 33: narrate the happy path + run curiosity meta-agent
        # so the cockpit's AGENT STATUS panel reflects this sweep.
        curiosity_action = None
        try:
            curiosity_action = _run_curiosity_step(
                universe=tuple(symbols),
                agent_audit=agent_audit,
                halted=False,
                halt_reasons=None,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("phase33 curiosity step failed: %s", exc)
        try:
            _emit_phase33_narration(
                cycle_id=str(agent_result.decision_id),
                strategy_name=strategy_name,
                live_regime=_live_regime,
                cockpit_regime=_to_cockpit_regime(_live_regime),
                sentiment=float(agent_result.research.sentiment or 0.0),
                approved_n=len(approved_syms),
                planned_n=len(planned),
                submitted_n=len(submitted),
                errors_n=len(errors),
                halted=False,
                halt_reasons=None,
                curiosity_action=curiosity_action,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("phase33 narration emit failed: %s", exc)
        # Refresh the §16 streak snapshot AFTER appending this run so the
        # dashboard always sees the latest day. Best-effort: a failure here
        # must not poison the actual run record.
        streak_dict: dict[str, Any] = {}
        try:
            streak = compute_paper_streak()
            streak_dict = streak.to_dict()
            (PAPER_LOG_DIR / "streak.json").write_text(
                json.dumps(streak_dict, indent=2, default=str)
            )
            log.info(
                "§16 streak: %d/%d clean paper days (longest %d)",
                streak.current_streak,
                streak.gate_target_days,
                streak.longest_streak,
            )
        except Exception as e:
            log.warning("could not refresh paper streak: %s", e)
        # Per-cycle snapshot: atomic JSON the cockpit reads on boot so the
        # dashboard isn't blank until the next cycle fires (§17, task 8).
        try:
            _write_snapshot(
                equity=record.get("account_equity"),
                buying_power=record.get("account_buying_power"),
                target_weights=record.get("target_weights"),
                streak=streak_dict,
                strategy=strategy_name,
                extras={
                    "halted": record.get("halted", False),
                    "decision_id": record.get("agent_decision_id"),
                },
            )
        except Exception as e:  # pragma: no cover - never let snapshot break the loop
            log.warning("could not write cycle snapshot: %s", e)
        return record
    finally:
        await broker.aclose()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _one_run(args) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    sentiment_scores = None
    if args.use_sentiment:
        try:
            from tools.fetch_sentiment import fetch_scores  # late import to avoid hard dep

            sentiment_scores = asyncio.run(
                fetch_scores(list(set().union(*STRATEGY_UNIVERSE.values())))
            )
            log.info("loaded %d sentiment scores", len(sentiment_scores))
        except Exception as e:
            log.warning("sentiment fetch failed (%s); falling back to neutral", e)

    return asyncio.run(
        run(
            args.strategy,
            dry_run=args.dry_run,
            use_sentiment=args.use_sentiment,
            sentiment_scores=sentiment_scores,
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    # Phase 32: default is now auto-selected from the wall clock. If the
    # caller passes ``--strategy`` we honour it; otherwise we route to
    # intraday-trend during RTH and mean-reversion outside.
    ap.add_argument(
        "--strategy",
        choices=STRATEGY_CHOICES,
        default=None,
        help=(
            "Trading strategy to run. Default: intraday-trend during "
            "RTH (9:30-16:00 ET, Mon-Fri), mean-reversion otherwise."
        ),
    )
    ap.add_argument("--dry-run", action="store_true", help="Plan orders but do not submit.")
    ap.add_argument("--use-sentiment", action="store_true", help="Apply real sentiment overlay.")
    ap.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously, sleeping --interval seconds between cycles. Honors cockpit pause flag.",
    )
    ap.add_argument(
        "--interval",
        type=int,
        default=900,
        help="Seconds between cycles in --loop mode (default 900 = 15 minutes).",
    )
    args = ap.parse_args()
    if args.strategy is None:
        args.strategy = _auto_default_strategy()
        log.info("strategy auto-selected: %s", args.strategy)

    if not args.loop:
        result = _one_run(args)
        print(json.dumps(result, indent=2, default=str))
        return 0 if not result.get("halted") else 1

    # Loop mode: run every --interval seconds, log results, honor pause flag.
    import time

    log.info(
        "paper-trade loop starting: strategy=%s dry_run=%s interval=%ds",
        args.strategy,
        args.dry_run,
        args.interval,
    )
    while True:
        try:
            result = _one_run(args)
            print(json.dumps(result, default=str), flush=True)
        except KeyboardInterrupt:
            log.info("loop interrupted, exiting")
            return 0
        except Exception:
            log.exception("cycle failed; sleeping then retrying")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
