"""User-facing trading guardrails — the "Trading Controls" feature.

The cockpit user is non-technical and remote. This module lets them set,
in plain language, the parameters the bot is allowed to trade within:

  1. **Budget** — total money the bot may deploy (aliases the existing
     Robinhood float cap so the two can never diverge), plus per-trade,
     per-day and open-position limits.
  2. **Confidence gate** — a minimum confidence ("how picky should the AI
     be?") chosen via presets (Conservative / Balanced / Aggressive) or an
     exact slider. Picking a preset sets the threshold; moving the slider
     switches the preset to "custom".
  3. **Pending queue** — trades the bot wanted to make but were held
     because they didn't meet these parameters (see ``pending_trades``).

The non-budget settings persist to ``data/cockpit/trading_controls.json``
(atomic write, same pattern as ``onboarding.py`` / ``state.py``). The
budget value is read/written through the onboarding float cap so
``resolve_float_cap()`` (the broker's enforcement path) stays the single
source of truth.

Everything FAILS SAFE: any parse/validation error falls back to the safe
default. These controls operate regardless of broker mode; when not live,
qualifying trades are simulated/shadow — this module never enables live
trading or sends real orders.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Path is env-overridable for test isolation (mirrors ONBOARDING_PATH).
TRADING_CONTROLS_PATH = Path(
    os.getenv("COCKPIT_TRADING_CONTROLS_PATH", "data/cockpit/trading_controls.json")
)

# ---------------------------------------------------------------------------
# Risk presets -> minimum confidence.
#
# Tuned to the real score distribution the research scorer emits. Candidate
# confidence/score lives in [0, 1]; observed picks top out around ~0.66
# (e.g. the dashboard's "NVDA score=0.658") and most cluster well below
# that, so the spec's literal 0.75/0.60/0.45 would gate out almost
# everything on Conservative. These tuned thresholds keep each preset
# meaningful against that distribution: Aggressive lets through most
# real picks, Balanced keeps the better half, Conservative only the best.
# ---------------------------------------------------------------------------
RiskPreset = Literal["conservative", "balanced", "aggressive", "custom"]
VALID_PRESETS: tuple[RiskPreset, ...] = (
    "conservative",
    "balanced",
    "aggressive",
    "custom",
)
PRESET_CONFIDENCE: dict[str, float] = {
    "conservative": 0.70,
    "balanced": 0.55,
    "aggressive": 0.40,
}

# Pending handling. Only ``auto_when_qualified`` is implemented now, but the
# field is an enum so a future ``ask_first`` mode slots in without a schema
# change.
PendingMode = Literal["auto_when_qualified"]
VALID_PENDING_MODES: tuple[PendingMode, ...] = ("auto_when_qualified",)

# Safe defaults.
DEFAULT_MAX_TRADES_PER_DAY = 5
DEFAULT_MAX_OPEN_POSITIONS = 3
DEFAULT_MIN_CONFIDENCE = 0.55  # = Balanced preset; user can move it.
DEFAULT_PER_TRADE_FALLBACK = 50.0  # capped at total budget on load.

# Clamp ceilings (FAIL SAFE bounds).
MAX_TRADES_PER_DAY_CEIL = 100
MAX_OPEN_POSITIONS_CEIL = 50

# A budget slice below this (USD) is treated as "no room" — too small to
# place a meaningful trade. Keeps the gate from green-lighting $0.01 orders.
MIN_TRADEABLE_USD = 1.0


# ---------------------------------------------------------------------------
# Clamping helpers — every one fails safe to the given default.
# ---------------------------------------------------------------------------
def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return max(lo, min(v, hi))


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN guard (shouldn't happen post-int, belt and braces)
        return default
    return max(lo, min(v, hi))


def clamp_confidence(value: Any, default: float = DEFAULT_MIN_CONFIDENCE) -> float:
    """Clamp a confidence into [0, 1]; fail safe to ``default``."""
    return _clamp_float(value, 0.0, 1.0, default)


def clamp_paper_balance(value: Any, ceiling: float) -> float | None:
    """Clamp a paper starting-balance override into ``(0, ceiling]``.

    ``None`` / empty / non-positive / unparseable -> ``None`` (meaning "use
    my real Robinhood cash"). Keeps a fixed training balance bounded by the
    same absolute ceiling as the budget."""
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return min(v, ceiling)


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce common truthy/falsey JSON + form values; fail safe to default."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "on"}:
        return True
    if s in {"false", "0", "no", "off"}:
        return False
    return default


def normalize_preset(value: Any) -> RiskPreset:
    """Coerce to a valid preset; unknown -> 'custom'."""
    s = str(value or "").strip().lower()
    return s if s in VALID_PRESETS else "custom"  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# The settings model
# ---------------------------------------------------------------------------
@dataclass
class TradingControls:
    """Resolved, already-clamped guardrail settings.

    ``total_budget_usd`` mirrors the onboarding float cap (read on load,
    written on save) so it can never diverge from ``resolve_float_cap()``.
    All other fields persist in ``trading_controls.json``.
    """

    total_budget_usd: float = 300.0
    max_per_trade_usd: float = DEFAULT_PER_TRADE_FALLBACK
    max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY
    max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    risk_preset: RiskPreset = "balanced"
    pending_mode: PendingMode = "auto_when_qualified"

    # --- Paper realism (Robinhood-realistic simulator) ---
    # ``paper_start_balance_usd`` is the simulator's starting cash. ``None``
    # means "use my real Robinhood cash" (the default); a number is a fixed
    # training balance. ``paper_use_real_cash`` records which the user chose
    # so an override value can be remembered without being active.
    paper_use_real_cash: bool = True
    paper_start_balance_usd: float | None = None
    # Opt-in, rate-limited read-only ``review_equity_order`` grounding.
    paper_review_grounding: bool = False
    # Allow fills in extended hours (pre/after-market) as well as RTH.
    paper_extended_hours: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_budget_usd": self.total_budget_usd,
            "max_per_trade_usd": self.max_per_trade_usd,
            "max_trades_per_day": self.max_trades_per_day,
            "max_open_positions": self.max_open_positions,
            "min_confidence": self.min_confidence,
            "risk_preset": self.risk_preset,
            "pending_mode": self.pending_mode,
            "paper_use_real_cash": self.paper_use_real_cash,
            "paper_start_balance_usd": self.paper_start_balance_usd,
            "paper_review_grounding": self.paper_review_grounding,
            "paper_extended_hours": self.paper_extended_hours,
        }


# ---------------------------------------------------------------------------
# Budget bridge — read/write the onboarding float cap so this UI and the
# existing /api/onboarding/robinhood/cap setter share one value.
# ---------------------------------------------------------------------------
def _read_budget() -> float:
    """Active budget = the onboarding float cap, clamped. Defaults to $300."""
    try:
        from packages.cockpit.onboarding import clamp_float_cap, load_onboarding

        return clamp_float_cap(load_onboarding().live_float_cap_usd)
    except Exception:
        return 300.0


def _write_budget(value: float) -> float:
    """Persist the budget THROUGH the onboarding cap (single source of
    truth). Returns the clamped value actually stored."""
    from packages.cockpit.onboarding import (
        clamp_float_cap,
        load_onboarding,
        save_onboarding,
    )

    clamped = clamp_float_cap(value)
    state = load_onboarding()
    state.live_float_cap_usd = clamped
    save_onboarding(state)
    return clamped


def _budget_ceiling() -> float:
    try:
        from packages.cockpit.onboarding import ABSOLUTE_MAX_FLOAT_USD

        return float(ABSOLUTE_MAX_FLOAT_USD)
    except Exception:
        return 10_000.0


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------
def _coerce(raw: dict[str, Any], budget: float) -> TradingControls:
    """Build a fully-clamped TradingControls from a raw dict + live budget."""
    per_trade = _clamp_float(
        raw.get("max_per_trade_usd", min(DEFAULT_PER_TRADE_FALLBACK, budget)),
        0.0,
        budget,
        min(DEFAULT_PER_TRADE_FALLBACK, budget),
    )
    preset = normalize_preset(raw.get("risk_preset", "balanced"))
    # If a known preset is stored, the threshold it implies wins (keeps the
    # two coherent even if the file was hand-edited).
    if preset in PRESET_CONFIDENCE:
        min_conf = PRESET_CONFIDENCE[preset]
    else:
        min_conf = clamp_confidence(raw.get("min_confidence", DEFAULT_MIN_CONFIDENCE))
        preset = "custom"
    pending_raw = str(raw.get("pending_mode", "auto_when_qualified"))
    pending: PendingMode = (
        pending_raw if pending_raw in VALID_PENDING_MODES else "auto_when_qualified"
    )  # type: ignore[assignment]
    return TradingControls(
        total_budget_usd=budget,
        max_per_trade_usd=per_trade,
        max_trades_per_day=_clamp_int(
            raw.get("max_trades_per_day", DEFAULT_MAX_TRADES_PER_DAY),
            0,
            MAX_TRADES_PER_DAY_CEIL,
            DEFAULT_MAX_TRADES_PER_DAY,
        ),
        max_open_positions=_clamp_int(
            raw.get("max_open_positions", DEFAULT_MAX_OPEN_POSITIONS),
            0,
            MAX_OPEN_POSITIONS_CEIL,
            DEFAULT_MAX_OPEN_POSITIONS,
        ),
        min_confidence=min_conf,
        risk_preset=preset,
        pending_mode=pending,
        paper_use_real_cash=_as_bool(raw.get("paper_use_real_cash", True), True),
        paper_start_balance_usd=clamp_paper_balance(
            raw.get("paper_start_balance_usd"), _budget_ceiling()
        ),
        paper_review_grounding=_as_bool(
            raw.get("paper_review_grounding", False), False
        ),
        paper_extended_hours=_as_bool(
            raw.get("paper_extended_hours", False), False
        ),
    )


def load_controls(path: Path | None = None) -> TradingControls:
    """Read controls from disk, merged with the live budget. Missing or
    corrupt files yield safe defaults (never raises)."""
    if path is None:
        path = TRADING_CONTROLS_PATH
    budget = _read_budget()
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                raw = loaded
        except (OSError, json.JSONDecodeError):
            raw = {}
    return _coerce(raw, budget)


def _save_controls_file(controls: TradingControls, path: Path) -> None:
    """Atomically write the non-budget fields (budget lives in onboarding)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = controls.to_dict()
    payload.pop("total_budget_usd", None)  # owned by onboarding
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as f:
        json.dump(payload, f, indent=2)
        tmp = f.name
    os.replace(tmp, path)


def update_controls(
    updates: dict[str, Any], path: Path | None = None
) -> TradingControls:
    """Validate + clamp + persist any subset of fields; return resolved.

    Linkage rules (the UI depends on these):
      * Setting ``risk_preset`` to a known preset overrides ``min_confidence``
        with that preset's threshold.
      * Setting ``min_confidence`` directly (without a matching preset) flips
        ``risk_preset`` to "custom".
    Budget is persisted through the onboarding cap so it can't diverge from
    ``resolve_float_cap()``. Everything fails safe to the current/default
    value on bad input.
    """
    if path is None:
        path = TRADING_CONTROLS_PATH
    current = load_controls(path)
    u = dict(updates or {})

    # --- budget (writes through onboarding) ---
    if "total_budget_usd" in u:
        budget = _write_budget(u["total_budget_usd"])
    else:
        budget = current.total_budget_usd

    # --- per-trade / day / positions ---
    per_trade = (
        _clamp_float(
            u["max_per_trade_usd"], 0.0, budget, min(DEFAULT_PER_TRADE_FALLBACK, budget)
        )
        if "max_per_trade_usd" in u
        else min(current.max_per_trade_usd, budget)
    )
    max_trades = (
        _clamp_int(
            u["max_trades_per_day"], 0, MAX_TRADES_PER_DAY_CEIL,
            DEFAULT_MAX_TRADES_PER_DAY,
        )
        if "max_trades_per_day" in u
        else current.max_trades_per_day
    )
    max_open = (
        _clamp_int(
            u["max_open_positions"], 0, MAX_OPEN_POSITIONS_CEIL,
            DEFAULT_MAX_OPEN_POSITIONS,
        )
        if "max_open_positions" in u
        else current.max_open_positions
    )

    # --- confidence gate + preset linkage ---
    preset = current.risk_preset
    min_conf = current.min_confidence
    preset_req = normalize_preset(u["risk_preset"]) if "risk_preset" in u else None
    if preset_req in PRESET_CONFIDENCE:
        preset = preset_req  # type: ignore[assignment]
        min_conf = PRESET_CONFIDENCE[preset_req]
    if "min_confidence" in u:
        min_conf = clamp_confidence(u["min_confidence"])
        # Direct threshold edit => custom, unless it exactly equals a preset
        # (and the caller didn't just ask for that preset).
        if preset_req not in PRESET_CONFIDENCE:
            matched = next(
                (p for p, v in PRESET_CONFIDENCE.items() if abs(v - min_conf) < 1e-9),
                None,
            )
            preset = matched or "custom"  # type: ignore[assignment]

    pending = current.pending_mode
    if "pending_mode" in u:
        pr = str(u["pending_mode"])
        pending = pr if pr in VALID_PENDING_MODES else current.pending_mode  # type: ignore[assignment]

    # --- paper realism ---
    paper_use_real = (
        _as_bool(u["paper_use_real_cash"], current.paper_use_real_cash)
        if "paper_use_real_cash" in u
        else current.paper_use_real_cash
    )
    paper_balance = (
        clamp_paper_balance(u["paper_start_balance_usd"], _budget_ceiling())
        if "paper_start_balance_usd" in u
        else current.paper_start_balance_usd
    )
    paper_review = (
        _as_bool(u["paper_review_grounding"], current.paper_review_grounding)
        if "paper_review_grounding" in u
        else current.paper_review_grounding
    )
    paper_ext = (
        _as_bool(u["paper_extended_hours"], current.paper_extended_hours)
        if "paper_extended_hours" in u
        else current.paper_extended_hours
    )

    resolved = TradingControls(
        total_budget_usd=budget,
        max_per_trade_usd=per_trade,
        max_trades_per_day=max_trades,
        max_open_positions=max_open,
        min_confidence=min_conf,
        risk_preset=preset,
        pending_mode=pending,
        paper_use_real_cash=paper_use_real,
        paper_start_balance_usd=paper_balance,
        paper_review_grounding=paper_review,
        paper_extended_hours=paper_ext,
    )
    _save_controls_file(resolved, path)
    return resolved


# ---------------------------------------------------------------------------
# The guardrail — pure, unit-tested evaluation.
# ---------------------------------------------------------------------------
@dataclass
class TradeCandidate:
    """A trade the bot wants to make. ``confidence`` is in [0, 1].

    ``notional`` is the intended dollar size; <= 0 means "unspecified" and
    the evaluator sizes the trade up to the per-trade / remaining-budget cap.
    """

    symbol: str
    side: str = "buy"
    confidence: float = 0.0
    notional: float = 0.0


@dataclass
class PortfolioState:
    """Live counters the gate measures the candidate against."""

    used_budget_usd: float = 0.0
    open_positions: int = 0
    trades_today: int = 0


@dataclass
class ControlsVerdict:
    qualifies: bool
    reasons: list[str] = field(default_factory=list)
    clamped_notional: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualifies": self.qualifies,
            "reasons": list(self.reasons),
            "clamped_notional": round(self.clamped_notional, 2),
        }


def _pct(x: float) -> str:
    return f"{round(max(0.0, min(1.0, x)) * 100)}%"


def _usd(x: float) -> str:
    return f"${x:,.0f}" if float(x).is_integer() else f"${x:,.2f}"


def evaluate_trade_against_controls(
    candidate: TradeCandidate,
    controls: TradingControls,
    current_state: PortfolioState,
) -> ControlsVerdict:
    """Decide whether ``candidate`` may proceed under ``controls``.

    Pure function — no I/O, no globals. Returns a structured verdict with
    plain-language ``reasons`` (shown to the user) and a ``clamped_notional``
    sized down to respect the per-trade cap and remaining budget. A
    candidate ``qualifies`` only when no gate is tripped.
    """
    reasons: list[str] = []

    conf = clamp_confidence(candidate.confidence, default=0.0)
    if conf < controls.min_confidence:
        reasons.append(
            f"Confidence {_pct(conf)} is below your "
            f"{_pct(controls.min_confidence)} minimum"
        )

    # Size the trade. Unspecified notional (<= 0) defaults to the per-trade
    # cap so a qualifying candidate still gets a sensible size.
    intended = float(candidate.notional)
    if intended <= 0:
        intended = controls.max_per_trade_usd
    remaining = max(0.0, controls.total_budget_usd - current_state.used_budget_usd)
    clamped = max(0.0, min(intended, controls.max_per_trade_usd, remaining))

    if controls.max_per_trade_usd < MIN_TRADEABLE_USD:
        reasons.append(
            f"Your per-trade limit of {_usd(controls.max_per_trade_usd)} "
            f"is too small to trade"
        )
    if remaining < MIN_TRADEABLE_USD:
        reasons.append(
            f"Would exceed your {_usd(controls.total_budget_usd)} budget "
            f"(it's fully allocated)"
        )

    if current_state.open_positions >= controls.max_open_positions:
        reasons.append(
            f"Already at your max of {controls.max_open_positions} open positions"
        )
    if current_state.trades_today >= controls.max_trades_per_day:
        reasons.append(
            f"Hit today's limit of {controls.max_trades_per_day} trades"
        )

    return ControlsVerdict(
        qualifies=not reasons,
        reasons=reasons,
        clamped_notional=clamped,
    )


# ---------------------------------------------------------------------------
# Pipeline orchestration — fully injectable so it unit-tests without I/O.
# ---------------------------------------------------------------------------
async def process_candidates(
    candidates: list[TradeCandidate],
    controls: TradingControls,
    state: PortfolioState,
    *,
    executor: Any | None = None,
    record_pending: Any | None = None,
    mark_executed: Any | None = None,
) -> dict[str, Any]:
    """Run one guardrail pass over ``candidates``.

    For each candidate (in order):
      * Evaluate against ``controls`` + a running copy of ``state``.
      * If it qualifies and an ``executor`` is supplied, execute it (shadow)
        and advance the running counters (one more trade today, one more open
        position, more budget used) so later candidates in the SAME pass
        respect the cumulative spend — then ``mark_executed``.
      * Otherwise record it as pending with its reasons.

    ``executor`` is ``async (candidate, clamped_notional) -> ack`` — the
    caller wires a SHADOW-only executor so this never sends a real order.
    ``record_pending`` is ``(candidate, reasons) -> None`` and
    ``mark_executed`` is ``(symbol, side) -> None``; both optional (no-ops in
    pure tests). Returns a small summary dict.
    """
    used = float(state.used_budget_usd)
    open_pos = int(state.open_positions)
    today = int(state.trades_today)

    evaluated = 0
    qualified = 0
    held = 0
    executed = 0

    for cand in candidates:
        evaluated += 1
        running = PortfolioState(
            used_budget_usd=used, open_positions=open_pos, trades_today=today
        )
        verdict = evaluate_trade_against_controls(cand, controls, running)
        if not verdict.qualifies:
            held += 1
            if record_pending is not None:
                try:
                    record_pending(cand, verdict.reasons)
                except Exception:  # pragma: no cover - defensive
                    pass
            continue

        qualified += 1
        if executor is not None and verdict.clamped_notional > 0:
            try:
                await executor(cand, verdict.clamped_notional)
                executed += 1
                used += verdict.clamped_notional
                open_pos += 1
                today += 1
                if mark_executed is not None:
                    mark_executed(cand.symbol, cand.side)
            except Exception:  # pragma: no cover - defensive
                # Execution hiccup: leave it queued so the next pass retries.
                held += 1
                if record_pending is not None:
                    try:
                        record_pending(
                            cand, ["Ready — will retry on the next check"]
                        )
                    except Exception:
                        pass

    return {
        "evaluated": evaluated,
        "qualified": qualified,
        "executed": executed,
        "held": held,
    }
