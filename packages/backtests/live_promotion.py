"""Phase 5 live promotion + canary capital (§15, §16, issue #8).

Two responsibilities:

1. **Live readiness gate** — Hard checks that MUST pass before
   ``ENABLE_LIVE_TRADING`` may flip on:
     * 60 consecutive trading days of paper history
     * max drawdown over the window < 8%
     * annualized Sharpe over the window > 0.8
     * the operator has explicitly set ``ENABLE_LIVE_TRADING=true`` in env
       (defense-in-depth: a gate verdict alone is not enough)

2. **Canary capital schedule** — Once promoted, allocate only a small fraction
   of available capital to live trading at first, ratcheting up only after
   each tier has held for ``dwell_days`` consecutive live days with the same
   acceptance thresholds (max DD < 8%, Sharpe > 0.8) measured on the LIVE
   curve.

The default canary schedule is **5% → 10% → 25% → 100%** with a 30-day dwell
per tier. The function ``canary_fraction`` returns the current ceiling; the
execution layer multiplies position sizing by this fraction.

All decisions are deterministic functions of (paper_equity, live_equity, now)
so they are trivially testable and replayable from audit data.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from packages.backtests.champion_challenger import (
    annualized_sharpe,
    max_drawdown,
)

# Acceptance thresholds from §16. Locked.
PAPER_MIN_DAYS = 60
PAPER_MAX_DD = 0.08
PAPER_MIN_SHARPE = 0.8

# Canary schedule. Each tuple = (capital_fraction, dwell_days_required_to_advance).
# The final tier (1.0) has no further advance.
DEFAULT_CANARY_SCHEDULE: tuple[tuple[float, int], ...] = (
    (0.05, 30),
    (0.10, 30),
    (0.25, 30),
    (1.00, 0),
)


# ---------------------------------------------------------------------------
# 1. Live readiness gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveReadinessVerdict:
    ready: bool
    reasons: list[str]
    metrics: dict[str, float]


def live_readiness_gate(
    paper_equity: pd.Series,
    *,
    min_days: int = PAPER_MIN_DAYS,
    max_dd: float = PAPER_MAX_DD,
    min_sharpe: float = PAPER_MIN_SHARPE,
    enable_flag: str | None = None,
) -> LiveReadinessVerdict:
    """Hard gate before any real capital may be deployed.

    ``enable_flag`` is the ``ENABLE_LIVE_TRADING`` env var. ALL conditions
    must hold. Defense in depth: a flag without metrics or metrics without
    a flag both fail closed.
    """
    if enable_flag is None:
        enable_flag = os.getenv("ENABLE_LIVE_TRADING", "")

    reasons: list[str] = []

    if enable_flag.strip().lower() not in {"true", "1", "yes", "on"}:
        reasons.append("ENABLE_LIVE_TRADING flag is not set to true")

    n = len(paper_equity)
    if n < min_days:
        reasons.append(
            f"only {n} paper trading days; need {min_days} consecutive"
        )
        return LiveReadinessVerdict(ready=False, reasons=reasons, metrics={})

    window = paper_equity.iloc[-min_days:]
    dd = max_drawdown(window)
    sharpe = annualized_sharpe(window)
    metrics = {
        "paper_days": float(n),
        "max_drawdown": dd,
        "sharpe": sharpe,
    }

    if dd >= max_dd:
        reasons.append(
            f"max drawdown {dd:.3f} ≥ allowed {max_dd:.3f} over last {min_days}d"
        )
    if sharpe <= min_sharpe:
        reasons.append(
            f"Sharpe {sharpe:.3f} ≤ required {min_sharpe:.3f} over last {min_days}d"
        )

    return LiveReadinessVerdict(
        ready=not reasons,
        reasons=reasons,
        metrics=metrics,
    )


def readiness_report(
    equity_values: Sequence[float],
    *,
    enable_flag: str | None = None,
    min_days: int = PAPER_MIN_DAYS,
    max_dd: float = PAPER_MAX_DD,
    min_sharpe: float = PAPER_MIN_SHARPE,
    realized: Mapping[str, Any] | None = None,
) -> dict:
    """JSON-able readiness view for callers that don't speak pandas.

    Builds the ``pd.Series`` the gate expects from a plain list of equity
    marks (so the cockpit never has to import pandas), runs
    ``live_readiness_gate`` and flattens the verdict into a display payload:
    ``ready``, ``reasons``, ``metrics`` (paper_days / max_drawdown / sharpe —
    the latter two ``None`` until there is enough history), the ``thresholds``
    being checked, and the read-only ``enable_live_trading`` flag state.

    ``realized`` is the optional FIFO round-trip stat block from
    ``performance_stats.realized_trade_stats``. When supplied, its REAL profit
    factor / expectancy / win rate (and measured/unmeasured counts) are surfaced
    under a ``realized`` key so the operator can SEE the true track record
    alongside the gate verdict. This is **display only**: the gate thresholds
    are unchanged and the realized numbers do NOT loosen or alter the ready
    decision — the gate stays exactly as fail-closed as before.

    Fail-safe: a short/empty series yields ``ready=False`` with a clear reason
    (never ``ready=True`` on uncertainty). This function only reads and
    reports — it never enables live trading or touches any flag.
    """
    clean: list[float] = []
    for v in equity_values:
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            clean.append(float(v))
    series = pd.Series(clean, dtype=float)

    flag = enable_flag if enable_flag is not None else os.getenv("ENABLE_LIVE_TRADING", "")
    flag_on = flag.strip().lower() in {"true", "1", "yes", "on"}

    verdict = live_readiness_gate(
        series,
        min_days=min_days,
        max_dd=max_dd,
        min_sharpe=min_sharpe,
        enable_flag=flag,
    )
    report = {
        "ready": verdict.ready,
        "reasons": verdict.reasons,
        "metrics": {
            "paper_days": len(series),
            "max_drawdown": verdict.metrics.get("max_drawdown"),
            "sharpe": verdict.metrics.get("sharpe"),
        },
        "thresholds": {
            "min_days": min_days,
            "max_dd": max_dd,
            "min_sharpe": min_sharpe,
        },
        "enable_live_trading": flag_on,
    }
    if realized is not None:
        report["realized"] = {
            "profit_factor": realized.get("profit_factor"),
            "expectancy": realized.get("expectancy"),
            "round_trip_win_rate": realized.get(
                "round_trip_win_rate", realized.get("win_rate")
            ),
            "closed_round_trips": realized.get(
                "closed_round_trips", realized.get("total_round_trips", 0)
            ),
            "unmeasured_round_trips": realized.get("unmeasured_round_trips", 0),
            "confidence": realized.get("confidence"),
            "insufficient_data": realized.get("insufficient_data", True),
        }
    return report


# ---------------------------------------------------------------------------
# 2. Canary capital schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryState:
    """Snapshot of the current canary tier.

    ``tier_index`` is 0-based. ``fraction`` is the capital ceiling in [0, 1].
    ``days_in_tier`` is how many live trading days the curve has spent in
    this tier so far. ``next_fraction`` is what advancing one tier would
    grant; ``None`` if already at the top.
    """

    tier_index: int
    fraction: float
    days_in_tier: int
    dwell_required: int
    next_fraction: float | None
    reasons: list[str] = field(default_factory=list)


def canary_fraction(
    live_equity: pd.Series,
    *,
    schedule: tuple[tuple[float, int], ...] = DEFAULT_CANARY_SCHEDULE,
    max_dd: float = PAPER_MAX_DD,
    min_sharpe: float = PAPER_MIN_SHARPE,
) -> CanaryState:
    """Return the current canary capital fraction.

    Advancement rule: to step from tier ``i`` to tier ``i+1``, the live
    equity series must have spent at least ``schedule[i][1]`` days in tier
    ``i`` AND the rolling Sharpe / max-DD over that window must still meet
    the §16 acceptance thresholds. Otherwise we stay put.

    The function is monotonic in ``live_equity`` length given the same
    inputs — we always start at tier 0 and walk forward. This means the
    caller does NOT need to remember the previous tier; the curve is the
    source of truth (auditable).
    """
    n = len(live_equity)
    current_tier = 0
    days_remaining_in_curve = n
    reasons: list[str] = []

    # Walk tiers forward. Each tier "consumes" dwell_days from the curve if
    # the metrics during those days satisfy the gate. The final tier never
    # advances.
    cursor_start = 0
    while current_tier < len(schedule) - 1:
        dwell = schedule[current_tier][1]
        if days_remaining_in_curve < dwell:
            reasons.append(
                f"tier {current_tier}: {days_remaining_in_curve}/{dwell} dwell days"
            )
            break

        window = live_equity.iloc[cursor_start : cursor_start + dwell]
        dd = max_drawdown(window)
        sharpe = annualized_sharpe(window)
        if dd >= max_dd:
            reasons.append(
                f"tier {current_tier}: live max-DD {dd:.3f} ≥ {max_dd:.3f}"
            )
            break
        if sharpe <= min_sharpe:
            reasons.append(
                f"tier {current_tier}: live Sharpe {sharpe:.3f} ≤ {min_sharpe:.3f}"
            )
            break

        # Tier passed; advance.
        cursor_start += dwell
        days_remaining_in_curve -= dwell
        current_tier += 1

    fraction, dwell_required = schedule[current_tier]
    next_fraction = (
        schedule[current_tier + 1][0]
        if current_tier + 1 < len(schedule)
        else None
    )
    return CanaryState(
        tier_index=current_tier,
        fraction=fraction,
        days_in_tier=days_remaining_in_curve,
        dwell_required=dwell_required,
        next_fraction=next_fraction,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# 3. Convenience: full promotion decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionDecision:
    """End-to-end answer to: how much real capital may we deploy right now?"""

    live_enabled: bool
    capital_fraction: float
    readiness: LiveReadinessVerdict
    canary: CanaryState | None


def decide_live_capital(
    paper_equity: pd.Series,
    live_equity: pd.Series | None = None,
    *,
    enable_flag: str | None = None,
) -> PromotionDecision:
    """Top-level: should the broker route any live orders, and at what size?

    Pipeline:
      1. Run the live-readiness gate against paper history. If it fails,
         return ``live_enabled=False, capital_fraction=0`` immediately.
      2. Otherwise, compute the current canary tier from the live equity
         curve (empty curve → tier 0, fraction 5%).
      3. Return a decision the execution layer can multiply against its
         intended notional.
    """
    readiness = live_readiness_gate(paper_equity, enable_flag=enable_flag)
    if not readiness.ready:
        return PromotionDecision(
            live_enabled=False,
            capital_fraction=0.0,
            readiness=readiness,
            canary=None,
        )

    series = live_equity if live_equity is not None else pd.Series(dtype=float)
    canary = canary_fraction(series)
    return PromotionDecision(
        live_enabled=True,
        capital_fraction=canary.fraction,
        readiness=readiness,
        canary=canary,
    )
