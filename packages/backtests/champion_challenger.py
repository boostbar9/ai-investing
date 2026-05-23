"""Champion/Challenger gating (§10, issues #1 + #2).

Two related gates protect the live strategy from drift and unproven challengers:

1. **Promotion gate** (#1) — A challenger may only replace the champion after
   ``min_days`` consecutive trading days of OOS outperformance on three
   metrics (Sharpe, max-DD, CAGR), with each margin exceeding its threshold.

2. **Sharpe-drop gate** (#2) — Nightly, the champion's rolling Sharpe is
   compared to its long-window baseline. If it falls by more than
   ``sharpe_drop_max`` (default 10%), CI blocks merges and the operator is
   paged. Mirrors §10's "blocks merges on Sharpe drop > 10%".

Both gates are deterministic functions of equity-curve series so they are
trivially testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def _rets(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def annualized_sharpe(equity: pd.Series) -> float:
    r = _rets(equity)
    if r.std() == 0 or len(r) == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * r.mean() / r.std())


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(abs(dd.min()))


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = len(equity) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


# ---------------------------------------------------------------------------
# Promotion gate (#1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionVerdict:
    promote: bool
    days_outperformed: int
    reasons: list[str]
    metrics: dict[str, float]


def promotion_gate(
    champion: pd.Series,
    challenger: pd.Series,
    *,
    min_days: int = 30,
    sharpe_margin: float = 0.10,  # challenger Sharpe must exceed by ≥ this
    dd_margin: float = 0.0,       # challenger max-DD must be ≤ champion's
    cagr_margin: float = 0.0,     # challenger CAGR must be ≥ champion's
) -> PromotionVerdict:
    """Decide whether a challenger can replace the live champion.

    The two series must share an index covering the OOS evaluation window.
    The verdict is intentionally conservative: ALL margins must hold for the
    LAST ``min_days`` trading days simultaneously.
    """
    if len(champion) != len(challenger):
        return PromotionVerdict(
            promote=False,
            days_outperformed=0,
            reasons=["series length mismatch"],
            metrics={},
        )
    n = len(champion)
    if n < min_days:
        return PromotionVerdict(
            promote=False,
            days_outperformed=n,
            reasons=[f"only {n} days of OOS data; need {min_days}"],
            metrics={},
        )

    window = slice(n - min_days, n)
    champ_win = champion.iloc[window]
    chal_win = challenger.iloc[window]

    metrics = {
        "champion_sharpe": annualized_sharpe(champ_win),
        "challenger_sharpe": annualized_sharpe(chal_win),
        "champion_max_dd": max_drawdown(champ_win),
        "challenger_max_dd": max_drawdown(chal_win),
        "champion_cagr": cagr(champ_win),
        "challenger_cagr": cagr(chal_win),
    }

    reasons: list[str] = []

    # Reject NaN/inf metrics outright. NaN comparisons always return False, so
    # the regression checks below would silently let a broken challenger through.
    # A bot that "promotes" on garbage data is the worst possible failure mode.
    for key, val in metrics.items():
        if val is None or (isinstance(val, float) and not math.isfinite(val)):
            reasons.append(f"{key} is not finite ({val!r}) — refusing to promote")
    if reasons:
        return PromotionVerdict(
            promote=False,
            days_outperformed=0,
            reasons=reasons,
            metrics=metrics,
        )

    if metrics["challenger_sharpe"] - metrics["champion_sharpe"] < sharpe_margin:
        reasons.append(
            f"sharpe margin "
            f"{metrics['challenger_sharpe'] - metrics['champion_sharpe']:.3f} "
            f"< required {sharpe_margin}"
        )
    if metrics["challenger_max_dd"] - metrics["champion_max_dd"] > dd_margin:
        reasons.append(
            f"max-DD regression: challenger "
            f"{metrics['challenger_max_dd']:.3f} > "
            f"champion {metrics['champion_max_dd']:.3f} + {dd_margin}"
        )
    if metrics["champion_cagr"] - metrics["challenger_cagr"] > cagr_margin:
        reasons.append(
            f"CAGR regression: challenger "
            f"{metrics['challenger_cagr']:.3f} < "
            f"champion {metrics['champion_cagr']:.3f} - {cagr_margin}"
        )

    return PromotionVerdict(
        promote=not reasons,
        days_outperformed=min_days if not reasons else 0,
        reasons=reasons,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Sharpe-drop gate (#2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharpeDropVerdict:
    blocked: bool
    drop_ratio: float
    baseline_sharpe: float
    recent_sharpe: float
    reason: str | None


def sharpe_drop_gate(
    equity: pd.Series,
    *,
    recent_window: int = 30,
    baseline_window: int = 252,
    sharpe_drop_max: float = 0.10,
) -> SharpeDropVerdict:
    """Block deploys / page operator when rolling Sharpe drops > 10%.

    ``drop_ratio`` = (baseline - recent) / |baseline|. Positive values mean a
    drop. The gate fires when ``drop_ratio > sharpe_drop_max``.

    If baseline Sharpe is ~0 we fall back to absolute deltas to avoid
    division blow-up.
    """
    if len(equity) < recent_window + 1:
        return SharpeDropVerdict(
            blocked=False,
            drop_ratio=0.0,
            baseline_sharpe=0.0,
            recent_sharpe=0.0,
            reason=f"insufficient history ({len(equity)} bars)",
        )

    recent = equity.iloc[-recent_window:]
    baseline = equity.iloc[-(baseline_window + recent_window) : -recent_window]
    if len(baseline) < 30:
        # not enough baseline yet; don't fire
        return SharpeDropVerdict(
            blocked=False,
            drop_ratio=0.0,
            baseline_sharpe=0.0,
            recent_sharpe=annualized_sharpe(recent),
            reason="insufficient baseline window",
        )

    s_base = annualized_sharpe(baseline)
    s_recent = annualized_sharpe(recent)

    if abs(s_base) < 1e-6:
        drop_ratio = float(s_base - s_recent)  # absolute
    else:
        drop_ratio = float((s_base - s_recent) / abs(s_base))

    if drop_ratio > sharpe_drop_max:
        return SharpeDropVerdict(
            blocked=True,
            drop_ratio=drop_ratio,
            baseline_sharpe=s_base,
            recent_sharpe=s_recent,
            reason=(
                f"rolling Sharpe dropped {drop_ratio * 100:.1f}% "
                f"(baseline {s_base:.2f} → recent {s_recent:.2f})"
            ),
        )
    return SharpeDropVerdict(
        blocked=False,
        drop_ratio=drop_ratio,
        baseline_sharpe=s_base,
        recent_sharpe=s_recent,
        reason=None,
    )
