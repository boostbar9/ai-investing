"""Phase 29: intraday walk-forward backtest + lookahead audit.

Tunes :class:`IntradayTrendFollowing` parameters on a rolling window of
intraday trading days, evaluates the refitted parameter set on a
held-out OOS slice of subsequent days, and gates promotion through the
same Sharpe-margin / drawdown-non-regression rule used by the daily
walk-forward.

Three intentional differences from ``packages/backtests/walk_forward.py``
(which operates on daily close-only series):

1. The "asset" here is a {symbol -> 5-min OHLCV} panel, not a single
   daily close series. Each session day is the smallest atomic unit
   used for the walk-forward split (a session can never be split in
   half — that would create same-day train/test contamination).

2. The strategy under test is intraday-aware: positions are forced
   flat by ``exit_block_minutes`` before the close, so the simulator
   doesn't need overnight gap handling. EOD-flat by construction
   means the equity curve is the product of per-day intraday returns.

3. A dedicated :func:`audit_lookahead` helper sanity-checks that no
   feature used by the strategy at bar ``t`` depends on bar ``t+k``
   information (k > 0). This is the #1 cause of "great backtest,
   awful live" — we run the audit before every promotion decision
   and refuse to promote if a leak is detected.

The cost model is the same per-side bps charged on turnover as the
daily harness. For 5-min bars on liquid US ETFs, 2bps slippage + 1bp
spread per side is realistic; turnover is typically 1-2 round-trips
per session for the ORB+VWAP-trail strategy, so cost drag is well
under 50bps/day net.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from packages.backtests.champion_challenger import (
    annualized_sharpe,
    max_drawdown,
    promotion_gate,
)
from packages.features.intraday import compute_intraday_features
from packages.strategies.intraday_trend import (
    DEFAULT_ENTRY_BLOCK_MIN,
    DEFAULT_EXIT_BLOCK_MIN,
    DEFAULT_OPENING_RANGE_MIN,
    DEFAULT_STOP_LOSS,
    IntradayTrendFollowing,
)

log = logging.getLogger("intraday_walk_forward")


# ---------------------------------------------------------------------------
# Parameter set + grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntradayParamSet:
    """Tunable knobs for :class:`IntradayTrendFollowing`.

    Kept intentionally small — the goal is to detect parameter drift,
    not to do a full hyperparameter search. Adding more knobs without
    adding more data is how walk-forward turns into curve-fitting.
    """

    opening_range_minutes: int = DEFAULT_OPENING_RANGE_MIN
    entry_block_minutes: int = DEFAULT_ENTRY_BLOCK_MIN
    exit_block_minutes: int = DEFAULT_EXIT_BLOCK_MIN
    stop_loss: float = DEFAULT_STOP_LOSS

    def as_dict(self) -> dict[str, float]:
        return {
            "opening_range_minutes": self.opening_range_minutes,
            "entry_block_minutes": self.entry_block_minutes,
            "exit_block_minutes": self.exit_block_minutes,
            "stop_loss": self.stop_loss,
        }

    def build(self) -> IntradayTrendFollowing:
        return IntradayTrendFollowing(
            opening_range_minutes=self.opening_range_minutes,
            entry_block_minutes=self.entry_block_minutes,
            exit_block_minutes=self.exit_block_minutes,
            stop_loss=self.stop_loss,
        )


DEFAULT_INTRADAY_GRID: tuple[IntradayParamSet, ...] = tuple(
    IntradayParamSet(
        opening_range_minutes=orm,
        entry_block_minutes=ebm,
        exit_block_minutes=DEFAULT_EXIT_BLOCK_MIN,
        stop_loss=sl,
    )
    for orm in (15, 30, 45)
    for ebm in (10, 15, 30)
    for sl in (0.008, 0.012, 0.018)
)
"""Default 3x3x3 = 27-point grid. Small on purpose."""


# ---------------------------------------------------------------------------
# Cost model (intraday-aware)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntradayCostModel:
    """Round-trip per-side bps applied to turnover on 5-min bars.

    For liquid US equity ETFs at the share counts a $300 daily float
    produces, 2bps slippage + 1bp spread per side is realistic. Override
    only with TCA evidence.
    """

    slippage_bps: float = 2.0
    spread_bps: float = 1.0
    commission_bps: float = 0.0

    @property
    def per_side_bps(self) -> float:
        return self.slippage_bps + self.spread_bps + self.commission_bps

    def apply(self, returns: pd.Series, signal: pd.Series) -> pd.Series:
        """Subtract turnover-weighted costs from each bar's return."""
        turnover = signal.diff().abs().fillna(signal.abs().fillna(0.0))
        cost = turnover * (self.per_side_bps / 10_000.0)
        return returns - cost


DEFAULT_INTRADAY_COST_MODEL = IntradayCostModel()


# ---------------------------------------------------------------------------
# Equity curve from a panel of intraday bars
# ---------------------------------------------------------------------------


def equity_from_intraday_panel(
    panel: dict[str, pd.DataFrame],
    params: IntradayParamSet,
    *,
    cost_model: IntradayCostModel | None = None,
) -> pd.Series:
    """Simulate the intraday strategy on a {symbol: OHLCV} panel.

    Returns the net equity curve (starts at 1.0) net of transaction costs.
    The simulator executes weights at the *next* bar — the same
    no-lookahead convention as the daily harness.
    """
    if not panel:
        return pd.Series([1.0], dtype=float)

    strat = params.build()
    weights = strat.generate_weights_for_panel(panel)
    if weights.empty:
        return pd.Series([1.0], index=[next(iter(panel.values())).index[0]])

    # Build aligned close panel so per-bar returns are unambiguous.
    closes = pd.concat(
        {sym: df["close"] for sym, df in panel.items()}, axis=1
    ).reindex(weights.index).ffill()
    bar_rets = closes.pct_change().fillna(0.0)

    # Execute on the NEXT bar after a weight change — no lookahead.
    held = weights.shift(1).fillna(0.0)
    gross = (held * bar_rets).sum(axis=1)

    # Turnover per bar = sum of per-symbol |weight change|.
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))

    cm = cost_model if cost_model is not None else DEFAULT_INTRADAY_COST_MODEL
    cost = turnover * (cm.per_side_bps / 10_000.0)
    net = gross - cost

    return (1.0 + net).cumprod()


# ---------------------------------------------------------------------------
# Walk-forward driver (split by session day, not by N-bar window)
# ---------------------------------------------------------------------------


def _session_days_in_panel(panel: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    """Distinct US/Eastern session dates present in any symbol's bars."""
    if not panel:
        return []
    days: set[pd.Timestamp] = set()
    for df in panel.values():
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            continue
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        et = idx.tz_convert("America/New_York")
        for d in et.normalize().unique():
            days.add(pd.Timestamp(d.date()))
    return sorted(days)


def _slice_panel_by_days(
    panel: dict[str, pd.DataFrame],
    days: list[pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    """Restrict each symbol's frame to bars whose ET session date is in ``days``."""
    if not days:
        return {sym: df.iloc[0:0].copy() for sym, df in panel.items()}
    keep = {pd.Timestamp(d).date() for d in days}
    out: dict[str, pd.DataFrame] = {}
    for sym, df in panel.items():
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        et = idx.tz_convert("America/New_York")
        mask = pd.Series(False, index=df.index)
        mask.values[:] = pd.Index(et.date).isin(keep)
        out[sym] = df.loc[mask].copy()
    return out


@dataclass
class IntradayWalkForwardResult:
    champion: IntradayParamSet
    challenger: IntradayParamSet
    promoted: bool
    reasons: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    lookahead_clean: bool = True
    lookahead_findings: list[str] = field(default_factory=list)


def _best_params_in_sample(
    panel: dict[str, pd.DataFrame],
    grid: tuple[IntradayParamSet, ...],
) -> tuple[IntradayParamSet, float]:
    best = grid[0]
    best_s = -float("inf")
    for params in grid:
        eq = equity_from_intraday_panel(panel, params)
        s = annualized_sharpe(eq)
        if max_drawdown(eq) > 0.40:
            s -= 1.0  # blow-up penalty
        if s > best_s:
            best_s = s
            best = params
    return best, best_s


def run_intraday_walk_forward(
    panel: dict[str, pd.DataFrame],
    *,
    champion: IntradayParamSet,
    grid: tuple[IntradayParamSet, ...] = DEFAULT_INTRADAY_GRID,
    in_sample_days: int = 20,
    out_of_sample_days: int = 5,
    sharpe_margin: float = 0.10,
) -> IntradayWalkForwardResult:
    """Refit on the last ``in_sample_days``, test on the next ``out_of_sample_days``.

    Splits by US/Eastern session date — never mid-session. Runs a
    lookahead audit before computing the promotion verdict so a leaking
    challenger gets blocked even if its OOS Sharpe looks great.
    """
    days = _session_days_in_panel(panel)
    needed = in_sample_days + out_of_sample_days
    if len(days) < needed:
        return IntradayWalkForwardResult(
            champion=champion,
            challenger=champion,
            promoted=False,
            reasons=[
                f"insufficient session days: have {len(days)}, need {needed}"
            ],
        )

    is_days = days[-needed:-out_of_sample_days]
    oos_days = days[-out_of_sample_days:]
    is_panel = _slice_panel_by_days(panel, is_days)
    oos_panel = _slice_panel_by_days(panel, oos_days)

    challenger, in_sample_sharpe = _best_params_in_sample(is_panel, grid)

    # Lookahead audit BEFORE promotion math — a leak invalidates the verdict.
    audit = audit_lookahead(oos_panel, challenger)

    champ_eq = equity_from_intraday_panel(oos_panel, champion)
    chal_eq = equity_from_intraday_panel(oos_panel, challenger)
    chal_eq, champ_eq = chal_eq.align(champ_eq, join="inner")

    verdict = promotion_gate(
        champ_eq,
        chal_eq,
        min_days=min(30, len(champ_eq)),
        sharpe_margin=sharpe_margin,
    )
    promoted = bool(verdict.promote and audit.clean)
    reasons = list(verdict.reasons)
    if not audit.clean:
        reasons.append(f"lookahead audit failed: {'; '.join(audit.findings)}")

    return IntradayWalkForwardResult(
        champion=champion,
        challenger=challenger,
        promoted=promoted,
        reasons=reasons,
        metrics=verdict.metrics,
        in_sample_sharpe=float(in_sample_sharpe),
        out_of_sample_sharpe=float(annualized_sharpe(chal_eq)),
        lookahead_clean=audit.clean,
        lookahead_findings=list(audit.findings),
    )


# ---------------------------------------------------------------------------
# Lookahead audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LookaheadAudit:
    """Result of a lookahead-leak check.

    ``clean`` is True when no test fired. ``findings`` carries one
    human-readable line per failure mode so a CI run can show the
    operator exactly where the leak was detected.
    """

    clean: bool
    findings: list[str]


def audit_lookahead(
    panel: dict[str, pd.DataFrame],
    params: IntradayParamSet,
) -> LookaheadAudit:
    """Sanity-check :class:`IntradayTrendFollowing` for future-information leaks.

    Three independent tests, each chosen to surface a different leak class:

    1. **Truncation invariance** — slicing the bar history to end at bar
       ``t`` must produce the same weight at ``t`` as the full panel.
       A look-ahead feature would change once the future bars appear.

    2. **Opening-range stability** — the ``opening_range_high`` value at
       any bar past the OR window must not change when later bars are
       removed. This catches accidental whole-session aggregations.

    3. **No same-bar reaction** — the very first bar of a session must
       carry weight 0 (no trade), since the OR window hasn't even
       started yet and no signal is computable.

    Returns a :class:`LookaheadAudit` summarising every failure. An
    empty ``findings`` list means the strategy passed.
    """
    findings: list[str] = []

    if not panel:
        return LookaheadAudit(clean=True, findings=findings)

    strat = params.build()

    # Truncation invariance: compare weights at a mid-session bar with
    # both the full panel and a panel truncated at that bar.
    for sym, df in panel.items():
        if len(df) < 30:
            continue
        full_weights = strat.generate_weights_for_panel({sym: df})
        if full_weights.empty:
            continue
        # Pick a bar a third of the way through so OR is established
        # but plenty of "future" bars exist to potentially leak.
        cut = len(df) // 3
        if cut < 10:
            continue
        truncated = {sym: df.iloc[: cut + 1].copy()}
        trunc_weights = strat.generate_weights_for_panel(truncated)
        if trunc_weights.empty:
            continue
        # Compare the last bar's weight in the truncated run with the
        # value at the same timestamp in the full run.
        ts = trunc_weights.index[-1]
        if ts not in full_weights.index:
            continue
        full_w = float(full_weights.loc[ts, sym])
        trunc_w = float(trunc_weights.loc[ts, sym])
        if abs(full_w - trunc_w) > 1e-9:
            findings.append(
                f"truncation_invariance: {sym} @ {ts} differs full={full_w:.6f} "
                f"trunc={trunc_w:.6f}"
            )

    # Opening-range stability: features at bar t past the OR window must
    # not depend on bars after t.
    for sym, df in panel.items():
        if len(df) < 30:
            continue
        full = compute_intraday_features(
            df, opening_range_minutes=params.opening_range_minutes
        )
        cut = len(df) // 2
        truncated = compute_intraday_features(
            df.iloc[: cut + 1],
            opening_range_minutes=params.opening_range_minutes,
        )
        ts = truncated.index[-1]
        if ts not in full.index:
            continue
        # Compare opening_range_high — most leak-prone aggregate.
        full_orh = full.loc[ts, "opening_range_high"]
        trunc_orh = truncated.loc[ts, "opening_range_high"]
        if pd.isna(full_orh) and pd.isna(trunc_orh):
            continue
        if pd.isna(full_orh) != pd.isna(trunc_orh) or (
            not pd.isna(full_orh)
            and abs(float(full_orh) - float(trunc_orh)) > 1e-9
        ):
            findings.append(
                f"opening_range_stability: {sym} @ {ts} differs "
                f"full={full_orh} trunc={trunc_orh}"
            )

    # No same-bar reaction: the OR window hasn't even started at the
    # first bar of a session, so the strategy must hold zero weight.
    for sym, df in panel.items():
        if df.empty:
            continue
        weights = strat.generate_weights_for_panel({sym: df})
        if weights.empty:
            continue
        first_w = float(weights.iloc[0, 0])
        if first_w > 1e-9:
            findings.append(
                f"no_same_bar_reaction: {sym} first-bar weight={first_w:.6f} "
                f"(must be 0.0)"
            )

    return LookaheadAudit(clean=not findings, findings=findings)
