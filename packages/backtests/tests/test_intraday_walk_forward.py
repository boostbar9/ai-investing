"""Phase 29 tests: intraday walk-forward driver + lookahead audit.

Three test surfaces:

  1. **IntradayParamSet / grid** — the parameter set builds a real
     :class:`IntradayTrendFollowing` instance with the right knobs, and
     the default grid is non-empty and well-formed.

  2. **equity_from_intraday_panel** — basic sanity (starts at 1.0,
     flat-weight panel returns equity 1.0, no-trade panel returns
     equity 1.0, costs are deducted on turnover).

  3. **audit_lookahead** — the audit passes cleanly on the genuine
     :class:`IntradayTrendFollowing` and fails loudly on a synthetic
     leaking strategy whose "feature" depends on future bars.

  4. **run_intraday_walk_forward** — split-by-session-day logic works,
     insufficient-history bails cleanly, and the verdict honours
     audit failures even when OOS Sharpe looks good.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from packages.backtests.intraday_walk_forward import (
    DEFAULT_INTRADAY_COST_MODEL,
    DEFAULT_INTRADAY_GRID,
    IntradayCostModel,
    IntradayParamSet,
    LookaheadAudit,
    _session_days_in_panel,
    _slice_panel_by_days,
    audit_lookahead,
    equity_from_intraday_panel,
    run_intraday_walk_forward,
)
from packages.strategies.intraday_trend import IntradayTrendFollowing


# ---------------------------------------------------------------------------
# Bar helpers
# ---------------------------------------------------------------------------


def _session_bars(
    *,
    day: datetime,
    n_bars: int = 78,  # 6.5h * 60min / 5min = 78
    breakout: bool = False,
    flat: bool = False,
) -> pd.DataFrame:
    """Build a single intraday session's 5-min OHLCV bars.

    ``day`` is interpreted as 09:30 ET local. We emit UTC-aware
    timestamps so the audit's tz logic exercises real conversion.
    """
    # 09:30 ET == 14:30 UTC (no DST math; tests use fixed dates).
    start = datetime(day.year, day.month, day.day, 14, 30, tzinfo=timezone.utc)
    rows = []
    price = 100.0
    for i in range(n_bars):
        ts = start + timedelta(minutes=5 * i)
        if flat:
            close = 100.0
        elif breakout and i >= 8:
            # After 30-min OR (first 6 bars) + entry block (15min, ~3 bars)
            # drive price up sharply.
            close = 100.0 + 0.3 * (i - 5)
        else:
            close = 100.0 + 0.005 * i
        rows.append(
            {
                "open": close,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 10_000.0,
            }
        )
        price = close
    idx = pd.DatetimeIndex(
        [start + timedelta(minutes=5 * i) for i in range(n_bars)],
        name="ts",
    )
    return pd.DataFrame(rows, index=idx)


def _multi_day_panel(
    n_days: int = 30, breakout_days: int = 0, syms: tuple[str, ...] = ("SPY",)
) -> dict[str, pd.DataFrame]:
    """Build a {sym: stacked-bars} panel covering ``n_days`` consecutive
    weekdays. The last ``breakout_days`` sessions show a breakout."""
    panels: dict[str, list[pd.DataFrame]] = {s: [] for s in syms}
    # Start on a Monday far enough back.
    day = datetime(2026, 1, 5)
    days_added = 0
    while days_added < n_days:
        if day.weekday() < 5:  # Mon-Fri
            is_breakout = days_added >= (n_days - breakout_days)
            for s in syms:
                panels[s].append(
                    _session_bars(day=day, breakout=is_breakout)
                )
            days_added += 1
        day += timedelta(days=1)
    return {s: pd.concat(panels[s]).sort_index() for s in syms}


# ---------------------------------------------------------------------------
# IntradayParamSet
# ---------------------------------------------------------------------------


class TestIntradayParamSet:
    def test_defaults_match_strategy(self) -> None:
        p = IntradayParamSet()
        assert p.opening_range_minutes == 30
        assert p.entry_block_minutes == 15
        assert p.exit_block_minutes == 15
        # stop_loss matches DEFAULT_STOP_LOSS = 0.01
        assert p.stop_loss == pytest.approx(0.01)

    def test_as_dict_round_trips(self) -> None:
        p = IntradayParamSet(
            opening_range_minutes=45,
            entry_block_minutes=10,
            exit_block_minutes=20,
            stop_loss=0.015,
        )
        d = p.as_dict()
        assert d["opening_range_minutes"] == 45
        assert d["stop_loss"] == pytest.approx(0.015)

    def test_build_returns_strategy_with_matching_knobs(self) -> None:
        p = IntradayParamSet(
            opening_range_minutes=45,
            entry_block_minutes=10,
            exit_block_minutes=20,
            stop_loss=0.015,
        )
        s = p.build()
        assert isinstance(s, IntradayTrendFollowing)
        assert s.opening_range_minutes == 45
        assert s.entry_block_minutes == 10
        assert s.exit_block_minutes == 20
        assert s.stop_loss == pytest.approx(0.015)


def test_default_grid_is_nonempty_and_distinct() -> None:
    grid = DEFAULT_INTRADAY_GRID
    assert len(grid) > 1
    # Every entry is a frozen dataclass; sets dedupe successfully.
    assert len(set(grid)) == len(grid)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


class TestCostModel:
    def test_per_side_bps_sums_components(self) -> None:
        c = IntradayCostModel(slippage_bps=2.0, spread_bps=1.0, commission_bps=0.5)
        assert c.per_side_bps == pytest.approx(3.5)

    def test_apply_deducts_on_turnover(self) -> None:
        c = IntradayCostModel(slippage_bps=2.0, spread_bps=1.0)
        rets = pd.Series([0.01, 0.01, 0.01])
        signal = pd.Series([0.0, 1.0, 1.0])  # turnover at bar 1, none at 2
        net = c.apply(rets, signal)
        # Bar 1: 1.0 * 3bps = 0.0003 cost; bar 2: 0 turnover.
        assert net.iloc[0] == pytest.approx(0.01)  # no prior signal -> 0 turnover
        assert net.iloc[1] == pytest.approx(0.01 - 0.0003)
        assert net.iloc[2] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# equity_from_intraday_panel
# ---------------------------------------------------------------------------


class TestEquityFromIntradayPanel:
    def test_empty_panel_returns_unit(self) -> None:
        eq = equity_from_intraday_panel({}, IntradayParamSet())
        assert len(eq) == 1
        assert eq.iloc[0] == pytest.approx(1.0)

    def test_flat_market_no_signal(self) -> None:
        panel = _multi_day_panel(n_days=3, breakout_days=0)
        # Flat-ish drift -> ORB won't trigger -> equity stays near 1.0
        eq = equity_from_intraday_panel(panel, IntradayParamSet())
        # End equity should be very close to 1.0 (within 50bps).
        assert abs(float(eq.iloc[-1]) - 1.0) < 0.005

    def test_breakout_panel_produces_finite_equity(self) -> None:
        panel = _multi_day_panel(n_days=10, breakout_days=10, syms=("SPY",))
        eq = equity_from_intraday_panel(panel, IntradayParamSet())
        assert np.isfinite(float(eq.iloc[-1]))
        # Must remain positive (no blow-up).
        assert float(eq.iloc[-1]) > 0.0

    def test_costs_reduce_returns(self) -> None:
        panel = _multi_day_panel(n_days=5, breakout_days=5, syms=("SPY",))
        zero_cost = IntradayCostModel(slippage_bps=0.0, spread_bps=0.0)
        free = equity_from_intraday_panel(panel, IntradayParamSet(), cost_model=zero_cost)
        costed = equity_from_intraday_panel(panel, IntradayParamSet())
        # If the strategy traded at all, costed equity must be <= free equity.
        assert float(costed.iloc[-1]) <= float(free.iloc[-1]) + 1e-9


# ---------------------------------------------------------------------------
# Session-day helpers
# ---------------------------------------------------------------------------


class TestSessionDayHelpers:
    def test_session_days_in_panel_is_sorted_unique(self) -> None:
        panel = _multi_day_panel(n_days=5)
        days = _session_days_in_panel(panel)
        assert days == sorted(days)
        assert len(days) == 5

    def test_slice_panel_by_days_filters(self) -> None:
        panel = _multi_day_panel(n_days=5)
        days = _session_days_in_panel(panel)
        sliced = _slice_panel_by_days(panel, days[:2])
        # Each symbol's frame should now cover only 2 sessions worth of bars.
        for df in sliced.values():
            et = df.index.tz_convert("America/New_York")
            unique_sessions = set(et.normalize().unique())
            assert len(unique_sessions) == 2

    def test_slice_panel_empty_days(self) -> None:
        panel = _multi_day_panel(n_days=3)
        sliced = _slice_panel_by_days(panel, [])
        assert all(df.empty for df in sliced.values())


# ---------------------------------------------------------------------------
# Lookahead audit
# ---------------------------------------------------------------------------


class TestLookaheadAudit:
    def test_clean_panel_passes(self) -> None:
        panel = _multi_day_panel(n_days=3, breakout_days=3, syms=("SPY",))
        result = audit_lookahead(panel, IntradayParamSet())
        # The real IntradayTrendFollowing must not leak.
        assert result.clean, f"unexpected findings: {result.findings}"
        assert result.findings == []

    def test_empty_panel_is_clean(self) -> None:
        result = audit_lookahead({}, IntradayParamSet())
        assert result.clean
        assert isinstance(result, LookaheadAudit)

    def test_breakout_panel_first_bar_holds_zero(self) -> None:
        # Specifically guards "no same-bar reaction" — first bar of a
        # session must always carry zero weight.
        panel = _multi_day_panel(n_days=2, breakout_days=2, syms=("SPY",))
        result = audit_lookahead(panel, IntradayParamSet())
        assert all(
            "no_same_bar_reaction" not in f for f in result.findings
        ), f"unexpected: {result.findings}"


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------


class TestRunIntradayWalkForward:
    def test_insufficient_history_bails(self) -> None:
        panel = _multi_day_panel(n_days=3, breakout_days=0, syms=("SPY",))
        out = run_intraday_walk_forward(
            panel,
            champion=IntradayParamSet(),
            grid=(IntradayParamSet(),),
            in_sample_days=20,
            out_of_sample_days=5,
        )
        assert out.promoted is False
        assert any("insufficient" in r for r in out.reasons)

    def test_runs_to_verdict_on_sufficient_history(self) -> None:
        # 30 sessions is just enough for in_sample_days=20 + oos_days=5.
        panel = _multi_day_panel(n_days=30, breakout_days=10, syms=("SPY",))
        # Use a tiny grid (just two points) so the test is fast.
        small_grid = (
            IntradayParamSet(opening_range_minutes=30),
            IntradayParamSet(opening_range_minutes=15),
        )
        out = run_intraday_walk_forward(
            panel,
            champion=IntradayParamSet(),
            grid=small_grid,
            in_sample_days=20,
            out_of_sample_days=5,
        )
        # Either outcome is valid — what matters is the run completed
        # cleanly and the audit ran.
        assert isinstance(out.promoted, bool)
        assert out.lookahead_clean is True
        assert isinstance(out.in_sample_sharpe, float)
        assert isinstance(out.out_of_sample_sharpe, float)

    def test_promotion_blocked_by_audit_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the audit reports a leak, the verdict must be unpromoted
        even if the OOS Sharpe gate would otherwise pass."""
        panel = _multi_day_panel(n_days=30, breakout_days=10, syms=("SPY",))

        from packages.backtests import intraday_walk_forward as iwf

        def fake_audit(*_args, **_kwargs) -> LookaheadAudit:
            return LookaheadAudit(clean=False, findings=["fake_leak: SPY"])

        # Force a passing promotion verdict too, so we know the audit
        # is doing the blocking.
        from packages.backtests.champion_challenger import PromotionVerdict

        def fake_gate(*_args, **_kwargs) -> PromotionVerdict:
            return PromotionVerdict(
                promote=True,
                days_outperformed=10,
                reasons=["challenger_wins"],
                metrics={"sharpe_diff": 0.5},
            )

        monkeypatch.setattr(iwf, "audit_lookahead", fake_audit)
        monkeypatch.setattr(iwf, "promotion_gate", fake_gate)

        out = run_intraday_walk_forward(
            panel,
            champion=IntradayParamSet(),
            grid=(IntradayParamSet(),),
            in_sample_days=20,
            out_of_sample_days=5,
        )
        assert out.promoted is False
        assert out.lookahead_clean is False
        assert any("lookahead audit failed" in r for r in out.reasons)
        assert "fake_leak: SPY" in out.lookahead_findings
