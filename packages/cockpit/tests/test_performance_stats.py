"""Unit tests for the realized-performance stats engine.

Every stat is exercised against a hand-computed fixture so the math is
pinned, plus the fail-safe paths: empty store -> insufficient_data, all-wins
-> profit factor undefined, segmentation correctness, and exclusion of
unsettled (return_eod is None) rows.
"""
from __future__ import annotations

import math

from packages.cockpit import performance_stats as ps


def _row(ret, *, ts="2026-05-01T12:00:00+00:00", mode=None, source=None,
         strategy=None, regime="risk_on", symbol="AAPL"):
    row = {
        "pick_id": f"p-{ts}-{symbol}",
        "ts": ts,
        "symbol": symbol,
        "regime_at_pick": regime,
        "return_eod": ret,
    }
    if mode is not None:
        row["mode"] = mode
    if source is not None:
        row["source"] = source
    if strategy is not None:
        row["strategy"] = strategy
    return row


# --- primitive math --------------------------------------------------------


def test_win_rate_basic():
    assert ps.win_rate([0.02, -0.01, 0.03, -0.02]) == 0.5
    assert ps.win_rate([0.02, 0.03]) == 1.0


def test_win_rate_excludes_scratches_and_empty():
    # one win, one loss, one scratch -> decided = 2 -> 0.5
    assert ps.win_rate([0.02, -0.02, 0.0]) == 0.5
    assert ps.win_rate([0.0, 0.0]) is None  # no decided trades
    assert ps.win_rate([]) is None


def test_avg_win_and_loss():
    rets = [0.02, 0.04, -0.01, -0.03]
    assert ps.avg_win(rets) == 0.03
    assert ps.avg_loss(rets) == -0.02
    assert ps.avg_win([-0.01]) is None
    assert ps.avg_loss([0.01]) is None


def test_profit_factor():
    # gross profit 0.06, gross loss 0.04 -> 1.5
    assert ps.profit_factor([0.02, 0.04, -0.01, -0.03]) == 1.5


def test_profit_factor_undefined_when_no_losses():
    assert ps.profit_factor([0.02, 0.03]) is None  # all wins -> undefined
    assert ps.profit_factor([]) is None
    assert ps.profit_factor([0.0]) is None  # no gross loss


def test_expectancy():
    assert math.isclose(ps.expectancy([0.02, -0.01, 0.03, -0.02]), 0.005)
    assert ps.expectancy([]) is None


def test_equity_curve_compounds():
    curve = ps.equity_curve([0.1, -0.5, 0.2])
    assert curve[0] == 1.0
    assert math.isclose(curve[1], 1.1)
    assert math.isclose(curve[2], 0.55)
    assert math.isclose(curve[3], 0.66)
    assert ps.equity_curve([]) == [1.0]


def test_max_drawdown():
    # 1.0 -> 1.1 -> 0.55 -> 0.66 ; worst dd from peak 1.1 to 0.55 = -0.5
    curve = ps.equity_curve([0.1, -0.5, 0.2])
    assert math.isclose(ps.max_drawdown(curve), -0.5)
    assert ps.max_drawdown([1.0]) is None  # < 2 points
    # monotonically rising -> no drawdown
    assert ps.max_drawdown(ps.equity_curve([0.1, 0.1])) == 0.0


def test_sharpe_known_series():
    rets = [0.01, -0.01, 0.01, -0.01]
    mean = 0.0
    # mean is 0 -> sharpe 0 (variance nonzero) -> 0.0, not None
    assert ps.sharpe(rets) == 0.0
    # constant series -> zero variance -> None
    assert ps.sharpe([0.01, 0.01]) is None
    assert ps.sharpe([0.01]) is None  # < 2


def test_sharpe_value():
    rets = [0.02, 0.04]
    mean = (0.02 + 0.04) / 2
    pstdev = math.sqrt(((0.02 - mean) ** 2 + (0.04 - mean) ** 2) / 2)
    expected = (mean / pstdev) * math.sqrt(252)
    assert math.isclose(ps.sharpe(rets), expected)


def test_total_return():
    curve = ps.equity_curve([0.1, 0.1])
    assert math.isclose(ps.total_return(curve), 1.21 - 1.0)
    assert ps.total_return([1.0]) is None


# --- compute_performance (the endpoint payload) ----------------------------


def test_empty_store_is_insufficient_data():
    rep = ps.compute_performance([])
    assert rep["insufficient_data"] is True
    o = rep["overall"]
    assert o["total_trades"] == 0
    assert o["win_rate"] is None
    assert o["profit_factor"] is None
    assert o["max_drawdown"] is None
    assert o["sharpe"] is None
    assert rep["equity_curve"] == []
    assert rep["by_mode"] == {}


def test_unsettled_rows_excluded_but_counted():
    rows = [_row(0.02), _row(None), _row(-0.01)]
    rep = ps.compute_performance(rows)
    o = rep["overall"]
    assert o["total_recorded"] == 3
    assert o["total_trades"] == 2  # the None row is excluded from stats
    assert o["wins"] == 1
    assert o["losses"] == 1


def test_all_wins_profit_factor_undefined_in_report():
    rep = ps.compute_performance([_row(0.02), _row(0.03)])
    o = rep["overall"]
    assert o["insufficient_data"] is False
    assert o["profit_factor"] is None
    assert o["win_rate"] == 1.0
    assert o["losses"] == 0


def test_overall_metrics_match_hand_computation():
    rows = [
        _row(0.02, ts="2026-05-01T12:00:00+00:00"),
        _row(-0.01, ts="2026-05-02T12:00:00+00:00"),
        _row(0.03, ts="2026-05-03T12:00:00+00:00"),
        _row(-0.02, ts="2026-05-04T12:00:00+00:00"),
    ]
    o = ps.compute_performance(rows)["overall"]
    assert o["total_trades"] == 4
    assert o["win_rate"] == 0.5
    assert o["avg_win"] == 0.025
    assert o["avg_loss"] == -0.015
    assert o["profit_factor"] == 0.05 / 0.03
    assert math.isclose(o["expectancy"], 0.005)
    # equity: 1.02, 1.0098, 1.040094, 1.01929212 -> dd from peak 1.040094
    assert o["max_drawdown"] is not None and o["max_drawdown"] < 0
    assert o["total_return"] is not None


def test_equity_curve_ordered_by_ts():
    # supplied out of order; engine must sort chronologically
    rows = [
        _row(0.03, ts="2026-05-03T12:00:00+00:00"),
        _row(0.02, ts="2026-05-01T12:00:00+00:00"),
        _row(-0.01, ts="2026-05-02T12:00:00+00:00"),
    ]
    curve = ps.compute_performance(rows)["equity_curve"]
    # first point is the synthetic start, then ascending ts
    ts_seq = [p["t"] for p in curve[1:]]
    assert ts_seq == sorted(ts_seq)
    assert curve[0]["equity"] == 1.0


def test_segmentation_by_mode_strategy_source_regime():
    rows = [
        _row(0.02, mode="shadow", strategy="ensemble", source="news", regime="risk_on"),
        _row(-0.01, mode="shadow", strategy="ensemble", source="scan", regime="chop"),
        _row(0.05, mode="paper", strategy="momentum", source="news", regime="risk_on"),
    ]
    rep = ps.compute_performance(rows)
    assert set(rep["by_mode"]) == {"shadow", "paper"}
    assert rep["by_mode"]["shadow"]["total_trades"] == 2
    assert rep["by_mode"]["paper"]["total_trades"] == 1
    assert set(rep["by_strategy"]) == {"ensemble", "momentum"}
    assert set(rep["by_source"]) == {"news", "scan"}
    assert rep["by_source"]["news"]["total_trades"] == 2
    assert set(rep["by_regime"]) == {"risk_on", "chop"}


def test_mode_defaults_to_shadow_when_absent():
    rep = ps.compute_performance([_row(0.02), _row(-0.01)])
    assert set(rep["by_mode"]) == {"shadow"}
    assert rep["by_mode"]["shadow"]["total_trades"] == 2


def test_source_segment_skips_rows_without_source():
    # no source field on any row -> by_source empty (no fabricated buckets)
    rep = ps.compute_performance([_row(0.02), _row(-0.01)])
    assert rep["by_source"] == {}


def test_deterministic_pure():
    rows = [_row(0.02, strategy="ensemble"), _row(-0.01, strategy="ensemble")]
    a = ps.compute_performance(rows)
    b = ps.compute_performance(rows)
    assert a == b
