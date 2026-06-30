"""Unit tests for the performance / track-record engine.

Two clearly-separated sections are exercised against hand-computed fixtures:

* **Section A — account performance** from a REAL equity series (known
  drawdown / total return / Sharpe) plus FIFO-matched order-ledger round-trips
  (known win rate / profit factor / expectancy; FIFO correctness; missing fill
  prices -> insufficient_data, never fabricated).
* **Section B — signal quality** from outcome rows (hit rate + real
  segmentation by source / regime / confidence bucket / horizon, with a missing
  field landing in an explicit ``"unknown"`` bucket, never collapsed/dropped).
"""
from __future__ import annotations

import math

from packages.cockpit import performance_stats as ps


def _row(ret, *, ts="2026-05-01T12:00:00+00:00", source=None, confidence=None,
         regime="risk_on", symbol="AAPL", r2h=None, mode=None):
    row = {
        "pick_id": f"p-{ts}-{symbol}",
        "ts": ts,
        "symbol": symbol,
        "regime_at_pick": regime,
        "return_eod": ret,
    }
    if source is not None:
        row["source"] = source
    if confidence is not None:
        row["confidence"] = confidence
    if r2h is not None:
        row["return_2h"] = r2h
    if mode is not None:
        row["mode"] = mode
    return row


def _trade(side, symbol, qty, price=None, ts="2026-05-01T12:00:00+00:00", **extra):
    row = {"side": side, "symbol": symbol, "qty": qty, "run_ts": ts}
    if price is not None:
        row["fill_price"] = price
    row.update(extra)
    return row


# --- primitive math --------------------------------------------------------


def test_win_rate_basic():
    assert ps.win_rate([0.02, -0.01, 0.03, -0.02]) == 0.5
    assert ps.win_rate([0.02, 0.03]) == 1.0


def test_win_rate_excludes_scratches_and_empty():
    assert ps.win_rate([0.02, -0.02, 0.0]) == 0.5
    assert ps.win_rate([0.0, 0.0]) is None
    assert ps.win_rate([]) is None


def test_avg_win_and_loss():
    rets = [0.02, 0.04, -0.01, -0.03]
    assert ps.avg_win(rets) == 0.03
    assert ps.avg_loss(rets) == -0.02
    assert ps.avg_win([-0.01]) is None
    assert ps.avg_loss([0.01]) is None


def test_profit_factor():
    assert ps.profit_factor([0.02, 0.04, -0.01, -0.03]) == 1.5


def test_profit_factor_undefined_when_no_losses():
    assert ps.profit_factor([0.02, 0.03]) is None
    assert ps.profit_factor([]) is None
    assert ps.profit_factor([0.0]) is None


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
    curve = ps.equity_curve([0.1, -0.5, 0.2])
    assert math.isclose(ps.max_drawdown(curve), -0.5)
    assert ps.max_drawdown([1.0]) is None
    assert ps.max_drawdown(ps.equity_curve([0.1, 0.1])) == 0.0


def test_sharpe_known_series():
    assert ps.sharpe([0.01, -0.01, 0.01, -0.01]) == 0.0
    assert ps.sharpe([0.01, 0.01]) is None
    assert ps.sharpe([0.01]) is None


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


# --- Section A: equity-series cleaning -------------------------------------


def test_clean_equity_series_sorts_dedupes_and_drops_nonnumeric():
    pts = [
        {"t": "2026-05-03", "equity": 101.0},
        {"t": "2026-05-01", "equity": 100.0},
        {"t": "2026-05-02", "equity": "oops"},   # dropped
        {"t": "2026-05-03", "equity": 101.5},    # same ts -> keep last
    ]
    out = ps.clean_equity_series(pts)
    assert [p["t"] for p in out] == ["2026-05-01", "2026-05-03"]
    assert out[-1]["equity"] == 101.5


# --- Section A: account performance from REAL equity series ----------------


def test_account_performance_real_drawdown_and_return():
    pts = [
        {"t": "2026-05-01", "equity": 100000.0},
        {"t": "2026-05-02", "equity": 101000.0},
        {"t": "2026-05-03", "equity": 99000.0},
        {"t": "2026-05-04", "equity": 101500.0},
    ]
    acct = ps.account_performance(pts, [])
    assert acct["insufficient_data"] is False
    assert acct["starting_equity"] == 100000.0
    assert acct["current_equity"] == 101500.0
    assert acct["peak_equity"] == 101500.0
    assert math.isclose(acct["total_return"], 0.015)
    # worst dd is peak 101000 -> 99000 = -2000/101000 (NOT -100%)
    assert math.isclose(acct["max_drawdown"], -2000.0 / 101000.0)
    assert acct["max_drawdown"] > -0.05   # realistic, not the old -100% artifact
    assert acct["sharpe"] is not None


def test_account_performance_insufficient_with_one_point():
    acct = ps.account_performance([{"t": "2026-05-01", "equity": 100000.0}], [])
    assert acct["insufficient_data"] is True
    assert acct["max_drawdown"] is None
    assert acct["total_return"] is None
    assert acct["sharpe"] is None


# --- Section A: FIFO round-trip matching -----------------------------------


def test_fifo_matches_buy_lots_to_sells():
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1"),
        _trade("buy", "AAPL", 10, 110.0, ts="t2"),
        _trade("sell", "AAPL", 15, 120.0, ts="t3"),
    ]
    fifo = ps.fifo_round_trips(trades)
    trips = fifo["round_trips"]
    assert len(trips) == 2
    # first sell-chunk closes the oldest lot (10 @ 100)
    assert trips[0]["qty"] == 10 and trips[0]["entry_price"] == 100.0
    assert math.isclose(trips[0]["pnl_dollars"], 200.0)
    assert math.isclose(trips[0]["pnl_pct"], 0.2)
    # remaining 5 close part of the 110 lot
    assert trips[1]["qty"] == 5 and trips[1]["entry_price"] == 110.0
    assert math.isclose(trips[1]["pnl_dollars"], 50.0)
    assert math.isclose(fifo["open_lots"], 5.0)


def test_realized_stats_win_rate_profit_factor_expectancy():
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1"),
        _trade("buy", "AAPL", 10, 110.0, ts="t2"),
        _trade("sell", "AAPL", 15, 120.0, ts="t3"),   # +200, +50
        _trade("buy", "MSFT", 10, 200.0, ts="t4"),
        _trade("sell", "MSFT", 10, 190.0, ts="t5"),    # -100
    ]
    rz = ps.realized_trade_stats(ps.fifo_round_trips(trades))
    assert rz["insufficient_data"] is False
    assert rz["total_round_trips"] == 3
    assert rz["wins"] == 2 and rz["losses"] == 1
    assert math.isclose(rz["win_rate"], 2 / 3)
    assert math.isclose(rz["gross_profit"], 250.0)
    assert math.isclose(rz["gross_loss"], 100.0)
    assert math.isclose(rz["profit_factor"], 2.5)
    assert math.isclose(rz["expectancy"], (200 + 50 - 100) / 3)
    assert math.isclose(rz["total_pnl_dollars"], 150.0)


def test_realized_stats_insufficient_when_no_fill_prices():
    trades = [
        _trade("buy", "AAPL", 10, ts="t1"),   # no fill_price
        _trade("sell", "AAPL", 10, ts="t2"),
    ]
    rz = ps.realized_trade_stats(ps.fifo_round_trips(trades))
    assert rz["insufficient_data"] is True
    assert rz["total_round_trips"] == 0
    assert rz["unpriced_fills"] == 2
    assert rz["note"] and "price" in rz["note"].lower()


def test_realized_stats_insufficient_when_no_sells():
    trades = [_trade("buy", "AAPL", 10, 100.0, ts="t1")]
    rz = ps.realized_trade_stats(ps.fifo_round_trips(trades))
    assert rz["insufficient_data"] is True
    assert rz["open_lots"] == 10.0
    assert rz["note"] is not None


# --- Section A: fill-source provenance (measured vs unmeasured round-trips) --


def test_realized_stats_known_fills_with_explicit_source_are_measured():
    # broker_fill + mark_estimate legs both count toward the real ratios.
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1", fill_source="broker_fill"),
        _trade("sell", "AAPL", 10, 120.0, ts="t2", fill_source="mark_estimate"),  # +200
        _trade("buy", "MSFT", 10, 200.0, ts="t3", fill_source="broker_fill"),
        _trade("sell", "MSFT", 10, 190.0, ts="t4", fill_source="broker_fill"),    # -100
    ]
    rz = ps.realized_trade_stats(ps.fifo_round_trips(trades))
    assert rz["insufficient_data"] is False
    assert rz["closed_round_trips"] == 2
    assert rz["unmeasured_round_trips"] == 0
    assert math.isclose(rz["gross_profit"], 200.0)
    assert math.isclose(rz["gross_loss"], 100.0)
    assert math.isclose(rz["profit_factor"], 2.0)
    assert math.isclose(rz["expectancy"], (200 - 100) / 2)
    assert math.isclose(rz["round_trip_win_rate"], 0.5)
    assert math.isclose(rz["avg_winner"], 200.0)
    assert math.isclose(rz["avg_loser"], -100.0)


def test_unknown_leg_excluded_and_counted_as_unmeasured():
    # A profitable-looking exit whose fill_source is explicitly "unknown" must
    # NOT be scored — it is reported as unmeasured and excluded from ratios.
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1", fill_source="broker_fill"),
        _trade("sell", "AAPL", 10, 130.0, ts="t2", fill_source="unknown"),  # excluded
        _trade("buy", "MSFT", 5, 200.0, ts="t3", fill_source="broker_fill"),
        _trade("sell", "MSFT", 5, 220.0, ts="t4", fill_source="broker_fill"),  # +100 measured
    ]
    fifo = ps.fifo_round_trips(trades)
    assert fifo["measured_round_trips"] == 1
    assert fifo["unmeasured_round_trips"] == 1
    rz = ps.realized_trade_stats(fifo)
    assert rz["closed_round_trips"] == 1
    assert rz["unmeasured_round_trips"] == 1
    # Only the MSFT +100 round-trip is graded; the AAPL unknown leg is ignored.
    assert math.isclose(rz["total_pnl_dollars"], 100.0)
    assert rz["wins"] == 1 and rz["losses"] == 0


def test_unknown_leg_never_fabricates_when_it_is_the_only_round_trip():
    # Fail-safe: the sole round-trip has an unknown leg -> insufficient_data,
    # NO fabricated win/loss, surfaced as unmeasured.
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1", fill_source="unknown"),
        _trade("sell", "AAPL", 10, 130.0, ts="t2", fill_source="broker_fill"),
    ]
    rz = ps.realized_trade_stats(ps.fifo_round_trips(trades))
    assert rz["insufficient_data"] is True
    assert rz["closed_round_trips"] == 0
    assert rz["unmeasured_round_trips"] == 1
    assert rz["profit_factor"] is None
    assert rz["expectancy"] is None
    assert rz["note"] and "unmeasured" in rz["note"].lower()


def test_backward_compat_old_rows_without_fill_source_still_measured():
    # Legacy ledger rows carry a fill_price but no fill_source. They must still
    # parse and remain measurable (a real historical price is never discarded).
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1"),   # no fill_source key
        _trade("sell", "AAPL", 10, 110.0, ts="t2"),  # no fill_source key
    ]
    rz = ps.realized_trade_stats(ps.fifo_round_trips(trades))
    assert rz["insufficient_data"] is False
    assert rz["closed_round_trips"] == 1
    assert rz["unmeasured_round_trips"] == 0
    assert math.isclose(rz["total_pnl_dollars"], 100.0)


def test_failsafe_missing_price_is_unmeasured_never_fabricated():
    # No price anywhere -> matched by qty but unmeasured; never a P&L value.
    trades = [
        _trade("buy", "AAPL", 10, ts="t1"),
        _trade("sell", "AAPL", 10, ts="t2"),
    ]
    fifo = ps.fifo_round_trips(trades)
    assert fifo["measured_round_trips"] == 0
    assert fifo["unmeasured_round_trips"] == 1
    assert fifo["round_trips"][0]["pnl_dollars"] is None
    rz = ps.realized_trade_stats(fifo)
    assert rz["insufficient_data"] is True
    assert rz["unpriced_fills"] == 2
    assert rz["closed_round_trips"] == 0


def test_low_confidence_below_measured_floor():
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1", fill_source="broker_fill"),
        _trade("sell", "AAPL", 10, 110.0, ts="t2", fill_source="broker_fill"),
    ]
    rz = ps.realized_trade_stats(ps.fifo_round_trips(trades))
    assert rz["closed_round_trips"] < ps.MIN_MEASURED_ROUND_TRIPS
    assert rz["confidence"] == "low"
    assert rz["note"] and "low-confidence" in rz["note"].lower()


def test_account_section_exposes_flattened_realized_fields():
    pts = [
        {"t": "2026-05-01", "equity": 100000.0},
        {"t": "2026-05-02", "equity": 101000.0},
    ]
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1", fill_source="broker_fill"),
        _trade("sell", "AAPL", 10, 110.0, ts="t2", fill_source="broker_fill"),
    ]
    acct = ps.account_performance(pts, trades)
    for key in (
        "round_trip_win_rate", "profit_factor", "expectancy", "avg_winner",
        "avg_loser", "closed_round_trips", "unmeasured_round_trips",
    ):
        assert key in acct, f"missing flattened account field {key!r}"
    assert acct["closed_round_trips"] == 1
    assert acct["round_trip_win_rate"] == 1.0


# --- Section B: signal quality / hit rate ----------------------------------


def test_signal_quality_hit_rate_and_segmentation():
    rows = [
        _row(0.02, source="news", confidence=0.8, regime="risk_on", r2h=0.01),
        _row(-0.01, source="news", confidence=0.2, regime="chop"),
        _row(0.03, confidence=0.5, regime="risk_on"),   # no source -> unknown
    ]
    sq = ps.signal_quality(rows, horizon="eod")
    assert sq["insufficient_data"] is False
    assert sq["total_recorded"] == 3 and sq["total_settled"] == 3
    assert math.isclose(sq["hit_rate"], 2 / 3)
    # source segmentation keeps a real 'unknown' bucket (never collapsed)
    assert set(sq["by_source"]) == {"news", "unknown"}
    assert sq["by_source"]["news"]["n"] == 2
    assert sq["by_source"]["unknown"]["n"] == 1
    # confidence buckets are real (low/medium/high)
    assert set(sq["by_confidence"]) == {"low (<0.34)", "medium (0.34-0.67)", "high (>=0.67)"}
    # regime segmentation
    assert set(sq["by_regime"]) == {"risk_on", "chop"}
    assert sq["by_regime"]["risk_on"]["n"] == 2
    # horizon breakdown: eod has 3, 2h has 1
    assert sq["by_horizon"]["eod"]["n"] == 3
    assert sq["by_horizon"]["2h"]["n"] == 1


def test_signal_quality_missing_regime_and_confidence_go_to_unknown():
    rows = [{"return_eod": 0.01}, {"return_eod": -0.02}]
    sq = ps.signal_quality(rows)
    assert set(sq["by_regime"]) == {"unknown"}
    assert set(sq["by_confidence"]) == {"unknown"}
    assert set(sq["by_source"]) == {"unknown"}


def test_signal_quality_empty_is_insufficient():
    sq = ps.signal_quality([])
    assert sq["insufficient_data"] is True
    assert sq["hit_rate"] is None
    assert sq["by_source"] == {}


def test_signal_quality_unsettled_rows_excluded():
    rows = [_row(0.02), _row(None), _row(-0.01)]
    sq = ps.signal_quality(rows)
    assert sq["total_recorded"] == 3
    assert sq["total_settled"] == 2


def test_signal_quality_is_never_compounded():
    # 10k tiny negative signal rows must NOT drive a -100% "equity"; signal
    # quality reports a hit rate, never an account equity field.
    rows = [_row(-0.001, ts=f"2026-05-01T00:{i:02d}:00+00:00", symbol="X") for i in range(60)]
    sq = ps.signal_quality(rows)
    assert "equity" not in sq and "current_equity" not in sq
    assert sq["hit_rate"] == 0.0  # all losers, but no -100% artifact


# --- Top-level report ------------------------------------------------------


def test_compute_performance_combines_both_sections():
    pts = [
        {"t": "2026-05-01", "equity": 100000.0},
        {"t": "2026-05-02", "equity": 101500.0},
    ]
    trades = [
        _trade("buy", "AAPL", 10, 100.0, ts="t1"),
        _trade("sell", "AAPL", 10, 110.0, ts="t2"),
    ]
    rows = [_row(0.02, source="news"), _row(-0.01, source="scan")]
    rep = ps.compute_performance(pts, trades, rows)
    assert set(rep) == {"mode", "insufficient_data", "account", "signal_quality"}
    assert rep["mode"] == "shadow"
    assert rep["account"]["insufficient_data"] is False
    assert math.isclose(rep["account"]["total_return"], 0.015)
    assert rep["account"]["realized"]["total_round_trips"] == 1
    assert rep["signal_quality"]["total_settled"] == 2


def test_compute_performance_empty_is_failsafe():
    rep = ps.compute_performance([], [], [])
    assert rep["insufficient_data"] is True
    assert rep["account"]["insufficient_data"] is True
    assert rep["account"]["max_drawdown"] is None
    assert rep["account"]["realized"]["insufficient_data"] is True
    assert rep["signal_quality"]["insufficient_data"] is True
    assert rep["mode"] == "shadow"


def test_compute_performance_detects_mode_but_never_live_by_accident():
    rep = ps.compute_performance([], [], [_row(0.01, mode="paper"), _row(0.02, mode="paper")])
    assert rep["mode"] == "paper"
    rep2 = ps.compute_performance([], [], [_row(0.01)])
    assert rep2["mode"] == "shadow"


def test_compute_performance_deterministic_pure():
    pts = [{"t": "2026-05-01", "equity": 100.0}, {"t": "2026-05-02", "equity": 101.0}]
    trades = [_trade("buy", "AAPL", 1, 100.0, ts="t1"), _trade("sell", "AAPL", 1, 101.0, ts="t2")]
    rows = [_row(0.02, source="news")]
    assert ps.compute_performance(pts, trades, rows) == ps.compute_performance(pts, trades, rows)


# --- per-lane realized split (under_radar vs mainstream) -------------------


def test_round_trips_carry_lane_and_catalyst_from_entry_leg():
    trades = [
        _trade("buy", "BCRX", 10, 1.50, ts="t1", lane="under_radar", catalyst_type="fda"),
        _trade("sell", "BCRX", 10, 1.80, ts="t2"),
    ]
    fifo = ps.fifo_round_trips(trades)
    rt = fifo["round_trips"][0]
    assert rt["lane"] == "under_radar"
    assert rt["catalyst_type"] == "fda"


def test_realized_by_lane_splits_two_lanes():
    trades = [
        # under_radar winner
        _trade("buy", "BCRX", 10, 1.00, ts="t1", lane="under_radar", catalyst_type="fda"),
        _trade("sell", "BCRX", 10, 1.50, ts="t2"),
        # mainstream loser
        _trade("buy", "AAPL", 1, 200.0, ts="t3", lane="mainstream", catalyst_type="none"),
        _trade("sell", "AAPL", 1, 180.0, ts="t4"),
    ]
    by_lane = ps.realized_by_lane(ps.fifo_round_trips(trades))
    assert set(by_lane) == {"under_radar", "mainstream"}
    assert by_lane["under_radar"]["round_trip_win_rate"] == 1.0
    assert by_lane["mainstream"]["round_trip_win_rate"] == 0.0


def test_realized_by_lane_legacy_rows_fall_into_unknown_bucket():
    # No lane tag at all (old ledger rows) -> 'unknown', never merged into a real lane.
    trades = [
        _trade("buy", "AAPL", 1, 100.0, ts="t1"),
        _trade("sell", "AAPL", 1, 110.0, ts="t2"),
    ]
    by_lane = ps.realized_by_lane(ps.fifo_round_trips(trades))
    assert set(by_lane) == {ps.UNKNOWN_BUCKET}


def test_realized_by_lane_empty_is_noop():
    assert ps.realized_by_lane({"round_trips": []}) == {}
    assert ps.realized_by_lane({}) == {}


def test_account_performance_exposes_realized_by_lane():
    pts = [{"t": "2026-05-01", "equity": 100.0}, {"t": "2026-05-02", "equity": 105.0}]
    trades = [
        _trade("buy", "BCRX", 10, 1.00, ts="t1", lane="under_radar", catalyst_type="fda"),
        _trade("sell", "BCRX", 10, 1.50, ts="t2"),
    ]
    acct = ps.account_performance(pts, trades)
    assert "realized_by_lane" in acct
    assert "under_radar" in acct["realized_by_lane"]


def test_legacy_rows_without_lane_still_parse_backward_compat():
    # Backward-compat: pre-existing ledger rows have no lane/catalyst keys and
    # must FIFO-match exactly as before (lane defaults to the unknown bucket).
    trades = [
        _trade("buy", "AAPL", 1, 100.0, ts="t1"),
        _trade("sell", "AAPL", 1, 110.0, ts="t2"),
    ]
    fifo = ps.fifo_round_trips(trades)
    assert len(fifo["round_trips"]) == 1
    rt = fifo["round_trips"][0]
    assert rt["lane"] == ps.UNKNOWN_BUCKET
    assert rt["pnl_dollars"] == 10.0
