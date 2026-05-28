"""Daily PnL + predicted-vs-actual tests."""

from __future__ import annotations

from datetime import date

from packages.shadow.pairing import PairedTrade
from packages.shadow.pnl import (
    DailyPnL,
    aggregate_daily,
    predicted_vs_actual,
)


def _trip(symbol: str, sell_ts: str, pnl: float) -> PairedTrade:
    # Construct so that (sell_px - buy_px) * qty == pnl with qty=1.
    return PairedTrade(
        symbol=symbol,
        buy_ts="2026-05-01T10:00:00Z",
        sell_ts=sell_ts,
        qty=1.0,
        buy_px=100.0,
        sell_px=100.0 + pnl,
    )


def test_aggregate_daily_empty() -> None:
    assert aggregate_daily([]) == []


def test_aggregate_daily_single_day() -> None:
    out = aggregate_daily([_trip("SPY", "2026-05-02T10:00:00Z", 5.0)])
    assert len(out) == 1
    assert out[0] == DailyPnL(day=date(2026, 5, 2), pnl=5.0, n_trades=1)


def test_aggregate_daily_sums_within_day() -> None:
    out = aggregate_daily(
        [
            _trip("SPY", "2026-05-02T10:00:00Z", 5.0),
            _trip("QQQ", "2026-05-02T14:00:00Z", -2.0),
        ]
    )
    assert len(out) == 1
    assert out[0].pnl == 3.0
    assert out[0].n_trades == 2


def test_aggregate_daily_fills_gaps() -> None:
    out = aggregate_daily(
        [
            _trip("SPY", "2026-05-01T10:00:00Z", 5.0),
            _trip("SPY", "2026-05-04T10:00:00Z", 1.0),
        ]
    )
    days = [r.day for r in out]
    assert days == [
        date(2026, 5, 1),
        date(2026, 5, 2),
        date(2026, 5, 3),
        date(2026, 5, 4),
    ]
    # Middle days have zero PnL and zero trades
    assert out[1].pnl == 0.0 and out[1].n_trades == 0
    assert out[2].pnl == 0.0 and out[2].n_trades == 0


def test_aggregate_daily_no_gap_filling() -> None:
    out = aggregate_daily(
        [
            _trip("SPY", "2026-05-01T10:00:00Z", 5.0),
            _trip("SPY", "2026-05-04T10:00:00Z", 1.0),
        ],
        fill_gaps=False,
    )
    assert len(out) == 2
    assert [r.day for r in out] == [date(2026, 5, 1), date(2026, 5, 4)]


def test_aggregate_handles_bad_timestamp() -> None:
    bad = PairedTrade("SPY", "bad-ts", "also-bad", 1.0, 100.0, 105.0)
    assert aggregate_daily([bad]) == []


def test_predicted_vs_actual_basic() -> None:
    paired = [
        _trip("SPY", "2026-05-02T10:00:00Z", 5.0),
        _trip("QQQ", "2026-05-02T10:00:00Z", -2.0),
    ]
    preds = [
        {"symbol": "SPY", "predicted_pnl": 4.0},
        {"symbol": "QQQ", "predicted_pnl": -1.0},
    ]
    out = predicted_vs_actual(preds, paired)
    assert len(out) == 2
    by_sym = {r.symbol: r for r in out}
    assert by_sym["SPY"].predicted_pnl == 4.0
    assert by_sym["SPY"].actual_pnl == 5.0
    assert by_sym["SPY"].matched is True


def test_predicted_vs_actual_unmatched_sides() -> None:
    paired = [_trip("SPY", "2026-05-02T10:00:00Z", 5.0)]
    preds = [{"symbol": "QQQ", "predicted_pnl": 3.0}]
    out = predicted_vs_actual(preds, paired)
    assert {r.symbol for r in out} == {"SPY", "QQQ"}
    for r in out:
        assert r.matched is False


def test_predicted_vs_actual_bad_rows_skipped() -> None:
    paired = [_trip("SPY", "2026-05-02T10:00:00Z", 5.0)]
    preds = [
        {"symbol": "SPY", "predicted_pnl": "not-a-number"},
        {"symbol": "", "predicted_pnl": 1.0},
        {"predicted_pnl": 1.0},  # no symbol
    ]
    out = predicted_vs_actual(preds, paired)
    assert len(out) == 1
    assert out[0].predicted_pnl == 0.0
    assert out[0].actual_pnl == 5.0


def test_predicted_vs_actual_sums_per_symbol() -> None:
    paired = [
        _trip("SPY", "2026-05-02T10:00:00Z", 3.0),
        _trip("SPY", "2026-05-03T10:00:00Z", 2.0),
    ]
    preds = [
        {"symbol": "SPY", "predicted_pnl": 1.0},
        {"symbol": "SPY", "predicted_pnl": 2.5},
    ]
    out = predicted_vs_actual(preds, paired)
    assert len(out) == 1
    assert out[0].predicted_pnl == 3.5
    assert out[0].actual_pnl == 5.0
