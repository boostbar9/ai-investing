"""Round-trip pairing tests."""

from __future__ import annotations

from packages.shadow.pairing import PairedTrade, pair_round_trips


def _t(ts: str, side: str, symbol: str, qty: float, px: float | None = None) -> dict:
    return {"ts": ts, "side": side, "symbol": symbol, "qty": qty, "limit_price": px}


def test_simple_buy_then_sell_pairs() -> None:
    trades = [
        _t("2026-05-01T10:00:00Z", "buy", "SPY", 10, 100.0),
        _t("2026-05-02T10:00:00Z", "sell", "SPY", 10, 105.0),
    ]
    out = pair_round_trips(trades)
    assert len(out) == 1
    p = out[0]
    assert isinstance(p, PairedTrade)
    assert p.symbol == "SPY"
    assert p.qty == 10
    assert p.pnl == 50.0


def test_sells_split_across_two_lots() -> None:
    # 5 + 10 bought, then 15 sold -> two paired trades
    trades = [
        _t("2026-05-01T10:00:00Z", "buy", "SPY", 5, 100.0),
        _t("2026-05-01T11:00:00Z", "buy", "SPY", 10, 102.0),
        _t("2026-05-02T10:00:00Z", "sell", "SPY", 15, 105.0),
    ]
    out = pair_round_trips(trades)
    assert len(out) == 2
    # FIFO -- first lot popped first
    assert out[0].qty == 5
    assert out[0].buy_px == 100.0
    assert out[1].qty == 10
    assert out[1].buy_px == 102.0
    total_pnl = sum(p.pnl for p in out)
    assert total_pnl == 5 * 5.0 + 10 * 3.0  # 25 + 30 = 55


def test_lot_split_when_sell_smaller_than_lot() -> None:
    trades = [
        _t("2026-05-01T10:00:00Z", "buy", "SPY", 10, 100.0),
        _t("2026-05-02T10:00:00Z", "sell", "SPY", 3, 110.0),
        _t("2026-05-03T10:00:00Z", "sell", "SPY", 4, 120.0),
    ]
    out = pair_round_trips(trades)
    assert len(out) == 2
    assert out[0].qty == 3
    assert out[0].pnl == 3 * 10.0
    assert out[1].qty == 4
    assert out[1].pnl == 4 * 20.0


def test_unmatched_sell_dropped() -> None:
    trades = [_t("2026-05-01T10:00:00Z", "sell", "SPY", 5, 100.0)]
    assert pair_round_trips(trades) == []


def test_different_symbols_isolated() -> None:
    trades = [
        _t("2026-05-01T10:00:00Z", "buy", "SPY", 10, 100.0),
        _t("2026-05-01T10:00:00Z", "buy", "QQQ", 5, 300.0),
        _t("2026-05-02T10:00:00Z", "sell", "QQQ", 5, 310.0),
    ]
    out = pair_round_trips(trades)
    assert len(out) == 1
    assert out[0].symbol == "QQQ"


def test_trades_resorted_by_ts() -> None:
    # Out-of-order input should still pair correctly
    trades = [
        _t("2026-05-02T10:00:00Z", "sell", "SPY", 10, 105.0),
        _t("2026-05-01T10:00:00Z", "buy", "SPY", 10, 100.0),
    ]
    out = pair_round_trips(trades)
    assert len(out) == 1
    assert out[0].pnl == 50.0


def test_missing_price_skipped() -> None:
    trades = [
        _t("2026-05-01T10:00:00Z", "buy", "SPY", 10, None),
        _t("2026-05-02T10:00:00Z", "sell", "SPY", 10, 105.0),
    ]
    # Buy had no price -> nothing to pair against
    assert pair_round_trips(trades) == []


def test_missing_qty_skipped() -> None:
    trades = [
        _t("2026-05-01T10:00:00Z", "buy", "SPY", "not-a-number", 100.0),
        _t("2026-05-02T10:00:00Z", "sell", "SPY", 10, 105.0),
    ]
    assert pair_round_trips(trades) == []


def test_zero_qty_skipped() -> None:
    trades = [
        _t("2026-05-01T10:00:00Z", "buy", "SPY", 0, 100.0),
        _t("2026-05-02T10:00:00Z", "sell", "SPY", 10, 105.0),
    ]
    assert pair_round_trips(trades) == []


def test_symbol_normalised_uppercase() -> None:
    trades = [
        _t("2026-05-01T10:00:00Z", "buy", "spy", 10, 100.0),
        _t("2026-05-02T10:00:00Z", "sell", "SPY", 10, 105.0),
    ]
    out = pair_round_trips(trades)
    assert len(out) == 1
    assert out[0].symbol == "SPY"
