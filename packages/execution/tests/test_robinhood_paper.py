"""Tests for the Robinhood-realistic paper simulator (``robinhood_paper``).

All Robinhood access is MOCKED -- there is NO live network here. A fake
read-broker supplies async ``equity_quote`` / ``equity_tradability`` /
``review_order``; the simulator's clock, fallback-price feed and starting
balance are injected. The safety properties under test:

  * Buys cross to the ask + slippage; sells hit the bid - slippage.
  * No live quote -> fall back to the parquet feed (``pricing_source``
    ``"fallback"``); no quote AND no fallback -> FAIL SAFE (``BrokerError``),
    never a fabricated fill.
  * Market closed (weekend / outside hours) -> ``BrokerError``, no fill.
  * Explicitly untradable symbol -> skipped.
  * Per-trade cap, budget, buying power, max-trades/day, max-open all size
    the order down (or refuse) -- never up.
  * Fractional rounding honors ``get_equity_tradability``.
  * Thin displayed size -> partial fill.
  * Starting balance defaults to real RH cash but an explicit override wins.
  * Resolved provenance keeps the same shape (extras ride alongside).
  * Selecting the backend NEVER places a real order / enables live.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.execution import robinhood_paper as rp
from packages.execution.broker import BrokerError, OrderRequest
from packages.execution.robinhood_paper import RobinhoodPaperBroker

# A weekday during regular trading hours (Mon 2026-06-29, 15:00 UTC = 11:00 ET).
OPEN_DT = datetime(2026, 6, 29, 15, 0, tzinfo=UTC)
# A Sunday -> market closed.
SUNDAY_DT = datetime(2026, 6, 28, 15, 0, tzinfo=UTC)


class _FakeReadBroker:
    """Stand-in for ``RobinhoodAgenticBroker``'s read-only surface.

    Records calls so tests can assert NO order-placing method was ever
    reached (there is none on this fake -- that's the point)."""

    def __init__(
        self,
        *,
        quote: dict | None = None,
        tradability: dict | None = None,
        review: dict | None = None,
    ) -> None:
        self._quote = quote
        self._tradability = tradability or {
            "tradable": True,
            "fractional": True,
            "known": True,
        }
        self._review = review
        self.quote_calls: list[str] = []
        self.review_calls: list[dict] = []

    async def equity_quote(self, symbol: str) -> dict | None:
        self.quote_calls.append(symbol)
        return self._quote

    async def equity_tradability(self, symbol: str) -> dict:
        return self._tradability

    async def review_order(self, **kwargs) -> dict | None:
        self.review_calls.append(kwargs)
        return self._review

    async def aclose(self) -> None:  # pragma: no cover - trivial
        pass


def _broker(
    *,
    quote: dict | None = None,
    tradability: dict | None = None,
    review: dict | None = None,
    start_balance: float = 10_000.0,
    clock: datetime = OPEN_DT,
    fallback: float | None = None,
) -> RobinhoodPaperBroker:
    read = _FakeReadBroker(quote=quote, tradability=tradability, review=review)
    return RobinhoodPaperBroker(
        read_broker=read,
        fallback_price=lambda _sym: fallback,
        start_balance=start_balance,
        clock=lambda: clock,
    )


@pytest.fixture(autouse=True)
def _isolate_controls(monkeypatch, tmp_path):
    """Point trading-controls at a tmp file and neutralize the live float
    cap so cap math is deterministic and no real user state is touched."""
    from packages.cockpit import trading_controls as tc

    monkeypatch.setattr(
        tc, "TRADING_CONTROLS_PATH", tmp_path / "trading_controls.json"
    )
    monkeypatch.setattr(rp, "_resolve_float_cap", lambda: 10_000.0)
    # Generous per-trade caps by default so sizing tests opt in explicitly.
    monkeypatch.setenv("COCKPIT_TRADING_CONTROLS_PATH", str(tmp_path / "tc.json"))
    return tmp_path


def _liquid_quote(**over) -> dict:
    q = {
        "symbol": "AAPL",
        "bid": 99.0,
        "ask": 101.0,
        "mid": 100.0,
        "last": 100.0,
        "bid_size": 1_000_000.0,
        "ask_size": 1_000_000.0,
    }
    q.update(over)
    return q


# ---------------------------------------------------------------------------
# market_is_open
# ---------------------------------------------------------------------------
def test_market_closed_on_sunday():
    assert rp.market_is_open(SUNDAY_DT) is False


def test_market_open_weekday_rth():
    assert rp.market_is_open(OPEN_DT) is True


def test_premarket_closed_in_rth_but_open_in_extended():
    # 12:00 UTC = 08:00 ET on a Monday -> before RTH open, within extended.
    premarket = datetime(2026, 6, 29, 12, 0, tzinfo=UTC)
    assert rp.market_is_open(premarket) is False
    assert rp.market_is_open(premarket, extended_hours=True) is True


# ---------------------------------------------------------------------------
# Fill pricing
# ---------------------------------------------------------------------------
async def test_buy_fills_at_ask_plus_slippage():
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    ack = await b.submit(
        OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1")
    )
    assert ack.status in ("filled", "partially_filled")
    meta = b.last_fill_meta
    assert meta["pricing_source"] == "rh_quote"
    expected = 101.0 * (1.0 + rp.SLIPPAGE_BPS / 10_000.0)
    assert meta["fill_price"] == pytest.approx(expected, rel=1e-9)


async def test_sell_fills_at_bid_minus_slippage():
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    # First buy to hold a position.
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=2.0, decision_id="d1"))
    ack = await b.submit(
        OrderRequest(symbol="AAPL", side="sell", qty=1.0, decision_id="d2")
    )
    assert ack.status in ("filled", "partially_filled")
    expected = 99.0 * (1.0 - rp.SLIPPAGE_BPS / 10_000.0)
    assert b.last_fill_meta["fill_price"] == pytest.approx(expected, rel=1e-9)


async def test_synthetic_spread_when_only_single_price():
    # No bid/ask, only a last price -> synthesize a half-spread around it.
    q = {"symbol": "AAPL", "bid": None, "ask": None, "mid": None, "last": 100.0}
    b = _broker(quote=q, start_balance=100_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1"))
    hs = rp.SYNTHETIC_HALF_SPREAD_BPS / 10_000.0
    slip = rp.SLIPPAGE_BPS / 10_000.0
    assert b.last_fill_meta["fill_price"] == pytest.approx(
        100.0 * (1.0 + hs + slip), rel=1e-9
    )


# ---------------------------------------------------------------------------
# Fallback + fail-safe pricing
# ---------------------------------------------------------------------------
async def test_fallback_pricing_when_no_quote():
    b = _broker(quote=None, fallback=50.0, start_balance=100_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1"))
    meta = b.last_fill_meta
    assert meta["pricing_source"] == "fallback"
    hs = rp.SYNTHETIC_HALF_SPREAD_BPS / 10_000.0
    slip = rp.SLIPPAGE_BPS / 10_000.0
    assert meta["fill_price"] == pytest.approx(50.0 * (1.0 + hs + slip), rel=1e-9)


async def test_fail_safe_when_no_quote_and_no_fallback():
    b = _broker(quote=None, fallback=None, start_balance=100_000.0)
    with pytest.raises(BrokerError):
        await b.submit(
            OrderRequest(symbol="ZZZZ", side="buy", qty=1.0, decision_id="d1")
        )
    assert b.last_fill_meta is None


# ---------------------------------------------------------------------------
# Gates: market hours, tradability
# ---------------------------------------------------------------------------
async def test_market_closed_raises_no_fill():
    b = _broker(quote=_liquid_quote(), clock=SUNDAY_DT, start_balance=100_000.0)
    with pytest.raises(BrokerError, match="market closed"):
        await b.submit(
            OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1")
        )


async def test_untradable_symbol_skipped():
    b = _broker(
        quote=_liquid_quote(),
        tradability={"tradable": False, "fractional": False, "known": True},
        start_balance=100_000.0,
    )
    with pytest.raises(BrokerError, match="not tradable"):
        await b.submit(
            OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1")
        )


# ---------------------------------------------------------------------------
# Caps, buying power, sizing
# ---------------------------------------------------------------------------
async def test_per_trade_cap_sizes_down(monkeypatch):
    from packages.cockpit import trading_controls as tc

    tc.update_controls(
        {"max_per_trade_usd": 50.0, "total_budget_usd": 10_000.0}
    )
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=10.0, decision_id="d1"))
    meta = b.last_fill_meta
    # 50 / ~101.02 ask ≈ 0.495 shares -> notional <= $50.
    assert meta["notional_usd"] <= 50.0 + 1e-6
    assert meta["partial"] is True


async def test_buying_power_caps_buy():
    # Only $30 cash -> can't buy a full $101 share even though caps allow it.
    b = _broker(quote=_liquid_quote(), start_balance=30.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=5.0, decision_id="d1"))
    meta = b.last_fill_meta
    assert meta["notional_usd"] <= 30.0 + 1e-6


async def test_max_trades_per_day_blocks_after_limit():
    from packages.cockpit import trading_controls as tc

    tc.update_controls(
        {"max_trades_per_day": 1, "total_budget_usd": 10_000.0, "max_per_trade_usd": 5_000.0}
    )
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1"))
    with pytest.raises(BrokerError, match="max trades"):
        await b.submit(
            OrderRequest(symbol="MSFT", side="buy", qty=1.0, decision_id="d2")
        )


async def test_max_open_positions_blocks_new_symbol():
    from packages.cockpit import trading_controls as tc

    tc.update_controls(
        {
            "max_open_positions": 1,
            "max_trades_per_day": 50,
            "total_budget_usd": 10_000.0,
            "max_per_trade_usd": 5_000.0,
        }
    )
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1"))
    with pytest.raises(BrokerError, match="max open"):
        await b.submit(
            OrderRequest(symbol="MSFT", side="buy", qty=1.0, decision_id="d2")
        )


# ---------------------------------------------------------------------------
# Fractional rounding
# ---------------------------------------------------------------------------
async def test_non_fractional_rounds_to_whole_shares():
    from packages.cockpit import trading_controls as tc

    tc.update_controls({"max_per_trade_usd": 250.0, "total_budget_usd": 10_000.0})
    b = _broker(
        quote=_liquid_quote(),
        tradability={"tradable": True, "fractional": False, "known": True},
        start_balance=100_000.0,
    )
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=10.0, decision_id="d1"))
    meta = b.last_fill_meta
    # $250 cap / ~$101 ask -> 2.47 shares -> floor to 2 whole shares.
    assert meta["filled_qty"] == 2.0
    assert meta["fractional"] is False


async def test_fractional_below_one_dollar_minimum_refused():
    from packages.cockpit import trading_controls as tc

    tc.update_controls({"max_per_trade_usd": 0.5, "total_budget_usd": 10_000.0})
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    with pytest.raises(BrokerError):
        await b.submit(
            OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1")
        )


# ---------------------------------------------------------------------------
# Partial fills on thin liquidity
# ---------------------------------------------------------------------------
async def test_thin_ask_size_partial_fill():
    from packages.cockpit import trading_controls as tc

    tc.update_controls({"max_per_trade_usd": 10_000.0, "total_budget_usd": 100_000.0})
    q = _liquid_quote(ask_size=0.25)
    b = _broker(quote=q, start_balance=1_000_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=5.0, decision_id="d1"))
    meta = b.last_fill_meta
    assert meta["filled_qty"] == pytest.approx(0.25)
    assert meta["partial"] is True


async def test_thin_bid_size_partial_sell():
    from packages.cockpit import trading_controls as tc

    tc.update_controls({"max_per_trade_usd": 10_000.0, "total_budget_usd": 100_000.0})
    b = _broker(quote=_liquid_quote(), start_balance=1_000_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=5.0, decision_id="d1"))
    # Now sell into a thin bid.
    b._read_broker._quote = _liquid_quote(bid_size=0.5)
    await b.submit(OrderRequest(symbol="AAPL", side="sell", qty=5.0, decision_id="d2"))
    assert b.last_fill_meta["filled_qty"] == pytest.approx(0.5)
    assert b.last_fill_meta["partial"] is True


# ---------------------------------------------------------------------------
# Starting balance resolution
# ---------------------------------------------------------------------------
async def test_injected_start_balance_used():
    b = _broker(quote=_liquid_quote(), start_balance=1234.0)
    state = b.account_state()
    # Ledger resolves lazily; force it via a positions read.
    await b.positions()
    state = b.account_state()
    assert state["start_balance_usd"] == 1234.0
    assert state["cash_usd"] == 1234.0


async def test_configured_override_beats_real_cash(monkeypatch):
    from packages.cockpit import trading_controls as tc

    tc.update_controls(
        {"paper_use_real_cash": False, "paper_start_balance_usd": 777.0}
    )

    async def _fake_real(*_a, **_k):
        return {"cash": 99999.0, "source": "rh_account", "connected": True}

    monkeypatch.setattr(rp, "fetch_real_rh_cash", _fake_real)
    bal, src = await rp.resolve_paper_start_balance()
    assert bal == 777.0
    assert src == "configured"


async def test_real_cash_used_when_no_override(monkeypatch):
    from packages.cockpit import trading_controls as tc

    tc.update_controls({"paper_use_real_cash": True})

    async def _fake_real(*_a, **_k):
        return {"cash": 4242.0, "source": "rh_account", "connected": True}

    monkeypatch.setattr(rp, "fetch_real_rh_cash", _fake_real)
    bal, src = await rp.resolve_paper_start_balance()
    assert bal == 4242.0
    assert src == "rh_cash"


async def test_real_cash_unavailable_falls_back_to_budget(monkeypatch):
    from packages.cockpit import trading_controls as tc

    tc.update_controls({"paper_use_real_cash": True})

    async def _fake_real(*_a, **_k):
        return {"cash": None, "source": "not_connected", "connected": False}

    monkeypatch.setattr(rp, "fetch_real_rh_cash", _fake_real)
    monkeypatch.setattr(rp, "_resolve_float_cap", lambda: 300.0)
    bal, src = await rp.resolve_paper_start_balance()
    assert bal == 300.0
    assert src == "budget"


# ---------------------------------------------------------------------------
# Provenance schema
# ---------------------------------------------------------------------------
async def test_fill_meta_schema_keys():
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1"))
    meta = b.last_fill_meta
    for key in (
        "pricing_source",
        "fill_price",
        "spread_bps",
        "slippage_bps",
        "requested_qty",
        "filled_qty",
        "partial",
        "notional_usd",
    ):
        assert key in meta


# ---------------------------------------------------------------------------
# Order-review grounding (opt-in, read-only)
# ---------------------------------------------------------------------------
async def test_review_grounding_called_only_when_enabled():
    from packages.cockpit import trading_controls as tc

    tc.update_controls(
        {"paper_review_grounding": True, "total_budget_usd": 10_000.0, "max_per_trade_usd": 5_000.0}
    )
    b = _broker(quote=_liquid_quote(), review={"ok": True}, start_balance=100_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1"))
    assert b._read_broker.review_calls  # review was consulted
    assert b.last_fill_meta["review_anchored"] is True


async def test_review_grounding_not_called_when_disabled():
    from packages.cockpit import trading_controls as tc

    tc.update_controls({"paper_review_grounding": False})
    b = _broker(quote=_liquid_quote(), review={"ok": True}, start_balance=100_000.0)
    await b.submit(OrderRequest(symbol="AAPL", side="buy", qty=1.0, decision_id="d1"))
    assert not b._read_broker.review_calls


# ---------------------------------------------------------------------------
# Invalid requests
# ---------------------------------------------------------------------------
async def test_invalid_request_rejected():
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    with pytest.raises(BrokerError, match="invalid order"):
        await b.submit(OrderRequest(symbol="", side="buy", qty=1.0, decision_id="d1"))
    with pytest.raises(BrokerError, match="invalid order"):
        await b.submit(
            OrderRequest(symbol="AAPL", side="hold", qty=1.0, decision_id="d2")
        )


async def test_sell_with_no_position_refused():
    b = _broker(quote=_liquid_quote(), start_balance=100_000.0)
    with pytest.raises(BrokerError):
        await b.submit(
            OrderRequest(symbol="AAPL", side="sell", qty=1.0, decision_id="d1")
        )


async def test_health_always_true_offline():
    b = _broker(quote=None, fallback=None)
    assert await b.health() is True


# ---------------------------------------------------------------------------
# Factory + backend wiring
# ---------------------------------------------------------------------------
def test_build_robinhood_paper_broker():
    b = rp.build_robinhood_paper_broker()
    assert isinstance(b, RobinhoodPaperBroker)
    assert b.name == "robinhood_paper"


def test_factory_resolves_robinhood_paper(monkeypatch):
    from packages.execution import broker_factory as bf

    monkeypatch.setenv("BROKER_BACKEND", "robinhood_paper")
    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, RobinhoodPaperBroker)
    assert sel.effective_backend == bf.BACKEND_ROBINHOOD_PAPER
    assert not sel.fell_back


def test_onboarding_accepts_robinhood_paper_backend():
    from packages.cockpit import onboarding as ob

    assert "robinhood_paper" in ob.VALID_BROKER_BACKENDS


def test_trading_controls_paper_fields_round_trip(tmp_path):
    from packages.cockpit import trading_controls as tc

    path = tmp_path / "tc.json"
    tc.update_controls(
        {
            "paper_use_real_cash": False,
            "paper_start_balance_usd": 1000.0,
            "paper_review_grounding": True,
            "paper_extended_hours": True,
        },
        path=path,
    )
    loaded = tc.load_controls(path)
    assert loaded.paper_use_real_cash is False
    assert loaded.paper_start_balance_usd == 1000.0
    assert loaded.paper_review_grounding is True
    assert loaded.paper_extended_hours is True
