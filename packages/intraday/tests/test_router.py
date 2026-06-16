"""Tests for the intraday router — Phase 28-R step 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from packages.intraday.router import (
    RouteResult,
    route_setups,
    shares_for_notional,
)
from packages.intraday.setup_finder import RankedSetup

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    return tmp_path / "intraday_router.jsonl"


@pytest.fixture
def enable_intraday(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests run with INTRADAY_MODE=1 unless they specifically opt out."""
    monkeypatch.setenv("INTRADAY_MODE", "1")


def _setup(
    symbol: str = "AAPL",
    score: float = 0.75,
    notional_usd: float = 100.0,
    reason: str = "ORB+1.0% | VWAP+0.5%",
) -> RankedSetup:
    return RankedSetup(
        symbol=symbol,
        score=score,
        components={
            "orb_breakout": 0.8,
            "vwap_align": 0.5,
            "news_sentiment": 0.6,
            "insider_cluster": 0.3,
        },
        notional_usd=notional_usd,
        reason=reason,
    )


class FakeAck:
    def __init__(self, broker_order_id: str = "ord-1") -> None:
        self.broker_order_id = broker_order_id


# ---------------------------------------------------------------------------
# shares_for_notional
# ---------------------------------------------------------------------------


class TestSharesForNotional:
    def test_zero_price(self) -> None:
        assert shares_for_notional(100.0, 0.0) == 0.0

    def test_negative_price(self) -> None:
        assert shares_for_notional(100.0, -5.0) == 0.0

    def test_zero_notional(self) -> None:
        assert shares_for_notional(0.0, 50.0) == 0.0

    def test_clean_division(self) -> None:
        assert shares_for_notional(100.0, 25.0) == 4.0

    def test_floor_division(self) -> None:
        # $100 / $30 = 3.33 -> 3 whole shares
        assert shares_for_notional(100.0, 30.0) == 3.0

    def test_under_one_share(self) -> None:
        # $100 / $150 = 0.66 -> 0 shares
        assert shares_for_notional(100.0, 150.0) == 0.0


# ---------------------------------------------------------------------------
# route_setups — disabled mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_setups_disabled_mode_is_noop(
    monkeypatch: pytest.MonkeyPatch, tmp_log: Path
) -> None:
    monkeypatch.delenv("INTRADAY_MODE", raising=False)
    calls: list[Any] = []

    async def submit(symbol: str, qty: float) -> Any:
        calls.append((symbol, qty))
        return FakeAck()

    result = await route_setups(
        [_setup()],
        submit_order=submit,
        price_lookup=lambda s: 100.0,
        log_path=tmp_log,
    )
    assert isinstance(result, RouteResult)
    assert result.submitted == []
    assert result.skipped == []
    assert result.errors == []
    assert calls == []
    # One audit row for "disabled".
    contents = tmp_log.read_text(encoding="utf-8").strip()
    assert contents
    rec = json.loads(contents)
    assert rec["action"] == "disabled"


# ---------------------------------------------------------------------------
# route_setups — happy path
# ---------------------------------------------------------------------------


class TestRouteSetupsHappyPath:
    @pytest.mark.asyncio
    async def test_submits_orders(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        submitted: list[tuple[str, float]] = []

        async def submit(symbol: str, qty: float) -> Any:
            submitted.append((symbol, qty))
            return FakeAck(broker_order_id=f"ord-{symbol}")

        result = await route_setups(
            [_setup("AAPL", notional_usd=100.0)],
            submit_order=submit,
            price_lookup=lambda s: 25.0,
            log_path=tmp_log,
        )
        # $100 / $25 = 4 shares
        assert submitted == [("AAPL", 4.0)]
        assert len(result.submitted) == 1
        assert result.submitted[0].broker_order_id == "ord-AAPL"
        assert result.submitted[0].qty == 4.0
        assert result.submitted[0].notional_usd == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_audit_log_contains_components(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        async def submit(symbol: str, qty: float) -> Any:
            return FakeAck()

        await route_setups(
            [_setup("MSFT", notional_usd=200.0)],
            submit_order=submit,
            price_lookup=lambda s: 50.0,
            log_path=tmp_log,
        )
        lines = tmp_log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["action"] == "submitted"
        assert rec["symbol"] == "MSFT"
        assert rec["qty"] == 4.0
        assert rec["components"]["orb_breakout"] == 0.8

    @pytest.mark.asyncio
    async def test_uppercases_symbols(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        called: list[str] = []

        async def submit(symbol: str, qty: float) -> Any:
            called.append(symbol)
            return FakeAck()

        await route_setups(
            [_setup("aapl", notional_usd=100.0)],
            submit_order=submit,
            price_lookup=lambda s: 25.0,
            log_path=tmp_log,
        )
        # The setup itself has symbol="aapl" but route_setups normalizes.
        assert called == ["AAPL"]


# ---------------------------------------------------------------------------
# Skip paths
# ---------------------------------------------------------------------------


class TestSkipPaths:
    @pytest.mark.asyncio
    async def test_skip_held(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        submitted: list[Any] = []

        async def submit(symbol: str, qty: float) -> Any:
            submitted.append((symbol, qty))
            return FakeAck()

        result = await route_setups(
            [_setup("AAPL"), _setup("MSFT")],
            submit_order=submit,
            price_lookup=lambda s: 25.0,
            held_symbols_getter=lambda: {"AAPL"},
            log_path=tmp_log,
        )
        assert submitted == [("MSFT", 4.0)]  # AAPL skipped
        assert len(result.submitted) == 1
        assert len(result.skipped) == 1
        assert result.skipped[0].action == "skip_held"
        assert result.skipped[0].symbol == "AAPL"

    @pytest.mark.asyncio
    async def test_skip_no_price(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        submitted: list[Any] = []

        async def submit(symbol: str, qty: float) -> Any:
            submitted.append((symbol, qty))
            return FakeAck()

        result = await route_setups(
            [_setup("AAPL")],
            submit_order=submit,
            price_lookup=lambda s: None,  # no quote
            log_path=tmp_log,
        )
        assert submitted == []
        assert len(result.skipped) == 1
        assert result.skipped[0].action == "skip_no_price"

    @pytest.mark.asyncio
    async def test_skip_zero_price(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        async def submit(symbol: str, qty: float) -> Any:
            raise AssertionError("should not submit")

        result = await route_setups(
            [_setup("AAPL")],
            submit_order=submit,
            price_lookup=lambda s: 0.0,
            log_path=tmp_log,
        )
        assert len(result.skipped) == 1
        assert result.skipped[0].action == "skip_no_price"

    @pytest.mark.asyncio
    async def test_skip_too_small_below_min_notional(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        async def submit(symbol: str, qty: float) -> Any:
            raise AssertionError("should not submit")

        # $100 budget, $200 share price -> qty=0
        result = await route_setups(
            [_setup("BRKA", notional_usd=100.0)],
            submit_order=submit,
            price_lookup=lambda s: 200.0,
            log_path=tmp_log,
        )
        assert len(result.skipped) == 1
        assert result.skipped[0].action == "skip_too_small"
        assert result.skipped[0].qty == 0.0

    @pytest.mark.asyncio
    async def test_skip_too_small_above_min_share_count(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        """Edge case: qty=1 but notional < MIN_NOTIONAL_USD."""

        async def submit(symbol: str, qty: float) -> Any:
            raise AssertionError("should not submit")

        # $4 notional, $4 price -> qty=1, effective notional $4 < $5 min
        result = await route_setups(
            [_setup("XYZ", notional_usd=4.0)],
            submit_order=submit,
            price_lookup=lambda s: 4.0,
            log_path=tmp_log,
        )
        assert len(result.skipped) == 1
        assert result.skipped[0].action == "skip_too_small"


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


class TestErrorPath:
    @pytest.mark.asyncio
    async def test_submit_error_recorded_but_other_orders_continue(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        async def submit(symbol: str, qty: float) -> Any:
            if symbol == "AAPL":
                raise RuntimeError("alpaca rejected")
            return FakeAck()

        result = await route_setups(
            [_setup("AAPL"), _setup("MSFT"), _setup("GOOG")],
            submit_order=submit,
            price_lookup=lambda s: 25.0,
            log_path=tmp_log,
        )
        assert {a.symbol for a in result.submitted} == {"MSFT", "GOOG"}
        assert len(result.errors) == 1
        assert result.errors[0].symbol == "AAPL"
        assert "alpaca rejected" in result.errors[0].reason

    @pytest.mark.asyncio
    async def test_audit_log_records_error(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        async def submit(symbol: str, qty: float) -> Any:
            raise RuntimeError("rate limited")

        await route_setups(
            [_setup("AAPL")],
            submit_order=submit,
            price_lookup=lambda s: 25.0,
            log_path=tmp_log,
        )
        lines = tmp_log.read_text(encoding="utf-8").strip().splitlines()
        rec = json.loads(lines[0])
        assert rec["action"] == "error"
        assert "rate limited" in rec["reason"]


# ---------------------------------------------------------------------------
# Mixed scenarios
# ---------------------------------------------------------------------------


class TestMixed:
    @pytest.mark.asyncio
    async def test_full_scenario_with_held_and_missing_and_submit(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        async def submit(symbol: str, qty: float) -> Any:
            return FakeAck(broker_order_id=f"ord-{symbol}")

        prices = {"AAPL": 25.0, "MSFT": 25.0, "GOOG": None}

        result = await route_setups(
            [
                _setup("AAPL"),  # held -> skip
                _setup("MSFT"),  # submitted
                _setup("GOOG"),  # no price -> skip
            ],
            submit_order=submit,
            price_lookup=lambda s: prices.get(s),
            held_symbols_getter=lambda: {"AAPL"},
            log_path=tmp_log,
        )
        assert {a.symbol for a in result.submitted} == {"MSFT"}
        assert {a.symbol for a in result.skipped} == {"AAPL", "GOOG"}
        assert {a.action for a in result.skipped} == {
            "skip_held",
            "skip_no_price",
        }

    @pytest.mark.asyncio
    async def test_force_enabled_overrides_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_log: Path
    ) -> None:
        monkeypatch.delenv("INTRADAY_MODE", raising=False)
        calls: list[Any] = []

        async def submit(symbol: str, qty: float) -> Any:
            calls.append(symbol)
            return FakeAck()

        result = await route_setups(
            [_setup("AAPL", notional_usd=100.0)],
            submit_order=submit,
            price_lookup=lambda s: 25.0,
            log_path=tmp_log,
            force_enabled=True,
        )
        assert len(result.submitted) == 1
        assert calls == ["AAPL"]

    @pytest.mark.asyncio
    async def test_dict_ack_extracts_order_id(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        async def submit(symbol: str, qty: float) -> Any:
            return {"broker_order_id": "dict-ord-42"}

        result = await route_setups(
            [_setup("AAPL")],
            submit_order=submit,
            price_lookup=lambda s: 25.0,
            log_path=tmp_log,
        )
        assert result.submitted[0].broker_order_id == "dict-ord-42"

    @pytest.mark.asyncio
    async def test_empty_setups_list(
        self, enable_intraday: None, tmp_log: Path
    ) -> None:
        async def submit(symbol: str, qty: float) -> Any:
            raise AssertionError("should not submit")

        result = await route_setups(
            [],
            submit_order=submit,
            price_lookup=lambda s: 100.0,
            log_path=tmp_log,
        )
        assert result.submitted == []
        assert result.skipped == []
        assert result.errors == []


# ---------------------------------------------------------------------------
# Log-path resolution
# ---------------------------------------------------------------------------


class TestLogPath:
    @pytest.mark.asyncio
    async def test_env_override(
        self,
        enable_intraday: None,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        custom = tmp_path / "sub" / "router.jsonl"
        monkeypatch.setenv("INTRADAY_ROUTER_LOG_PATH", str(custom))

        async def submit(symbol: str, qty: float) -> Any:
            return FakeAck()

        await route_setups(
            [_setup("AAPL")],
            submit_order=submit,
            price_lookup=lambda s: 25.0,
        )
        assert custom.exists()
