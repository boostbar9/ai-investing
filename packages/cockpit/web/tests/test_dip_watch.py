"""Tests for Phase 25 dip_watch buy-back logic."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from packages.cockpit.web import dip_watch


@pytest.fixture
def fresh_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dip_watch._WatcherStore:
    """Replace the module-level singleton with a tmp-backed store."""
    store = dip_watch._WatcherStore(path=tmp_path / "watchers.json")
    monkeypatch.setattr(dip_watch, "WATCHERS", store)
    return store


# ---- arm() ------------------------------------------------------------------


def test_arm_creates_watcher(fresh_store: dip_watch._WatcherStore) -> None:
    w = dip_watch.arm(
        symbol="AAPL", exit_price=100.0, exit_pnl_pct=0.03, qty=10, dip_pct=0.02
    )
    assert w is not None
    assert w.symbol == "AAPL"
    assert w.exit_price == 100.0
    assert w.target_price == pytest.approx(98.0)
    assert w.qty == 10
    all_w = fresh_store.all()
    assert "AAPL" in all_w


def test_arm_rejects_invalid_inputs(fresh_store: dip_watch._WatcherStore) -> None:
    assert dip_watch.arm(symbol="", exit_price=100, exit_pnl_pct=0.03, qty=10) is None
    assert dip_watch.arm(symbol="X", exit_price=0, exit_pnl_pct=0.03, qty=10) is None
    assert dip_watch.arm(symbol="X", exit_price=100, exit_pnl_pct=0.03, qty=0) is None


def test_arm_replaces_existing_watcher(fresh_store: dip_watch._WatcherStore) -> None:
    dip_watch.arm(symbol="AAPL", exit_price=100, exit_pnl_pct=0.03, qty=10)
    w2 = dip_watch.arm(symbol="AAPL", exit_price=110, exit_pnl_pct=0.05, qty=5)
    assert w2 is not None
    assert fresh_store.all()["AAPL"].exit_price == 110.0
    assert len(fresh_store.all()) == 1


def test_arm_uses_default_dip_pct(
    fresh_store: dip_watch._WatcherStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DIP_WATCH_DIP_PCT", raising=False)
    w = dip_watch.arm(symbol="AAPL", exit_price=100, exit_pnl_pct=0.03, qty=10)
    assert w is not None
    assert w.dip_pct == 0.015  # default 1.5%
    assert w.target_price == pytest.approx(98.5)


# ---- Persistence -------------------------------------------------------------


def test_watchers_persist_across_store_instances(tmp_path: Path) -> None:
    p = tmp_path / "w.json"
    s1 = dip_watch._WatcherStore(path=p)
    w = dip_watch.Watcher(
        symbol="AAPL",
        exit_price=100.0,
        exit_pnl_pct=0.03,
        dip_pct=0.02,
        target_price=98.0,
        armed_at="2026-06-01T00:00:00+00:00",
        expires_at="2026-06-03T00:00:00+00:00",
        qty=10,
    )
    s1.put(w)

    s2 = dip_watch._WatcherStore(path=p)
    assert "AAPL" in s2.all()


def test_expire_removes_stale_watchers(tmp_path: Path) -> None:
    s = dip_watch._WatcherStore(path=tmp_path / "w.json")
    expired = dip_watch.Watcher(
        symbol="OLD",
        exit_price=100.0,
        exit_pnl_pct=0.03,
        dip_pct=0.02,
        target_price=98.0,
        armed_at="2026-01-01T00:00:00+00:00",
        expires_at="2026-01-02T00:00:00+00:00",
        qty=10,
    )
    fresh = dip_watch.Watcher(
        symbol="NEW",
        exit_price=100.0,
        exit_pnl_pct=0.03,
        dip_pct=0.02,
        target_price=98.0,
        armed_at="2026-06-01T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        qty=10,
    )
    s.put(expired)
    s.put(fresh)
    removed = s.expire(datetime(2026, 6, 1, tzinfo=UTC))
    assert len(removed) == 1
    assert removed[0].symbol == "OLD"
    assert "NEW" in s.all()
    assert "OLD" not in s.all()


# ---- run_tick ----------------------------------------------------------------


def test_run_tick_fires_when_price_hits_target(
    fresh_store: dip_watch._WatcherStore,
) -> None:
    dip_watch.arm(
        symbol="AAPL", exit_price=100.0, exit_pnl_pct=0.03, qty=10, dip_pct=0.02
    )
    # Target = 98.0

    buys: list[tuple[str, float]] = []

    async def _submit(symbol: str, qty: float):
        buys.append((symbol, qty))

        class _Ack:
            broker_order_id = "ok"

        return _Ack()

    def _price(symbol: str) -> float | None:
        return 97.5  # below target

    r = asyncio.run(
        dip_watch.run_tick(
            price_lookup=_price, submit_buy=_submit, size_fraction=1.0
        )
    )
    assert r.fired == 1
    assert buys == [("AAPL", 10.0)]
    # Watcher should be gone now
    assert "AAPL" not in fresh_store.all()


def test_run_tick_does_not_fire_above_target(
    fresh_store: dip_watch._WatcherStore,
) -> None:
    dip_watch.arm(
        symbol="AAPL", exit_price=100.0, exit_pnl_pct=0.03, qty=10, dip_pct=0.02
    )

    async def _submit(symbol: str, qty: float):
        raise AssertionError("should not fire")

    def _price(symbol: str) -> float | None:
        return 99.0  # above target 98.0

    r = asyncio.run(dip_watch.run_tick(price_lookup=_price, submit_buy=_submit))
    assert r.checked == 1
    assert r.fired == 0
    assert "AAPL" in fresh_store.all()


def test_run_tick_skips_missing_prices(fresh_store: dip_watch._WatcherStore) -> None:
    dip_watch.arm(symbol="AAPL", exit_price=100.0, exit_pnl_pct=0.03, qty=10)

    def _price(symbol: str) -> float | None:
        return None

    r = asyncio.run(dip_watch.run_tick(price_lookup=_price, submit_buy=None))
    assert r.checked == 1
    assert r.fired == 0
    assert "AAPL" in fresh_store.all()


def test_run_tick_expires_stale_watchers(
    fresh_store: dip_watch._WatcherStore,
) -> None:
    # Arm with negative TTL by manipulating the watcher directly.
    w = dip_watch.Watcher(
        symbol="OLD",
        exit_price=100.0,
        exit_pnl_pct=0.03,
        dip_pct=0.02,
        target_price=98.0,
        armed_at="2026-01-01T00:00:00+00:00",
        expires_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat(timespec="seconds"),
        qty=10,
    )
    fresh_store.put(w)

    def _price(symbol: str) -> float | None:
        return 95.0

    r = asyncio.run(dip_watch.run_tick(price_lookup=_price, submit_buy=None))
    assert r.expired == 1
    assert "OLD" not in fresh_store.all()


def test_clear_removes_one_or_all(fresh_store: dip_watch._WatcherStore) -> None:
    dip_watch.arm(symbol="AAPL", exit_price=100, exit_pnl_pct=0.03, qty=10)
    dip_watch.arm(symbol="MSFT", exit_price=200, exit_pnl_pct=0.04, qty=5)
    assert dip_watch.clear("AAPL") == 1
    assert "AAPL" not in fresh_store.all()
    assert "MSFT" in fresh_store.all()
    assert dip_watch.clear(None) == 1
    assert fresh_store.all() == {}


def test_snapshot_returns_expected_shape(fresh_store: dip_watch._WatcherStore) -> None:
    dip_watch.arm(symbol="AAPL", exit_price=100, exit_pnl_pct=0.03, qty=10)
    snap = dip_watch.snapshot()
    assert "watchers" in snap
    assert "config" in snap
    assert len(snap["watchers"]) == 1
    assert snap["watchers"][0]["symbol"] == "AAPL"
