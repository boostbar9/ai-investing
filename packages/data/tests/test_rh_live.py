"""Tests for the Robinhood read-only live market-data facade.

All Robinhood access is mocked (NO live network): a ``FakeBroker`` stands in
for the in-process read-only broker. We verify the spec's guarantees:

* RH is the PRIMARY source when fresh+available, tagged ``rh_quotes`` etc.
* On RH failure / disabled toggle we FAIL SAFE to the fallback -- never to a
  bearish or fabricated value.
* Provenance (source / age / stale) is tagged on every result.
* Cache TTL + serve-stale-on-error labels a served-stale RH value ``stale``.
* Saved-scan candidate sourcing degrades cleanly to empty (additive only).
"""
from __future__ import annotations

import pytest

from packages.data import health as health_mod
from packages.data import rh_live
from packages.data.cache import TTLCache
from packages.data.health import SourceRegistry


class FakeBroker:
    """Stand-in for RobinhoodAgenticBroker's read-only surface."""

    def __init__(self, **overrides):
        self._overrides = overrides
        self.calls: list[str] = []

    async def equity_quote(self, symbol):
        self.calls.append("equity_quote")
        return self._overrides.get("quote", {"symbol": symbol, "last": 123.0, "mid": 123.0})

    async def equity_historicals(self, symbol, *, start_time=None, interval=None, span=None):
        self.calls.append("equity_historicals")
        return self._overrides.get(
            "bars",
            [{"close_price": 10.0}, {"close_price": 11.0}, {"close_price": 12.0}],
        )

    async def equity_fundamentals(self, symbol):
        self.calls.append("equity_fundamentals")
        return self._overrides.get("fundamentals", {"symbol": symbol, "pe_ratio": 20.0})

    async def earnings_calendar(self, symbol=None):
        self.calls.append("earnings_calendar")
        return self._overrides.get("earnings", [{"symbol": "AAPL", "date": "2026-07-01"}])

    async def indexes(self):
        self.calls.append("indexes")
        return self._overrides.get(
            "indexes", [{"symbol": "VIX", "id": "vix-id-1"}]
        )

    async def index_quotes(self, instrument_ids):
        self.calls.append("index_quotes")
        return self._overrides.get("index_quotes", [{"last_price": 17.5}])

    async def scans(self):
        self.calls.append("scans")
        return self._overrides.get("scans", [])

    async def run_scan(self, scan_id):
        self.calls.append("run_scan")
        return self._overrides.get("scan_results", {}).get(scan_id, [])


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """Isolate cache + registry + toggles so tests don't touch process-wide
    state or disk, and default every RH source to enabled."""
    cache = TTLCache(default_ttl_s=300.0, use_disk=False)
    registry = SourceRegistry()
    monkeypatch.setattr(rh_live, "get_cache", lambda: cache)
    monkeypatch.setattr(rh_live, "get_registry", lambda: registry)
    monkeypatch.setattr(rh_live, "is_enabled", lambda name: True)
    rh_live.reset_for_test()
    yield
    rh_live.reset_for_test()


async def _fallback_factory(value, label="yfinance"):
    async def _fb():
        return value, label

    return _fb


# ---------------------------------------------------------------------------
# RH-primary selection + provenance
# ---------------------------------------------------------------------------
async def test_quote_rh_primary_when_available():
    rh_live.set_broker_for_test(FakeBroker())
    res = await rh_live.get_quote("AAPL")
    assert res.ok is True
    assert res.source == rh_live.SRC_QUOTES
    assert res.is_rh is True
    assert res.stale is False
    assert res.value["last"] == 123.0


async def test_bars_rh_primary_value_is_list():
    rh_live.set_broker_for_test(FakeBroker())
    res = await rh_live.get_bars("AAPL", interval="day", span="year")
    assert res.ok and res.source == rh_live.SRC_BARS
    assert isinstance(res.value, list) and len(res.value) == 3


async def test_fundamentals_rh_primary():
    rh_live.set_broker_for_test(FakeBroker())
    res = await rh_live.get_fundamentals("AAPL")
    assert res.ok and res.source == rh_live.SRC_FUNDAMENTALS
    assert res.value["pe_ratio"] == 20.0


# ---------------------------------------------------------------------------
# Fail-safe to fallback (NEVER bearish / fabricated)
# ---------------------------------------------------------------------------
async def test_quote_falls_back_when_rh_returns_nothing():
    rh_live.set_broker_for_test(FakeBroker(quote=None))
    fb = await _fallback_factory({"symbol": "AAPL", "last": 99.0}, "yfinance")
    res = await rh_live.get_quote("AAPL", fallback=fb)
    assert res.ok is True
    assert res.source == "yfinance"          # fallback provenance, not RH
    assert res.value["last"] == 99.0


async def test_quote_falls_back_when_rh_raises():
    class Boom(FakeBroker):
        async def equity_quote(self, symbol):
            raise RuntimeError("rh down")

    rh_live.set_broker_for_test(Boom())
    fb = await _fallback_factory({"last": 50.0}, "parquet")
    res = await rh_live.get_quote("AAPL", fallback=fb)
    assert res.source == "parquet" and res.value["last"] == 50.0


async def test_no_rh_and_no_fallback_is_absent_not_bearish():
    rh_live.set_broker_for_test(FakeBroker(quote=None))
    res = await rh_live.get_quote("AAPL")  # no fallback
    assert res.ok is False
    assert res.source == "none"
    assert res.value is None                # absent, never a fabricated/bearish number


# ---------------------------------------------------------------------------
# Disabled toggle short-circuits to fallback
# ---------------------------------------------------------------------------
async def test_disabled_toggle_short_circuits_to_fallback(monkeypatch):
    monkeypatch.setattr(rh_live, "is_enabled", lambda name: name != rh_live.SRC_QUOTES)
    broker = FakeBroker()
    rh_live.set_broker_for_test(broker)
    fb = await _fallback_factory({"last": 7.0}, "yfinance")
    res = await rh_live.get_quote("AAPL", fallback=fb)
    assert res.source == "yfinance"
    assert "equity_quote" not in broker.calls   # RH never even called


# ---------------------------------------------------------------------------
# Cache TTL + serve-stale labeling
# ---------------------------------------------------------------------------
async def test_quote_served_from_cache_within_ttl():
    broker = FakeBroker()
    rh_live.set_broker_for_test(broker)
    await rh_live.get_quote("AAPL")
    await rh_live.get_quote("AAPL")
    assert broker.calls.count("equity_quote") == 1   # second served from cache


async def test_stale_rh_value_served_and_labeled_stale(monkeypatch):
    # Tiny TTL so the cached value goes stale immediately.
    monkeypatch.setattr(rh_live, "TTL_QUOTES_S", 0.0)
    broker = FakeBroker()
    rh_live.set_broker_for_test(broker)
    first = await rh_live.get_quote("AAPL")
    assert first.source == rh_live.SRC_QUOTES

    # Now RH starts failing: a stale cached value must be served + labeled.
    class Boom(FakeBroker):
        async def equity_quote(self, symbol):
            raise RuntimeError("rh down")

    boom = Boom()
    boom._overrides = {}
    # Reuse the same cache (fixture-bound) but swap the broker.
    rh_live.set_broker_for_test(boom)
    second = await rh_live.get_quote("AAPL")
    assert second.stale is True
    assert second.source == "stale"
    assert second.value["last"] == 123.0     # last good value, not fabricated


# ---------------------------------------------------------------------------
# Scans: additive candidate sourcing degrades cleanly
# ---------------------------------------------------------------------------
async def test_scan_candidates_empty_degrades_cleanly():
    rh_live.set_broker_for_test(FakeBroker(scans=[]))
    res = await rh_live.get_scan_candidates()
    assert res.ok is True            # clean degrade, NOT an error
    assert res.value == []           # contributes nothing -> universe unchanged


async def test_scan_candidates_returns_symbols():
    broker = FakeBroker(
        scans=[{"id": "s1", "name": "momentum"}],
        scan_results={"s1": [{"symbol": "NVDA"}, "amd", {"symbol": "NVDA"}]},
    )
    rh_live.set_broker_for_test(broker)
    res = await rh_live.get_scan_candidates()
    assert res.ok and res.source == rh_live.SRC_SCANS
    assert res.value == ["NVDA", "AMD"]   # deduped + uppercased


async def test_scan_candidates_disconnected_is_absent(monkeypatch):
    from packages.execution import robinhood as rh_mod

    monkeypatch.setattr(rh_mod, "is_connected", lambda: False)
    rh_live.set_broker_for_test(None)  # no injected broker => consult is_connected
    res = await rh_live.get_scan_candidates()
    assert res.ok is False and res.value == []


# ---------------------------------------------------------------------------
# Regime inputs
# ---------------------------------------------------------------------------
async def test_rh_daily_closes_parses_bars():
    rh_live.set_broker_for_test(FakeBroker())
    closes = await rh_live.rh_daily_closes("SPY", days=90)
    assert closes == [10.0, 11.0, 12.0]


async def test_rh_vix_level_resolves_and_reads():
    rh_live.set_broker_for_test(FakeBroker())
    level = await rh_live.rh_vix_level()
    assert level == 17.5


async def test_rh_vix_level_none_when_no_vix_index():
    rh_live.set_broker_for_test(FakeBroker(indexes=[{"symbol": "DJX", "id": "x"}]))
    assert await rh_live.rh_vix_level() is None


# ---------------------------------------------------------------------------
# rh_bars: get_equity_historicals param build + empty-on-closed-market
# ---------------------------------------------------------------------------
class _RecordingBroker(FakeBroker):
    """FakeBroker that records the kwargs passed to equity_historicals."""

    def __init__(self, **overrides):
        super().__init__(**overrides)
        self.historicals_args: list[dict] = []

    async def equity_historicals(self, symbol, *, start_time=None, interval=None, span=None):
        self.calls.append("equity_historicals")
        self.historicals_args.append(
            {"symbol": symbol, "start_time": start_time, "interval": interval, "span": span}
        )
        return self._overrides.get(
            "bars", [{"close_price": 10.0}, {"close_price": 11.0}, {"close_price": 12.0}]
        )


async def test_daily_closes_always_sends_start_time():
    # get_equity_historicals REQUIRES symbols + start_time; the regime
    # daily-closes path must always supply start_time (interval/span too).
    broker = _RecordingBroker()
    rh_live.set_broker_for_test(broker)
    closes = await rh_live.rh_daily_closes("SPY", days=30)
    assert closes == [10.0, 11.0, 12.0]
    args = broker.historicals_args[-1]
    assert args["start_time"] is not None and args["start_time"].endswith("Z")
    assert args["interval"] == "day" and args["span"] == "year"


async def test_bars_empty_market_closed_is_success_empty_not_failure():
    # Empty bars (market closed -- Sunday) is a SUCCESSFUL-but-empty response:
    # record a health SUCCESS, fall back quietly, NEVER a recorded failure.
    rh_live.set_broker_for_test(FakeBroker(bars=[]))
    fb = await _fallback_factory([{"close_price": 5.0}], "parquet")
    res = await rh_live.get_bars("AAPL", start_time="2025-01-01", fallback=fb)
    assert res.ok is True and res.source == "parquet"   # quiet fallback
    snap = rh_live.get_registry().snapshot(rh_live.SRC_BARS)
    assert snap["consecutive_failures"] == 0
    assert snap["total_successes"] >= 1
    assert snap["status"] != "down"


async def test_bars_empty_no_fallback_is_absent_not_down():
    rh_live.set_broker_for_test(FakeBroker(bars=[]))
    res = await rh_live.get_bars("AAPL", start_time="2025-01-01")  # no fallback
    assert res.ok is False and res.source == "none" and res.value is None
    snap = rh_live.get_registry().snapshot(rh_live.SRC_BARS)
    assert snap["consecutive_failures"] == 0   # empty != failure
    assert snap["status"] != "down"


async def test_bars_real_error_records_failure_and_falls_back():
    class Boom(FakeBroker):
        async def equity_historicals(self, symbol, *, start_time=None, interval=None, span=None):
            raise RuntimeError("rh historicals down")

    rh_live.set_broker_for_test(Boom())
    fb = await _fallback_factory([{"close_price": 9.0}], "yfinance")
    res = await rh_live.get_bars("AAPL", start_time="2025-01-01", fallback=fb)
    assert res.source == "yfinance"          # fail safe, never fabricated
    snap = rh_live.get_registry().snapshot(rh_live.SRC_BARS)
    assert snap["consecutive_failures"] >= 1  # a genuine error IS a failure


# ---------------------------------------------------------------------------
# rh_indexes: id-resolution + parse, OK-when-reachable, real-failure
# ---------------------------------------------------------------------------
async def test_index_id_resolution_and_quote_parse():
    # Mixed catalog (e.g. NDX + VIX); resolve VIX's id (instrument_id key) and
    # parse the level from the returned quote.
    broker = FakeBroker(
        indexes=[
            {"symbol": "NDX", "id": "ndx-id"},
            {"symbol": "VIX", "instrument_id": "vix-2"},
        ],
        index_quotes=[{"value": 19.25}],
    )
    rh_live.set_broker_for_test(broker)
    assert await rh_live.rh_vix_level() == 19.25
    assert "index_quotes" in broker.calls


async def test_indexes_ok_when_reachable_without_vix():
    # get_indexes works (NDX present) but no VIX -> rh_indexes reports OK and we
    # fall back to yfinance ^VIX (None here). NOT a health failure.
    broker = FakeBroker(indexes=[{"symbol": "NDX", "id": "ndx-id"}])
    rh_live.set_broker_for_test(broker)
    assert await rh_live.rh_vix_level() is None
    snap = rh_live.get_registry().snapshot(rh_live.SRC_INDEXES)
    assert snap["status"] == "ok"
    assert snap["consecutive_failures"] == 0
    assert "index_quotes" not in broker.calls   # no VIX id -> no quote call


async def test_indexes_empty_records_failure():
    # get_indexes returning nothing IS a real RH outage -> recorded failure.
    rh_live.set_broker_for_test(FakeBroker(indexes=[]))
    assert await rh_live.rh_vix_level() is None
    snap = rh_live.get_registry().snapshot(rh_live.SRC_INDEXES)
    assert snap["consecutive_failures"] >= 1


async def test_indexes_closed_market_vix_empty_value_is_success():
    # Live closed-market (Sunday) shape: get_indexes lists VIX but its quote
    # carries an empty current_value (no numeric level). This is NOT a failure:
    # rh_indexes must report OK and rh_vix_level returns None so regime falls
    # back to yfinance ^VIX.
    broker = FakeBroker(
        indexes=[
            {
                "id": "3b912aa2-88f9-4682-8ae3-e39520bdf4db",
                "symbol": "VIX",
                "name": "",
                "current_value": "",
                "trade_halted": False,
            },
            {"id": "cc1fd266", "symbol": "I00001US"},
        ],
        index_quotes=[{"current_value": ""}],  # closed market -> no numeric level
    )
    rh_live.set_broker_for_test(broker)
    assert await rh_live.rh_vix_level() is None
    snap = rh_live.get_registry().snapshot(rh_live.SRC_INDEXES)
    assert snap["status"] == "ok"
    assert snap["consecutive_failures"] == 0
    assert "index_quotes" in broker.calls  # VIX id resolved -> quote attempted


# ---------------------------------------------------------------------------
# Sync->async bridge: callable repeatedly without "Event loop is closed"
# ---------------------------------------------------------------------------
def test_bridge_vix_provider_repeatable_no_loop_closed():
    rh_live.reset_for_test()
    rh_live.set_broker_for_test(FakeBroker())
    try:
        # Repeated calls must NOT raise "Event loop is closed": the worker loop
        # is persistent (the old asyncio.run-per-call closed it each time).
        for _ in range(4):
            assert rh_live.regime_vix_provider() == 17.5
    finally:
        rh_live.reset_for_test()


def test_bridge_price_provider_repeatable_no_loop_closed():
    rh_live.reset_for_test()
    rh_live.set_broker_for_test(FakeBroker())
    try:
        for _ in range(4):
            assert rh_live.regime_price_provider("SPY", days=90) == [10.0, 11.0, 12.0]
    finally:
        rh_live.reset_for_test()


def test_bridge_falls_back_to_none_when_rh_errors():
    class Boom(FakeBroker):
        async def indexes(self):
            raise RuntimeError("rh boom")

    rh_live.reset_for_test()
    rh_live.set_broker_for_test(Boom())
    try:
        # Bridge error -> None -> caller uses its yfinance default (fail safe).
        assert rh_live.regime_vix_provider() is None
        assert rh_live.regime_vix_provider() is None   # still no loop-closed
    finally:
        rh_live.reset_for_test()
