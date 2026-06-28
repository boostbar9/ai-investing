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
