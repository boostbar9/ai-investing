"""Tests for the INDEPENDENT micro-cap catalyst universe feeding under_radar.

These cover the universe-sourced lane added on top of PR #22:

  * :func:`candidates_from_under_radar_universe` — runs proactively-sourced
    seeds (earnings/IPO calendars, RH scans) through the UNCHANGED PR #22
    pipeline (catalyst classifier -> ETF/mega-cap exclusion -> MANDATORY
    liquidity/price/spread gate), tags ``lane="under_radar"`` + catalyst fields,
    caps at ``RADAR_MAX_CANDIDATES``, and reports the post-gate count.
  * :func:`_gather_under_radar_universe` — probes read-only sub-sources, each
    degrading independently and fail-safe; records per-sub-source provenance.

All read-only responses are MOCKED — no network.
"""

from __future__ import annotations

import pytest

from packages.agents.research_sweep import (
    _gather_under_radar_universe,
    _RadarSeed,
    candidates_from_under_radar_universe,
)
from packages.data.rh_live import Provenanced


def _liq(price, dvol, spread, *, market_cap=None, is_etf=False):
    d: dict = {
        "price": price,
        "avg_dollar_volume": dvol,
        "spread_pct": spread,
        "is_etf": is_etf,
    }
    if market_cap is not None:
        d["market_cap"] = market_cap
    return d


# ---------------------------------------------------------------------------
# candidates_from_under_radar_universe — the pipeline core
# ---------------------------------------------------------------------------


def test_earnings_calendar_lowprice_smallcap_surfaces():
    seeds = [
        _RadarSeed(
            symbol="BCRX",
            catalyst_text="BCRX upcoming quarterly results / earnings report"
            " scheduled for 2024-02-15",
            source="earnings_calendar",
        )
    ]
    liq = {"BCRX": _liq(1.50, 5_000_000, 0.01, market_cap=3e8)}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol=liq
    )
    assert [c.symbol for c in cands] == ["BCRX"]
    c = cands[0]
    assert c.lane == "under_radar"
    assert c.catalyst_type == "earnings"
    assert c.catalyst_score > 0.0
    assert c.confidence == c.catalyst_score
    assert "under_radar" in c.sources
    assert "earnings_calendar" in c.sources
    assert after_gate == 1


def test_megacap_excluded_by_cap():
    seeds = [
        _RadarSeed(
            symbol="AAPL",
            catalyst_text="AAPL upcoming quarterly results earnings report",
            source="earnings_calendar",
        )
    ]
    liq = {"AAPL": _liq(190.0, 1e10, 0.001, market_cap=3e12)}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol=liq
    )
    assert cands == []
    assert after_gate == 0


def test_index_etf_excluded():
    seeds = [
        _RadarSeed(
            symbol="XBI",
            catalyst_text="XBI upcoming quarterly results earnings report",
            source="rh_scan",
        )
    ]
    # ETF flag set on the resolved fundamentals row -> excluded even though the
    # price/cap would otherwise pass.
    liq = {"XBI": _liq(2.0, 5_000_000, 0.01, market_cap=1e8, is_etf=True)}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol=liq
    )
    assert cands == []
    assert after_gate == 0


def test_low_volume_excluded():
    seeds = [
        _RadarSeed(
            symbol="LOWV",
            catalyst_text="LOWV gets FDA approval for new therapy",
            source="news_catalyst",
        )
    ]
    liq = {"LOWV": _liq(2.0, 100_000, 0.01, market_cap=1e8)}
    cands, _ = candidates_from_under_radar_universe(seeds, liquidity_by_symbol=liq)
    assert cands == []


def test_wide_spread_excluded():
    seeds = [
        _RadarSeed(
            symbol="WIDE",
            catalyst_text="WIDE wins major government contract award",
            source="news_catalyst",
        )
    ]
    liq = {"WIDE": _liq(2.0, 5_000_000, 0.25, market_cap=1e8)}
    cands, _ = candidates_from_under_radar_universe(seeds, liquidity_by_symbol=liq)
    assert cands == []


def test_missing_liquidity_failsafe_excluded():
    seeds = [
        _RadarSeed(
            symbol="NODATA",
            catalyst_text="NODATA FDA approval granted",
            source="earnings_calendar",
        )
    ]
    # No liquidity entry at all => fail-safe exclude.
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}
    )
    assert cands == []
    assert after_gate == 0


def test_no_catalyst_not_surfaced():
    # A scan-sourced symbol with NO catalyst text never surfaces, even though
    # its price/volume/spread/cap would all pass the gate.
    seeds = [_RadarSeed(symbol="TINY", catalyst_text="", source="rh_scan")]
    liq = {"TINY": _liq(1.50, 5_000_000, 0.01, market_cap=3e8)}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol=liq
    )
    assert cands == []
    assert after_gate == 0


def test_scan_symbol_surfaces_only_when_calendar_supplies_catalyst():
    # Same symbol appears in a catalyst-less scan AND the earnings calendar:
    # the calendar's real dated catalyst lets it surface, tagged with both subs.
    seeds = [
        _RadarSeed(symbol="BCRX", catalyst_text="", source="rh_scan"),
        _RadarSeed(
            symbol="BCRX",
            catalyst_text="BCRX upcoming quarterly results earnings report",
            source="earnings_calendar",
        ),
    ]
    liq = {"BCRX": _liq(1.50, 5_000_000, 0.01, market_cap=3e8)}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol=liq
    )
    assert [c.symbol for c in cands] == ["BCRX"]
    assert {"rh_scan", "earnings_calendar"} <= set(cands[0].sources)
    assert after_gate == 1


def test_max_candidates_cap_respected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RADAR_MAX_CANDIDATES", "2")
    seeds = []
    liq = {}
    for i in range(5):
        sym = f"SYM{i}"
        seeds.append(
            _RadarSeed(
                symbol=sym,
                catalyst_text=f"{sym} receives FDA approval for drug",
                source="earnings_calendar",
            )
        )
        liq[sym] = _liq(1.50, 5_000_000, 0.01, market_cap=3e8)
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol=liq
    )
    # All 5 pass the gate, but only 2 are surfaced.
    assert after_gate == 5
    assert len(cands) == 2


def test_empty_seeds_is_noop():
    assert candidates_from_under_radar_universe([], liquidity_by_symbol={}) == (
        [],
        0,
    )


# ---------------------------------------------------------------------------
# _gather_under_radar_universe — read-only sub-source probing + provenance
# ---------------------------------------------------------------------------


class _FakeFinnhub:
    has_key = True

    def __init__(self, *a, **k) -> None:
        pass

    async def get_earnings_calendar(self, frm, to):
        return [
            {"symbol": "BCRX", "date": "2024-02-15"},
            {"symbol": "ABCD", "date": "2024-02-16"},
            {"no_symbol": True},  # skipped, no usable ticker
        ]

    async def get_ipo_calendar(self, frm, to):
        return [{"symbol": "NEWCO", "name": "New Co Inc", "date": "2024-02-20"}]

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_gather_universe_collects_seeds_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setattr(
        "packages.data.adapters.finnhub.FinnhubAdapter", _FakeFinnhub
    )

    async def _fake_earnings(*a, **k):
        return Provenanced(value=[{"symbol": "ERN1"}], source="rh_earnings", ok=True)

    async def _fake_scans(*a, **k):
        return Provenanced(value=["SCN1", "SCN2"], source="rh_scans", ok=True)

    monkeypatch.setattr("packages.data.rh_live.get_earnings", _fake_earnings)
    monkeypatch.setattr("packages.data.rh_live.get_scan_candidates", _fake_scans)

    seeds, prov = await _gather_under_radar_universe()
    syms = {s.symbol for s in seeds}
    assert {"BCRX", "ABCD", "NEWCO", "ERN1", "SCN1", "SCN2"} <= syms
    assert prov["earnings_calendar"] == 2
    assert prov["ipo"] == 1
    assert prov["rh_earnings"] == 1
    assert prov["rh_scan"] == 2


@pytest.mark.asyncio
async def test_gather_universe_degrades_when_subsource_unreachable(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unreachable sub-source is skipped gracefully; the others still
    return seeds and the lane never fabricates anything."""
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

    class _BoomFinnhub(_FakeFinnhub):
        async def get_earnings_calendar(self, frm, to):
            raise RuntimeError("finnhub calendar down")

    monkeypatch.setattr(
        "packages.data.adapters.finnhub.FinnhubAdapter", _BoomFinnhub
    )

    async def _boom_earnings(*a, **k):
        raise RuntimeError("rh earnings down")

    async def _fake_scans(*a, **k):
        return Provenanced(value=["SCN1"], source="rh_scans", ok=True)

    monkeypatch.setattr("packages.data.rh_live.get_earnings", _boom_earnings)
    monkeypatch.setattr("packages.data.rh_live.get_scan_candidates", _fake_scans)

    # Must not raise; the reachable sub-sources (IPO + scan) still contribute.
    seeds, prov = await _gather_under_radar_universe()
    syms = {s.symbol for s in seeds}
    assert "NEWCO" in syms  # IPO survived
    assert "SCN1" in syms   # scan survived
    # The two failed sub-sources simply did not record a count (no fabrication).
    assert "earnings_calendar" not in prov
    assert "rh_earnings" not in prov
    assert prov["ipo"] == 1
    assert prov["rh_scan"] == 1


@pytest.mark.asyncio
async def test_gather_universe_no_finnhub_key_skips_finnhub(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)

    async def _empty_earnings(*a, **k):
        return Provenanced(value=[], source="none", ok=False)

    async def _empty_scans(*a, **k):
        return Provenanced(value=[], source="none", ok=False)

    monkeypatch.setattr("packages.data.rh_live.get_earnings", _empty_earnings)
    monkeypatch.setattr("packages.data.rh_live.get_scan_candidates", _empty_scans)

    seeds, prov = await _gather_under_radar_universe()
    # No Finnhub key -> no finnhub sub-source provenance, no seeds from it.
    assert "earnings_calendar" not in prov
    assert "ipo" not in prov
    assert seeds == []
