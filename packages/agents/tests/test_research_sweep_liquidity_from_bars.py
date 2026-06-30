"""Tests for deriving liquidity (price / $-volume / spread) from price bars.

The free data tier (Finnhub/RH) rarely exposes a dollar-volume or bid/ask
spread for micro-caps, so the mandatory :func:`liquidity_gate` fail-safe
excluded every under_radar candidate. These tests cover the fix: COMPUTE the
gate's inputs from the daily bars we already fetch for free, then feed the
UNCHANGED gate. Genuinely liquid micro-caps pass; junk is still rejected; a name
with too few / failed bars is fail-safe excluded (never fabricated).

All bars are MOCKED — no network.
"""

from __future__ import annotations

import pytest

from packages.agents import catalyst
from packages.agents.research_sweep import (
    _RadarSeed,
    candidates_from_under_radar_universe,
    liquidity_from_bars,
)


def _bars(close, volume, *, hl=None, n=20):
    """``n`` oldest-first daily bar rows at ``close`` with ``volume`` shares and
    an intraday ``hl`` high-low span (defaults to a tight 0.5% of close)."""
    span = hl if hl is not None else close * 0.005
    return [
        {
            "close_price": str(close),
            "high_price": str(close + span / 2.0),
            "low_price": str(close - span / 2.0),
            "volume": str(volume),
            "begins_at": f"2024-01-{(i % 28) + 1:02d}T00:00:00Z",
        }
        for i in range(n)
    ]


def _seed(sym):
    return _RadarSeed(
        symbol=sym,
        catalyst_text=f"{sym} upcoming quarterly results / earnings report"
        " scheduled for 2024-02-15",
        source="earnings_calendar",
    )


# ---------------------------------------------------------------------------
# liquidity_from_bars — the pure compute helper
# ---------------------------------------------------------------------------


def test_liquid_microcap_profile_computed():
    # $3 close, 500k shares/day -> $1.5M ADV; tight high-low -> small spread.
    prof = liquidity_from_bars(_bars(3.0, 500_000, hl=0.06))
    assert prof is not None
    assert prof["price"] == pytest.approx(3.0)
    assert prof["avg_dollar_volume"] == pytest.approx(1_500_000.0)
    # median (high-low)/close = 0.06/3.0 = 0.02
    assert prof["spread_pct"] == pytest.approx(0.02, abs=1e-6)


def test_fewer_than_min_bars_is_missing():
    # < 5 usable bars -> the whole profile is MISSING (fail-safe), never faked.
    assert liquidity_from_bars(_bars(3.0, 500_000, n=4)) is None


def test_empty_or_none_bars_is_missing():
    assert liquidity_from_bars(None) is None
    assert liquidity_from_bars([]) is None


def test_bars_without_volume_are_unusable():
    rows = [{"close_price": "3.0", "high_price": "3.05", "low_price": "2.95"}] * 20
    # No volume -> no usable bars for ADV -> missing (fail-safe exclude).
    assert liquidity_from_bars(rows) is None


def test_real_quote_preferred_over_proxy():
    # A real bid/ask quote is preferred over the median high-low proxy.
    prof = liquidity_from_bars(
        _bars(3.0, 500_000, hl=0.30),  # proxy would be ~10%
        quote={"bid": 2.99, "ask": 3.01},
    )
    assert prof is not None
    # mid 3.0, spread 0.02/3.0 ~= 0.006667 from the quote, not the wide proxy.
    assert prof["spread_pct"] == pytest.approx(0.006667, abs=1e-5)


def test_adv_window_override_respected(monkeypatch):
    # Last 10 bars are heavy, earlier 10 are light. A window of 10 only sees the
    # heavy tail; the default window of 20 averages everything down.
    light = _bars(3.0, 100_000, n=10)
    heavy = _bars(3.0, 1_000_000, n=10)
    rows = light + heavy

    monkeypatch.setenv("RADAR_ADV_WINDOW_DAYS", "10")
    prof = liquidity_from_bars(rows)
    assert prof is not None
    assert prof["avg_dollar_volume"] == pytest.approx(3_000_000.0)

    monkeypatch.delenv("RADAR_ADV_WINDOW_DAYS", raising=False)
    prof_full = liquidity_from_bars(rows)
    assert prof_full is not None
    # 10*0.3M + 10*3M over 20 = (3M + 30M)/20 = 1.65M
    assert prof_full["avg_dollar_volume"] == pytest.approx(1_650_000.0)


# ---------------------------------------------------------------------------
# candidates_from_under_radar_universe — bar-fed gate + funnel provenance
# ---------------------------------------------------------------------------


def test_liquid_microcap_passes_gate_from_bars():
    seeds = [_seed("BCRX")]
    bars = {"BCRX": _bars(3.0, 500_000, hl=0.06)}
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}, bars_by_symbol=bars, funnel=funnel
    )
    assert [c.symbol for c in cands] == ["BCRX"]
    assert after_gate == 1
    assert funnel["priced_from_bars"] == 1
    assert funnel["after_gate"] == 1
    assert funnel["excluded_price"] == 0
    assert funnel["excluded_volume"] == 0
    assert funnel["excluded_spread"] == 0
    assert funnel["excluded_missing_bars"] == 0


def test_thin_name_excluded_volume():
    # $3 close but only 1k shares/day -> $3k ADV < $1M -> excluded_volume.
    seeds = [_seed("THIN")]
    bars = {"THIN": _bars(3.0, 1_000, hl=0.01)}
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}, bars_by_symbol=bars, funnel=funnel
    )
    assert cands == []
    assert after_gate == 0
    assert funnel["excluded_volume"] == 1
    assert funnel["priced_from_bars"] == 1


def test_wide_highlow_excluded_spread():
    # Plenty of dollar-volume but a 13%+ median high-low span -> excluded_spread.
    seeds = [_seed("WIDE")]
    bars = {"WIDE": _bars(3.0, 500_000, hl=0.40)}
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}, bars_by_symbol=bars, funnel=funnel
    )
    assert cands == []
    assert after_gate == 0
    assert funnel["excluded_spread"] == 1


def test_sub_floor_price_excluded_price():
    # $0.30 close is below the $0.50 floor -> excluded_price (ample volume).
    seeds = [_seed("LOW")]
    bars = {"LOW": _bars(0.30, 5_000_000, hl=0.002)}
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}, bars_by_symbol=bars, funnel=funnel
    )
    assert cands == []
    assert after_gate == 0
    assert funnel["excluded_price"] == 1


def test_too_few_bars_excluded_missing_bars():
    # < 5 usable bars -> fail-safe excluded, attributed to missing bars (NOT
    # fabricated, NOT counted as priced_from_bars).
    seeds = [_seed("FEW")]
    bars = {"FEW": _bars(3.0, 500_000, n=3)}
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}, bars_by_symbol=bars, funnel=funnel
    )
    assert cands == []
    assert after_gate == 0
    assert funnel["excluded_missing_bars"] == 1
    assert funnel["priced_from_bars"] == 0


def test_failed_fetch_absent_bars_excluded_missing_bars():
    # No bars entry at all (failed/empty fetch) -> excluded_missing_bars.
    seeds = [_seed("GONE")]
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}, bars_by_symbol={}, funnel=funnel
    )
    assert cands == []
    assert funnel["excluded_missing_bars"] == 1


def test_full_funnel_mixed_universe():
    seeds = [_seed(s) for s in ("LIQ", "THIN", "WIDE", "LOW", "FEW")]
    bars = {
        "LIQ": _bars(3.0, 500_000, hl=0.06),
        "THIN": _bars(3.0, 1_000, hl=0.01),
        "WIDE": _bars(3.0, 500_000, hl=0.40),
        "LOW": _bars(0.30, 5_000_000, hl=0.002),
        "FEW": _bars(3.0, 500_000, n=3),
    }
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol={}, bars_by_symbol=bars, funnel=funnel
    )
    assert [c.symbol for c in cands] == ["LIQ"]
    assert after_gate == 1
    assert funnel == {
        "priced_from_bars": 4,
        "excluded_price": 1,
        "excluded_volume": 1,
        "excluded_spread": 1,
        "excluded_missing_bars": 1,
        "after_gate": 1,
    }


def test_direct_fields_skip_bar_derivation():
    # When liquidity_by_symbol already has all three fields, bars are not used
    # and priced_from_bars stays 0 (existing behaviour unchanged).
    seeds = [_seed("BCRX")]
    liq = {
        "BCRX": {
            "price": 1.50,
            "avg_dollar_volume": 5_000_000,
            "spread_pct": 0.01,
            "market_cap": 3e8,
        }
    }
    funnel: dict = {}
    cands, after_gate = candidates_from_under_radar_universe(
        seeds, liquidity_by_symbol=liq, bars_by_symbol={"BCRX": _bars(9.9, 1)},
        funnel=funnel,
    )
    assert [c.symbol for c in cands] == ["BCRX"]
    assert after_gate == 1
    assert funnel["priced_from_bars"] == 0


# ---------------------------------------------------------------------------
# The gate thresholds + fail-safe stay UNCHANGED — we only feed it better data.
# ---------------------------------------------------------------------------


def test_gate_thresholds_unchanged(monkeypatch):
    monkeypatch.delenv("RADAR_MIN_PRICE", raising=False)
    monkeypatch.delenv("RADAR_MIN_DOLLAR_VOL", raising=False)
    monkeypatch.delenv("RADAR_MAX_SPREAD_PCT", raising=False)
    assert catalyst.min_price() == 0.50
    assert catalyst.min_dollar_vol() == 1_000_000.0
    assert catalyst.max_spread_pct() == 0.035


def test_gate_failsafe_still_excludes_on_missing():
    # The gate itself is untouched: any missing input still fail-safe excludes.
    assert not catalyst.liquidity_gate(
        price=None, avg_dollar_volume=2_000_000, spread_pct=0.01
    ).passed
    assert not catalyst.liquidity_gate(
        price=3.0, avg_dollar_volume=None, spread_pct=0.01
    ).passed
    assert not catalyst.liquidity_gate(
        price=3.0, avg_dollar_volume=2_000_000, spread_pct=None
    ).passed
