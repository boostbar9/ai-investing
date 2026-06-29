"""Mock-only tests for the Robinhood fundamentals + earnings research signal.

NO live Robinhood / network access: the pure enrichment helpers are driven
with fixture dicts, and the one facade test injects a ``FakeBroker``. We assert
the spec's guarantees:

  * compliant + high rel-volume => fields persisted, provenance set, and the
    scorer awards a modest positive.
  * ``financial_status_description="Noncompliant"`` => compliance_ok False, a
    "delisting/compliance risk" flag, a strong negative in the scorer that can
    push a pick below min_confidence, and NEVER a boost.
  * earnings today/tomorrow => negative + "earnings imminent" flag.
  * fundamentals feed unavailable / erroring => candidates untouched, neutral
    (0.0) scorer contribution, nothing fabricated.
  * the [-0.20, +0.08] clamp band holds for extreme inputs.
  * with the phase OFF (no ``fundamentals_source``) the scorer output is
    byte-for-byte identical to before this feature.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.agents.research_sweep import (
    Candidate,
    _apply_earnings_proximity,
    _apply_fundamentals_enrichment,
    _compliance_ok,
    _gather_rh_fundamentals,
    _next_earnings,
    _parse_fundamentals_row,
)
from packages.cockpit.web import autonomy
from packages.data import rh_live


def _cand(symbol: str = "AAA", **kw) -> Candidate:
    base = {
        "symbol": symbol,
        "signal_kind": "sentiment",
        "thesis": "stub",
        "confidence": 0.5,
    }
    base.update(kw)
    return Candidate(**base)


# ---------------------------------------------------------------------------
# Pure parsing: fundamentals row -> fields
# ---------------------------------------------------------------------------
def test_parse_fundamentals_row_derives_relvol_and_distance() -> None:
    row = {
        "market_cap": "2.5e12",
        "pe_ratio": "28.5",
        "float": "1.6e10",
        "volume": "200",
        "average_volume_30_days": "100",
        "high_52_weeks": "200.0",
        "low_52_weeks": "100.0",
        "last_trade_price": "150.0",
        "sector": "Technology",
        "financial_status_indicator": "N",
        "financial_status_description": "Normal",
    }
    parsed = _parse_fundamentals_row(row)
    assert parsed["market_cap"] == pytest.approx(2.5e12)
    assert parsed["pe_ratio"] == pytest.approx(28.5)
    assert parsed["rel_volume"] == pytest.approx(2.0)
    # 150 vs 200 high => -25% from high; 150 vs 100 low => +50% above low.
    assert parsed["pct_from_52w_high"] == pytest.approx(-0.25)
    assert parsed["pct_above_52w_low"] == pytest.approx(0.5)
    assert parsed["sector"] == "Technology"
    assert parsed["compliance_ok"] is True


def test_parse_fundamentals_row_skips_missing_without_fabricating() -> None:
    # No volume/avg => no rel_volume key; no prices => no 52w distances.
    parsed = _parse_fundamentals_row({"pe_ratio": "10"})
    assert "rel_volume" not in parsed
    assert "pct_from_52w_high" not in parsed
    assert parsed["pe_ratio"] == pytest.approx(10.0)
    # Absent status => innocent (compliant), no fabricated bad flag.
    assert parsed["compliance_ok"] is True


def test_compliance_ok_detects_noncompliant() -> None:
    bad, status = _compliance_ok(
        {"financial_status_indicator": "CC4",
         "financial_status_description": "Noncompliant"}
    )
    assert bad is False and status == "Noncompliant"
    # Non-normal indicator with no description still trips.
    bad2, _ = _compliance_ok({"financial_status_indicator": "D"})
    assert bad2 is False
    # Normal / empty are fine.
    assert _compliance_ok({"financial_status_indicator": "N"})[0] is True
    assert _compliance_ok({})[0] is True


# ---------------------------------------------------------------------------
# Pure enrichment: stamping onto candidates
# ---------------------------------------------------------------------------
def test_apply_fundamentals_compliant_high_volume() -> None:
    c = _cand("AAA")
    funds = {
        "AAA": {
            "market_cap": 3e9,
            "pe_ratio": 18.0,
            "volume": 300,
            "average_volume_30_days": 100,
            "high_52_weeks": 200,
            "last_trade_price": 150,
            "sector": "Energy",
            "financial_status_description": "Normal",
        }
    }
    (out,) = _apply_fundamentals_enrichment([c], funds)
    assert out.fundamentals_source == "rh_fundamentals"
    assert out.rel_volume == pytest.approx(3.0)
    assert out.compliance_ok is True
    assert out.risk_flag == ""  # nothing alarming


def test_apply_fundamentals_noncompliant_sets_risk_flag() -> None:
    c = _cand("BAD")
    funds = {"BAD": {"financial_status_description": "Noncompliant"}}
    (out,) = _apply_fundamentals_enrichment([c], funds)
    assert out.compliance_ok is False
    assert out.risk_flag == "delisting/compliance risk"


def test_apply_fundamentals_missing_symbol_untouched() -> None:
    c = _cand("AAA")
    out = _apply_fundamentals_enrichment([c], {"OTHER": {"pe_ratio": 5}})
    assert out[0].fundamentals_source == ""  # neutral, phase didn't touch it
    assert out[0].compliance_ok is True


def test_apply_fundamentals_empty_feed_is_noop() -> None:
    c = _cand("AAA")
    assert _apply_fundamentals_enrichment([c], {}) == [c]
    assert c.fundamentals_source == ""


# ---------------------------------------------------------------------------
# Earnings proximity
# ---------------------------------------------------------------------------
def test_next_earnings_picks_soonest_future() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    rows = [
        {"report_date": "2025-03-01"},  # past, ignored
        {"report_date": "2026-08-15"},
        {"report_date": "2026-06-30"},
    ]
    date_iso, days = _next_earnings(rows, now=now)
    assert date_iso == "2026-06-30"
    assert days == 29


def test_next_earnings_no_future_is_unknown() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    date_iso, days = _next_earnings([{"date": "2020-01-01"}], now=now)
    assert date_iso == "" and days is None


def test_apply_earnings_proximity_imminent_flag() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    c = _cand("AAA")
    out = _apply_earnings_proximity(
        [c], {"AAA": [{"report_date": "2026-06-02"}]}, now=now
    )
    assert out[0].next_earnings_date == "2026-06-02"
    assert out[0].days_to_earnings == 1
    assert out[0].risk_flag == "earnings imminent"


def test_apply_earnings_proximity_does_not_clobber_delisting() -> None:
    now = datetime(2026, 6, 1, tzinfo=UTC)
    c = _cand("AAA", risk_flag="delisting/compliance risk")
    out = _apply_earnings_proximity(
        [c], {"AAA": [{"report_date": "2026-06-02"}]}, now=now
    )
    # Higher-priority safety flag wins.
    assert out[0].risk_flag == "delisting/compliance risk"


# ---------------------------------------------------------------------------
# Scorer integration (guardrail-first, bounded, fail-safe-neutral)
# ---------------------------------------------------------------------------
def _score(**kw) -> float:
    return autonomy._score_candidate(_cand(**kw).to_dict())[0]


def test_scorer_neutral_when_phase_off_byte_for_byte() -> None:
    # A candidate that never went through fundamentals scores exactly like a
    # bare candidate -- the new block must be a no-op.
    bare = _cand(symbol="AAA", confidence=0.5).to_dict()
    s_bare, r_bare, f_bare = autonomy._score_candidate(bare)
    # Same dict but minus the new keys entirely (legacy shape).
    legacy = {k: v for k, v in bare.items()
              if k not in {
                  "fundamentals_source", "market_cap", "pe_ratio",
                  "float_shares", "rel_volume", "pct_from_52w_high",
                  "pct_above_52w_low", "sector", "compliance_status",
                  "compliance_ok", "next_earnings_date", "days_to_earnings",
              }}
    s_legacy, r_legacy, f_legacy = autonomy._score_candidate(legacy)
    assert (s_bare, r_bare, f_bare) == (s_legacy, r_legacy, f_legacy)
    assert "fundamentals" not in f_bare


def test_scorer_high_volume_modest_positive() -> None:
    base = _score(symbol="AAA", confidence=0.5)
    hi = _score(
        symbol="AAA", confidence=0.5, fundamentals_source="rh_fundamentals",
        rel_volume=2.1, market_cap=3e9, pct_from_52w_high=-0.2,
    )
    assert hi > base
    # bounded: <= base + clamp ceiling.
    assert hi <= base + 0.08 + 1e-9


def test_scorer_noncompliant_strong_negative_can_veto() -> None:
    base = _score(symbol="AAA", confidence=0.20)
    bad = _score(
        symbol="AAA", confidence=0.20, fundamentals_source="rh_fundamentals",
        compliance_ok=False, compliance_status="Noncompliant",
    )
    assert bad < base
    # -0.15 penalty pushes a 0.20 base below a typical 0.15 min_confidence.
    assert bad <= 0.06
    # Even with otherwise-bullish volume it never flips to a net boost.
    bad_with_vol = _score(
        symbol="AAA", confidence=0.20, fundamentals_source="rh_fundamentals",
        compliance_ok=False, rel_volume=3.0, pct_from_52w_high=-0.3,
    )
    assert bad_with_vol < base


def test_scorer_earnings_imminent_negative() -> None:
    base = _score(symbol="AAA", confidence=0.5)
    soon = _score(
        symbol="AAA", confidence=0.5, fundamentals_source="rh_fundamentals",
        days_to_earnings=1,
    )
    assert soon < base


def test_scorer_clamp_band_holds_for_extremes() -> None:
    base = _score(symbol="AAA", confidence=0.5)
    # Everything bad at once: clamp floor is -0.20.
    worst = _score(
        symbol="AAA", confidence=0.5, fundamentals_source="rh_fundamentals",
        compliance_ok=False, days_to_earnings=0,
        rel_volume=0.1, market_cap=1e6,
    )
    assert worst == pytest.approx(base - 0.20, abs=1e-9)
    # Everything good: clamp ceiling is +0.08.
    best = _score(
        symbol="AAA", confidence=0.5, fundamentals_source="rh_fundamentals",
        rel_volume=9.0, pct_from_52w_high=-0.5,
    )
    assert best == pytest.approx(base + 0.06, abs=1e-9)  # 0.04+0.02, under cap


# ---------------------------------------------------------------------------
# Facade gather: injected broker, no network. Feed-unavailable => empty/neutral.
# ---------------------------------------------------------------------------
class _FakeBroker:
    def __init__(self, fundamentals=None, raise_on=None):
        self._f = fundamentals or {}
        self._raise = raise_on or set()

    async def equity_fundamentals(self, symbol):
        if "fundamentals" in self._raise:
            raise RuntimeError("boom")
        return self._f.get(symbol.upper())


@pytest.fixture()
def _rh_isolated(monkeypatch):
    from packages.data.cache import TTLCache
    from packages.data.health import SourceRegistry

    cache = TTLCache(default_ttl_s=300.0, use_disk=False)
    monkeypatch.setattr(rh_live, "get_cache", lambda: cache)
    monkeypatch.setattr(rh_live, "get_registry", lambda: SourceRegistry())
    monkeypatch.setattr(rh_live, "is_enabled", lambda name: True)
    rh_live.reset_for_test()
    yield
    rh_live.reset_for_test()


async def test_gather_fundamentals_via_injected_broker(_rh_isolated) -> None:
    rh_live.set_broker_for_test(
        _FakeBroker(fundamentals={"AAA": {"symbol": "AAA", "pe_ratio": 12.0}})
    )
    out = await _gather_rh_fundamentals(["AAA"])
    assert out["AAA"]["pe_ratio"] == 12.0


async def test_gather_fundamentals_feed_error_is_empty(_rh_isolated) -> None:
    rh_live.set_broker_for_test(_FakeBroker(raise_on={"fundamentals"}))
    # Erroring feed => empty map, no crash, nothing fabricated.
    assert await _gather_rh_fundamentals(["AAA"]) == {}


async def test_gather_fundamentals_no_broker_is_empty(_rh_isolated) -> None:
    rh_live.set_broker_for_test(None)
    assert await _gather_rh_fundamentals(["AAA"]) == {}
