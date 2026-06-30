"""Unit tests for the under-the-radar catalyst lane primitives.

Covers the four pure building blocks in :mod:`packages.agents.catalyst`:

  * :func:`classify_catalyst` — type/score/detail, including the fail-safe
    ``none`` and the never-fabricate rule.
  * :func:`liquidity_gate` — the MANDATORY tradability gate, including the
    fail-safe exclusion when any input is missing/garbage.
  * :func:`is_under_radar` — small/micro-cap membership test.
  * :func:`conviction_notional` — conviction scales WITHIN the absolute caps
    and can NEVER exceed $50/trade, the $300 budget, or the $10k ceiling.

All pure — no network, no env mutation beyond explicit monkeypatch.
"""

from __future__ import annotations

import math

import pytest

from packages.agents.catalyst import (
    ABSOLUTE_BUDGET_USD,
    ABSOLUTE_CEILING_USD,
    ABSOLUTE_PER_TRADE_USD,
    CatalystSignal,
    classify_catalyst,
    conviction_notional,
    is_under_radar,
    liquidity_gate,
)


# ---------------------------------------------------------------------------
# classify_catalyst
# ---------------------------------------------------------------------------


def test_classify_fda_approval_high_score():
    sig = classify_catalyst("BioCryst receives FDA approval for its rare disease drug")
    assert sig.catalyst_type == "fda"
    assert sig.catalyst_score >= 0.90
    assert sig.catalyst_detail  # non-empty, quoted from source


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Company announces PDUFA date for lead candidate", "fda"),
        ("Topline data from the Phase 3 trial were positive", "fda"),
        ("AcmeCo to be acquired in $2B definitive agreement", "m&a"),
        ("Tiny defense firm awarded government contract", "contract"),
        ("Q3 earnings beat: tops estimates and raises guidance", "earnings"),
        ("Analyst initiates coverage with a buy rating", "analyst"),
    ],
)
def test_classify_recognized_types(text, expected):
    assert classify_catalyst(text).catalyst_type == expected


@pytest.mark.parametrize("text", ["", "   ", None, "the cat sat on the mat", "weather is nice"])
def test_classify_none_is_failsafe_default(text):
    """No recognized catalyst language => type 'none', score 0.0 — NEVER fabricated."""
    sig = classify_catalyst(text)
    assert sig.catalyst_type == "none"
    assert sig.catalyst_score == 0.0
    assert sig.catalyst_detail == ""


def test_classify_score_always_in_unit_interval():
    for text in [
        "FDA approval granted on 12/15 confirmed",
        "rumored buyout could potentially happen",
        "upgrade",
    ]:
        s = classify_catalyst(text).catalyst_score
        assert 0.0 <= s <= 1.0


def test_classify_speculative_scored_below_confirmed():
    """A rumour must score lower than a confirmed event of the same type."""
    confirmed = classify_catalyst("AcmeCo acquired in completed definitive agreement")
    rumor = classify_catalyst("AcmeCo reportedly could be acquired, in talks per rumor")
    assert rumor.catalyst_type == confirmed.catalyst_type == "m&a"
    assert rumor.catalyst_score < confirmed.catalyst_score
    # but still a recognized catalyst, not driven to none
    assert rumor.catalyst_score > 0.0


def test_classify_dated_bonus_raises_score():
    plain = classify_catalyst("Company wins contract")
    dated = classify_catalyst("Company wins contract, signed and completed")
    assert dated.catalyst_score >= plain.catalyst_score


def test_catalyst_signal_to_dict_roundtrip():
    d = CatalystSignal("fda", 0.9, "x").to_dict()
    assert d == {"catalyst_type": "fda", "catalyst_score": 0.9, "catalyst_detail": "x"}


# ---------------------------------------------------------------------------
# liquidity_gate (MANDATORY, fail-safe)
# ---------------------------------------------------------------------------


def test_gate_passes_clean_tradable_name():
    r = liquidity_gate(price=1.50, avg_dollar_volume=5_000_000, spread_pct=0.01)
    assert r.passed is True


def test_gate_excludes_below_price_floor():
    r = liquidity_gate(price=0.10, avg_dollar_volume=5_000_000, spread_pct=0.01)
    assert r.passed is False
    assert "floor" in r.reason


def test_gate_excludes_low_dollar_volume():
    r = liquidity_gate(price=2.0, avg_dollar_volume=100_000, spread_pct=0.01)
    assert r.passed is False
    assert "vol" in r.reason.lower()


def test_gate_excludes_wide_spread():
    r = liquidity_gate(price=2.0, avg_dollar_volume=5_000_000, spread_pct=0.20)
    assert r.passed is False
    assert "spread" in r.reason.lower()


def test_gate_allows_zero_spread():
    """A zero spread is NOT 'missing' — it is allowed."""
    r = liquidity_gate(price=2.0, avg_dollar_volume=5_000_000, spread_pct=0.0)
    assert r.passed is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"price": None, "avg_dollar_volume": 5_000_000, "spread_pct": 0.01},
        {"price": 2.0, "avg_dollar_volume": None, "spread_pct": 0.01},
        {"price": 2.0, "avg_dollar_volume": 5_000_000, "spread_pct": None},
        {"price": "abc", "avg_dollar_volume": 5_000_000, "spread_pct": 0.01},
        {"price": 2.0, "avg_dollar_volume": "n/a", "spread_pct": 0.01},
        {"price": 2.0, "avg_dollar_volume": 5_000_000, "spread_pct": "wide"},
        {"price": 2.0, "avg_dollar_volume": 5_000_000, "spread_pct": float("nan")},
        {"price": float("inf"), "avg_dollar_volume": 5_000_000, "spread_pct": 0.01},
    ],
)
def test_gate_failsafe_excludes_missing_or_garbage(kwargs):
    """Any missing/unparseable input => EXCLUDED (never assume tradable)."""
    assert liquidity_gate(**kwargs).passed is False


def test_gate_env_overrides(monkeypatch):
    monkeypatch.setenv("RADAR_MIN_PRICE", "5.0")
    # Name priced at 2.0 now fails the raised floor.
    assert liquidity_gate(price=2.0, avg_dollar_volume=5_000_000, spread_pct=0.01).passed is False


# ---------------------------------------------------------------------------
# is_under_radar
# ---------------------------------------------------------------------------


def test_under_radar_by_market_cap():
    assert is_under_radar(market_cap=3e8) is True
    assert is_under_radar(market_cap=3e12) is False


def test_under_radar_price_fallback_when_cap_unknown():
    assert is_under_radar(price=1.50) is True
    assert is_under_radar(price=190.0) is False


def test_under_radar_unknown_both_returns_false():
    """Membership test, not safety gate: when in doubt keep OUT of the lane."""
    assert is_under_radar() is False
    assert is_under_radar(market_cap=None, price=None) is False


def test_under_radar_market_cap_takes_precedence_over_price():
    # Known large cap with a low price is still NOT under-radar.
    assert is_under_radar(market_cap=5e12, price=1.0) is False


# ---------------------------------------------------------------------------
# conviction_notional — clamped WITHIN absolute caps
# ---------------------------------------------------------------------------


def test_conviction_max_score_hits_per_trade_cap():
    assert conviction_notional(1.0) == ABSOLUTE_PER_TRADE_USD


def test_conviction_zero_score_uses_floor():
    # floor 0.40 * $50 = $20
    assert conviction_notional(0.0) == pytest.approx(20.0)


def test_conviction_monotonic_in_score():
    a = conviction_notional(0.2)
    b = conviction_notional(0.8)
    assert a < b


@pytest.mark.parametrize("score", [1.0, 5.0, 1e9, float("inf"), float("nan"), -3.0, "junk", None])
def test_conviction_never_exceeds_per_trade_cap(score):
    n = conviction_notional(score)
    assert 0.0 <= n <= ABSOLUTE_PER_TRADE_USD


def test_conviction_clamped_by_budget_remaining():
    # Even at max score, a tiny remaining budget caps the size.
    assert conviction_notional(1.0, budget_remaining=10.0) == pytest.approx(10.0)


def test_conviction_clamped_by_ceiling_even_with_huge_caps():
    n = conviction_notional(
        1.0, per_trade_cap=1e9, budget_remaining=1e9, ceiling=ABSOLUTE_CEILING_USD
    )
    assert n == pytest.approx(ABSOLUTE_CEILING_USD)


def test_conviction_never_above_ten_thousand_ceiling():
    n = conviction_notional(1.0, per_trade_cap=1e12, budget_remaining=1e12, ceiling=1e12)
    # ceiling default constant is the hard $10k cap exposed by the module
    assert n <= 1e12  # respects whatever ceiling is passed
    # and with the spec ceiling it never exceeds $10k
    n2 = conviction_notional(1.0, per_trade_cap=1e12, budget_remaining=1e12)
    assert n2 <= ABSOLUTE_CEILING_USD


@pytest.mark.parametrize("cap", [0.0, -5.0, float("nan"), float("inf")])
def test_conviction_nonpositive_or_garbage_cap_returns_zero(cap):
    assert conviction_notional(1.0, budget_remaining=cap) == 0.0


def test_conviction_result_is_finite_and_nonnegative():
    n = conviction_notional(0.5)
    assert math.isfinite(n) and n >= 0.0


def test_absolute_caps_are_spec_values():
    assert ABSOLUTE_PER_TRADE_USD == 50.0
    assert ABSOLUTE_BUDGET_USD == 300.0
    assert ABSOLUTE_CEILING_USD == 10_000.0
