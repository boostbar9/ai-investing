"""Unit tests for the confidence-gated policy (Phase 13).

We deliberately pin thresholds and weights via constructor / direct
assertions rather than env overrides so the tests stay independent of
whatever a developer has exported locally. The composite math is the
real risk surface; thresholds + sizing are mechanically simpler so they
get lighter coverage.
"""
from __future__ import annotations

import math

import pytest

from packages.agents.policy import (
    CONFIDENCE_WEIGHTS,
    ConfidenceGatedPolicy,
    PolicyDecision,
    composite_confidence,
)

# ---------------------------------------------------------------------------
# composite_confidence: each component, regime inversion, missing data
# ---------------------------------------------------------------------------


def test_composite_zero_when_all_inputs_zero() -> None:
    """Nothing in -> nothing out. Sanity floor."""
    score, parts = composite_confidence(
        candidate=None,
        regime="bull",
        regime_confidence=0.0,
        ensemble_weight=0.0,
    )
    assert score == 0.0
    assert parts == {
        "candidate": 0.0,
        "regime": 0.0,
        "trust": 0.0,
        "ensemble_alignment": 0.0,
    }


def test_composite_perfect_bull_signal_maxes_out() -> None:
    """All signals firing in bull regime -> ~1.0 (sum of weights)."""
    candidate = {
        "symbol": "NVDA",
        "confidence": 1.0,
        "reddit_trust": 1.0,
        "corroborated": True,
    }
    score, parts = composite_confidence(
        candidate=candidate,
        regime="bull",
        regime_confidence=1.0,
        ensemble_weight=0.05,
    )
    assert score == pytest.approx(1.0)
    assert parts["candidate"] == 1.0
    assert parts["regime"] == 1.0
    assert parts["trust"] == 1.0  # 1.0 + 0.3 corroboration, clipped at 1.0
    assert parts["ensemble_alignment"] == 1.0


def test_composite_chop_regime_contributes_neutral_half() -> None:
    """Chop is regime-neutral: contributes 0.5 regardless of posterior."""
    _, parts = composite_confidence(
        candidate={"confidence": 0.0},
        regime="chop",
        regime_confidence=0.99,  # high posterior, should be ignored
        ensemble_weight=0.0,
    )
    assert parts["regime"] == 0.5

    # And again with a low posterior.
    _, parts2 = composite_confidence(
        candidate={"confidence": 0.0},
        regime="chop",
        regime_confidence=0.0,
        ensemble_weight=0.0,
    )
    assert parts2["regime"] == 0.5


def test_composite_bear_regime_inverts_posterior() -> None:
    """In bear/crisis, a confident bearish regime SUBTRACTS from buy score.

    Mapped 1 - posterior, so a strong bear (0.9) -> 0.1 contribution.
    This is the whole reason regime is part of the blend.
    """
    _, bear_parts = composite_confidence(
        candidate={"confidence": 1.0},  # candidate is rock-solid bullish
        regime="bear",
        regime_confidence=0.9,
        ensemble_weight=0.0,
    )
    assert bear_parts["regime"] == pytest.approx(0.1)

    _, crisis_parts = composite_confidence(
        candidate={"confidence": 1.0},
        regime="crisis",
        regime_confidence=0.95,
        ensemble_weight=0.0,
    )
    assert crisis_parts["regime"] == pytest.approx(0.05)


def test_composite_corroboration_bonus_lifts_modest_trust() -> None:
    """Corroborated story should pull trust component above raw reddit_trust."""
    _, parts = composite_confidence(
        candidate={
            "confidence": 0.0,
            "reddit_trust": 0.5,
            "corroborated": True,
        },
        regime="bull",
        regime_confidence=0.0,
        ensemble_weight=0.0,
    )
    # 0.5 + 0.3 = 0.8, clipped within [0,1]
    assert parts["trust"] == pytest.approx(0.8)


def test_composite_ensemble_alignment_is_binary() -> None:
    """Any non-trivial ensemble weight -> full 1.0 alignment vote."""
    _, parts_aligned = composite_confidence(
        candidate={"confidence": 0.0},
        regime="bull",
        regime_confidence=0.0,
        ensemble_weight=0.001,  # exactly 1bps boundary
    )
    assert parts_aligned["ensemble_alignment"] == 1.0

    _, parts_silent = composite_confidence(
        candidate={"confidence": 0.0},
        regime="bull",
        regime_confidence=0.0,
        ensemble_weight=0.00001,  # below 1bps
    )
    assert parts_silent["ensemble_alignment"] == 0.0


def test_composite_handles_missing_and_garbage_fields() -> None:
    """Bad JSON shouldn't crash the policy."""
    score, _ = composite_confidence(
        candidate={"confidence": "not a number", "reddit_trust": None},
        regime="bull",
        regime_confidence=float("nan"),  # NaN tolerated
        ensemble_weight=None,  # type: ignore[arg-type]
    )
    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)


def test_composite_weights_sum_to_one() -> None:
    """The blend weights must sum to 1.0 so composite stays in [0, 1]."""
    assert sum(CONFIDENCE_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ConfidenceGatedPolicy.decide: thresholds, regime gating, held-only sells
# ---------------------------------------------------------------------------


def _policy(**kw: float) -> ConfidenceGatedPolicy:
    """Build a policy with predictable thresholds for tests."""
    defaults: dict[str, float | int] = {
        "buy_threshold": 0.65,
        "sell_threshold": 0.35,
        "max_positions": 10,
        "cash_floor": 0.05,
    }
    defaults.update(kw)
    return ConfidenceGatedPolicy(**defaults)  # type: ignore[arg-type]


def test_decide_buy_when_above_threshold() -> None:
    p = _policy()
    decisions = p.decide(
        sweep_candidates=[
            {
                "symbol": "NVDA",
                "confidence": 1.0,
                "reddit_trust": 1.0,
                "corroborated": True,
            }
        ],
        ensemble_weights={"NVDA": 0.10},
        current_holdings=set(),
        regime="bull",
        regime_confidence=0.95,
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d.symbol == "NVDA"
    assert d.action == "buy"
    assert d.composite_confidence >= 0.65


def test_decide_hold_when_in_middle_zone() -> None:
    """Composite between sell_threshold and buy_threshold -> HOLD."""
    p = _policy()
    decisions = p.decide(
        sweep_candidates=[
            {"symbol": "AAPL", "confidence": 0.5, "reddit_trust": 0.5}
        ],
        ensemble_weights={},
        current_holdings=set(),
        regime="chop",
        regime_confidence=0.5,
    )
    assert len(decisions) == 1
    assert decisions[0].action == "hold"


def test_decide_sell_only_emitted_for_held_symbols() -> None:
    """A low-confidence non-held symbol must NOT produce a SELL."""
    p = _policy()
    # Held + low signal -> SELL
    decisions = p.decide(
        sweep_candidates=[],
        ensemble_weights={},
        current_holdings={"XYZ"},
        regime="bear",
        regime_confidence=0.8,
    )
    actions = {d.symbol: d.action for d in decisions}
    assert actions == {"XYZ": "sell"}

    # Not held + low signal -> HOLD, never SELL (no-op SELL would pollute log).
    decisions2 = p.decide(
        sweep_candidates=[
            {"symbol": "WEAK", "confidence": 0.05}
        ],
        ensemble_weights={},
        current_holdings=set(),
        regime="bear",
        regime_confidence=0.8,
    )
    assert all(d.action != "sell" for d in decisions2)


def test_decide_crisis_regime_makes_buys_much_harder() -> None:
    """Same strong candidate signal should BUY in bull, HOLD in crisis."""
    candidate = [{
        "symbol": "QQQ",
        "confidence": 0.85,
        "reddit_trust": 0.6,
        "corroborated": True,
    }]
    p = _policy()
    bull = p.decide(
        sweep_candidates=candidate,
        ensemble_weights={"QQQ": 0.05},
        current_holdings=set(),
        regime="bull",
        regime_confidence=0.9,
    )
    crisis = p.decide(
        sweep_candidates=candidate,
        ensemble_weights={"QQQ": 0.05},
        current_holdings=set(),
        regime="crisis",
        regime_confidence=0.9,
    )
    assert bull[0].action == "buy"
    # In crisis the threshold multiplier is 1.80, so effective_buy ~= 1.0
    # and the bear-inverted regime component drops too. Should NOT BUY.
    assert crisis[0].action != "buy"


def test_decide_universe_is_union_of_sweep_ensemble_holdings() -> None:
    """We should evaluate every symbol we have an opinion on."""
    p = _policy()
    decisions = p.decide(
        sweep_candidates=[{"symbol": "AAA", "confidence": 0.1}],
        ensemble_weights={"BBB": 0.10},
        current_holdings={"CCC"},
        regime="chop",
        regime_confidence=0.5,
    )
    symbols = {d.symbol for d in decisions}
    assert symbols == {"AAA", "BBB", "CCC"}


def test_decide_rejects_invalid_thresholds() -> None:
    """sell >= buy is a contradiction; must raise at construction time."""
    with pytest.raises(ValueError):
        ConfidenceGatedPolicy(buy_threshold=0.5, sell_threshold=0.5)
    with pytest.raises(ValueError):
        ConfidenceGatedPolicy(buy_threshold=0.3, sell_threshold=0.7)


# ---------------------------------------------------------------------------
# to_target_weights: equal-weighting, cap, cash_floor, SELL emission
# ---------------------------------------------------------------------------


def _buy(sym: str, conf: float = 0.9) -> PolicyDecision:
    return PolicyDecision(symbol=sym, action="buy", composite_confidence=conf)


def _sell(sym: str) -> PolicyDecision:
    return PolicyDecision(symbol=sym, action="sell", composite_confidence=0.1)


def _hold(sym: str) -> PolicyDecision:
    return PolicyDecision(symbol=sym, action="hold", composite_confidence=0.5)


def test_to_target_weights_equal_weights_buys_minus_cash_floor() -> None:
    p = _policy(cash_floor=0.05)
    weights = p.to_target_weights([_buy("A"), _buy("B"), _buy("C")])
    # 0.95 spread evenly across 3 buys
    expected = (1.0 - 0.05) / 3
    assert weights["A"] == pytest.approx(expected, abs=1e-5)
    assert weights["B"] == pytest.approx(expected, abs=1e-5)
    assert weights["C"] == pytest.approx(expected, abs=1e-5)
    # Sum should be (1 - cash_floor), not 1.0 -- cash is reserve.
    assert sum(weights.values()) == pytest.approx(1.0 - 0.05, abs=1e-5)


def test_to_target_weights_caps_to_max_positions_keeping_best() -> None:
    """When more BUYs than the cap allows, keep highest composites."""
    p = _policy(max_positions=2)
    decisions = [
        _buy("LOW", conf=0.66),
        _buy("MID", conf=0.80),
        _buy("HIGH", conf=0.99),
    ]
    weights = p.to_target_weights(decisions)
    # HIGH + MID kept, LOW dropped.
    assert "HIGH" in weights and "MID" in weights
    assert "LOW" not in weights


def test_to_target_weights_emits_zero_for_sells() -> None:
    """SELL must surface as explicit 0.0 so rebalancer flattens position."""
    p = _policy()
    weights = p.to_target_weights([_buy("A"), _sell("B")])
    assert weights["B"] == 0.0
    assert weights["A"] > 0.0


def test_to_target_weights_holds_emit_nothing() -> None:
    """HOLD means 'keep current'; rebalancer treats absence as no-op."""
    p = _policy()
    weights = p.to_target_weights([_hold("X"), _hold("Y")])
    assert weights == {}


def test_to_target_weights_no_buys_just_sells() -> None:
    """All SELL, no BUY -> only flatten signals, no new positions."""
    p = _policy()
    weights = p.to_target_weights([_sell("A"), _sell("B")])
    assert weights == {"A": 0.0, "B": 0.0}


# ---------------------------------------------------------------------------
# PolicyDecision.to_dict: serialization for decision log
# ---------------------------------------------------------------------------


def test_policy_decision_to_dict_shape() -> None:
    d = PolicyDecision(
        symbol="NVDA",
        action="buy",
        composite_confidence=0.8123456,
        components={"candidate": 0.9, "regime": 0.7, "trust": 0.6, "ensemble_alignment": 1.0},
        reason="strong signal",
    )
    out = d.to_dict()
    assert out["symbol"] == "NVDA"
    assert out["action"] == "buy"
    assert out["confidence"] == 0.8123  # rounded to 4dp
    assert set(out["components"]) == {"candidate", "regime", "trust", "ensemble_alignment"}
    assert out["reason"] == "strong signal"


# ---------------------------------------------------------------------------
# Phase 14: optional calibrator wiring
# ---------------------------------------------------------------------------


def test_decide_uses_calibrator_when_provided() -> None:
    """A calibrator that halves every input should reduce composite_confidence
    in the resulting PolicyDecision -- and raw_confidence should hold the
    pre-calibration value so we can see what changed."""
    p = _policy(buy_threshold=0.65)
    # Halving calibrator: 0.9 raw -> 0.45 calibrated.
    p.calibrator = lambda x: x * 0.5  # type: ignore[method-assign]
    decisions = p.decide(
        sweep_candidates=[
            {"symbol": "NVDA", "confidence": 1.0, "reddit_trust": 1.0, "corroborated": True}
        ],
        ensemble_weights={"NVDA": 0.10},
        current_holdings=set(),
        regime="bull",
        regime_confidence=0.95,
    )
    d = next(x for x in decisions if x.symbol == "NVDA")
    # Raw composite would be ~1.0; calibrated should be ~0.5 -- below the 0.65
    # buy threshold even though the raw signal screams BUY. That's the whole
    # point: thresholds now speak in true-probability space.
    assert d.raw_confidence is not None
    assert d.raw_confidence >= 0.9
    assert d.composite_confidence == pytest.approx(d.raw_confidence * 0.5, abs=1e-9)
    assert d.action == "hold"  # calibration killed the BUY trigger


def test_decide_no_calibrator_leaves_raw_confidence_none() -> None:
    """Without a calibrator (Phase 13 default), raw_confidence stays None.
    This keeps the decision log lean and lets us distinguish unfitted
    runs from runs where calibration was active."""
    p = _policy()
    decisions = p.decide(
        sweep_candidates=[{"symbol": "NVDA", "confidence": 0.9}],
        ensemble_weights={"NVDA": 0.10},
        current_holdings=set(),
        regime="bull",
        regime_confidence=0.9,
    )
    for d in decisions:
        assert d.raw_confidence is None


def test_decide_calibrator_failure_falls_back_to_raw() -> None:
    """A buggy calibrator must not crash a paper-trade cycle; the policy
    should log + fall back to raw composite for the affected symbol."""
    p = _policy(buy_threshold=0.65)

    def boom(x: float) -> float:
        raise RuntimeError("intentionally broken calibrator")

    p.calibrator = boom  # type: ignore[method-assign]
    decisions = p.decide(
        sweep_candidates=[
            {"symbol": "NVDA", "confidence": 1.0, "reddit_trust": 1.0, "corroborated": True}
        ],
        ensemble_weights={"NVDA": 0.10},
        current_holdings=set(),
        regime="bull",
        regime_confidence=0.95,
    )
    d = next(x for x in decisions if x.symbol == "NVDA")
    # Composite falls back to raw, which clears the threshold -> BUY.
    assert d.action == "buy"
    assert d.composite_confidence >= 0.65


def test_decide_calibrator_output_clipped_to_unit_interval() -> None:
    """A pathological calibrator returning >1 should be clipped at 1.0,
    not pollute downstream code that assumes composite is in [0, 1]."""
    p = _policy()
    p.calibrator = lambda x: x + 5.0  # type: ignore[method-assign]
    decisions = p.decide(
        sweep_candidates=[{"symbol": "NVDA", "confidence": 0.5}],
        ensemble_weights={"NVDA": 0.10},
        current_holdings=set(),
        regime="bull",
        regime_confidence=0.5,
    )
    for d in decisions:
        assert 0.0 <= d.composite_confidence <= 1.0


def test_policy_decision_to_dict_includes_raw_confidence_when_set() -> None:
    """raw_confidence should appear in to_dict ONLY when populated (additive
    schema -- Phase 13 readers don't see a new field)."""
    d_with = PolicyDecision(
        symbol="NVDA", action="buy", composite_confidence=0.7,
        components={}, reason="ok", raw_confidence=0.9,
    )
    out_with = d_with.to_dict()
    assert out_with["raw_confidence"] == 0.9

    d_without = PolicyDecision(
        symbol="NVDA", action="buy", composite_confidence=0.7,
        components={}, reason="ok",
    )
    out_without = d_without.to_dict()
    assert "raw_confidence" not in out_without
