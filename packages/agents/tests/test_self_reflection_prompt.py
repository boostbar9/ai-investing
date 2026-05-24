"""Pin the prompt self-reflection injection contract.

The cockpit feeds ``summarize_scorecard().to_jsonable()`` straight into the
prompt builders. If the injection ever drops out — or worse, hallucinates
content on cold start — the agents would silently drift. These tests pin:

  * empty / None / fresh scorecard → injection is OMITTED entirely.
  * a real scorecard → the block is present and contains the rendered
    hit-rate and PnL.
  * each prompt builder (research / strategy / risk / execution / discovery)
    accepts the kwarg and emits the same block.
"""

from __future__ import annotations

from uuid import uuid4

from packages.agents.prompts import (
    discovery_prompt,
    execution_prompt,
    research_prompt,
    risk_prompt,
    self_reflection_block,
    strategy_prompt,
)
from packages.shared.schemas import (
    DiscoveryInput,
    ExecutionInput,
    Order,
    Position,
    ResearchInput,
    RiskInput,
    Signal,
    StrategyInput,
)


def _good_summary() -> dict:
    return {
        "n_runs": 8,
        "n_signals": 16,
        "hit_rate_5d": 0.6,
        "avg_pnl_bps_5d": 42.3,
        "avg_pnl_bps_1d": 12.0,
        "regime_bias": {"bull": 5, "chop": 3},
        "last_run_ts": "2026-05-23T20:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# self_reflection_block
# ---------------------------------------------------------------------------


def test_reflection_block_empty_on_none() -> None:
    """No scorecard at all → no injection. Prevents fabricating stats."""
    assert self_reflection_block(None) == ""


def test_reflection_block_empty_on_zero_runs() -> None:
    """A summary that exists but contains no scored runs must NOT inject."""
    summary = {"n_runs": 0, "n_signals": 0, "hit_rate_5d": None,
               "avg_pnl_bps_5d": None, "avg_pnl_bps_1d": None,
               "regime_bias": {}, "last_run_ts": None}
    assert self_reflection_block(summary) == ""


def test_reflection_block_renders_stats() -> None:
    """A real summary must render hit-rate as a percent and PnL in bps."""
    text = self_reflection_block(_good_summary())
    assert "RECENT SELF-ASSESSMENT" in text
    assert "60%" in text  # 0.6 → 60%
    assert "+42.3 bps" in text
    assert "+12.0 bps" in text
    assert "bull:5" in text
    assert "chop:3" in text


def test_reflection_block_handles_missing_pnl_gracefully() -> None:
    """A scorecard where some horizons haven't matured must still render
    without crashing."""
    summary = _good_summary()
    summary["avg_pnl_bps_1d"] = None
    text = self_reflection_block(summary)
    assert "n/a" in text


def test_reflection_block_handles_no_regime_bias() -> None:
    """Empty regime_bias dict must produce 'none' rather than crash."""
    summary = _good_summary()
    summary["regime_bias"] = {}
    text = self_reflection_block(summary)
    assert "none" in text


# ---------------------------------------------------------------------------
# Each prompt builder accepts the kwarg and injects when given a summary.
# ---------------------------------------------------------------------------


def _research_input() -> ResearchInput:
    return ResearchInput(
        decision_id=uuid4(),
        symbols=["SPY", "QQQ"],
        lookback_days=5,
    )


def _strategy_input() -> StrategyInput:
    return StrategyInput(
        decision_id=uuid4(),
        regime="bull",
        universe=["SPY", "QQQ"],
        features={"mom_12_1": 0.18},
    )


def _risk_input() -> RiskInput:
    return RiskInput(
        decision_id=uuid4(),
        candidates=[Signal(symbol="SPY", side="buy", strength=0.5, rationale="mom_12_1=0.18")],
        positions=[Position(symbol="SPY", qty=0.0, avg_price=0.0)],
    )


def _execution_input() -> ExecutionInput:
    return ExecutionInput(
        decision_id=uuid4(),
        approved_orders=[Order(symbol="SPY", side="buy", qty=10)],
    )


def _discovery_input() -> DiscoveryInput:
    return DiscoveryInput(
        decision_id=uuid4(),
        regime="bull",
        universe=["SPY", "QQQ"],
        features={"sentiment": 0.2},
        recent_thesis="momentum is constructive",
    )


def test_research_prompt_omits_block_without_summary() -> None:
    """Cold start (no summary) → no injection so prompt stays minimal."""
    text = research_prompt(_research_input())
    assert "RECENT SELF-ASSESSMENT" not in text


def test_research_prompt_injects_with_summary() -> None:
    text = research_prompt(_research_input(), scorecard_summary=_good_summary())
    assert "RECENT SELF-ASSESSMENT" in text
    assert "60%" in text


def test_strategy_prompt_injects_with_summary() -> None:
    text = strategy_prompt(_strategy_input(), scorecard_summary=_good_summary())
    assert "RECENT SELF-ASSESSMENT" in text
    # Strategy-specific content must remain intact.
    assert "Strategy Agent" in text
    assert "REGIME PLAYBOOK" in text


def test_risk_prompt_injects_with_summary() -> None:
    text = risk_prompt(_risk_input(), scorecard_summary=_good_summary())
    assert "RECENT SELF-ASSESSMENT" in text
    assert "Risk Agent" in text


def test_execution_prompt_injects_with_summary() -> None:
    text = execution_prompt(_execution_input(), scorecard_summary=_good_summary())
    assert "RECENT SELF-ASSESSMENT" in text
    assert "Execution Agent" in text


def test_discovery_prompt_injects_with_summary() -> None:
    text = discovery_prompt(_discovery_input(), scorecard_summary=_good_summary())
    assert "RECENT SELF-ASSESSMENT" in text
    assert "Discovery Agent" in text


def test_all_prompts_keep_json_contract_after_injection() -> None:
    """The JSON contract block ('Reply with ONE JSON object only') must NOT
    be displaced — model-side, that's the single most important rule."""
    s = _good_summary()
    for text in (
        research_prompt(_research_input(), scorecard_summary=s),
        strategy_prompt(_strategy_input(), scorecard_summary=s),
        risk_prompt(_risk_input(), scorecard_summary=s),
        execution_prompt(_execution_input(), scorecard_summary=s),
        discovery_prompt(_discovery_input(), scorecard_summary=s),
    ):
        assert "ONE JSON object only" in text
        assert "OUTPUT JSON SCHEMA:" in text
