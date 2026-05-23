from uuid import uuid4

from packages.shared.schemas import (
    ExecutionOutput,
    RiskOutput,
    Signal,
    StrategyOutput,
)


def test_strategy_output_roundtrip():
    out = StrategyOutput(
        decision_id=uuid4(),
        signals=[Signal(symbol="SPY", side="buy", strength=0.7, rationale="trend up")],
    )
    assert out.signals[0].strength == 0.7


def test_risk_halt_requires_reason_optional():
    # Halt may be True with a reason; it's not strictly required by schema but conventionally set.
    out = RiskOutput(decision_id=uuid4(), approved=[], rejected=[], halted=True, halt_reason="VIX > 40")
    assert out.halted


def test_execution_output_empty_fills():
    out = ExecutionOutput(decision_id=uuid4(), fills=[], audit_id=uuid4())
    assert out.fills == []
