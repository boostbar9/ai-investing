"""End-to-end Temporal workflow tests using the in-memory dev server.

These tests exercise the full Research → Strategy → Risk → HITL → Execution
topology, with the LLM activities monkeypatched to deterministic stubs so
no external network or model is needed.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from packages.agents.temporal_workflow import (
    TradeCycleInput,
    TradeCycleResult,
    TradeCycleWorkflow,
)

# ---------------------------------------------------------------------------
# Deterministic fake activities (replace the LLM-bound ones)
# ---------------------------------------------------------------------------


@activity.defn(name="agent.research")
async def fake_research(payload: dict) -> dict:
    return {
        "decision_id": payload["decision_id"],
        "thesis": "fake thesis",
        "sentiment": 0.5,
        "citations": [],
    }


@activity.defn(name="agent.strategy")
async def fake_strategy(payload: dict) -> dict:
    return {
        "decision_id": payload["decision_id"],
        "signals": [
            {"symbol": "SPY", "side": "buy", "strength": 0.7, "rationale": "fake"},
            {"symbol": "XLE", "side": "buy", "strength": 0.6, "rationale": "fake"},
        ],
    }


@activity.defn(name="agent.strategy")
async def fake_strategy_empty(payload: dict) -> dict:
    return {"decision_id": payload["decision_id"], "signals": []}


@activity.defn(name="agent.risk")
async def fake_risk_approve_all(payload: dict) -> dict:
    return {
        "decision_id": payload["decision_id"],
        "approved": payload["candidates"],
        "rejected": [],
        "halted": False,
        "halt_reason": None,
    }


@activity.defn(name="agent.risk")
async def fake_risk_halt(payload: dict) -> dict:
    return {
        "decision_id": payload["decision_id"],
        "approved": [],
        "rejected": payload["candidates"],
        "halted": True,
        "halt_reason": "drawdown halt",
    }


@activity.defn(name="agent.execution")
async def fake_execution(payload: dict) -> dict:
    return {
        "decision_id": payload["decision_id"],
        "fills": [
            {
                "symbol": o["symbol"],
                "side": o["side"],
                "qty": o["qty"],
                "price": 100.0,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
            for o in payload["approved_orders"]
        ],
        "audit_id": "11111111-1111-1111-1111-111111111111",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _input(timeout_s: int = 5) -> TradeCycleInput:
    return TradeCycleInput(
        decision_id=str(uuid.uuid4()),
        symbols=["SPY", "XLE"],
        regime="bull",
        positions=[],
        features={"mom_20": 0.1},
        approval_timeout_seconds=timeout_s,
    )


async def _run(env: WorkflowEnvironment, activities, signal=None) -> TradeCycleResult:
    client: Client = env.client
    queue = "test-queue"
    async with Worker(
        client,
        task_queue=queue,
        workflows=[TradeCycleWorkflow],
        activities=activities,
    ):
        wf_id = f"trade-{uuid.uuid4()}"
        handle = await client.start_workflow(
            TradeCycleWorkflow.run,
            _input(),
            id=wf_id,
            task_queue=queue,
            execution_timeout=timedelta(seconds=30),
        )
        if signal is not None:
            name, arg = signal
            # Small delay so workflow reaches wait_condition first.
            if arg is None:
                await client.get_workflow_handle(wf_id).signal(name)
            else:
                await client.get_workflow_handle(wf_id).signal(name, arg)
        return await handle.result()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workflow_full_happy_path():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env,
            activities=[fake_research, fake_strategy, fake_risk_approve_all, fake_execution],
            signal=("approve", ["SPY", "XLE"]),
        )
    assert result.halted is False
    assert result.n_signals == 2
    assert result.n_approved == 2
    assert result.n_fills == 2


@pytest.mark.asyncio
async def test_workflow_empty_signals_short_circuits():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env,
            activities=[fake_research, fake_strategy_empty, fake_risk_approve_all, fake_execution],
        )
    assert result.n_signals == 0
    assert result.n_fills == 0


@pytest.mark.asyncio
async def test_workflow_risk_halt_blocks_execution():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env,
            activities=[fake_research, fake_strategy, fake_risk_halt, fake_execution],
        )
    assert result.halted is True
    assert result.halt_reason == "drawdown halt"
    assert result.n_fills == 0


@pytest.mark.asyncio
async def test_workflow_operator_deny():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env,
            activities=[fake_research, fake_strategy, fake_risk_approve_all, fake_execution],
            signal=("deny", None),
        )
    assert result.halt_reason == "operator denied"
    assert result.n_fills == 0


@pytest.mark.asyncio
async def test_workflow_partial_approval():
    """Operator approves only one of the candidates."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        result = await _run(
            env,
            activities=[fake_research, fake_strategy, fake_risk_approve_all, fake_execution],
            signal=("approve", ["SPY"]),
        )
    assert result.n_approved == 1
    assert result.n_fills == 1


@pytest.mark.asyncio
async def test_workflow_approval_timeout():
    """No signal arrives → workflow exits with timeout halt_reason."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        client: Client = env.client
        queue = "timeout-queue"
        payload = TradeCycleInput(
            decision_id=str(uuid.uuid4()),
            symbols=["SPY", "XLE"],
            regime="bull",
            positions=[],
            features={"mom_20": 0.1},
            approval_timeout_seconds=2,
        )
        async with Worker(
            client,
            task_queue=queue,
            workflows=[TradeCycleWorkflow],
            activities=[
                fake_research,
                fake_strategy,
                fake_risk_approve_all,
                fake_execution,
            ],
        ):
            result = await client.execute_workflow(
                TradeCycleWorkflow.run,
                payload,
                id=f"trade-{uuid.uuid4()}",
                task_queue=queue,
                execution_timeout=timedelta(seconds=30),
            )
    assert result.halt_reason == "approval timeout"
    assert result.n_fills == 0
