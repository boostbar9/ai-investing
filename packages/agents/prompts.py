"""Agent prompt templates (§5).

Each prompt:
  - Is JSON-only (the router calls Ollama with ``format=json``).
  - Embeds the exact JSON schema the response must conform to (echoed from
    Pydantic's ``model_json_schema()`` at runtime so they cannot drift).
  - Includes the ``decision_id`` so the model can echo it back (we still
    overwrite the value post-parse to prevent injection).
  - Is short — Ollama context budgets are tight on consumer GPUs.

The actual call path is::

    runner = build_research_runner(router)
    out: ResearchOutput = await runner(input)
"""

from __future__ import annotations

import json

from packages.shared.schemas import (
    ExecutionInput,
    ExecutionOutput,
    ResearchInput,
    ResearchOutput,
    RiskInput,
    RiskOutput,
    StrategyInput,
    StrategyOutput,
)

# ---------------------------------------------------------------------------
# System preambles
# ---------------------------------------------------------------------------

_BASE_RULES = """You are a deterministic JSON producer.
Hard rules:
  1. Reply with ONE JSON object only. No prose, no markdown fences.
  2. Conform exactly to the provided JSON Schema. Unknown keys are forbidden.
  3. Echo `decision_id` from the input verbatim.
  4. If you are uncertain, return safe defaults (empty arrays, neutral
     sentiment 0.0, no signals) rather than hallucinating.
  5. Never invent prices, fills, or citation URLs."""


def _schema_block(model: type) -> str:
    schema = model.model_json_schema()
    return json.dumps(schema, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

def research_prompt(payload: ResearchInput) -> str:
    return f"""{_BASE_RULES}

ROLE: Research Agent.
TASK: Read recent context for the given symbols and emit a thesis plus a
sentiment score in [-1, 1]. Cite only URLs already known to you; otherwise
return an empty citations list.

INPUT:
{payload.model_dump_json()}

OUTPUT JSON SCHEMA:
{_schema_block(ResearchOutput)}
"""


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def strategy_prompt(payload: StrategyInput) -> str:
    return f"""{_BASE_RULES}

ROLE: Strategy Agent.
TASK: Given the current regime and feature dict, produce trade signals
across the universe. Each signal must include a side, a strength in [0,1],
and a one-sentence rationale that names the feature(s) driving it. If the
regime is `crisis`, return an empty signals list.

INPUT:
{payload.model_dump_json()}

OUTPUT JSON SCHEMA:
{_schema_block(StrategyOutput)}
"""


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

def risk_prompt(payload: RiskInput) -> str:
    return f"""{_BASE_RULES}

ROLE: Risk Agent. NOTE: position sizing is computed by the deterministic
v3.1 engine (Kelly * regime_mult * vol_target / realized_vol). Your job is
only to APPROVE or REJECT candidate signals based on portfolio constraints
(concentration, correlation, hard halts). Never invent sizes.

TASK: Partition `candidates` into `approved` and `rejected`. If drawdown,
gross exposure, or any §15 hard-rule check trips, set `halted=true` and
populate `halt_reason`.

INPUT:
{payload.model_dump_json()}

OUTPUT JSON SCHEMA:
{_schema_block(RiskOutput)}
"""


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execution_prompt(payload: ExecutionInput) -> str:
    return f"""{_BASE_RULES}

ROLE: Execution Agent. NOTE: actual order placement happens in the broker
abstraction. Your job is to plan slicing/routing notes and emit empty
`fills` (the broker will fill them in). Never invent prices or fill counts.

TASK: For each approved order, decide slicing strategy (e.g. TWAP 5min,
single market) and return the order list unchanged plus an empty `fills`
array.

INPUT:
{payload.model_dump_json()}

OUTPUT JSON SCHEMA:
{_schema_block(ExecutionOutput)}
"""
