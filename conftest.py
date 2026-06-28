"""Repo-wide pytest fixtures.

The local-LLM thesis enrichment (``packages/agents/llm_thesis.py``) is ON
by default in production, but tests must exercise the deterministic
rule-based path unless they explicitly opt in with mocks. We force
``ENABLE_AGENT_LLM=false`` for every test so that:

  * a developer box (or CI runner) that happens to have Ollama reachable
    can never let a real model perturb scoring/ordering assertions, and
  * the protected yfinance-ordering and scorer tests see byte-for-byte
    the same rule-based behavior they always did.

Tests that want to drive the LLM path re-enable the flag locally with
``monkeypatch.setenv("ENABLE_AGENT_LLM", "true")`` AND mock
``LLMRouter.generate_json`` so no real Ollama call is ever made.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _llm_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_AGENT_LLM", "false")
