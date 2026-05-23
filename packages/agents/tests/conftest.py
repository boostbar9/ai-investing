"""Test fixtures for the agents package.

The router + runner tests pin the original \u00a75 spec model names
(deepseek-r1:70b, qwen2.5:72b, llama3.3:70b, mistral-large). Force the
``workstation`` hardware profile for the whole agents test directory so
those assertions remain meaningful regardless of the dev's HARDWARE_PROFILE
override.

Tests that exercise other profiles (test_model_profiles.py) pass the
profile explicitly and never read from env.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _force_workstation_profile(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    if request.node.fspath.basename == "test_model_profiles.py":
        # That file controls its own env via monkeypatch \u2014 don't shadow it.
        return
    monkeypatch.setenv("HARDWARE_PROFILE", "workstation")
