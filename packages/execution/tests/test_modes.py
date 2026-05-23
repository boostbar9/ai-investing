"""Tests for the per-strategy execution mode wiring."""

from __future__ import annotations

import pytest

from packages.execution.modes import (
    _DEFAULTS,
    ExecutionMode,
    all_modes,
    get_mode,
    resolve_mode,
    set_mode,
)


@pytest.fixture(autouse=True)
def _clear_modes(monkeypatch):
    _DEFAULTS.clear()
    monkeypatch.delenv("EXEC_MODE_DEFAULT", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    yield
    _DEFAULTS.clear()


def test_default_is_paper():
    assert get_mode("trend") is ExecutionMode.PAPER


def test_env_default_override(monkeypatch):
    monkeypatch.setenv("EXEC_MODE_DEFAULT", "shadow")
    assert get_mode("trend") is ExecutionMode.SHADOW


def test_set_mode_overrides_env(monkeypatch):
    monkeypatch.setenv("EXEC_MODE_DEFAULT", "shadow")
    set_mode("trend", ExecutionMode.PAPER)
    assert get_mode("trend") is ExecutionMode.PAPER


def test_all_modes_returns_snapshot():
    set_mode("trend", ExecutionMode.PAPER)
    set_mode("sentiment", ExecutionMode.SHADOW)
    snap = all_modes()
    assert snap == {"trend": ExecutionMode.PAPER, "sentiment": ExecutionMode.SHADOW}
    snap["trend"] = ExecutionMode.LIVE  # mutating snapshot must not leak
    assert get_mode("trend") is ExecutionMode.PAPER


def test_parse_unknown_falls_back_to_paper():
    assert ExecutionMode.parse("garbage") is ExecutionMode.PAPER
    assert ExecutionMode.parse(None) is ExecutionMode.PAPER
    assert ExecutionMode.parse("LIVE") is ExecutionMode.LIVE


def test_resolve_live_requires_gate_and_env(monkeypatch):
    set_mode("trend", ExecutionMode.LIVE)
    # No env, no gate → downgraded.
    d = resolve_mode("trend", live_gate_passed=False)
    assert d.effective is ExecutionMode.PAPER
    assert d.downgraded
    assert "ENABLE_LIVE_TRADING" in d.reason


def test_resolve_live_with_env_but_no_gate(monkeypatch):
    set_mode("trend", ExecutionMode.LIVE)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    d = resolve_mode("trend", live_gate_passed=False)
    assert d.effective is ExecutionMode.PAPER
    assert "gate not passed" in d.reason


def test_resolve_live_passes_when_both_satisfied(monkeypatch):
    set_mode("trend", ExecutionMode.LIVE)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    d = resolve_mode("trend", live_gate_passed=True)
    assert d.effective is ExecutionMode.LIVE
    assert not d.downgraded


def test_resolve_shadow_is_never_upgraded(monkeypatch):
    set_mode("trend", ExecutionMode.SHADOW)
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    d = resolve_mode("trend", live_gate_passed=True)
    assert d.effective is ExecutionMode.SHADOW
