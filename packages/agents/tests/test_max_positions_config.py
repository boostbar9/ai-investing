"""Focus-tightening: fail-safe POLICY_MAX_POSITIONS parsing.

Honors the engine's own reflection lesson ("Tighten focus_count — pick
fewer, higher-conviction symbols") by lowering the default concurrent
position cap, while guaranteeing misconfiguration can NEVER widen risk to
effectively-unlimited concentration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from packages.agents import policy  # noqa: E402
from packages.agents.policy import (  # noqa: E402
    DEFAULT_MAX_POSITIONS,
    MAX_POSITIONS_CEILING,
    _safe_max_positions,
)


def test_default_is_tighter_than_old_ten() -> None:
    """The new default concentrates capital (was 10) into ~3-5 names."""
    assert DEFAULT_MAX_POSITIONS == 5
    assert 3 <= DEFAULT_MAX_POSITIONS <= 5


def test_unset_uses_default() -> None:
    assert _safe_max_positions(None) == DEFAULT_MAX_POSITIONS
    assert _safe_max_positions("") == DEFAULT_MAX_POSITIONS
    assert _safe_max_positions("   ") == DEFAULT_MAX_POSITIONS


def test_valid_override_is_honored() -> None:
    assert _safe_max_positions("3") == 3
    assert _safe_max_positions(" 4 ") == 4


@pytest.mark.parametrize("bad", ["abc", "3.5", "ten", "1e9", None and "x"])
def test_unparseable_falls_back_to_default(bad) -> None:
    if bad is None:
        return
    assert _safe_max_positions(bad) == DEFAULT_MAX_POSITIONS


@pytest.mark.parametrize("bad", ["0", "-1", "-100"])
def test_non_positive_falls_back_to_default(bad) -> None:
    """Zero/negative would mean 'no positions' or nonsense — fail safe to
    the default rather than silently halting or inverting the cap."""
    assert _safe_max_positions(bad) == DEFAULT_MAX_POSITIONS


def test_absurdly_large_is_clamped_to_ceiling() -> None:
    """A fat-fingered huge value must never uncap concurrency."""
    assert _safe_max_positions("100000") == MAX_POSITIONS_CEILING
    assert _safe_max_positions(str(MAX_POSITIONS_CEILING + 1)) == MAX_POSITIONS_CEILING
    # Exactly at the ceiling is allowed.
    assert _safe_max_positions(str(MAX_POSITIONS_CEILING)) == MAX_POSITIONS_CEILING


def test_module_max_positions_within_safe_bounds() -> None:
    """Whatever the live env, the resolved cap is always bounded."""
    assert 1 <= policy.MAX_POSITIONS <= MAX_POSITIONS_CEILING


def test_env_override_resolves_through_safe_parser(monkeypatch) -> None:
    """Reloading the module with a misconfigured env still yields a safe,
    bounded cap (never unlimited)."""
    import importlib

    monkeypatch.setenv("POLICY_MAX_POSITIONS", "-5")
    reloaded = importlib.reload(policy)
    try:
        assert reloaded.MAX_POSITIONS == reloaded.DEFAULT_MAX_POSITIONS
    finally:
        monkeypatch.delenv("POLICY_MAX_POSITIONS", raising=False)
        importlib.reload(policy)
