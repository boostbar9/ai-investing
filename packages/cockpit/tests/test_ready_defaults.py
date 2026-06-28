"""Training-ready defaults + shadow-safety invariants.

These lock the "ready to run out of the box" behavior without enabling any
live path:

  * Fresh config => engine ``robinhood_paper``, real RH cash on, Balanced
    confidence (>=55%), caps unchanged.
  * The default broker resolves to the READ-ONLY Robinhood-realistic sim and
    is always shadow / never live.
  * Auto-started autopilot stays shadow: a strategy on SHADOW is never
    upgraded; a strategy requesting LIVE is downgraded to PAPER unless the
    promotion gate AND ENABLE_LIVE_TRADING both clear.
  * Go-Live still requires the gate (cannot enable live without it).
  * Fail safe: corrupt / uncertain config resolves to the shadow default.

Mocks only; no live network.
"""

from __future__ import annotations

import pytest

from packages.cockpit import onboarding as ob
from packages.cockpit import trading_controls as tc
from packages.execution import broker_factory as bf
from packages.execution.modes import ExecutionMode, resolve_mode


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """Onboarding (not env) drives backend; live trading stays off."""
    monkeypatch.delenv("BROKER_BACKEND", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.delenv("ROBINHOOD_FORCE_LIVE_GATE", raising=False)


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(ob, "ONBOARDING_PATH", tmp_path / "onboarding.json")
    monkeypatch.setattr(tc, "TRADING_CONTROLS_PATH", tmp_path / "tc.json")
    return tmp_path


# --------------------------------------------------------------------------
# Defaults resolve correctly
# --------------------------------------------------------------------------
def test_onboarding_default_backend_is_robinhood_paper(isolated):
    s = ob.OnboardingState()
    assert s.broker_backend == "robinhood_paper"
    # And mode/cap stay safe.
    assert s.rh_mode == "shadow"
    assert s.live_float_cap_usd == pytest.approx(300.0)
    # Missing file => same default (load path).
    assert ob.load_onboarding().broker_backend == "robinhood_paper"


def test_corrupt_onboarding_falls_back_to_shadow_default(isolated):
    (isolated / "onboarding.json").write_text("{ not valid json")
    s = ob.load_onboarding()
    assert s.broker_backend == "robinhood_paper"
    assert s.rh_mode == "shadow"


def test_unknown_backend_value_falls_back_to_default(isolated):
    (isolated / "onboarding.json").write_text('{"broker_backend": "etrade"}')
    assert ob.load_onboarding().broker_backend == "robinhood_paper"


def test_trading_controls_defaults_are_training_ready(isolated):
    c = tc.load_controls()
    assert c.risk_preset == "balanced"
    assert c.min_confidence == pytest.approx(0.55)
    assert c.paper_use_real_cash is True
    assert c.paper_start_balance_usd is None
    # Caps unchanged.
    assert c.total_budget_usd == pytest.approx(300.0)
    assert c.max_per_trade_usd == pytest.approx(50.0)
    assert c.max_trades_per_day == 5
    assert c.max_open_positions == 3


def test_paper_balance_override_still_available(isolated):
    c = tc.update_controls({"paper_start_balance_usd": 1000.0})
    assert c.paper_start_balance_usd == pytest.approx(1000.0)
    # Blank clears it back to "use real cash".
    c2 = tc.update_controls({"paper_start_balance_usd": ""})
    assert c2.paper_start_balance_usd is None


# --------------------------------------------------------------------------
# Default broker is the read-only sim, always shadow / never live
# --------------------------------------------------------------------------
def test_default_broker_resolves_to_readonly_sim_shadow(isolated):
    from packages.execution.robinhood_paper import RobinhoodPaperBroker

    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, RobinhoodPaperBroker)
    assert sel.effective_backend == bf.BACKEND_ROBINHOOD_PAPER
    assert not sel.fell_back

    status = bf.active_broker_status()
    assert status["shadow"] is True
    assert status["live"] is False


def test_robinhood_paper_broker_has_no_live_order_path():
    """The realistic engine must never expose place_/cancel_ order calls."""
    from packages.execution.robinhood_paper import RobinhoodPaperBroker

    for attr in dir(RobinhoodPaperBroker):
        assert not attr.startswith("place_")
        assert not attr.startswith("cancel_")


# --------------------------------------------------------------------------
# Go-Live still requires the promotion gate (the auto-on autopilot is shadow)
# --------------------------------------------------------------------------
def test_shadow_strategy_never_auto_upgraded(monkeypatch):
    from packages.execution import modes

    monkeypatch.setattr(modes, "_DEFAULTS", {"s": ExecutionMode.SHADOW})
    d = resolve_mode("s", live_gate_passed=True, env_enable_live=True)
    assert d.effective is ExecutionMode.SHADOW


def test_live_request_downgraded_without_enable_live(monkeypatch):
    from packages.execution import modes

    monkeypatch.setattr(modes, "_DEFAULTS", {"s": ExecutionMode.LIVE})
    # Gate passed but ENABLE_LIVE_TRADING off => stays paper.
    d = resolve_mode("s", live_gate_passed=True, env_enable_live=False)
    assert d.effective is ExecutionMode.PAPER
    assert d.downgraded


def test_live_request_downgraded_without_gate(monkeypatch):
    from packages.execution import modes

    monkeypatch.setattr(modes, "_DEFAULTS", {"s": ExecutionMode.LIVE})
    # ENABLE_LIVE_TRADING on but promotion gate not passed => stays paper.
    d = resolve_mode("s", live_gate_passed=False, env_enable_live=True)
    assert d.effective is ExecutionMode.PAPER
    assert d.downgraded


def test_live_only_when_both_gates_clear(monkeypatch):
    from packages.execution import modes

    monkeypatch.setattr(modes, "_DEFAULTS", {"s": ExecutionMode.LIVE})
    d = resolve_mode("s", live_gate_passed=True, env_enable_live=True)
    assert d.effective is ExecutionMode.LIVE


def test_unknown_mode_fails_safe_to_paper():
    """Uncertain/garbage requested mode resolves to the safe PAPER default."""
    assert ExecutionMode.parse("definitely-not-a-mode") is ExecutionMode.PAPER
    assert ExecutionMode.parse(None) is ExecutionMode.PAPER
