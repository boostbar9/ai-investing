"""Tests for the active-broker selection seam (``broker_factory``).

The factory is the single place the autonomy loop picks a broker. The
safety properties under test:

  * Default / unset / unknown / error -> Alpaca paper (preserves existing
    behavior + all current tests).
  * ``robinhood`` selected but NOT connected / no agentic account / build
    failure -> fail safe back to Alpaca paper, never crash.
  * ``robinhood`` selected AND connected AND agentic account resolved ->
    the Robinhood broker.
  * Selecting Robinhood does NOT by itself enable live -- the resolved
    broker stays SHADOW without ENABLE_LIVE_TRADING + the promotion gate.
  * ``active_broker_status`` reports backend, shadow/live, cap, masked acct.
"""

from __future__ import annotations

import json

import pytest

from packages.execution import broker_factory as bf
from packages.execution import robinhood as rh_mod
from packages.execution.broker import AlpacaPaperBroker
from packages.execution.modes import ExecutionMode
from packages.execution.robinhood import RobinhoodAgenticBroker
from packages.execution.robinhood_token import TokenSet

# The user's REAL agentic account number (confirmed live). Used here only
# as a fixture value -- never hardcoded in source.
AGENTIC_ACCT = "668863863"
MARGIN_ACCT = "5SA87845"
MANAGED_ACCT = "181701389106"


@pytest.fixture
def isolated_onboarding(monkeypatch, tmp_path):
    """Point onboarding state at a tmp file so backend/account selection
    tests don't touch the real user state."""
    from packages.cockpit import onboarding as ob

    path = tmp_path / "onboarding.json"
    monkeypatch.setattr(ob, "ONBOARDING_PATH", path)
    return path


def _write_onboarding(path, *, backend="alpaca_paper", account="", mode="shadow"):
    path.write_text(
        json.dumps(
            {
                "completed": True,
                "robinhood_status": "granted",
                "live_float_cap_usd": 300.0,
                "rh_mode": mode,
                "broker_backend": backend,
                "rh_account_number": account,
            }
        )
    )


def _good_tokens() -> TokenSet:
    import time

    return TokenSet(
        access_token="acc", refresh_token="ref", expires_at=time.time() + 3600
    )


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """Ensure the BROKER_BACKEND env override is unset so onboarding wins,
    and live trading stays off by default."""
    monkeypatch.delenv("BROKER_BACKEND", raising=False)
    monkeypatch.delenv("ENABLE_LIVE_TRADING", raising=False)
    monkeypatch.delenv("ROBINHOOD_FORCE_LIVE_GATE", raising=False)


# ---------------------------------------------------------------------------
# Default / fail-safe selection -> Alpaca paper
# ---------------------------------------------------------------------------


def test_default_is_robinhood_paper(isolated_onboarding):
    """No config file at all -> Robinhood-realistic paper simulator (the
    new training-ready default). Read-only; never live."""
    from packages.execution.robinhood_paper import RobinhoodPaperBroker

    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, RobinhoodPaperBroker)
    assert sel.effective_backend == bf.BACKEND_ROBINHOOD_PAPER
    assert not sel.fell_back


def test_explicit_alpaca_paper_still_selectable(isolated_onboarding):
    _write_onboarding(isolated_onboarding, backend="alpaca_paper")
    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, AlpacaPaperBroker)
    assert sel.effective_backend == bf.BACKEND_ALPACA_PAPER
    assert not sel.fell_back


def test_unknown_backend_falls_back_to_paper(isolated_onboarding, monkeypatch):
    monkeypatch.setenv("BROKER_BACKEND", "etrade")
    broker = bf.resolve_active_broker()
    assert isinstance(broker, AlpacaPaperBroker)


def test_corrupt_onboarding_falls_back_to_shadow_default(isolated_onboarding):
    """Corrupt config => the safe shadow default (Robinhood-realistic sim),
    which is read-only and never live."""
    from packages.execution.robinhood_paper import RobinhoodPaperBroker

    isolated_onboarding.write_text("{ not valid json")
    broker = bf.resolve_active_broker()
    assert isinstance(broker, RobinhoodPaperBroker)


def test_robinhood_selected_but_not_connected_falls_back(
    isolated_onboarding, monkeypatch
):
    """The headline fail-safe: select robinhood but no tokens -> paper."""
    _write_onboarding(isolated_onboarding, backend="robinhood", account=AGENTIC_ACCT)
    monkeypatch.setattr(rh_mod, "load_tokens", lambda: None)
    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, AlpacaPaperBroker)
    assert sel.backend == bf.BACKEND_ROBINHOOD
    assert sel.effective_backend == bf.BACKEND_ALPACA_PAPER
    assert sel.fell_back


def test_robinhood_connected_but_no_account_falls_back(
    isolated_onboarding, monkeypatch
):
    """Connected but no agentic account resolved -> paper (the order path
    would refuse anyway; surface it as a fallback)."""
    _write_onboarding(isolated_onboarding, backend="robinhood", account="")
    monkeypatch.setattr(rh_mod, "load_tokens", _good_tokens)
    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, AlpacaPaperBroker)
    assert sel.fell_back


# ---------------------------------------------------------------------------
# Robinhood selected AND connected AND account resolved -> Robinhood broker
# ---------------------------------------------------------------------------


def test_robinhood_selected_and_ready(isolated_onboarding, monkeypatch):
    _write_onboarding(isolated_onboarding, backend="robinhood", account=AGENTIC_ACCT)
    monkeypatch.setattr(rh_mod, "load_tokens", _good_tokens)
    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, RobinhoodAgenticBroker)
    assert sel.effective_backend == bf.BACKEND_ROBINHOOD
    assert not sel.fell_back
    # The broker targets the stored agentic account.
    assert sel.broker._account_number == AGENTIC_ACCT


def test_robinhood_selected_stays_shadow_without_live_gate(
    isolated_onboarding, monkeypatch
):
    """Selecting robinhood does NOT enable live. Even with rh_mode=live
    in onboarding, the broker stays shadow until ENABLE_LIVE_TRADING +
    the promotion gate clear."""
    _write_onboarding(
        isolated_onboarding, backend="robinhood", account=AGENTIC_ACCT, mode="live"
    )
    monkeypatch.setattr(rh_mod, "load_tokens", _good_tokens)
    # ENABLE_LIVE_TRADING is unset (autouse fixture) -> resolve_mode gate fails.
    sel = bf.resolve_broker_selection()
    assert isinstance(sel.broker, RobinhoodAgenticBroker)
    # The broker requested LIVE but the gate keeps it shadow.
    assert sel.broker._mode is ExecutionMode.LIVE
    assert sel.broker._is_shadow() is True


# ---------------------------------------------------------------------------
# Status surface
# ---------------------------------------------------------------------------


def test_status_default_robinhood_paper(isolated_onboarding):
    """Default now resolves to the read-only Robinhood-realistic sim, which
    is always shadow / never live."""
    status = bf.active_broker_status()
    assert status["effective_backend"] == bf.BACKEND_ROBINHOOD_PAPER
    assert status["shadow"] is True
    assert status["live"] is False
    assert status["cap_usd"] == 300.0
    assert status["account_masked"] is None


def test_status_robinhood_masks_account(isolated_onboarding, monkeypatch):
    _write_onboarding(isolated_onboarding, backend="robinhood", account=AGENTIC_ACCT)
    monkeypatch.setattr(rh_mod, "load_tokens", _good_tokens)
    status = bf.active_broker_status()
    assert status["effective_backend"] == bf.BACKEND_ROBINHOOD
    assert status["shadow"] is True  # no live gate -> shadow
    assert status["account_masked"] == "••••3863"
    assert status["account_masked"].endswith(AGENTIC_ACCT[-4:])
    # Full account number is never exposed.
    assert AGENTIC_ACCT not in status["account_masked"]
