"""Active-broker selection for the autonomy / execution loop.

Historically every live path hard-coded :class:`AlpacaPaperBroker`. This
module introduces a single selection seam -- :func:`resolve_active_broker`
-- so the loop can trade through a different backend (today: Robinhood
agentic) *without* changing any default behavior.

Safety model (do not weaken):

* **Default / unset / any error -> Alpaca paper.** The selection is opt-in;
  a fresh install, an unreadable config, or a build failure all resolve to
  the existing paper broker. This preserves current runtime + tests.
* **Selecting ``robinhood`` does NOT enable live trading.** The Robinhood
  broker is built in whatever mode onboarding says (SHADOW by default), and
  its live path is still gated by ``resolve_mode`` + ``ENABLE_LIVE_TRADING``
  exactly like Alpaca. Selecting the backend only changes *which* broker
  receives orders, not whether they're live.
* **Fail safe to paper.** If Robinhood is selected but not connected (no
  tokens), or has no resolvable agentic account, or the broker fails to
  build, we log a clear warning and fall back to Alpaca paper rather than
  crashing the loop or silently trading nowhere.

The backend is read from ``OnboardingState.broker_backend`` (persisted in
the cockpit onboarding store, like ``rh_mode``), with a ``BROKER_BACKEND``
env override for CI / ops. Values: ``alpaca_paper`` (default), ``robinhood``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from packages.execution.broker import AlpacaPaperBroker, Broker

logger = logging.getLogger(__name__)

# Recognized backend identifiers. Anything else resolves to the default.
BACKEND_ALPACA_PAPER = "alpaca_paper"
BACKEND_ROBINHOOD = "robinhood"


@dataclass(frozen=True)
class BrokerSelection:
    """Resolved active-broker decision, surfaced for status display.

    ``backend`` is the *requested* backend; ``effective_backend`` is what
    we actually built (they differ when a Robinhood request fell back to
    paper). ``reason`` is human-readable for the cockpit."""

    broker: Broker
    backend: str
    effective_backend: str
    reason: str

    @property
    def fell_back(self) -> bool:
        return self.backend != self.effective_backend


def _read_backend() -> str:
    """Resolve the requested backend: env override wins, else onboarding.

    Fails safe to ``alpaca_paper`` on any error so a corrupt config can
    never wedge broker construction."""
    env = os.getenv("BROKER_BACKEND", "").strip().lower()
    if env:
        return env
    try:
        from packages.cockpit.onboarding import load_onboarding

        return str(load_onboarding().broker_backend or BACKEND_ALPACA_PAPER).lower()
    except Exception as exc:  # pragma: no cover - belt and braces
        logger.warning(
            "broker backend resolve failed (%s) -- defaulting to alpaca_paper",
            exc.__class__.__name__,
        )
        return BACKEND_ALPACA_PAPER


def _build_robinhood_or_fallback() -> BrokerSelection:
    """Build the Robinhood broker, falling back to paper on any problem.

    Order of fail-safe checks:
      1. not connected (no usable token) -> paper
      2. broker build raises -> paper
      3. no resolvable agentic account -> paper (reads would work but the
         order path would refuse; surface that as a fallback so the loop
         doesn't silently route to a broker that can't trade)
    """
    try:
        from packages.execution import robinhood as rh
    except Exception as exc:  # pragma: no cover - import guard
        return BrokerSelection(
            AlpacaPaperBroker(),
            BACKEND_ROBINHOOD,
            BACKEND_ALPACA_PAPER,
            f"robinhood import failed ({exc.__class__.__name__}) -- using paper",
        )

    if not rh.is_connected():
        return BrokerSelection(
            AlpacaPaperBroker(),
            BACKEND_ROBINHOOD,
            BACKEND_ALPACA_PAPER,
            "robinhood not connected -- using paper (connect in Settings)",
        )

    try:
        broker = rh.build_broker_from_settings()
    except Exception as exc:
        logger.warning(
            "robinhood broker build failed (%s) -- falling back to paper",
            exc.__class__.__name__,
        )
        return BrokerSelection(
            AlpacaPaperBroker(),
            BACKEND_ROBINHOOD,
            BACKEND_ALPACA_PAPER,
            f"robinhood build failed ({exc.__class__.__name__}) -- using paper",
        )

    if not rh.resolve_agentic_account_number():
        return BrokerSelection(
            AlpacaPaperBroker(),
            BACKEND_ROBINHOOD,
            BACKEND_ALPACA_PAPER,
            "no agentic account resolved -- using paper "
            "(run agentic-account discovery in Settings)",
        )

    return BrokerSelection(
        broker,
        BACKEND_ROBINHOOD,
        BACKEND_ROBINHOOD,
        "robinhood active (still shadow unless the live gate authorizes)",
    )


def resolve_broker_selection() -> BrokerSelection:
    """Resolve the active broker plus selection metadata (for status).

    See module docstring for the safety model. Never raises -- any error
    resolves to Alpaca paper."""
    try:
        backend = _read_backend()
    except Exception:  # pragma: no cover - _read_backend already guards
        backend = BACKEND_ALPACA_PAPER

    if backend == BACKEND_ROBINHOOD:
        return _build_robinhood_or_fallback()

    # Default / unset / unrecognized -> existing Alpaca paper behavior.
    reason = (
        "alpaca paper (default)"
        if backend in (BACKEND_ALPACA_PAPER, "")
        else f"unknown backend {backend!r} -- using alpaca paper"
    )
    return BrokerSelection(
        AlpacaPaperBroker(), backend, BACKEND_ALPACA_PAPER, reason
    )


def resolve_active_broker() -> Broker:
    """Return the active broker for the autonomy / execution loop.

    Thin wrapper over :func:`resolve_broker_selection` for call sites that
    only need the broker object. Default behavior (unset / error) is the
    existing :class:`AlpacaPaperBroker`."""
    return resolve_broker_selection().broker


def active_broker_status() -> dict[str, Any]:
    """Read-only status snapshot of the active broker + safety posture.

    Reports the selected backend, effective backend (after fail-safe
    fallback), whether the broker is shadow or live, the resolved float
    cap, and the targeted agentic account number MASKED to its last 4
    digits. Never raises; never enables trading. Safe for a status
    endpoint to call on every request."""
    sel = resolve_broker_selection()
    out: dict[str, Any] = {
        "backend": sel.backend,
        "effective_backend": sel.effective_backend,
        "fell_back": sel.fell_back,
        "reason": sel.reason,
        "shadow": True,
        "live": False,
        "cap_usd": None,
        "account_masked": None,
    }

    # Float cap + Robinhood-specific posture (only meaningful when the
    # effective backend is Robinhood).
    try:
        from packages.execution import robinhood as rh

        out["cap_usd"] = rh.resolve_float_cap()
        if sel.effective_backend == BACKEND_ROBINHOOD:
            broker = sel.broker
            # ``_is_shadow`` consults the same live gate the order path uses.
            is_shadow = (
                broker._is_shadow()
                if hasattr(broker, "_is_shadow")
                else True
            )
            out["shadow"] = bool(is_shadow)
            out["live"] = not bool(is_shadow)
            acct = rh.resolve_agentic_account_number()
            if acct:
                out["account_masked"] = "••••" + acct[-4:]
    except Exception as exc:  # pragma: no cover - status must never raise
        logger.warning(
            "active_broker_status posture read failed (%s)",
            exc.__class__.__name__,
        )

    return out


__all__ = [
    "BACKEND_ALPACA_PAPER",
    "BACKEND_ROBINHOOD",
    "BrokerSelection",
    "active_broker_status",
    "resolve_active_broker",
    "resolve_broker_selection",
]
