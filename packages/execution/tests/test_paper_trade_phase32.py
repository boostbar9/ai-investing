"""Phase 32: tests for the live-bot fixes diagnosed from the 2026-06-02
decision ledger.

Three independent bugs surfaced when reading
``data/paper_log/decisions.jsonl`` for the trading day:

  1. Every cycle ran ``strategy: mean-reversion`` even though the
     argparse choices include ``intraday-trend``. Root cause: the
     argparse default was ``mean-reversion`` and the Windows service
     invokes the script with no ``--strategy`` flag. Fix: auto-select
     ``intraday-trend`` during RTH, ``mean-reversion`` otherwise.
  2. Half of all sweeps halted on ``agent_halt: sentiment floor
     breached``. Default floor of -0.5 was tuned for the old multi-day
     strategy. Intraday strategies want to *trade* through bearish
     sentiment, not halt on it. Fix: relax the floor to -1.0 for
     intraday strategies.
  3. Cockpit chip showed ``risk_on`` while every decision logged
     ``regime: chop`` — the cockpit and the trader speak two different
     vocabularies. Fix: publish the cockpit-vocab translation alongside
     the HMM label so the operator UI and the decision ledger agree.
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

import pytest_asyncio  # noqa: F401  (registers asyncio mode)
from uuid import uuid4

from packages.agents.paper_bridge import make_risk_agent
from packages.shared.schemas import RiskInput, Signal
from tools.paper_trade import (
    _HMM_TO_COCKPIT,
    _auto_default_strategy,
    _is_rth,
    _to_cockpit_regime,
)


# --- Fix 1: auto-default strategy --------------------------------------------


@pytest.mark.parametrize(
    "iso,expected_rth",
    [
        # Mon 2026-06-01 13:30 UTC = 09:30 ET — the open. RTH starts here.
        ("2026-06-01T13:30:00+00:00", True),
        # Mon 2026-06-01 15:00 UTC = 11:00 ET — deep RTH.
        ("2026-06-01T15:00:00+00:00", True),
        # Mon 2026-06-01 19:59 UTC = 15:59 ET — last RTH minute.
        ("2026-06-01T19:59:00+00:00", True),
        # Mon 2026-06-01 20:00 UTC = 16:00 ET — close, RTH ends here.
        ("2026-06-01T20:00:00+00:00", False),
        # Mon 2026-06-01 09:00 UTC = 05:00 ET — premarket.
        ("2026-06-01T09:00:00+00:00", False),
        # Mon 2026-06-01 22:00 UTC = 18:00 ET — after-hours.
        ("2026-06-01T22:00:00+00:00", False),
        # Sat 2026-05-30 17:00 UTC — weekend, never RTH.
        ("2026-05-30T17:00:00+00:00", False),
        # Sun 2026-05-31 17:00 UTC — weekend, never RTH.
        ("2026-05-31T17:00:00+00:00", False),
    ],
)
def test_is_rth_boundary_conditions(iso: str, expected_rth: bool) -> None:
    now = datetime.fromisoformat(iso)
    assert _is_rth(now) is expected_rth


def test_auto_default_strategy_is_intraday_during_rth() -> None:
    """The 2026-06-02 bug: during RTH the bot defaulted to
    mean-reversion and never picked intraday-trend. This test pins the
    fix — RTH always routes to intraday-trend."""
    # Tue 2026-06-02 14:00 UTC = 10:00 ET, definitively inside RTH.
    rth_now = datetime(2026, 6, 2, 14, 0, tzinfo=UTC)
    assert _auto_default_strategy(rth_now) == "intraday-trend"


def test_auto_default_strategy_is_mean_reversion_after_hours() -> None:
    """After-hours and weekends should fall back to mean-reversion so
    the advisory loop still does something useful overnight, but
    won't try to run intraday signals against stale bars."""
    # Mon 2026-06-01 02:00 UTC = Sun 22:00 ET, after hours.
    ah_now = datetime(2026, 6, 1, 2, 0, tzinfo=UTC)
    assert _auto_default_strategy(ah_now) == "mean-reversion"

    # Sat 2026-05-30 17:00 UTC, weekend.
    weekend = datetime(2026, 5, 30, 17, 0, tzinfo=UTC)
    assert _auto_default_strategy(weekend) == "mean-reversion"


# --- Fix 3: regime vocabulary translation ------------------------------------


@pytest.mark.parametrize(
    "hmm,cockpit",
    [
        ("bull", "risk_on"),
        ("bear", "risk_off"),
        ("chop", "neutral"),
        ("crisis", "volatile"),
    ],
)
def test_regime_translation_known_labels(hmm: str, cockpit: str) -> None:
    """Every label in the HMM vocabulary must map to exactly one
    cockpit-vocab label. This is the bridge that fixes the operator-UI
    vs trader-decision divergence we saw on 2026-06-02."""
    assert _to_cockpit_regime(hmm) == cockpit


def test_regime_translation_is_case_insensitive() -> None:
    """The HMM module sometimes emits title-cased labels (``Bull``);
    the cockpit must still get a clean lowercase translation."""
    assert _to_cockpit_regime("Bull") == "risk_on"
    assert _to_cockpit_regime("CHOP") == "neutral"


def test_regime_translation_unknown_label_degrades_to_neutral() -> None:
    """Defensive: unknown labels (e.g. ``unknown`` when the panel was
    empty) must never raise — they degrade to ``neutral`` so the chip
    stays informative."""
    assert _to_cockpit_regime("unknown") == "neutral"
    assert _to_cockpit_regime("") == "neutral"
    assert _to_cockpit_regime("garbage-from-future-version") == "neutral"


def test_regime_translation_table_is_total_over_hmm_vocab() -> None:
    """Pin the table size. If a future commit adds a new HMM label
    without updating the mapping, this test fails and forces the dev
    to think about the cockpit-side display."""
    hmm_vocab = {"bull", "bear", "chop", "crisis"}
    assert set(_HMM_TO_COCKPIT.keys()) == hmm_vocab
    cockpit_vocab = {"risk_on", "risk_off", "neutral", "volatile"}
    assert set(_HMM_TO_COCKPIT.values()) == cockpit_vocab


# --- Fix 2: relaxed sentiment floor for intraday strategies ------------------


@pytest.mark.asyncio
async def test_sentiment_floor_minus_one_does_not_halt_on_mildly_bearish():
    """Phase 32 floor: with min_sentiment=-1.0 (the new intraday
    default), aggregate sentiment of -0.6 must NOT halt the sweep.

    The 2026-06-02 live log showed 11/22 sweeps halted on aggregate
    sentiment dipping under -0.5. The relaxed floor lets the intraday
    trend-follower keep working through a bearish-news regime — which
    is often when the best short-momentum setups appear.
    """
    fn = make_risk_agent(research_sentiment=-0.6, min_sentiment=-1.0)
    out = await fn(
        RiskInput(
            decision_id=uuid4(),
            positions=[],
            candidates=[Signal(symbol="SPY", side="sell", strength=0.3, rationale="x")],
        )
    )
    assert out.halted is False
    assert len(out.approved) == 1


@pytest.mark.asyncio
async def test_sentiment_floor_minus_one_still_halts_on_extreme_panic():
    """The floor still exists — it's just looser. At sentiment <= -1.0
    we treat it as a true crisis signal (every source negative,
    aggregate below the wall) and halt regardless of strategy."""
    fn = make_risk_agent(research_sentiment=-1.01, min_sentiment=-1.0)
    out = await fn(
        RiskInput(
            decision_id=uuid4(),
            positions=[],
            candidates=[Signal(symbol="SPY", side="buy", strength=0.3, rationale="x")],
        )
    )
    assert out.halted is True


@pytest.mark.asyncio
async def test_old_floor_still_halts_for_multiday_strategies():
    """Multi-day strategies still pass min_sentiment=-0.5 — the relaxed
    floor is opt-in via the call site in paper_trade.py, not a global
    change. Pin this so a future refactor doesn't accidentally relax
    the floor for trend-following / mean-reversion / sector-rotation."""
    fn = make_risk_agent(research_sentiment=-0.6, min_sentiment=-0.5)
    out = await fn(
        RiskInput(
            decision_id=uuid4(),
            positions=[],
            candidates=[Signal(symbol="SPY", side="buy", strength=0.3, rationale="x")],
        )
    )
    assert out.halted is True
