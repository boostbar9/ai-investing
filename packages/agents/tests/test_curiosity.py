"""Phase 33: tests for the curiosity meta-agent.

Curiosity is a pure decision function: read in a snapshot of sweep
state, return one of four action kinds. These tests pin every branch
of the decision tree and the durable action log round-trip.

The bot's safety rails are intentionally strict, so curiosity is the
operator's main lever for "do something" on chop days. We want each
branch of decide() to be unambiguously triggerable and the cap on
cumulative relaxation to actually bite.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from packages.agents.curiosity import (
    IDLE_STREAK_THRESHOLD,
    MAX_CUMULATIVE_RELAXATION,
    WATCHLIST_STALE_S,
    CuriosityAction,
    CuriosityInput,
    decide,
    log_action,
    read_recent_actions,
)


# --- Helpers -----------------------------------------------------------------


def _baseline(**overrides) -> CuriosityInput:
    """A 'quiet bot, no stall' state. Tests bend exactly one knob."""
    base = dict(
        idle_streak=0,
        watchlist_age_s=0.0,
        cumulative_relaxation=0.0,
        dominant_rejection="",
        universe=("AAPL", "MSFT"),
        wildcard_pool=("NVDA", "TSLA", "AMD", "META", "GOOG", "AMZN"),
        last_reflection_age_s=0.0,
    )
    base.update(overrides)
    return CuriosityInput(**base)


# --- Branch 1: wildcard_scan -------------------------------------------------


def test_stale_watchlist_triggers_wildcard_scan() -> None:
    state = _baseline(watchlist_age_s=WATCHLIST_STALE_S + 1)
    action = decide(state, rng=random.Random(0))
    assert action.kind == "wildcard_scan"
    assert action.payload["symbols"]
    assert len(action.payload["symbols"]) <= 5


def test_wildcard_symbols_are_outside_universe() -> None:
    state = _baseline(
        watchlist_age_s=WATCHLIST_STALE_S + 1,
        universe=("AAPL", "MSFT", "NVDA"),
        wildcard_pool=("NVDA", "TSLA", "AMD"),  # NVDA must be excluded
    )
    action = decide(state, rng=random.Random(0))
    assert action.kind == "wildcard_scan"
    assert "NVDA" not in action.payload["symbols"]


def test_wildcard_skipped_if_pool_exhausted_by_universe() -> None:
    """If the only pool symbols are already in the universe, fall
    through to the next decision branch."""
    state = _baseline(
        watchlist_age_s=WATCHLIST_STALE_S + 1,
        universe=("AAPL", "MSFT", "NVDA", "TSLA"),
        wildcard_pool=("NVDA", "TSLA"),  # both already in universe
    )
    action = decide(state, rng=random.Random(0))
    assert action.kind == "noop"


def test_wildcard_skipped_if_pool_empty() -> None:
    state = _baseline(
        watchlist_age_s=WATCHLIST_STALE_S + 1, wildcard_pool=tuple()
    )
    action = decide(state, rng=random.Random(0))
    assert action.kind == "noop"


# --- Branch 2: lower_threshold ----------------------------------------------


def test_idle_streak_triggers_lower_threshold() -> None:
    state = _baseline(
        idle_streak=IDLE_STREAK_THRESHOLD,
        dominant_rejection="atr",
        cumulative_relaxation=0.0,
    )
    action = decide(state)
    assert action.kind == "lower_threshold"
    assert action.payload["filter"] == "atr"
    assert action.payload["relaxation_step"] == pytest.approx(0.10)
    assert action.payload["new_cumulative"] == pytest.approx(0.10)


def test_lower_threshold_skipped_below_idle_streak() -> None:
    state = _baseline(
        idle_streak=IDLE_STREAK_THRESHOLD - 1,
        dominant_rejection="atr",
    )
    action = decide(state)
    assert action.kind == "noop"


def test_lower_threshold_skipped_without_known_rejection() -> None:
    """We don't randomly relax filters. The orchestrator must tell us
    which filter is the bottleneck."""
    state = _baseline(
        idle_streak=IDLE_STREAK_THRESHOLD, dominant_rejection=""
    )
    action = decide(state)
    assert action.kind == "noop"


def test_cumulative_relaxation_cap_respected() -> None:
    """At 0.20 cumulative + 0.10 step = 0.30 cap (allowed). One more
    call at 0.25 + 0.10 = 0.35 must NOT relax."""
    ok = _baseline(
        idle_streak=IDLE_STREAK_THRESHOLD,
        dominant_rejection="vwap",
        cumulative_relaxation=0.20,
    )
    assert decide(ok).kind == "lower_threshold"

    capped = _baseline(
        idle_streak=IDLE_STREAK_THRESHOLD,
        dominant_rejection="vwap",
        cumulative_relaxation=0.25,
    )
    assert decide(capped).kind == "noop"


def test_cumulative_relaxation_exactly_at_cap_is_allowed() -> None:
    """Floating-point ergonomics: 0.20 + 0.10 == 0.30 must not be
    rejected by a strict inequality."""
    state = _baseline(
        idle_streak=IDLE_STREAK_THRESHOLD,
        dominant_rejection="cluster",
        cumulative_relaxation=MAX_CUMULATIVE_RELAXATION - 0.10,
    )
    action = decide(state)
    assert action.kind == "lower_threshold"


# --- Branch 3: narrate_blockers ---------------------------------------------


def test_long_reflection_silence_triggers_narrate() -> None:
    state = _baseline(last_reflection_age_s=31 * 60)
    action = decide(state)
    assert action.kind == "narrate_blockers"


def test_short_reflection_silence_does_not_narrate() -> None:
    state = _baseline(last_reflection_age_s=29 * 60)
    action = decide(state)
    assert action.kind == "noop"


# --- Branch 4: noop ----------------------------------------------------------


def test_quiet_state_is_noop() -> None:
    """Baseline state should always be noop. Curiosity respects
    correct silence."""
    action = decide(_baseline())
    assert action.kind == "noop"


# --- Priority ordering -------------------------------------------------------


def test_wildcard_wins_over_lower_threshold() -> None:
    """When both conditions fire, wildcard_scan must come first \u2014
    it's the cheapest unblock and runs before we touch filter floors."""
    state = _baseline(
        watchlist_age_s=WATCHLIST_STALE_S + 1,
        idle_streak=IDLE_STREAK_THRESHOLD,
        dominant_rejection="atr",
    )
    action = decide(state, rng=random.Random(0))
    assert action.kind == "wildcard_scan"


def test_lower_threshold_wins_over_narrate() -> None:
    state = _baseline(
        idle_streak=IDLE_STREAK_THRESHOLD,
        dominant_rejection="sentiment",
        last_reflection_age_s=99 * 60,
    )
    action = decide(state)
    assert action.kind == "lower_threshold"


# --- Action log round-trip --------------------------------------------------


def test_log_action_and_read_recent_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CURIOSITY_ACTION_LOG", str(tmp_path / "ca.jsonl"))
    a = CuriosityAction(kind="noop", rationale="quiet", payload={})
    log_action(a)
    rows = read_recent_actions()
    assert len(rows) == 1
    assert rows[0].kind == "noop"
    assert rows[0].rationale == "quiet"
    assert rows[0].ts  # auto-stamped


def test_read_recent_actions_newest_first(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CURIOSITY_ACTION_LOG", str(tmp_path / "ca.jsonl"))
    log_action(CuriosityAction(kind="noop", rationale="first", payload={}))
    log_action(CuriosityAction(kind="wildcard_scan", rationale="second", payload={"symbols": ["X"]}))
    log_action(CuriosityAction(kind="lower_threshold", rationale="third", payload={"filter": "atr"}))
    rows = read_recent_actions()
    assert [r.rationale for r in rows] == ["third", "second", "first"]


def test_read_recent_actions_respects_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CURIOSITY_ACTION_LOG", str(tmp_path / "ca.jsonl"))
    for i in range(10):
        log_action(CuriosityAction(kind="noop", rationale=f"r{i}", payload={}))
    rows = read_recent_actions(limit=3)
    assert len(rows) == 3
    # newest three: r9, r8, r7
    assert [r.rationale for r in rows] == ["r9", "r8", "r7"]


def test_read_recent_actions_missing_file_returns_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CURIOSITY_ACTION_LOG", str(tmp_path / "never.jsonl"))
    assert read_recent_actions() == []


def test_read_recent_actions_skips_corrupt_lines(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "ca.jsonl"
    monkeypatch.setenv("CURIOSITY_ACTION_LOG", str(target))
    log_action(CuriosityAction(kind="noop", rationale="ok", payload={}))
    with target.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write("\n")
    log_action(CuriosityAction(kind="wildcard_scan", rationale="also ok", payload={}))
    rows = read_recent_actions()
    assert [r.rationale for r in rows] == ["also ok", "ok"]


def test_log_action_creates_parent_dir(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "deep" / "nested" / "ca.jsonl"
    monkeypatch.setenv("CURIOSITY_ACTION_LOG", str(target))
    log_action(CuriosityAction(kind="noop", rationale="x", payload={}))
    assert target.exists()
