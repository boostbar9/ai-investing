"""Unit tests for ensemble agent auto-reweighting
(:mod:`packages.learning.agent_weights`).

All synthetic; no network and no writes outside ``tmp_path``. Covers the
guardrail family the task mandates: cold-start equal weights, shrinkage
toward uniform, bounded movement, renormalisation to 1, and degrade-to-
equal on empty/degenerate data — plus persistence and the apply-point
multiplier.
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.learning.agent_weights import (
    MAX_WEIGHT_FACTOR,
    MIN_SAMPLES_PER_AGENT,
    MIN_WEIGHT_FACTOR,
    agent_influence_multiplier,
    compute_agent_weights,
    current_agent_weights,
    load_agent_weights,
    reweight_from_outcomes,
)


def _row(agents: list[str], correct: bool | None, **kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "pick_id": kw.get("pick_id", "p"),
        "symbol": "SPY",
        "agents_voted": agents,
        "correct": correct,
        "return_eod": 0.01 if correct else -0.01,
        "return_2h": 0.0,
    }
    base.update(kw)
    return base


def _outcomes(agent: str, wins: int, losses: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(wins):
        rows.append(_row([agent], True, pick_id=f"{agent}-w{i}"))
    for i in range(losses):
        rows.append(_row([agent], False, pick_id=f"{agent}-l{i}"))
    return rows


# ---------------------------------------------------------------------------
# Degrade-to-equal / empty
# ---------------------------------------------------------------------------


def test_empty_degrades_to_equal() -> None:
    res = compute_agent_weights([])
    assert res["agents"] == {}
    assert res["n_agents"] == 0
    assert res["cold_start"] is True


def test_single_agent_gets_full_equal_share() -> None:
    res = compute_agent_weights(_outcomes("research", 30, 10))
    rec = res["agents"]["research"]
    assert rec["weight"] == 1.0
    assert rec["factor"] == 1.0


# ---------------------------------------------------------------------------
# Cold start: below the floor an agent keeps an equal say
# ---------------------------------------------------------------------------


def test_cold_start_keeps_equal_weight() -> None:
    # "rookie" has very few decided picks but a perfect record; it must
    # still hold an equal say, not be boosted.
    rows = _outcomes("veteran", 40, 20) + _outcomes("rookie", 3, 0)
    res = compute_agent_weights(rows)
    rookie = res["agents"]["rookie"]
    assert rookie["cold_start"] is True
    assert rookie["decided"] < MIN_SAMPLES_PER_AGENT
    assert "equal say" in rookie["reason"]
    # Two agents, rookie pinned to an equal share (factor 1.0 pre-norm) means
    # it should not be the most influential despite a perfect record.
    assert rookie["factor"] <= res["agents"]["veteran"]["factor"] + 1e-9


def test_overall_cold_start_until_one_agent_clears_floor() -> None:
    res = compute_agent_weights(_outcomes("a", 5, 5) + _outcomes("b", 4, 4))
    assert res["cold_start"] is True
    for rec in res["agents"].values():
        assert rec["factor"] == 1.0


# ---------------------------------------------------------------------------
# Higher accuracy -> more influence (with enough samples)
# ---------------------------------------------------------------------------


def test_winner_outweighs_loser() -> None:
    rows = _outcomes("winner", 45, 5) + _outcomes("loser", 5, 45)
    res = compute_agent_weights(rows)
    assert res["agents"]["winner"]["factor"] > res["agents"]["loser"]["factor"]
    assert res["agents"]["winner"]["factor"] > 1.0
    assert res["agents"]["loser"]["factor"] < 1.0
    assert res["cold_start"] is False


# ---------------------------------------------------------------------------
# Bounded movement: never above max, never silenced below min
# ---------------------------------------------------------------------------


def test_bounded_movement_extremes() -> None:
    # Perfect vs hopeless, lots of samples — bounds must still hold.
    rows = _outcomes("perfect", 200, 0) + _outcomes("hopeless", 0, 200)
    res = compute_agent_weights(rows)
    for rec in res["agents"].values():
        assert MIN_WEIGHT_FACTOR - 1e-6 <= rec["factor"] <= MAX_WEIGHT_FACTOR + 1e-6
    assert res["agents"]["hopeless"]["factor"] >= MIN_WEIGHT_FACTOR
    assert res["agents"]["hopeless"]["weight"] > 0  # never zeroed out


# ---------------------------------------------------------------------------
# Renormalisation: weights sum to 1, factors average 1.0
# ---------------------------------------------------------------------------


def test_weights_sum_to_one() -> None:
    rows = (
        _outcomes("a", 40, 10)
        + _outcomes("b", 25, 25)
        + _outcomes("c", 10, 40)
    )
    res = compute_agent_weights(rows)
    total = sum(r["weight"] for r in res["agents"].values())
    assert abs(total - 1.0) < 1e-6
    mean_factor = sum(r["factor"] for r in res["agents"].values()) / res["n_agents"]
    assert abs(mean_factor - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Shrinkage: more samples at the same win rate -> larger move
# ---------------------------------------------------------------------------


def test_shrinkage_more_samples_moves_more() -> None:
    # Same 80% win rate; "big" has 5× the samples of "small". Compared
    # head-to-head against an identical neutral partner, big should earn a
    # higher factor because shrinkage trusts it more.
    big = compute_agent_weights(
        _outcomes("x", 160, 40) + _outcomes("neutral", 100, 100)
    )["agents"]["x"]["factor"]
    small = compute_agent_weights(
        _outcomes("x", 24, 6) + _outcomes("neutral", 100, 100)
    )["agents"]["x"]["factor"]
    assert big > small


# ---------------------------------------------------------------------------
# "What changed" notes vs previous
# ---------------------------------------------------------------------------


def test_changes_note_emitted_on_movement() -> None:
    previous = {"agents": {"winner": {"factor": 1.0}, "loser": {"factor": 1.0}}}
    rows = _outcomes("winner", 45, 5) + _outcomes("loser", 5, 45)
    res = compute_agent_weights(rows, previous=previous)
    assert res["changes"]
    assert res["agents"]["winner"]["delta"] is not None


# ---------------------------------------------------------------------------
# apply-point multiplier
# ---------------------------------------------------------------------------


def test_influence_multiplier_failsafe() -> None:
    assert agent_influence_multiplier([], {"a": 2.0}) == 1.0
    assert agent_influence_multiplier(["a"], None) == 1.0
    # Unknown agent defaults to neutral.
    assert agent_influence_multiplier(["unknown"], {"a": 2.0}) == 1.0


def test_influence_multiplier_averages_and_bounds() -> None:
    assert agent_influence_multiplier(["a", "b"], {"a": 2.0, "b": 1.0}) == 1.5
    # Clamped to the bounded band.
    assert agent_influence_multiplier(["a"], {"a": 99.0}) == MAX_WEIGHT_FACTOR
    assert agent_influence_multiplier(["a"], {"a": 0.0}) == MIN_WEIGHT_FACTOR


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_reweight_persists_and_loads(tmp_path: Path) -> None:
    out = tmp_path / "outcomes.jsonl"
    wpath = tmp_path / "agent_weights.json"
    rows = _outcomes("winner", 45, 5) + _outcomes("loser", 5, 45)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    status = reweight_from_outcomes(outcomes_path=out, weights_path=wpath)
    assert wpath.exists()
    assert status["updated"]
    loaded = load_agent_weights(wpath)
    assert loaded["agents"]["winner"]["factor"] == status["agents"]["winner"]["factor"]
    cur = current_agent_weights(wpath)
    assert cur["winner"] > cur["loser"]


def test_load_missing_is_cold_start(tmp_path: Path) -> None:
    res = load_agent_weights(tmp_path / "nope.json")
    assert res["cold_start"] is True
    assert res["agents"] == {}
    assert current_agent_weights(tmp_path / "nope.json") == {}


def test_reweight_empty_journal_degrades(tmp_path: Path) -> None:
    status = reweight_from_outcomes(
        outcomes_path=tmp_path / "missing.jsonl",
        weights_path=tmp_path / "w.json",
    )
    assert status["cold_start"] is True
    assert status["agents"] == {}
