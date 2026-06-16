"""Phase 30 tests: intraday reward -> bandit pipeline.

Surfaces:

  1. ``compute_reward`` — threshold math is correct, NaN/None/junk
     return None (skip, don't poison the bandit).

  2. Ledger I/O — round-trips a known set of pick_ids, tolerates
     missing files, ignores junk lines.

  3. ``apply_outcomes_to_bandit`` — credits the bandit with the right
     features+reward, dedupes via the ledger on re-run, skips
     unsettled rows and rows with no features. Hit/miss/flat tallies
     match.

  4. ``apply_daily_outcomes`` (the public CLI entry point) — loads
     from a real outcomes.jsonl and returns a structured report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.learning.intraday_reward import (
    REWARD_FLAT,
    REWARD_HIT,
    REWARD_MISS,
    ApplyReport,
    append_ledger_entry,
    apply_daily_outcomes,
    apply_outcomes_to_bandit,
    compute_reward,
    load_applied_pick_ids,
)

# ---------------------------------------------------------------------------
# compute_reward
# ---------------------------------------------------------------------------


class TestComputeReward:
    def test_hit_at_threshold(self) -> None:
        assert compute_reward(0.005) == REWARD_HIT

    def test_hit_above_threshold(self) -> None:
        assert compute_reward(0.02) == REWARD_HIT
        assert compute_reward(0.5) == REWARD_HIT

    def test_miss_at_threshold(self) -> None:
        assert compute_reward(-0.005) == REWARD_MISS

    def test_miss_below_threshold(self) -> None:
        assert compute_reward(-0.05) == REWARD_MISS
        assert compute_reward(-0.5) == REWARD_MISS

    def test_flat_between_thresholds(self) -> None:
        assert compute_reward(0.0) == REWARD_FLAT
        assert compute_reward(0.004) == REWARD_FLAT
        assert compute_reward(-0.004) == REWARD_FLAT

    def test_none_returns_none(self) -> None:
        assert compute_reward(None) is None

    def test_nan_returns_none(self) -> None:
        assert compute_reward(float("nan")) is None

    def test_junk_returns_none(self) -> None:
        assert compute_reward("not-a-number") is None  # type: ignore[arg-type]

    def test_string_number_coerces(self) -> None:
        # JSONL rows often deserialise floats as strings if the writer
        # chose to quote them. Make sure we still get a useful reward.
        assert compute_reward("0.01") == REWARD_HIT  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class TestLedger:
    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        out = load_applied_pick_ids(tmp_path / "nope.jsonl")
        assert out == set()

    def test_append_then_load_round_trips(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        append_ledger_entry(
            {"pick_id": "abc123", "reward": 1.0, "features": ["x"]}, ledger
        )
        append_ledger_entry(
            {"pick_id": "def456", "reward": -1.0, "features": ["y"]}, ledger
        )
        assert load_applied_pick_ids(ledger) == {"abc123", "def456"}

    def test_load_skips_junk_lines(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text(
            '{"pick_id": "ok1"}\n'
            "not-json\n"
            "   \n"
            '{"pick_id": "ok2"}\n'
            '{"no_pick_id": true}\n',
            encoding="utf-8",
        )
        assert load_applied_pick_ids(ledger) == {"ok1", "ok2"}


# ---------------------------------------------------------------------------
# apply_outcomes_to_bandit
# ---------------------------------------------------------------------------


def _make_row(
    pick_id: str,
    return_eod: float | None,
    features: list[str] | None = None,
    *,
    symbol: str = "SPY",
) -> dict:
    return {
        "pick_id": pick_id,
        "symbol": symbol,
        "return_eod": return_eod,
        "agents_voted": features if features is not None else ["research"],
    }


class TestApplyOutcomesToBandit:
    def test_credits_bandit_with_features_and_reward(
        self, tmp_path: Path
    ) -> None:
        ledger = tmp_path / "ledger.jsonl"
        calls: list[tuple[list[str], float]] = []

        def fake_update(feats, reward, **_kw):
            calls.append((list(feats), float(reward)))

        rows = [
            _make_row("p1", 0.02, ["research", "analyst_bullish"]),  # hit
            _make_row("p2", -0.02, ["insider"]),                      # miss
            _make_row("p3", 0.001, ["reddit_trust"]),                 # flat
        ]
        report = apply_outcomes_to_bandit(
            rows, ledger_path=ledger, bandit_update=fake_update
        )

        assert report.applied == 3
        assert report.hits == 1
        assert report.misses == 1
        assert report.flats == 1
        # Calls in row order, each with the row's features.
        assert calls[0] == (["research", "analyst_bullish"], REWARD_HIT)
        assert calls[1] == (["insider"], REWARD_MISS)
        assert calls[2] == (["reddit_trust"], REWARD_FLAT)

    def test_ledger_dedupes_on_rerun(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        calls: list[tuple] = []

        def fake_update(feats, reward, **_kw):
            calls.append((tuple(feats), reward))

        rows = [
            _make_row("p1", 0.02, ["research"]),
            _make_row("p2", -0.02, ["insider"]),
        ]
        first = apply_outcomes_to_bandit(
            rows, ledger_path=ledger, bandit_update=fake_update
        )
        assert first.applied == 2

        # Second run over the same rows must NOT re-update the bandit.
        second = apply_outcomes_to_bandit(
            rows, ledger_path=ledger, bandit_update=fake_update
        )
        assert second.applied == 0
        assert second.skipped_already_applied == 2
        assert len(calls) == 2  # still just the first run's two calls

    def test_skips_unsettled_rows(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"

        def fake_update(*_a, **_kw):
            pytest.fail("bandit must not be updated for unsettled rows")

        rows = [
            _make_row("p1", None, ["research"]),
            _make_row("p2", float("nan"), ["research"]),
        ]
        report = apply_outcomes_to_bandit(
            rows, ledger_path=ledger, bandit_update=fake_update
        )
        assert report.applied == 0
        assert report.skipped_unsettled == 2

    def test_skips_rows_without_features(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        calls: list = []

        def fake_update(feats, reward, **_kw):
            calls.append(reward)

        rows = [
            _make_row("p1", 0.02, []),       # no features
            _make_row("p2", 0.02, ["research"]),
        ]
        report = apply_outcomes_to_bandit(
            rows, ledger_path=ledger, bandit_update=fake_update
        )
        assert report.applied == 1
        assert report.skipped_no_features == 1
        assert calls == [REWARD_HIT]

    def test_skips_rows_without_pick_id(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        calls: list = []

        def fake_update(feats, reward, **_kw):
            calls.append(reward)

        rows = [
            {"return_eod": 0.02, "agents_voted": ["research"]},  # no pick_id
            _make_row("p2", 0.02, ["research"]),
        ]
        report = apply_outcomes_to_bandit(
            rows, ledger_path=ledger, bandit_update=fake_update
        )
        assert report.applied == 1  # only the row with a pick_id was applied

    def test_string_feature_is_normalized_to_list(self, tmp_path: Path) -> None:
        """Some outcome writers store agents_voted as a single string —
        the helper must wrap it in a list rather than iterating chars."""
        ledger = tmp_path / "ledger.jsonl"
        calls: list = []

        def fake_update(feats, reward, **_kw):
            calls.append(list(feats))

        rows = [{"pick_id": "p1", "return_eod": 0.02, "agents_voted": "research"}]
        apply_outcomes_to_bandit(
            rows, ledger_path=ledger, bandit_update=fake_update
        )
        assert calls == [["research"]]

    def test_report_to_dict_round_trips(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        rows = [_make_row("p1", 0.02, ["research"])]
        report = apply_outcomes_to_bandit(
            rows, ledger_path=ledger, bandit_update=lambda *a, **k: None
        )
        d = report.to_dict()
        assert d["applied"] == 1
        assert d["hits"] == 1
        # JSON-serialisable.
        json.dumps(d)


# ---------------------------------------------------------------------------
# apply_daily_outcomes (loads from disk)
# ---------------------------------------------------------------------------


def test_apply_daily_outcomes_loads_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outcomes_path = tmp_path / "outcomes.jsonl"
    rows = [
        {
            "pick_id": "p1",
            "symbol": "SPY",
            "return_eod": 0.02,
            "agents_voted": ["research"],
        },
        {
            "pick_id": "p2",
            "symbol": "QQQ",
            "return_eod": -0.02,
            "agents_voted": ["insider"],
        },
    ]
    outcomes_path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    ledger_path = tmp_path / "ledger.jsonl"

    # Stub the real bandit module so we don't touch its on-disk state.
    calls: list = []

    def fake_update(feats, reward, **_kw):
        calls.append((list(feats), reward))

    from packages.learning import intraday_reward as ir

    monkeypatch.setattr(ir.cockpit_bandit, "update_with_outcome", fake_update)

    report = apply_daily_outcomes(
        outcomes_path=outcomes_path, ledger_path=ledger_path
    )
    assert isinstance(report, ApplyReport)
    assert report.applied == 2
    assert report.hits == 1
    assert report.misses == 1
    assert len(calls) == 2
    # Ledger file was written.
    assert ledger_path.exists()
    assert load_applied_pick_ids(ledger_path) == {"p1", "p2"}
