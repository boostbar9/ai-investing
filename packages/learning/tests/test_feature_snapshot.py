"""Phase 34: tests for the candidate feature snapshot sink.

These tests pin the schema the ranker depends on. If you add or
rename a feature key, the trainer must keep working against old
on-disk rows \u2014 these tests force that contract to be explicit.
"""
from __future__ import annotations

import json
from pathlib import Path

from packages.learning.feature_snapshot import (
    CATEGORICAL_KEYS,
    FEATURE_KEYS,
    append_snapshots,
    extract_features_from_candidate,
    iter_snapshots,
    load_snapshots,
)


def _full_candidate(symbol: str = "AAPL") -> dict:
    """A maximal candidate dict carrying every feature key the scorer
    reads. Lets us assert the snapshot round-trips faithfully."""
    return {
        "symbol": symbol,
        "confidence": 0.62,
        "corroborated": True,
        "corroboration_score": 0.55,
        "reddit_trust": 0.71,
        "analyst_mean_rating": 2.1,
        "analyst_num": 14,
        "analyst_recent_action": "upgrade",
        "insider_form4_30d": 4,
        "insider_net_shares": 12500.0,
        "stocktwits_trending": True,
        "yahoo_news_count": 7,
    }


# --- extraction -------------------------------------------------------------


def test_extract_features_returns_all_canonical_keys() -> None:
    feats = extract_features_from_candidate(_full_candidate())
    assert set(feats.keys()) == set(FEATURE_KEYS)


def test_extract_coerces_booleans_to_int() -> None:
    """corroborated and stocktwits_trending are bool-ish on disk; the
    ranker treats them as 0/1 numerics."""
    feats = extract_features_from_candidate(_full_candidate())
    assert feats["corroborated"] == 1
    assert feats["stocktwits_trending"] == 1
    falsey = extract_features_from_candidate(
        {"symbol": "X", "corroborated": False, "stocktwits_trending": False}
    )
    assert falsey["corroborated"] == 0
    assert falsey["stocktwits_trending"] == 0


def test_extract_handles_missing_keys_as_none() -> None:
    feats = extract_features_from_candidate({"symbol": "X"})
    # Numeric/optional keys collapse to None (NaN-fill territory for
    # the trainer); bool flags default to 0.
    assert feats["confidence"] is None
    assert feats["reddit_trust"] is None
    assert feats["analyst_action"] is None
    assert feats["corroborated"] == 0
    assert feats["stocktwits_trending"] == 0


def test_extract_tolerates_unparseable_numerics() -> None:
    """A stray string in a numeric field must NOT crash; we want
    None so the trainer can NaN-fill that cell."""
    feats = extract_features_from_candidate(
        {"symbol": "X", "confidence": "nope", "analyst_num": "n/a"}
    )
    assert feats["confidence"] is None
    assert feats["analyst_num"] is None


def test_analyst_action_is_categorical() -> None:
    """analyst_action carries an enum-ish label that the trainer
    label-encodes \u2014 contract test that it lives in CATEGORICAL_KEYS."""
    assert "analyst_action" in CATEGORICAL_KEYS


# --- write / read round-trip -----------------------------------------------


def test_append_and_iter_round_trip(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "snap.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(target))
    n = append_snapshots(
        decision_id="DEC-1",
        regime="bull",
        rows=[_full_candidate("AAPL"), _full_candidate("MSFT")],
    )
    assert n == 2
    rows = list(iter_snapshots())
    assert len(rows) == 2
    assert {r["symbol"] for r in rows} == {"AAPL", "MSFT"}
    assert all(r["decision_id"] == "DEC-1" for r in rows)
    assert all(r["regime"] == "bull" for r in rows)


def test_append_skips_rows_without_symbol(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(tmp_path / "snap.jsonl"))
    n = append_snapshots(
        decision_id="DEC", regime="chop", rows=[{"confidence": 0.5}]
    )
    assert n == 0


def test_append_creates_parent_dir(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "deeply" / "nested" / "snap.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(target))
    append_snapshots(
        decision_id="DEC", regime="bull", rows=[_full_candidate("AAPL")]
    )
    assert target.exists()


def test_iter_snapshots_missing_file_yields_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(tmp_path / "absent.jsonl"))
    assert list(iter_snapshots()) == []


def test_iter_snapshots_skips_corrupt_lines(tmp_path: Path, monkeypatch) -> None:
    """A half-written line on a crashed write must NOT poison the
    whole training table. The reader skips and continues."""
    target = tmp_path / "snap.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(target))
    append_snapshots(
        decision_id="DEC-1", regime="bull", rows=[_full_candidate("AAPL")]
    )
    with target.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write("\n")
        fh.write("{partial: ...\n")
    append_snapshots(
        decision_id="DEC-2", regime="bull", rows=[_full_candidate("MSFT")]
    )
    rows = list(iter_snapshots())
    assert {r["symbol"] for r in rows} == {"AAPL", "MSFT"}


def test_snapshot_features_match_canonical_keys(tmp_path: Path, monkeypatch) -> None:
    """A snapshot row's ``features`` block must be keyed exactly by
    FEATURE_KEYS so the trainer can build a stable matrix."""
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(tmp_path / "snap.jsonl"))
    append_snapshots(
        decision_id="DEC", regime="bull", rows=[_full_candidate()]
    )
    rows = load_snapshots()
    assert set(rows[0]["features"].keys()) == set(FEATURE_KEYS)


def test_jsonl_is_one_object_per_line(tmp_path: Path, monkeypatch) -> None:
    """Some downstream tools read JSONL line-by-line, so the writer
    must NEVER emit pretty-printed multi-line objects."""
    target = tmp_path / "snap.jsonl"
    monkeypatch.setenv("FEATURE_SNAPSHOT_PATH", str(target))
    append_snapshots(
        decision_id="DEC",
        regime="bull",
        rows=[_full_candidate("A"), _full_candidate("B")],
    )
    lines = [ln for ln in target.read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 2
    for ln in lines:
        json.loads(ln)  # each line parses standalone
