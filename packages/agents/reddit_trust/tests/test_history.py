"""Tests for the trust-history JSONL store.

The store must be:
  * Append-only (no row ever rewritten on a write)
  * Crash-tolerant on read (one bad line doesn't poison the file)
  * Silent on IO failure (return [] rather than raising)
  * Trim-correct (prune keeps the LAST N rows, not the first)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.agents.reddit_trust import history as h_mod
from packages.agents.reddit_trust.history import (
    HISTORY_WINDOW,
    HistoryEntry,
    TrustHistory,
)


@pytest.fixture
def isolated_history(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "hist.jsonl"
    monkeypatch.setattr(h_mod, "HISTORY_PATH", p)
    return p


def _entry(author="alice", post_id="p1", *, accurate=None) -> HistoryEntry:
    return HistoryEntry(
        author=author,
        post_id=post_id,
        symbol="SPY",
        direction=1,
        confidence_at_signal=0.6,
        created_at="2026-05-01T12:00:00+00:00",
        outcome_return=(0.02 if accurate is not None else None),
        accurate=accurate,
    )


def test_record_creates_parent_dir(monkeypatch, tmp_path):
    nested = tmp_path / "deep" / "nope" / "hist.jsonl"
    monkeypatch.setattr(h_mod, "HISTORY_PATH", nested)
    hist = TrustHistory()
    hist.record(_entry())
    assert nested.exists()


def test_record_and_load_round_trip(isolated_history):
    hist = TrustHistory()
    hist.record(_entry(post_id="p1", accurate=True))
    hist.record(_entry(post_id="p2", accurate=False))
    rows = hist.load()
    assert len(rows) == 2
    assert rows[0]["post_id"] == "p1"
    assert rows[0]["accurate"] is True
    assert rows[1]["accurate"] is False


def test_load_missing_file_returns_empty(isolated_history):
    hist = TrustHistory()
    assert hist.load() == []


def test_load_skips_malformed_lines(isolated_history):
    """One bad mid-file line must not lose the rest."""
    isolated_history.write_text(
        json.dumps({"author": "a"}) + "\n"
        "garbage line\n"
        "\n"
        + json.dumps({"author": "b"}) + "\n"
    )
    hist = TrustHistory()
    rows = hist.load()
    assert [r["author"] for r in rows] == ["a", "b"]


# ---------------------------------------------------------------------------
# author_accuracy
# ---------------------------------------------------------------------------


def test_author_accuracy_unknown_returns_none(isolated_history):
    hist = TrustHistory()
    hist.record(_entry(author="alice", accurate=True))
    acc, n = hist.author_accuracy("nobody")
    assert acc is None
    assert n == 0


def test_author_accuracy_skips_unlabeled(isolated_history):
    """Entries with accurate=None aren't yet graded -- they shouldn't
    count in the denominator."""
    hist = TrustHistory()
    hist.record(_entry(author="alice", post_id="p1", accurate=None))
    hist.record(_entry(author="alice", post_id="p2", accurate=True))
    acc, n = hist.author_accuracy("alice")
    assert n == 1
    assert acc == pytest.approx(1.0)


def test_author_accuracy_windows_to_recent(isolated_history):
    """Old observations beyond HISTORY_WINDOW are ignored so behavior
    shifts (a once-good author goes bad) get caught quickly."""
    hist = TrustHistory()
    # First HISTORY_WINDOW+10 entries are all wrong; last 10 are all right.
    # author_accuracy should reflect only the last HISTORY_WINDOW.
    for i in range(HISTORY_WINDOW + 10):
        hist.record(_entry(author="x", post_id=f"old{i}", accurate=False))
    for i in range(10):
        hist.record(_entry(author="x", post_id=f"new{i}", accurate=True))
    acc, n = hist.author_accuracy("x")
    assert n == HISTORY_WINDOW
    # Window has 10 correct out of HISTORY_WINDOW
    assert acc == pytest.approx(10 / HISTORY_WINDOW)


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------


def test_prune_keeps_most_recent(isolated_history):
    hist = TrustHistory()
    for i in range(10):
        hist.record(_entry(author=f"a{i}", post_id=f"p{i}"))
    removed = hist.prune(max_rows=5)
    assert removed == 5
    rows = hist.load()
    assert len(rows) == 5
    # The last 5 are the "newest" -- their post_ids should be p5..p9.
    assert [r["post_id"] for r in rows] == [f"p{i}" for i in range(5, 10)]


def test_prune_noop_when_under_cap(isolated_history):
    hist = TrustHistory()
    hist.record(_entry())
    assert hist.prune(max_rows=100) == 0


def test_to_json_round_trip():
    e = _entry(accurate=True)
    parsed = json.loads(e.to_json())
    assert parsed["author"] == e.author
    assert parsed["confidence_at_signal"] == pytest.approx(0.6)
    assert parsed["accurate"] is True
