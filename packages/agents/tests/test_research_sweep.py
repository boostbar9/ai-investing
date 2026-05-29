"""Tests for the boot-time research sweep.

The sweep is the user-visible 'what's interesting today?' tile, so we
cover:

  * Pure helpers (confidence, thesis, candidates_from_sentiment,
    merge_portfolio_candidates) -- deterministic, no I/O.
  * Async orchestration with injected fakes for adapter + portfolio.
  * Persistence round-trip + the dashboard never-crashes contract
    (corrupt files load as None / defaults).
  * Resilience: timeout marks status='failed' instead of raising;
    adapter exceptions don't crash the sweep.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from packages.agents import research_sweep as rs_mod
from packages.agents.research_sweep import (
    Candidate,
    SweepResult,
    _confidence,
    _thesis_line,
    candidates_from_sentiment,
    load_status,
    load_sweep,
    merge_portfolio_candidates,
    run_sweep,
    save_status,
    save_sweep,
)
from packages.data.adapters.base import NewsItem

# ---------------------------------------------------------------------------
# _confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,mentions,expected_floor,expected_ceil",
    [
        (1.0, 20, 0.99, 1.01),  # max magnitude + saturated mentions -> ~1.0
        (-1.0, 20, 0.99, 1.01),  # symmetric: negative magnitude matters
        (0.0, 0, -0.01, 0.01),  # nothing -> ~0
        (0.5, 5, 0.4, 0.5),  # moderate
    ],
)
def test_confidence_in_expected_range(
    score: float, mentions: int, expected_floor: float, expected_ceil: float
) -> None:
    c = _confidence(score, mentions)
    assert expected_floor <= c <= expected_ceil


def test_confidence_is_bounded_unit_interval() -> None:
    """Even pathological inputs must stay in [0, 1] -- downstream code
    treats confidence as a probability-like quantity."""
    assert 0.0 <= _confidence(99.0, 9999) <= 1.0
    assert 0.0 <= _confidence(-99.0, 9999) <= 1.0
    assert 0.0 <= _confidence(0.0, -5) <= 1.0


def test_confidence_mention_saturation() -> None:
    """100 mentions shouldn't beat 20 mentions on confidence -- both are
    saturated. That's the pump-detection guardrail."""
    a = _confidence(0.5, 20)
    b = _confidence(0.5, 100)
    assert a == b


# ---------------------------------------------------------------------------
# _thesis_line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected_word",
    [
        (0.8, "bullish"),
        (0.2, "mildly bullish"),
        (-0.8, "bearish"),
        (-0.2, "mildly bearish"),
        (0.0, "mixed"),
    ],
)
def test_thesis_line_picks_appropriate_bias(
    score: float, expected_word: str
) -> None:
    line = _thesis_line("AAPL", score, 10)
    assert expected_word in line.lower()
    assert "AAPL" in line


def test_thesis_includes_symbol_score_and_mentions() -> None:
    line = _thesis_line("NVDA", 0.42, 7)
    assert "NVDA" in line
    assert "7" in line
    assert "0.42" in line or "+0.42" in line


# ---------------------------------------------------------------------------
# candidates_from_sentiment
# ---------------------------------------------------------------------------


def test_candidates_filters_below_min_mentions() -> None:
    agg = {
        "AAPL": {"score": 0.6, "n": 2, "headlines": ["a"]},  # below MIN
        "NVDA": {"score": 0.5, "n": 8, "headlines": ["b"]},
    }
    out = candidates_from_sentiment(agg, min_mentions=3)
    symbols = {c.symbol for c in out}
    assert "AAPL" not in symbols
    assert "NVDA" in symbols


def test_candidates_sorted_by_confidence_desc() -> None:
    """Highest-confidence candidate must come first -- the dashboard
    relies on this ordering."""
    agg = {
        "WEAK": {"score": 0.15, "n": 5, "headlines": []},
        "STRONG": {"score": 0.9, "n": 15, "headlines": []},
        "MID": {"score": 0.5, "n": 8, "headlines": []},
    }
    out = candidates_from_sentiment(agg, min_mentions=3)
    confidences = [c.confidence for c in out]
    assert confidences == sorted(confidences, reverse=True)
    assert out[0].symbol == "STRONG"


def test_candidates_deterministic_tiebreak_by_symbol() -> None:
    """Ties on confidence break alphabetically so test output is
    reproducible across runs."""
    agg = {
        "ZZZ": {"score": 0.5, "n": 8, "headlines": []},
        "AAA": {"score": 0.5, "n": 8, "headlines": []},
    }
    out = candidates_from_sentiment(agg, min_mentions=3)
    assert [c.symbol for c in out] == ["AAA", "ZZZ"]


def test_candidates_truncated_to_max() -> None:
    agg = {
        f"SYM{i}": {"score": 0.5, "n": 5, "headlines": []} for i in range(50)
    }
    out = candidates_from_sentiment(agg, min_mentions=3, max_candidates=5)
    assert len(out) == 5


def test_candidate_signal_kind_default_sentiment() -> None:
    agg = {"AAPL": {"score": 0.6, "n": 5, "headlines": []}}
    out = candidates_from_sentiment(agg, min_mentions=3)
    assert out[0].signal_kind == "sentiment"


def test_candidate_keeps_sample_headlines_limit() -> None:
    """No more than 5 sample headlines so the JSON file doesn't grow."""
    agg = {
        "AAPL": {
            "score": 0.6,
            "n": 20,
            "headlines": [f"line {i}" for i in range(20)],
        }
    }
    out = candidates_from_sentiment(agg, min_mentions=3)
    assert len(out[0].sample_headlines) == 5


# ---------------------------------------------------------------------------
# merge_portfolio_candidates
# ---------------------------------------------------------------------------


def test_portfolio_overlap_retagged_to_portfolio() -> None:
    base = [
        Candidate(symbol="AAPL", signal_kind="sentiment", thesis="x",
                  confidence=0.3),
        Candidate(symbol="NVDA", signal_kind="sentiment", thesis="y",
                  confidence=0.5),
    ]
    merged = merge_portfolio_candidates(base, portfolio_symbols=["AAPL"])
    aapl = next(c for c in merged if c.symbol == "AAPL")
    assert aapl.signal_kind == "portfolio"


def test_portfolio_overlap_floors_confidence_at_06() -> None:
    """The dashboard always shows held positions with fresh signal --
    that's the whole point of the floor."""
    base = [
        Candidate(symbol="AAPL", signal_kind="sentiment", thesis="x",
                  confidence=0.1)
    ]
    merged = merge_portfolio_candidates(base, portfolio_symbols=["AAPL"])
    assert merged[0].confidence >= 0.6


def test_portfolio_overlap_case_insensitive() -> None:
    """Brokers can return upper-or-lower-case; the merge must be
    case-insensitive."""
    base = [Candidate(symbol="aapl", signal_kind="sentiment", thesis="x",
                      confidence=0.3)]
    merged = merge_portfolio_candidates(base, portfolio_symbols=["AAPL"])
    assert merged[0].signal_kind == "portfolio"


def test_portfolio_overlap_does_not_lower_confidence() -> None:
    """If sentiment already gave us 0.9, the floor must not pull it down."""
    base = [Candidate(symbol="AAPL", signal_kind="sentiment", thesis="x",
                      confidence=0.9)]
    merged = merge_portfolio_candidates(base, portfolio_symbols=["AAPL"])
    assert merged[0].confidence == 0.9


def test_portfolio_with_no_overlap_is_unchanged() -> None:
    base = [Candidate(symbol="AAPL", signal_kind="sentiment", thesis="x",
                      confidence=0.5)]
    merged = merge_portfolio_candidates(base, portfolio_symbols=["MSFT"])
    assert merged[0].signal_kind == "sentiment"
    assert merged[0].confidence == 0.5


# ---------------------------------------------------------------------------
# run_sweep: orchestration
# ---------------------------------------------------------------------------


def _fake_news(symbol: str, headline: str, when: datetime) -> NewsItem:
    return NewsItem(
        symbol=symbol,
        ts=when,
        headline=headline,
        url="https://example.com",
        source="test",
    )


@pytest.mark.asyncio
async def test_run_sweep_happy_path_returns_candidates() -> None:
    now = datetime.now(UTC)
    fake_items = [
        _fake_news("NVDA", f"NVDA bullish earnings rally {i}", now)
        for i in range(8)
    ]
    fake_adapter = AsyncMock()
    fake_adapter.fetch_all.return_value = fake_items
    fake_adapter.aclose.return_value = None

    result = await run_sweep(
        adapter=fake_adapter,
        portfolio_symbols=["AAPL"],
    )

    assert result.status == "done"
    assert result.error == ""
    assert len(result.candidates) >= 1
    nvda = next((c for c in result.candidates if c.symbol == "NVDA"), None)
    assert nvda is not None
    assert nvda.mentions >= 3


@pytest.mark.asyncio
async def test_run_sweep_swallows_adapter_failure() -> None:
    """A flaky adapter must NOT crash the sweep -- we get an empty
    candidate list but ``status='done'`` (graceful degrade)."""
    fake_adapter = AsyncMock()
    fake_adapter.fetch_all.side_effect = RuntimeError("reddit is down")
    fake_adapter.aclose.return_value = None

    result = await run_sweep(
        adapter=fake_adapter,
        portfolio_symbols=[],
    )
    assert result.status == "done"
    assert result.candidates == []


@pytest.mark.asyncio
async def test_run_sweep_marks_held_positions_even_without_news() -> None:
    """Even with zero news, the sweep should still report portfolio
    symbols on the result so the dashboard can show 'no fresh signal
    on your N holdings' rather than going blank."""
    fake_adapter = AsyncMock()
    fake_adapter.fetch_all.return_value = []
    fake_adapter.aclose.return_value = None

    result = await run_sweep(
        adapter=fake_adapter,
        portfolio_symbols=["AAPL", "NVDA"],
    )
    assert result.portfolio_symbols == ["AAPL", "NVDA"]
    assert result.status == "done"


@pytest.mark.asyncio
async def test_run_sweep_timeout_marks_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the budget is busted the sweep returns status='failed' --
    NOT an exception. The dashboard surfaces this as a yellow tile."""

    async def _slow(_self, **_kw):
        await asyncio.sleep(5.0)
        return []

    # Force the sweep budget down to 0.05s and make fetch_all sleep past it.
    monkeypatch.setattr(rs_mod, "RESEARCH_SWEEP_TIMEOUT_S", 0.05)

    fake_adapter = AsyncMock()

    async def _too_slow(*_a, **_kw):
        await asyncio.sleep(5.0)
        return []

    fake_adapter.fetch_all.side_effect = _too_slow
    fake_adapter.aclose.return_value = None

    result = await run_sweep(
        adapter=fake_adapter, portfolio_symbols=[]
    )
    assert result.status == "failed"
    assert "timed out" in result.error


# ---------------------------------------------------------------------------
# Persistence (atomic save/load with monkeypatched paths)
# ---------------------------------------------------------------------------


@pytest.fixture
def sweep_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    sweep_p = tmp_path / "research_sweep.json"
    status_p = tmp_path / "research_sweep_status.json"
    monkeypatch.setattr(rs_mod, "SWEEP_PATH", sweep_p)
    monkeypatch.setattr(rs_mod, "SWEEP_STATUS_PATH", status_p)
    return sweep_p, status_p


def test_save_then_load_sweep_round_trips(
    sweep_paths: tuple[Path, Path],
) -> None:
    _sweep_p, _ = sweep_paths
    result = SweepResult(
        status="done",
        started_at="2026-05-28T15:00:00+00:00",
        finished_at="2026-05-28T15:00:30+00:00",
        duration_s=30.0,
        candidates=[
            Candidate(
                symbol="AAPL",
                signal_kind="sentiment",
                thesis="bullish chatter",
                confidence=0.7,
            )
        ],
        portfolio_symbols=["AAPL"],
    )
    save_sweep(result)
    loaded = load_sweep()
    assert loaded is not None
    assert loaded["status"] == "done"
    assert loaded["candidates"][0]["symbol"] == "AAPL"
    assert loaded["portfolio_symbols"] == ["AAPL"]


def test_load_sweep_returns_none_when_missing(
    sweep_paths: tuple[Path, Path],
) -> None:
    assert load_sweep() is None


def test_load_sweep_returns_none_on_corrupt_json(
    sweep_paths: tuple[Path, Path],
) -> None:
    sweep_p, _ = sweep_paths
    sweep_p.write_text("not json", encoding="utf-8")
    assert load_sweep() is None


def test_save_status_writes_heartbeat(
    sweep_paths: tuple[Path, Path],
) -> None:
    save_status("running", detail="fetching news")
    status = load_status()
    assert status["status"] == "running"
    assert status["detail"] == "fetching news"
    assert status["updated_at"]  # non-empty ISO


def test_load_status_defaults_when_missing(
    sweep_paths: tuple[Path, Path],
) -> None:
    s = load_status()
    assert s == {"status": "idle", "detail": "", "updated_at": ""}


def test_load_status_defaults_on_corruption(
    sweep_paths: tuple[Path, Path],
) -> None:
    _, status_p = sweep_paths
    status_p.write_text("garbage", encoding="utf-8")
    s = load_status()
    assert s["status"] == "idle"


def test_atomic_save_leaves_no_tmp_files(
    sweep_paths: tuple[Path, Path],
) -> None:
    sweep_p, _ = sweep_paths
    save_sweep(
        SweepResult(
            status="done",
            started_at="x",
            finished_at="y",
            duration_s=1.0,
            candidates=[],
            portfolio_symbols=[],
        )
    )
    leftovers = list(sweep_p.parent.glob("*.tmp"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# Candidate.to_dict round-trips JSON
# ---------------------------------------------------------------------------


def test_candidate_to_dict_serializes_cleanly() -> None:
    c = Candidate(
        symbol="AAPL",
        signal_kind="portfolio",
        thesis="hello",
        confidence=0.8,
        sentiment_score=0.5,
        mentions=10,
        sources=["reddit"],
        sample_headlines=["a", "b"],
        created_at="2026-05-28T15:00:00+00:00",
    )
    d = c.to_dict()
    encoded = json.dumps(d)
    assert "AAPL" in encoded
    assert "portfolio" in encoded


# ---------------------------------------------------------------------------
# _atomic_write_json Windows-safety: relative-path handling + retry-on-EACCES
# ---------------------------------------------------------------------------


def test_atomic_write_resolves_relative_target_to_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original Windows bug: caller passed a relative Path; the temp
    file was created absolute (because tempfile.NamedTemporaryFile
    resolves ``dir=`` against CWD) and os.replace then failed with
    'Access is denied' because the two sides looked like different roots
    once the cockpit's CWD-juggling kicked in. We now resolve() before
    doing anything so this can never happen again."""
    monkeypatch.chdir(tmp_path)
    rel_target = Path("data/cockpit/heartbeat.json")
    rs_mod._atomic_write_json(rel_target, {"ok": True})

    # File must exist at the *resolved* absolute location.
    abs_target = (tmp_path / rel_target).resolve()
    assert abs_target.exists()
    assert json.loads(abs_target.read_text(encoding="utf-8")) == {"ok": True}
    # No leftover tmp files in the directory.
    assert list(abs_target.parent.glob("*.tmp")) == []


def test_atomic_write_retries_on_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the Windows sharing-violation: ``os.replace`` raises
    PermissionError once, then succeeds. The retry loop must absorb the
    transient error and complete normally."""
    target = tmp_path / "status.json"

    real_replace = os.replace
    attempts = {"n": 0}

    def flaky_replace(src, dst):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise PermissionError(5, "Access is denied (simulated)")
        return real_replace(src, dst)

    monkeypatch.setattr(rs_mod.os, "replace", flaky_replace)
    # Tighten the retry cadence so this test stays fast.
    monkeypatch.setattr(rs_mod, "_REPLACE_RETRY_SLEEP_S", 0.001)
    monkeypatch.setattr(rs_mod, "_REPLACE_RETRY_BUDGET_S", 0.5)

    rs_mod._atomic_write_json(target, {"v": 42})

    assert attempts["n"] == 2, "second attempt should have succeeded"
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 42}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_gives_up_after_retry_budget_and_cleans_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the destination is held open forever (pathological case), we
    must eventually raise the PermissionError instead of looping forever.
    The temp file must be cleaned up so we don't leak tmp* on every
    failed write."""
    target = tmp_path / "locked.json"

    def always_denied(src, dst):
        raise PermissionError(5, "Access is denied (permanent)")

    monkeypatch.setattr(rs_mod.os, "replace", always_denied)
    monkeypatch.setattr(rs_mod, "_REPLACE_RETRY_SLEEP_S", 0.001)
    monkeypatch.setattr(rs_mod, "_REPLACE_RETRY_BUDGET_S", 0.01)

    with pytest.raises(PermissionError):
        rs_mod._atomic_write_json(target, {"v": 1})

    # The destination should NOT exist (we never landed the write).
    assert not target.exists()
    # Temp files MUST be cleaned up so we don't leak them on every
    # failed write. The dashboard polls 1 Hz; even a small leak adds up.
    assert list(tmp_path.glob("*.tmp")) == []
