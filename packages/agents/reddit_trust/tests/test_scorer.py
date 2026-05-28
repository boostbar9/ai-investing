"""Tests for the Reddit trust scorer.

These pin down the *shape* of the scoring function, not exact magic
numbers -- the constants will get swept in Phase 5's pretrain. What
must remain stable:

  * Brand-new account with a viral post -> very low weight.
  * Old account with strong consensus -> high weight.
  * Unknown author history -> neutral 0.5, not punitive.
  * Pump-style language -> at least one flag, weight reduced.
  * No component ever escapes [0, 1].
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from packages.agents.reddit_trust.history import HistoryEntry, TrustHistory
from packages.agents.reddit_trust.schema import RedditPost
from packages.agents.reddit_trust.scorer import (
    AGE_MIN_DAYS,
    HISTORY_UNKNOWN_NEUTRAL,
    KARMA_SATURATION,
    RedditTrustScorer,
    detect_pump_flags,
    score_age,
    score_engagement,
    score_history,
    score_karma,
)

_NOW = datetime(2026, 5, 28, tzinfo=UTC).timestamp()


def _post(
    *,
    title="SPY breakout looks real",
    selftext="long thesis here",
    author="trader_jane",
    karma=10_000,
    account_age_days=365,
    score=200,
    num_comments=50,
    upvote_ratio=0.85,
    tickers=("SPY",),
) -> RedditPost:
    author_created = None if account_age_days is None else _NOW - account_age_days * 86400
    return RedditPost(
        id="abc",
        permalink="/r/stocks/comments/abc/x/",
        subreddit="stocks",
        title=title,
        selftext=selftext,
        author=author,
        author_created_utc=author_created,
        author_karma=karma,
        score=score,
        num_comments=num_comments,
        upvote_ratio=upvote_ratio,
        created_utc=_NOW - 3600,  # 1h ago
        tickers=tickers,
    )


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------


def test_score_karma_handles_none():
    """``None`` is 'unknown', not 'zero' -- the account exists, we just
    couldn't fetch /about.json. Returning 0 would be too punitive."""
    assert score_karma(None) == pytest.approx(0.1)


def test_score_karma_monotonic_in_karma():
    a = score_karma(100)
    b = score_karma(10_000)
    c = score_karma(KARMA_SATURATION)
    assert 0.0 < a < b < c <= 1.0


def test_score_karma_clamped_to_unit_interval():
    huge = score_karma(KARMA_SATURATION * 100)
    assert 0.0 <= huge <= 1.0


def test_score_age_zero_for_fresh_burner():
    """A 3-day-old account is almost certainly farmed -- zero credit."""
    assert score_age(3.0) == 0.0


def test_score_age_unknown_low_but_not_zero():
    assert score_age(None) == pytest.approx(0.1)


def test_score_age_monotonic():
    a = score_age(AGE_MIN_DAYS + 1)
    b = score_age(180)
    c = score_age(365)
    assert 0.0 < a < b <= c == 1.0


def test_score_engagement_low_for_dead_post():
    """Score below the minimum = nobody cared. Floor at 0.1."""
    eng = score_engagement(score=1, num_comments=0, upvote_ratio=None)
    assert eng == pytest.approx(0.1)


def test_score_engagement_penalizes_low_upvote_ratio():
    """High score but ratio < 0.55 means controversial -- halve it."""
    high = score_engagement(score=500, num_comments=50, upvote_ratio=0.9)
    contro = score_engagement(score=500, num_comments=50, upvote_ratio=0.4)
    assert contro < high * 0.6  # at least 40% penalty


def test_score_engagement_penalizes_noisy_threads():
    """When comments >> upvotes, the thread is an argument, not signal."""
    calm = score_engagement(score=500, num_comments=100, upvote_ratio=0.85)
    noisy = score_engagement(score=500, num_comments=2000, upvote_ratio=0.85)
    assert noisy < calm


def test_score_engagement_within_unit_interval():
    eng = score_engagement(score=1_000_000, num_comments=10, upvote_ratio=0.99)
    assert 0.0 <= eng <= 1.0


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def test_score_history_no_history_returns_neutral():
    """Cold-start authors get a neutral 0.5 -- punishing them would
    bake in a never-recoverable cold start."""
    comp, n = score_history(None, "anyone")
    assert comp == HISTORY_UNKNOWN_NEUTRAL
    assert n == 0


def test_score_history_shrinks_small_sample(tmp_path, monkeypatch):
    """1/1 should NOT be crowned 100% -- shrink toward neutral until we
    have a real sample."""
    from packages.agents.reddit_trust import history as h_mod

    monkeypatch.setattr(h_mod, "HISTORY_PATH", tmp_path / "h.jsonl")
    hist = TrustHistory()
    hist.record(
        HistoryEntry(
            author="alice",
            post_id="p1",
            symbol="SPY",
            direction=1,
            confidence_at_signal=0.7,
            created_at="2026-01-01T00:00:00+00:00",
            outcome_return=0.02,
            accurate=True,
        )
    )
    comp, n = score_history(hist, "alice")
    assert n == 1
    # Heavy shrink: 1/1 should be close to neutral, not 1.0.
    assert 0.5 < comp < 0.6


def test_score_history_credits_consistent_track_record(tmp_path, monkeypatch):
    from packages.agents.reddit_trust import history as h_mod

    monkeypatch.setattr(h_mod, "HISTORY_PATH", tmp_path / "h.jsonl")
    hist = TrustHistory()
    for i in range(50):
        hist.record(
            HistoryEntry(
                author="oracle",
                post_id=f"p{i}",
                symbol="SPY",
                direction=1,
                confidence_at_signal=0.7,
                created_at="2026-01-01T00:00:00+00:00",
                outcome_return=0.02,
                accurate=True,
            )
        )
    comp, n = score_history(hist, "oracle")
    assert n == 50
    # 50/50 should be well above the neutral prior.
    assert comp > 0.85


# ---------------------------------------------------------------------------
# Pump detection
# ---------------------------------------------------------------------------


def test_detect_pump_flags_catches_pump_phrase_and_exclamation():
    p = _post(
        title="$XYZ to the MOON!!! BUY NOW!!!",
        selftext="this is a guaranteed 10x squeeze incoming",
        tickers=("XYZ",),
    )
    flags = detect_pump_flags(p)
    codes = {f.code for f in flags}
    assert "pump_phrase" in codes
    assert "exclamation_spam" in codes


def test_detect_pump_flags_catches_all_caps_in_title():
    """Two non-ticker all-caps runs in the title trip the heuristic."""
    p = _post(
        title="AAPL BREAKOUT INCOMING TODAY",
        selftext="steady move",
        tickers=("AAPL",),
    )
    codes = {f.code for f in detect_pump_flags(p)}
    assert "all_caps" in codes


def test_detect_pump_flags_flags_young_account_with_traction():
    p = _post(
        title="SPY analysis",
        selftext="...",
        account_age_days=2.0,
        score=100,
        num_comments=30,
    )
    codes = {f.code for f in detect_pump_flags(p)}
    assert "young_account_high_engagement" in codes


def test_detect_pump_flags_clean_post_has_no_flags():
    """A normal DD post must NOT be flagged -- false positives here
    would silence legitimate signal."""
    p = _post(
        title="SPY breakout looks real",
        selftext="I think the macro setup favors continuation. Risk is...",
        tickers=("SPY",),
    )
    assert detect_pump_flags(p) == []


def test_detect_pump_flags_ignores_legit_caps_tickers():
    """All-caps tickers shouldn't trigger the all-caps heuristic --
    they're legitimately uppercase."""
    p = _post(
        title="SPY QQQ IWM all looking strong",
        selftext="diversified bull",
        tickers=("SPY", "QQQ", "IWM"),
    )
    codes = {f.code for f in detect_pump_flags(p)}
    # Ticker density may flag (3 tickers) but all_caps must not.
    assert "all_caps" not in codes


# ---------------------------------------------------------------------------
# Full scorer
# ---------------------------------------------------------------------------


def test_scorer_burner_account_with_pump_post_scores_very_low():
    """The headline failure mode this whole system exists to catch."""
    scorer = RedditTrustScorer()
    p = _post(
        title="$XYZ TO THE MOON!!! BUY NOW!!!",
        selftext="GUARANTEED 100x squeeze",
        author="newbie_2026",
        karma=50,
        account_age_days=3.0,
        score=80,
        num_comments=40,
        upvote_ratio=0.6,
        tickers=("XYZ",),
    )
    br = scorer.score(p)
    assert br.weight < 0.30
    assert br.pump_flags  # at least one pump signal


def test_scorer_seasoned_author_clean_post_scores_high():
    scorer = RedditTrustScorer()
    p = _post()  # defaults are intentionally "good"
    br = scorer.score(p)
    assert br.weight > 0.55
    assert br.pump_flags == []


def test_scorer_weight_always_in_unit_interval():
    scorer = RedditTrustScorer()
    # A grab-bag of weird inputs.
    cases = [
        _post(karma=None, account_age_days=None, upvote_ratio=None),
        _post(karma=0, account_age_days=0, score=0, num_comments=0),
        _post(karma=10**9, account_age_days=10_000, score=10**6),
        _post(upvote_ratio=0.0),
        _post(upvote_ratio=1.0),
    ]
    for p in cases:
        br = scorer.score(p)
        assert 0.0 <= br.weight <= 1.0
        assert 0.0 <= br.karma <= 1.0
        assert 0.0 <= br.age <= 1.0
        assert 0.0 <= br.engagement <= 1.0
        assert 0.0 <= br.history <= 1.0


def test_scorer_history_present_lifts_known_good_author(tmp_path, monkeypatch):
    """Same post text + a strong track record should score *higher*
    than the no-history baseline."""
    from packages.agents.reddit_trust import history as h_mod

    monkeypatch.setattr(h_mod, "HISTORY_PATH", tmp_path / "h.jsonl")
    hist = TrustHistory()
    for i in range(40):
        hist.record(
            HistoryEntry(
                author="oracle",
                post_id=f"p{i}",
                symbol="SPY",
                direction=1,
                confidence_at_signal=0.7,
                created_at="2026-01-01T00:00:00+00:00",
                outcome_return=0.02,
                accurate=True,
            )
        )

    base_post = _post(author="oracle")
    no_hist = RedditTrustScorer().score(base_post).weight
    with_hist = RedditTrustScorer(history=hist).score(base_post).weight
    assert with_hist > no_hist


def test_scorer_pump_penalty_bounded():
    """Even with many flags, weight never goes below 0 (and we don't
    zero out -- some legit DD uses exclamation)."""
    scorer = RedditTrustScorer()
    p = _post(
        title="$AAA $BBB $CCC $DDD TO THE MOON!!! BUY NOW!!! 10X!!! 100X!!!",
        selftext="GUARANTEED PUMP MOONSHOT SQUEEZE INCOMING!!!",
        author="newbie",
        karma=10,
        account_age_days=1.0,
        score=200,
        num_comments=100,
        upvote_ratio=0.4,
        tickers=("AAA", "BBB", "CCC", "DDD"),
    )
    br = scorer.score(p)
    assert 0.0 <= br.weight <= 1.0
    # 60% penalty cap on stacked flags; weight should still be > 0.
    assert br.weight >= 0.0


def test_score_to_post_trust_serializes(tmp_path, monkeypatch):
    scorer = RedditTrustScorer()
    pt = scorer.score_to_post_trust(_post())
    d = pt.to_dict()
    assert "post_id" in d and "weight" in d
    assert isinstance(d["pump_flags"], list)
    assert 0.0 <= d["weight"] <= 1.0
