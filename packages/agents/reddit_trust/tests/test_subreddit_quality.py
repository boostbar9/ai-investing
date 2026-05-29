"""Tests for the Phase 10 tiered subreddit roster."""

from __future__ import annotations

from packages.agents.reddit_trust.subreddit_quality import (
    DEFAULT_SWEEP_ROSTER,
    SUBREDDIT_QUALITY,
    SubredditQuality,
    fetch_roster,
    quality_for,
)


def test_known_high_quality_subs_score_at_top():
    assert quality_for("SecurityAnalysis").multiplier == 1.0
    assert quality_for("ValueInvesting").multiplier == 1.0
    assert quality_for("investing").multiplier == 1.0


def test_known_wsb_subs_are_downweighted_but_not_zero():
    q = quality_for("wallstreetbets")
    assert 0.4 < q.multiplier < 0.7
    assert q.tier == "wsb"


def test_penny_subs_are_lowest():
    q = quality_for("pennystocks")
    assert q.multiplier <= 0.5
    assert q.tier == "penny"


def test_unknown_sub_falls_back_to_general_tier():
    q = quality_for("a_brand_new_sub_we_never_saw")
    assert q.multiplier == SubredditQuality(
        "x", 0.85, "general"
    ).multiplier
    assert q.tier == "general"


def test_quality_lookup_is_case_insensitive():
    a = quality_for("wallstreetbets")
    b = quality_for("WALLSTREETBETS")
    c = quality_for("WallStreetBets")
    assert a.multiplier == b.multiplier == c.multiplier


def test_empty_subreddit_name_does_not_crash():
    q = quality_for("")
    assert q.subreddit == ""
    assert q.multiplier > 0.0


def test_fetch_roster_is_deduped_and_ordered():
    roster = fetch_roster()
    assert roster[0] == "SecurityAnalysis"
    assert "wallstreetbets" in roster
    # No duplicates.
    assert len(roster) == len({s.lower() for s in roster})


def test_fetch_roster_appends_extras_after_defaults():
    roster = fetch_roster(["NVDA_Stock", "investing"])
    # The duplicate "investing" must NOT appear twice.
    assert sum(1 for s in roster if s.lower() == "investing") == 1
    # The new per-ticker sub appears after the static roster.
    nvda_idx = roster.index("NVDA_Stock")
    investing_idx = next(
        i for i, s in enumerate(roster) if s.lower() == "investing"
    )
    assert nvda_idx > investing_idx


def test_all_default_roster_subs_have_quality_entries():
    # Every sub the default sweep hits must have a documented quality
    # tier — fall-through to the general default is fine for unknown
    # subs but is a smell for ones we've explicitly listed.
    for sub in DEFAULT_SWEEP_ROSTER:
        q = quality_for(sub)
        assert 0.4 <= q.multiplier <= 1.0, sub


def test_all_quality_multipliers_are_bounded():
    for sub, mult in SUBREDDIT_QUALITY.items():
        assert 0.4 <= mult <= 1.0, f"{sub} out of bounds: {mult}"
