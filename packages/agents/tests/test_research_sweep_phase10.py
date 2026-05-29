"""Tests for the Phase 10 enrichment helper.

Network paths (Yahoo / EDGAR / StockTwits / per-ticker Reddit) are
unit-tested in their adapter modules. This module focuses on the
*pure* enrichment function so we can assert candidate-shape changes
without exercising 4+ network mocks at once.
"""

from __future__ import annotations

from packages.agents.research_sweep import (
    Candidate,
    _apply_phase10_enrichment,
)


def _cand(symbol: str) -> Candidate:
    return Candidate(
        symbol=symbol,
        signal_kind="buy",
        thesis="x",
        confidence=0.5,
        sentiment_score=0.4,
        mentions=5,
        sources=["rss"],
        sample_headlines=["a"],
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_enrichment_with_no_signals_is_a_no_op():
    cands = [_cand("NVDA"), _cand("TSLA")]
    out = _apply_phase10_enrichment(
        cands,
        yahoo_signals={},
        insider_form4={},
        stocktwits_trending=[],
    )
    assert len(out) == 2
    for c in out:
        assert c.analyst_mean_rating == 0.0
        assert c.analyst_num == 0
        assert c.insider_form4_30d == 0
        assert c.stocktwits_trending is False


def test_enrichment_stamps_analyst_fields():
    cands = [_cand("NVDA")]
    out = _apply_phase10_enrichment(
        cands,
        yahoo_signals={
            "NVDA": {
                "analyst": {
                    "mean_rating": 1.6,
                    "num_analysts": 50,
                    "target_mean": 150.0,
                    "recent_action": "upgrade",
                    "recent_firm": "Morgan Stanley",
                },
                "insider": {},
                "news_count": 8,
            }
        },
        insider_form4={},
        stocktwits_trending=[],
    )
    c = out[0]
    assert c.analyst_mean_rating == 1.6
    assert c.analyst_num == 50
    assert c.analyst_target_mean == 150.0
    assert c.analyst_recent_action == "upgrade"
    assert c.analyst_recent_firm == "Morgan Stanley"
    assert c.yahoo_news_count == 8


def test_enrichment_stamps_insider_form4():
    out = _apply_phase10_enrichment(
        [_cand("NVDA")],
        yahoo_signals={
            "NVDA": {
                "analyst": {},
                "insider": {
                    "net_shares": 75_000,
                    "buy_count": 6,
                    "sell_count": 1,
                },
                "news_count": 0,
            }
        },
        insider_form4={
            "NVDA": {"count": 4, "latest": "2026-01-15", "cik": "..."},
        },
        stocktwits_trending=[],
    )
    c = out[0]
    assert c.insider_net_shares == 75_000
    assert c.insider_buy_count == 6
    assert c.insider_sell_count == 1
    assert c.insider_form4_30d == 4


def test_enrichment_marks_stocktwits_trending():
    out = _apply_phase10_enrichment(
        [_cand("NVDA"), _cand("BORING")],
        yahoo_signals={},
        insider_form4={},
        stocktwits_trending=[
            {
                "symbol": "NVDA",
                "title": "NVIDIA",
                "watchlist_count": 1_500_000,
            }
        ],
    )
    nvda = next(c for c in out if c.symbol == "NVDA")
    boring = next(c for c in out if c.symbol == "BORING")
    assert nvda.stocktwits_trending is True
    assert nvda.stocktwits_watchlist == 1_500_000
    assert boring.stocktwits_trending is False
    assert boring.stocktwits_watchlist == 0


def test_enrichment_is_case_insensitive_on_symbol():
    out = _apply_phase10_enrichment(
        [_cand("nvda")],
        yahoo_signals={
            "NVDA": {
                "analyst": {"mean_rating": 2.0},
                "insider": {},
                "news_count": 3,
            }
        },
        insider_form4={"NVDA": {"count": 2}},
        stocktwits_trending=[
            {"symbol": "NVDA", "watchlist_count": 1}
        ],
    )
    c = out[0]
    assert c.analyst_mean_rating == 2.0
    assert c.insider_form4_30d == 2
    assert c.stocktwits_trending is True


def test_enrichment_tolerates_malformed_signal_blobs():
    """Sweep gatherers can return partial dicts when an adapter half-
    fails. Enrichment must not crash on missing keys or wrong types."""
    out = _apply_phase10_enrichment(
        [_cand("NVDA")],
        yahoo_signals={
            "NVDA": {
                "analyst": None,    # weird but possible
                "insider": None,
                "news_count": None,
            }
        },
        insider_form4={"NVDA": None},  # type: ignore[dict-item]
        stocktwits_trending=[],
    )
    c = out[0]
    assert c.analyst_mean_rating == 0.0
    assert c.insider_net_shares == 0.0
    assert c.insider_form4_30d == 0
    assert c.yahoo_news_count == 0


def test_enrichment_preserves_existing_candidate_fields():
    cands = [_cand("NVDA")]
    original = cands[0].thesis
    out = _apply_phase10_enrichment(
        cands,
        yahoo_signals={"NVDA": {"analyst": {"mean_rating": 1.0}, "insider": {}, "news_count": 1}},
        insider_form4={},
        stocktwits_trending=[],
    )
    assert out[0].thesis == original
    assert out[0].confidence == 0.5
