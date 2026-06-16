"""Tests for the intraday morning setup finder — Phase 28-R step 4."""

from __future__ import annotations

import pytest

from packages.intraday.setup_finder import (
    WEIGHTS,
    CandidateInput,
    SetupFinderResult,
    composite_score,
    find_morning_setups,
    is_intraday_mode_enabled,
    rank_candidates,
    score_insider_cluster,
    score_news_sentiment,
    score_orb_breakout,
    score_vwap_align,
)

# ---------------------------------------------------------------------------
# Pure scorers
# ---------------------------------------------------------------------------


class TestOrbBreakout:
    def test_below_orb_returns_zero(self) -> None:
        assert score_orb_breakout(99.0, 100.0) == 0.0

    def test_exactly_at_orb_returns_zero(self) -> None:
        assert score_orb_breakout(100.0, 100.0) == 0.0

    def test_small_breakout_ramps_from_zero(self) -> None:
        # 0.1% breakout — under the 0.25% threshold, so it should
        # ramp linearly between 0 and 0.2.
        s = score_orb_breakout(100.10, 100.0)
        assert 0.0 < s < 0.2

    def test_quarter_pct_breakout_hits_two_tenths(self) -> None:
        s = score_orb_breakout(100.25, 100.0)
        assert s == pytest.approx(0.2, abs=1e-6)

    def test_one_and_a_half_pct_caps_at_one(self) -> None:
        s = score_orb_breakout(101.5, 100.0)
        assert s == pytest.approx(1.0)

    def test_runaway_breakout_still_caps(self) -> None:
        assert score_orb_breakout(120.0, 100.0) == 1.0

    def test_zero_or_negative_orb_returns_zero(self) -> None:
        assert score_orb_breakout(100.0, 0.0) == 0.0
        assert score_orb_breakout(100.0, -10.0) == 0.0


class TestVwapAlign:
    def test_at_or_below_vwap_is_zero(self) -> None:
        assert score_vwap_align(99.0, 100.0) == 0.0
        assert score_vwap_align(100.0, 100.0) == 0.0

    def test_half_pct_above_is_half(self) -> None:
        assert score_vwap_align(100.5, 100.0) == pytest.approx(0.5)

    def test_one_pct_above_caps_at_one(self) -> None:
        assert score_vwap_align(101.0, 100.0) == pytest.approx(1.0)

    def test_two_pct_above_still_caps(self) -> None:
        assert score_vwap_align(102.0, 100.0) == pytest.approx(1.0)


class TestNewsSentiment:
    def test_none_score_is_zero(self) -> None:
        assert score_news_sentiment(None, 0) == 0.0

    def test_no_headlines_is_zero(self) -> None:
        assert score_news_sentiment(0.5, 0) == 0.0

    def test_neutral_with_one_headline(self) -> None:
        # sentiment 0 -> mapped 0.5; n=1 -> confidence 0.5; product 0.25.
        assert score_news_sentiment(0.0, 1) == pytest.approx(0.25)

    def test_positive_with_many_headlines(self) -> None:
        # sentiment +1 -> mapped 1.0; n=5 -> confidence 1.0.
        assert score_news_sentiment(1.0, 5) == pytest.approx(1.0)

    def test_negative_clips_to_zero_floor(self) -> None:
        # sentiment -1 maps to 0; confidence multiplies through 0.
        assert score_news_sentiment(-1.0, 3) == 0.0


class TestInsiderCluster:
    def test_none_is_zero(self) -> None:
        assert score_insider_cluster(None, 0.5) == 0.0

    def test_zero_confidence_is_zero(self) -> None:
        assert score_insider_cluster(1.0, 0.0) == 0.0

    def test_full_signal_and_confidence(self) -> None:
        assert score_insider_cluster(1.0, 1.0) == 1.0

    def test_neutral_score_half(self) -> None:
        # 0 maps to 0.5; * 1.0 conf = 0.5
        assert score_insider_cluster(0.0, 1.0) == pytest.approx(0.5)


class TestCompositeScore:
    def test_zero_components(self) -> None:
        assert composite_score(
            {
                "orb_breakout": 0.0,
                "vwap_align": 0.0,
                "news_sentiment": 0.0,
                "insider_cluster": 0.0,
            }
        ) == 0.0

    def test_full_components_sum_to_one(self) -> None:
        # WEIGHTS sum to 1.0, so all-ones composite = 1.0
        assert composite_score(
            dict.fromkeys(WEIGHTS, 1.0)
        ) == pytest.approx(1.0)

    def test_only_orb_weight(self) -> None:
        s = composite_score(
            {
                "orb_breakout": 1.0,
                "vwap_align": 0.0,
                "news_sentiment": 0.0,
                "insider_cluster": 0.0,
            }
        )
        assert s == pytest.approx(WEIGHTS["orb_breakout"])

    def test_weights_are_normalized(self) -> None:
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------


def _good_candidate(
    symbol: str = "AAPL",
    close: float = 101.0,
    orb_high: float = 100.0,
    vwap: float = 100.5,
    adv_usd_20d: float = 5e9,
    news_score: float | None = 0.6,
    news_n: int = 4,
    insider_score: float | None = 0.4,
    insider_confidence: float = 0.5,
) -> CandidateInput:
    return CandidateInput(
        symbol=symbol,
        close=close,
        orb_high=orb_high,
        vwap=vwap,
        adv_usd_20d=adv_usd_20d,
        news_score=news_score,
        news_n=news_n,
        insider_score=insider_score,
        insider_confidence=insider_confidence,
    )


class TestRankCandidates:
    def test_empty_universe(self) -> None:
        setups, rejected = rank_candidates([], equity_usd=100_000)
        assert setups == []
        assert rejected == []

    def test_single_passing_candidate(self) -> None:
        setups, rejected = rank_candidates(
            [_good_candidate()], equity_usd=100_000
        )
        assert len(setups) == 1
        assert setups[0].symbol == "AAPL"
        # 1% of 100k = 1000 capped at $300 -> $300 / 1 slot = $300.
        assert setups[0].notional_usd == pytest.approx(300.0)
        assert rejected == []

    def test_illiquid_rejected(self) -> None:
        cand = _good_candidate(symbol="XYZ", adv_usd_20d=1e7)  # $10M
        setups, rejected = rank_candidates([cand], equity_usd=100_000)
        assert setups == []
        assert len(rejected) == 1
        assert rejected[0]["reason"] == "illiquid"
        assert rejected[0]["symbol"] == "XYZ"

    def test_no_price_signal_rejected(self) -> None:
        # close <= ORB -> ORB score is 0 -> price gate fails
        cand = _good_candidate(close=99.0, orb_high=100.0)
        setups, rejected = rank_candidates([cand], equity_usd=100_000)
        assert setups == []
        assert any(r["reason"] == "no_price_signal" for r in rejected)

    def test_vwap_below_rejected_even_with_orb(self) -> None:
        # ORB breakout fires but price below VWAP -> price gate fails
        cand = _good_candidate(close=100.5, orb_high=100.0, vwap=101.0)
        setups, rejected = rank_candidates([cand], equity_usd=100_000)
        assert setups == []
        assert any(r["reason"] == "no_price_signal" for r in rejected)

    def test_top_k_selected_by_score(self) -> None:
        # Build 5 candidates with descending news scores so ranking is
        # determined.
        candidates = [
            _good_candidate(symbol=sym, news_score=score, news_n=5)
            for sym, score in [
                ("AAA", 1.0),
                ("BBB", 0.8),
                ("CCC", 0.6),
                ("DDD", 0.4),
                ("EEE", 0.2),
            ]
        ]
        setups, _ = rank_candidates(
            candidates, equity_usd=100_000, top_k=3
        )
        assert [s.symbol for s in setups] == ["AAA", "BBB", "CCC"]

    def test_split_notional_across_k(self) -> None:
        candidates = [
            _good_candidate(symbol=f"S{i}", news_score=1.0 - i * 0.1, news_n=5)
            for i in range(3)
        ]
        setups, _ = rank_candidates(
            candidates, equity_usd=100_000, top_k=3
        )
        # min($300, 100k*1%=1000) = $300; split 3 ways -> $100 each.
        assert all(s.notional_usd == pytest.approx(100.0) for s in setups)

    def test_small_equity_uses_equity_floor(self) -> None:
        # 1% of $5000 = $50; floor is min($50, $300) = $50.
        # With 1 slot: $50 per slot, above $25 MIN_PER_POSITION -> 1 setup.
        setups, _ = rank_candidates(
            [_good_candidate()], equity_usd=5_000.0, top_k=1
        )
        assert len(setups) == 1
        assert setups[0].notional_usd == pytest.approx(50.0)

    def test_tiny_equity_falls_below_min_per_position(self) -> None:
        # 1% of $1000 = $10; split across 3 names -> $3.33 each, below
        # MIN_PER_POSITION_USD. Should return zero setups.
        candidates = [
            _good_candidate(symbol=f"S{i}", news_score=1.0)
            for i in range(3)
        ]
        setups, rejected = rank_candidates(
            candidates, equity_usd=1_000.0, top_k=3
        )
        assert setups == []
        assert all(
            r["reason"] == "sub_min_per_position" for r in rejected
        )

    def test_mixed_pass_and_fail(self) -> None:
        candidates = [
            _good_candidate(symbol="GOOD1", news_score=1.0, news_n=5),
            _good_candidate(symbol="ILLIQ", adv_usd_20d=1e6),
            _good_candidate(symbol="GOOD2", news_score=0.5, news_n=2),
            _good_candidate(symbol="NOORB", close=99.0, orb_high=100.0),
        ]
        setups, rejected = rank_candidates(
            candidates, equity_usd=100_000, top_k=3
        )
        symbols = {s.symbol for s in setups}
        assert "GOOD1" in symbols
        assert "GOOD2" in symbols
        assert "ILLIQ" not in symbols
        assert "NOORB" not in symbols
        rejected_reasons = {r["reason"] for r in rejected}
        assert "illiquid" in rejected_reasons
        assert "no_price_signal" in rejected_reasons

    def test_components_rounded(self) -> None:
        setups, _ = rank_candidates(
            [_good_candidate()], equity_usd=100_000
        )
        for v in setups[0].components.values():
            # Rounded to 4 decimal places.
            assert v == round(v, 4)

    def test_reason_string_includes_signals(self) -> None:
        setups, _ = rank_candidates(
            [_good_candidate(news_score=0.7, news_n=3, insider_score=0.6)],
            equity_usd=100_000,
        )
        r = setups[0].reason
        assert "ORB" in r
        assert "VWAP" in r
        assert "news" in r
        assert "insider" in r


# ---------------------------------------------------------------------------
# Orchestrator: find_morning_setups
# ---------------------------------------------------------------------------


class TestFindMorningSetups:
    def _price_lookup_factory(
        self, data: dict[str, dict[str, float]]
    ):
        def _lookup(symbol: str) -> dict[str, float] | None:
            return data.get(symbol)

        return _lookup

    def test_happy_path_with_all_providers(self) -> None:
        prices = {
            "AAPL": {
                "close": 200.0,
                "orb_high": 198.0,
                "vwap": 199.0,
                "adv_usd_20d": 5e10,
            },
            "MSFT": {
                "close": 400.0,
                "orb_high": 395.0,
                "vwap": 398.0,
                "adv_usd_20d": 5e10,
            },
        }
        sentiments = {"AAPL": (0.8, 5), "MSFT": (0.5, 3)}
        insiders = {"AAPL": (0.5, 0.6), "MSFT": (0.0, 0.0)}

        result = find_morning_setups(
            universe=["AAPL", "MSFT"],
            equity_usd=100_000,
            price_lookup=self._price_lookup_factory(prices),
            sentiment_lookup=lambda s: sentiments.get(s, (None, 0)),
            insider_lookup=lambda s: insiders.get(s, (None, 0.0)),
        )
        assert isinstance(result, SetupFinderResult)
        assert {s.symbol for s in result.setups} == {"AAPL", "MSFT"}
        # AAPL should rank first (better news + insider).
        assert result.setups[0].symbol == "AAPL"
        assert result.ts != ""

    def test_missing_price_bundle_rejected(self) -> None:
        result = find_morning_setups(
            universe=["AAPL", "MISSING"],
            equity_usd=100_000,
            price_lookup=self._price_lookup_factory(
                {
                    "AAPL": {
                        "close": 200.0,
                        "orb_high": 198.0,
                        "vwap": 199.0,
                        "adv_usd_20d": 5e10,
                    }
                }
            ),
        )
        assert [s.symbol for s in result.setups] == ["AAPL"]
        assert any(
            r["symbol"] == "MISSING" and r["reason"] == "no_bars"
            for r in result.rejected
        )

    def test_malformed_bars_rejected(self) -> None:
        prices = {
            "BAD": {
                "close": 200.0,
                "orb_high": "not_a_number",
                "vwap": 199.0,
                "adv_usd_20d": 5e10,
            }
        }
        result = find_morning_setups(
            universe=["BAD"],
            equity_usd=100_000,
            price_lookup=self._price_lookup_factory(prices),  # type: ignore[arg-type]
        )
        assert result.setups == []
        assert any(r["reason"] == "malformed_bars" for r in result.rejected)

    def test_symbols_uppercased(self) -> None:
        prices = {
            "AAPL": {
                "close": 200.0,
                "orb_high": 198.0,
                "vwap": 199.0,
                "adv_usd_20d": 5e10,
            }
        }
        result = find_morning_setups(
            universe=["aapl"],
            equity_usd=100_000,
            price_lookup=self._price_lookup_factory(prices),
        )
        assert result.setups[0].symbol == "AAPL"

    def test_no_providers_uses_price_only(self) -> None:
        prices = {
            "AAPL": {
                "close": 200.0,
                "orb_high": 198.0,
                "vwap": 199.0,
                "adv_usd_20d": 5e10,
            }
        }
        result = find_morning_setups(
            universe=["AAPL"],
            equity_usd=100_000,
            price_lookup=self._price_lookup_factory(prices),
        )
        assert len(result.setups) == 1
        # News + insider components zero out.
        assert result.setups[0].components["news_sentiment"] == 0.0
        assert result.setups[0].components["insider_cluster"] == 0.0

    def test_nan_bars_rejected(self) -> None:
        prices = {
            "NAN": {
                "close": float("nan"),
                "orb_high": 100.0,
                "vwap": 100.0,
                "adv_usd_20d": 5e10,
            }
        }
        result = find_morning_setups(
            universe=["NAN"],
            equity_usd=100_000,
            price_lookup=self._price_lookup_factory(prices),
        )
        assert result.setups == []
        assert any(r["reason"] == "nan_in_bars" for r in result.rejected)


# ---------------------------------------------------------------------------
# Mode flag
# ---------------------------------------------------------------------------


class TestIntradayMode:
    def test_default_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("INTRADAY_MODE", raising=False)
        assert is_intraday_mode_enabled() is False

    def test_explicit_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTRADAY_MODE", "0")
        assert is_intraday_mode_enabled() is False

    def test_explicit_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("INTRADAY_MODE", "1")
        assert is_intraday_mode_enabled() is True

    def test_arbitrary_value_treated_as_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("INTRADAY_MODE", "true")
        assert is_intraday_mode_enabled() is False
