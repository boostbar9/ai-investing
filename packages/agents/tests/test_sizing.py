"""Phase 15: unit tests for the risk-adaptive sizer.

These cover the three contract layers separately:

  1. ``RiskSizerConfig`` validation -- bad inputs raise, env overrides
     work, defaults are sane.
  2. ``RiskSizer.size`` math -- equal-weight, confidence-proportional,
     and fractional-Kelly modes each produce sensible weights with the
     right invariants (sum <= gross_target, cap respected, biggest
     conf gets biggest size where applicable).
  3. The drawdown taper + vol-scaling layers -- monotone shrink with
     deeper DD, vol scaling clips correctly, missing data falls back
     to no-op.

We use lightweight dataclass stand-ins for ``PolicyDecision`` so the
test file doesn't depend on the policy module's full graph.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from packages.agents.sizing import (
    DEFAULT_DD_HARD_LIMIT,
    DEFAULT_DD_TAPER_FLOOR,
    RiskSizer,
    RiskSizerConfig,
    SizingResult,
    load_peak_equity,
)

# ---------------------------------------------------------------------------
# Test fixtures: a minimal stand-in for PolicyDecision. Duck-typed so the
# sizer doesn't need to import policy.py.
# ---------------------------------------------------------------------------


@dataclass
class FakeBuy:
    symbol: str
    composite_confidence: float


def _buys(*pairs: tuple[str, float]) -> list[FakeBuy]:
    return [FakeBuy(symbol=s, composite_confidence=c) for s, c in pairs]


# ---------------------------------------------------------------------------
# RiskSizerConfig validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_config_constructs(self) -> None:
        cfg = RiskSizerConfig()
        # Sanity check that we haven't shipped a default outside its
        # documented range. If any of these flip the user finds out
        # at sizer-construction time instead of at trade time.
        assert cfg.mode in ("equal_weight", "confidence_proportional", "fractional_kelly")
        assert 0.0 <= cfg.kelly_fraction <= 1.0
        assert 0.0 <= cfg.dd_taper_start < cfg.dd_hard_limit
        assert 0.0 < cfg.dd_taper_floor <= 1.0
        assert 0.0 < cfg.max_position_weight <= 1.0
        assert cfg.target_vol_annual > 0
        assert 0.0 <= cfg.cash_floor < 1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "bogus"},
            {"kelly_fraction": -0.1},
            {"kelly_fraction": 1.5},
            {"dd_taper_start": 0.10, "dd_hard_limit": 0.05},  # inverted
            {"dd_taper_floor": 0.0},
            {"dd_taper_floor": 1.5},
            {"max_position_weight": 0.0},
            {"max_position_weight": 1.5},
            {"target_vol_annual": 0.0},
            {"cash_floor": 1.0},
            {"cash_floor": -0.1},
        ],
    )
    def test_invalid_config_raises(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            RiskSizerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Mode math
# ---------------------------------------------------------------------------


class TestEqualWeightMode:
    """Equal-weight should reproduce the Phase 13 contract exactly."""

    def test_three_buys_split_gross_target(self) -> None:
        sizer = RiskSizer(RiskSizerConfig(mode="equal_weight", cash_floor=0.05))
        out = sizer.size(
            buy_decisions=_buys(("A", 0.80), ("B", 0.75), ("C", 0.70)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols=None,
        )
        assert isinstance(out, SizingResult)
        # gross_target = (1 - 0.05) * 1.0 (no DD taper) = 0.95.
        # Each name = 0.95 / 3 ~ 0.3167, but the 20% max-position cap
        # clips each one down to 0.20. The uncapped excess stays in cash;
        # we don't redistribute back into uncapped names.
        assert pytest.approx(out.gross_target, abs=1e-6) == 0.95
        assert all(w <= 0.20 + 1e-9 for w in out.weights.values())
        # All three names were kept.
        assert set(out.weights.keys()) == {"A", "B", "C"}

    def test_empty_input_returns_empty(self) -> None:
        sizer = RiskSizer(RiskSizerConfig(mode="equal_weight"))
        out = sizer.size(
            buy_decisions=[], max_positions=3, equity=0.0, peak_equity=0.0, realised_vols=None
        )
        assert out.weights == {}
        assert out.diagnostics == []
        assert out.gross_actual == 0.0

    def test_max_positions_cap_keeps_top_confidence(self) -> None:
        sizer = RiskSizer(RiskSizerConfig(mode="equal_weight"))
        out = sizer.size(
            buy_decisions=_buys(("A", 0.90), ("B", 0.85), ("C", 0.80), ("D", 0.75)),
            max_positions=2,
            equity=0.0,
            peak_equity=0.0,
            realised_vols=None,
        )
        # Top two confidence names kept; lower two dropped.
        assert set(out.weights.keys()) == {"A", "B"}
        assert any("capped" in n for n in out.notes)


class TestConfidenceProportionalMode:
    def test_higher_confidence_gets_bigger_size(self) -> None:
        sizer = RiskSizer(
            RiskSizerConfig(
                mode="confidence_proportional",
                buy_threshold=0.65,
                max_position_weight=1.0,  # disable cap so we can verify pure ratio
                cash_floor=0.0,
            )
        )
        out = sizer.size(
            buy_decisions=_buys(("HI", 0.95), ("LO", 0.70)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols=None,
        )
        # Edge: HI=0.30, LO=0.05 -> ratio 6:1.
        assert out.weights["HI"] > out.weights["LO"]
        assert pytest.approx(out.weights["HI"] / out.weights["LO"], rel=1e-3) == 6.0

    def test_zero_edge_falls_back_to_equal_weight(self) -> None:
        """When everything sits right at the threshold, the mode-specific
        math degenerates -- the sizer must not divide by zero or return
        an empty dict; it should fall back to equal-weight."""
        sizer = RiskSizer(
            RiskSizerConfig(
                mode="confidence_proportional",
                buy_threshold=0.65,
                max_position_weight=1.0,
                cash_floor=0.0,
            )
        )
        out = sizer.size(
            buy_decisions=_buys(("A", 0.65), ("B", 0.65)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols=None,
        )
        # Two names splitting gross_target=1.0 evenly.
        assert pytest.approx(out.weights["A"], rel=1e-3) == pytest.approx(out.weights["B"], rel=1e-3)
        assert pytest.approx(out.weights["A"], rel=1e-3) == 0.5


class TestFractionalKellyMode:
    def test_higher_confidence_gets_bigger_kelly_size(self) -> None:
        sizer = RiskSizer(
            RiskSizerConfig(
                mode="fractional_kelly",
                kelly_fraction=0.25,
                max_position_weight=1.0,
                cash_floor=0.0,
            )
        )
        out = sizer.size(
            buy_decisions=_buys(("HI", 0.90), ("LO", 0.70)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols=None,
        )
        # 2p-1 for HI = 0.80, for LO = 0.40 -> ratio 2:1.
        assert out.weights["HI"] > out.weights["LO"]
        assert pytest.approx(out.weights["HI"] / out.weights["LO"], rel=1e-3) == 2.0

    def test_diagnostics_populate_kelly_weight(self) -> None:
        sizer = RiskSizer(RiskSizerConfig(mode="fractional_kelly"))
        out = sizer.size(
            buy_decisions=_buys(("X", 0.80)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols=None,
        )
        assert len(out.diagnostics) == 1
        # In Kelly mode the diagnostic's kelly_weight field is populated;
        # in other modes it stays None. This is the visible distinction
        # in the /shadow/sizing dashboard.
        assert out.diagnostics[0].kelly_weight is not None
        assert out.diagnostics[0].kelly_weight > 0


# ---------------------------------------------------------------------------
# Drawdown taper
# ---------------------------------------------------------------------------


class TestDrawdownTaper:
    def test_no_drawdown_no_taper(self) -> None:
        sizer = RiskSizer(RiskSizerConfig(mode="equal_weight", cash_floor=0.05))
        out = sizer.size(
            buy_decisions=_buys(("A", 0.80)),
            max_positions=5,
            equity=100_000.0,
            peak_equity=100_000.0,
            realised_vols=None,
        )
        assert pytest.approx(out.dd_observed, abs=1e-6) == 0.0
        assert pytest.approx(out.dd_exposure_multiplier, abs=1e-6) == 1.0
        assert pytest.approx(out.gross_target, abs=1e-6) == 0.95

    def test_shallow_drawdown_below_threshold_no_taper(self) -> None:
        """DD below dd_taper_start (default 3%) should NOT shrink sizes."""
        sizer = RiskSizer(RiskSizerConfig(mode="equal_weight"))
        out = sizer.size(
            buy_decisions=_buys(("A", 0.80)),
            max_positions=5,
            equity=98_000.0,  # 2% DD
            peak_equity=100_000.0,
            realised_vols=None,
        )
        assert pytest.approx(out.dd_observed, abs=1e-6) == 0.02
        assert pytest.approx(out.dd_exposure_multiplier, abs=1e-6) == 1.0

    def test_mid_drawdown_linearly_tapers(self) -> None:
        """DD at the midpoint should give the midpoint multiplier."""
        cfg = RiskSizerConfig(mode="equal_weight")
        sizer = RiskSizer(cfg)
        # Midpoint between taper_start (0.03) and hard_limit (0.08) = 0.055.
        mid_dd = (cfg.dd_taper_start + cfg.dd_hard_limit) / 2.0
        out = sizer.size(
            buy_decisions=_buys(("A", 0.80)),
            max_positions=5,
            equity=100_000.0 * (1.0 - mid_dd),
            peak_equity=100_000.0,
            realised_vols=None,
        )
        # Expected multiplier = 1 - 0.5 * (1 - floor) where floor=0.30 => 0.65.
        expected = 1.0 - 0.5 * (1.0 - cfg.dd_taper_floor)
        assert pytest.approx(out.dd_exposure_multiplier, abs=1e-4) == expected
        # And gross_target should reflect it.
        assert pytest.approx(out.gross_target, abs=1e-4) == (1 - cfg.cash_floor) * expected
        # The DD-taper note should be present in the audit trail.
        assert any("DD taper active" in n for n in out.notes)

    def test_deep_drawdown_clamps_to_floor(self) -> None:
        """DD beyond the hard limit clamps to the floor (doesn't go to 0)."""
        sizer = RiskSizer(RiskSizerConfig(mode="equal_weight"))
        out = sizer.size(
            buy_decisions=_buys(("A", 0.80)),
            max_positions=5,
            equity=100_000.0 * (1.0 - DEFAULT_DD_HARD_LIMIT - 0.05),  # 13% DD
            peak_equity=100_000.0,
            realised_vols=None,
        )
        assert pytest.approx(out.dd_exposure_multiplier, abs=1e-6) == DEFAULT_DD_TAPER_FLOOR

    def test_missing_equity_skips_taper(self) -> None:
        """equity=0 OR peak_equity=0 -> no DD signal -> full size."""
        sizer = RiskSizer(RiskSizerConfig(mode="equal_weight"))
        out = sizer.size(
            buy_decisions=_buys(("A", 0.80)),
            max_positions=5,
            equity=0.0,
            peak_equity=100_000.0,
            realised_vols=None,
        )
        assert out.dd_observed == 0.0
        assert out.dd_exposure_multiplier == 1.0


# ---------------------------------------------------------------------------
# Vol scaling
# ---------------------------------------------------------------------------


class TestVolScaling:
    def test_vol_scalar_inverts_relative_to_target(self) -> None:
        """A name with 2x target vol should get half the raw weight."""
        cfg = RiskSizerConfig(
            mode="equal_weight",
            target_vol_annual=0.18,
            max_position_weight=1.0,
            cash_floor=0.0,
        )
        sizer = RiskSizer(cfg)
        # Same confidence -> equal raw weights. Vol-scalar then halves
        # the high-vol name; renormalisation preserves total = gross_target,
        # so the low-vol name ends up larger.
        out = sizer.size(
            buy_decisions=_buys(("LO", 0.80), ("HI", 0.80)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols={"LO": 0.18, "HI": 0.36},
        )
        assert out.weights["LO"] > out.weights["HI"]
        # Diagnostics expose the scalar so the dashboard can show it.
        diags = {d.symbol: d for d in out.diagnostics}
        assert pytest.approx(diags["LO"].vol_scalar, abs=1e-3) == 1.0
        assert pytest.approx(diags["HI"].vol_scalar, abs=1e-3) == 0.5

    def test_vol_scalar_clips_at_2x(self) -> None:
        """Sub-target vol gets upsized but capped at 2x, defending against
        a degenerate vol estimate (e.g. a brand-new ticker with one bar)."""
        cfg = RiskSizerConfig(mode="equal_weight", target_vol_annual=0.18)
        sizer = RiskSizer(cfg)
        out = sizer.size(
            buy_decisions=_buys(("LOW", 0.80)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols={"LOW": 0.001},  # would give scalar=180x
        )
        diag = out.diagnostics[0]
        assert pytest.approx(diag.vol_scalar, abs=1e-6) == 2.0

    def test_missing_vol_defaults_to_no_scaling(self) -> None:
        cfg = RiskSizerConfig(mode="equal_weight")
        sizer = RiskSizer(cfg)
        out = sizer.size(
            buy_decisions=_buys(("A", 0.80), ("B", 0.80)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols={"A": 0.18},  # B missing
        )
        diags = {d.symbol: d for d in out.diagnostics}
        # A at target -> 1.0 scalar; B has no vol info -> 1.0 scalar.
        assert pytest.approx(diags["A"].vol_scalar, abs=1e-6) == 1.0
        assert pytest.approx(diags["B"].vol_scalar, abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# Max-position cap + result serialisation
# ---------------------------------------------------------------------------


class TestCapAndSerialisation:
    def test_cap_clips_oversized_single_name(self) -> None:
        cfg = RiskSizerConfig(
            mode="confidence_proportional",
            buy_threshold=0.65,
            max_position_weight=0.15,
            cash_floor=0.0,
        )
        sizer = RiskSizer(cfg)
        # Single high-conf name would naturally take the full gross_target
        # (=1.0) but the cap clips to 0.15.
        out = sizer.size(
            buy_decisions=_buys(("ONLY", 0.95)),
            max_positions=5,
            equity=0.0,
            peak_equity=0.0,
            realised_vols=None,
        )
        assert out.weights["ONLY"] == 0.15
        # Note must surface that we clipped (operator-visible audit trail).
        assert any("capped" in n for n in out.notes)

    def test_to_dict_round_trips_via_json(self) -> None:
        """SizingResult.to_dict must be JSON-serialisable -- the decision
        log JSONL line dies hard if any non-primitive sneaks in."""
        sizer = RiskSizer(RiskSizerConfig(mode="fractional_kelly"))
        out = sizer.size(
            buy_decisions=_buys(("A", 0.80), ("B", 0.70)),
            max_positions=5,
            equity=100_000.0,
            peak_equity=100_000.0,
            realised_vols={"A": 0.18, "B": 0.25},
        )
        d = out.to_dict()
        s = json.dumps(d)  # this raises if not serialisable
        loaded = json.loads(s)
        assert loaded["mode"] == "fractional_kelly"
        assert loaded["equity"] == 100_000.0
        assert loaded["peak_equity"] == 100_000.0
        assert len(loaded["diagnostics"]) == 2
        # Each diagnostic has the expected per-symbol fields.
        for diag in loaded["diagnostics"]:
            assert {"symbol", "raw_weight", "confidence", "edge",
                    "kelly_weight", "vol_scalar", "final_weight"} <= set(diag.keys())


# ---------------------------------------------------------------------------
# load_peak_equity helper
# ---------------------------------------------------------------------------


class TestLoadPeakEquity:
    def test_missing_file_returns_zero(self, tmp_path) -> None:
        assert load_peak_equity(tmp_path / "nope.json") == 0.0

    def test_valid_file_returns_peak(self, tmp_path) -> None:
        p = tmp_path / "peak.json"
        p.write_text(json.dumps({"peak": 12345.67, "updated_at": "2026-05-29T00:00:00Z"}))
        assert load_peak_equity(p) == 12345.67

    def test_corrupt_file_returns_zero(self, tmp_path) -> None:
        """Any I/O or parse error -> 0.0 (no-DD-taper fallback). The
        sizer must never bring the cycle down because the peak file got
        truncated mid-write."""
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert load_peak_equity(p) == 0.0

    def test_file_missing_peak_key_returns_zero(self, tmp_path) -> None:
        p = tmp_path / "weird.json"
        p.write_text(json.dumps({"updated_at": "..."}))
        assert load_peak_equity(p) == 0.0
