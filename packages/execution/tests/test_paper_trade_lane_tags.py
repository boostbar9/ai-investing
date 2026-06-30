"""Tests for ledger lane/catalyst tagging in tools.paper_trade.load_lane_tags.

The under-the-radar lane stamps each shadow order with the lane + catalyst
that surfaced its symbol, so performance_stats can split win rate / profit
factor BY LANE. This is READ-ONLY (reads the persisted research-sweep) and
fail-safe (any missing/corrupt sweep yields ``{}`` -> no tag -> 'unknown' lane).
"""

from __future__ import annotations

import tools.paper_trade as pt


def test_load_lane_tags_maps_symbol_to_lane_and_catalyst(monkeypatch):
    sweep = {
        "candidates": [
            {
                "symbol": "bcrx",
                "lane": "Under_Radar",
                "catalyst_type": "FDA",
                "catalyst_score": 0.98,
            },
            {"symbol": "AAPL", "lane": "mainstream", "catalyst_type": "none"},
        ]
    }
    monkeypatch.setattr(
        "packages.agents.research_sweep.load_sweep", lambda *a, **k: sweep
    )
    tags = pt.load_lane_tags()
    assert tags["BCRX"] == {
        "lane": "under_radar",
        "catalyst_type": "fda",
        "catalyst_score": 0.98,
    }
    assert tags["AAPL"]["lane"] == "mainstream"


def test_load_lane_tags_failsafe_on_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("corrupt")

    monkeypatch.setattr("packages.agents.research_sweep.load_sweep", boom)
    assert pt.load_lane_tags() == {}


def test_load_lane_tags_failsafe_on_non_dict(monkeypatch):
    monkeypatch.setattr(
        "packages.agents.research_sweep.load_sweep", lambda *a, **k: None
    )
    assert pt.load_lane_tags() == {}


def test_load_lane_tags_skips_rows_without_symbol(monkeypatch):
    sweep = {"candidates": [{"lane": "under_radar"}, {"symbol": "", "lane": "x"}]}
    monkeypatch.setattr(
        "packages.agents.research_sweep.load_sweep", lambda *a, **k: sweep
    )
    assert pt.load_lane_tags() == {}
