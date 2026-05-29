"""Tests for the Phase 10 /data-sources cockpit page + API."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv


@pytest.fixture
def fake_sweep_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point the sweep file at a temp path so the API has fixture data
    to read.
    """
    from packages.agents import research_sweep as rs

    f = tmp_path / "research_sweep.json"
    f.write_text(
        json.dumps(
            {
                "status": "done",
                "started_at": "2026-05-28T12:00:00+00:00",
                "finished_at": "2026-05-28T12:00:42+00:00",
                "duration_s": 42.0,
                "portfolio_symbols": ["SPY"],
                "candidates": [
                    {"symbol": "NVDA", "thesis": "x"},
                    {"symbol": "TSLA", "thesis": "y"},
                ],
                "error": "",
                "sources_meta": {
                    "rss_news": {
                        "ok": True,
                        "count": 35,
                        "latency_ms": 240.0,
                    },
                    "yahoo_news": {
                        "ok": True,
                        "count": 16,
                        "latency_ms": 1100.0,
                    },
                    "reddit_rich": {
                        "ok": False,
                        "count": 0,
                        "latency_ms": 8000.0,
                    },
                    "stocktwits": {
                        "ok": True,
                        "count": 30,
                        "latency_ms": 90.0,
                    },
                    "sec_form4": {
                        "ok": True,
                        "count": 7,
                        "latency_ms": 1500.0,
                    },
                },
            }
        )
    )
    monkeypatch.setattr(rs, "SWEEP_PATH", f)
    return f


def test_data_sources_page_renders(fake_sweep_file):
    c = TestClient(srv.app)
    r = c.get("/data-sources")
    assert r.status_code == 200
    body = r.text.lower()
    # Page-specific markers — guards against regressions where a
    # template refactor accidentally points the route somewhere else.
    assert "data sources" in body
    assert "subreddit roster" in body
    assert "/api/data-sources/snapshot" in body


def test_data_sources_snapshot_includes_all_canonical_sources(
    fake_sweep_file,
):
    c = TestClient(srv.app)
    r = c.get("/api/data-sources/snapshot")
    assert r.status_code == 200
    data = r.json()
    names = [s["name"] for s in data["sources"]]
    # The six Phase-10 sources must all be present in canonical order.
    assert names[:6] == [
        "rss_news",
        "yahoo_news",
        "sec_form4",
        "reddit_rich",
        "reddit_per_ticker",
        "stocktwits",
    ]


def test_data_sources_snapshot_reflects_sweep_meta(fake_sweep_file):
    c = TestClient(srv.app)
    data = c.get("/api/data-sources/snapshot").json()
    by_name = {s["name"]: s for s in data["sources"]}
    assert by_name["rss_news"]["ok"] is True
    assert by_name["rss_news"]["count"] == 35
    assert by_name["reddit_rich"]["ok"] is False
    # Sources that did NOT appear in the sweep are marked not-present.
    assert by_name["reddit_per_ticker"]["present"] is False
    assert data["candidate_count"] == 2
    assert data["sweep_status"] == "done"


def test_data_sources_snapshot_includes_subreddit_roster(
    fake_sweep_file,
):
    c = TestClient(srv.app)
    data = c.get("/api/data-sources/snapshot").json()
    names = [r["name"] for r in data["subreddit_roster"]]
    assert "SecurityAnalysis" in names
    assert "wallstreetbets" in names
    # Each entry must have the multiplier the trust scorer will apply.
    sa = next(
        r for r in data["subreddit_roster"]
        if r["name"] == "SecurityAnalysis"
    )
    assert sa["multiplier"] == 1.0
    assert sa["tier"] == "high_quality"


def test_data_sources_snapshot_when_no_sweep_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The page must render an empty-but-valid response on first boot
    before any sweep has run."""
    from packages.agents import research_sweep as rs

    missing = tmp_path / "never_written.json"
    monkeypatch.setattr(rs, "SWEEP_PATH", missing)

    c = TestClient(srv.app)
    r = c.get("/api/data-sources/snapshot")
    assert r.status_code == 200
    data = r.json()
    assert data["sweep_status"] == ""
    assert data["candidate_count"] == 0
    # All canonical sources still listed, but with present=False.
    assert all(not s["present"] for s in data["sources"])
