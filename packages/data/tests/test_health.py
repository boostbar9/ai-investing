"""Tests for the per-source health registry + toggle persistence."""

from __future__ import annotations

import pytest

from packages.data import health as health_mod
from packages.data.health import SourceRegistry


@pytest.fixture
def reg() -> SourceRegistry:
    return SourceRegistry()


@pytest.fixture(autouse=True)
def _isolated_toggles(tmp_path, monkeypatch):
    """Point toggle persistence at a temp file so tests never touch the
    real runtime store and stay independent of each other."""
    monkeypatch.setattr(
        health_mod, "_TOGGLE_PATH", tmp_path / "data_sources.json"
    )


def test_unseen_source_is_optimistic(reg):
    snap = reg.snapshot("never_seen")
    assert snap["status"] == "ok"          # nothing failed yet
    assert snap["success_rate"] == 1.0
    assert snap["last_success_ts"] is None
    assert snap["enabled"] is True


def test_success_recorded(reg):
    reg.record_attempt("finnhub")
    reg.record_success("finnhub", latency_ms=12.5)
    snap = reg.snapshot("finnhub")
    assert snap["status"] == "ok"
    assert snap["total_successes"] == 1
    assert snap["last_latency_ms"] == 12.5
    assert snap["last_success_ts"] is not None


def test_three_failures_marks_down(reg):
    for _ in range(3):
        reg.record_failure("reddit", "HTTP 429")
    snap = reg.snapshot("reddit")
    assert snap["status"] == "down"
    assert snap["consecutive_failures"] == 3
    assert snap["last_error"] == "HTTP 429"


def test_single_failure_marks_degraded(reg):
    reg.record_success("yahoo_news")
    reg.record_failure("yahoo_news", "boom")
    assert reg.snapshot("yahoo_news")["status"] == "degraded"


def test_success_after_failures_recovers(reg):
    reg.record_failure("src", "x")
    reg.record_failure("src", "y")
    reg.record_success("src")
    snap = reg.snapshot("src")
    assert snap["consecutive_failures"] == 0
    assert snap["last_error"] is None


def test_error_is_redacted_in_registry(reg):
    reg.record_failure("finnhub", "GET ?token=supersecret failed")
    last = reg.snapshot("finnhub")["last_error"]
    assert "supersecret" not in last
    assert "token=***" in last


def test_success_rate_rolls(reg):
    for _ in range(8):
        reg.record_success("src")
    for _ in range(2):
        reg.record_failure("src", "e")
    # 8/10 = 0.8
    assert reg.snapshot("src")["success_rate"] == 0.8


def test_disabled_source_status(reg):
    health_mod.set_enabled("reddit", False)
    assert reg.snapshot("reddit")["status"] == "disabled"
    assert reg.snapshot("reddit")["enabled"] is False


# ---- toggle persistence (fail-safe default = enabled) ----


def test_default_enabled_when_no_file():
    assert health_mod.is_enabled("anything") is True


def test_set_enabled_persists():
    health_mod.set_enabled("reddit", False)
    assert health_mod.is_enabled("reddit") is False
    health_mod.set_enabled("reddit", True)
    assert health_mod.is_enabled("reddit") is True


def test_toggle_flips():
    assert health_mod.is_enabled("stocktwits") is True
    assert health_mod.toggle("stocktwits") is False
    assert health_mod.is_enabled("stocktwits") is False
    assert health_mod.toggle("stocktwits") is True


def test_corrupt_toggle_file_fails_safe(tmp_path, monkeypatch):
    p = tmp_path / "broken.json"
    p.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setattr(health_mod, "_TOGGLE_PATH", p)
    # Unreadable/corrupt store -> everything reads as enabled.
    assert health_mod.is_enabled("reddit") is True
