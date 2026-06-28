"""Phase 36a \u2014 Unified trading view.

Tests for the ``/api/trading/unified-snapshot`` endpoint that joins
Alpaca account state with the shadow decision ledger so the cockpit's
primary trading page can show one story per cycle.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv


def test_unified_snapshot_returns_documented_shape() -> None:
    """All eight top-level sections + errors map are always present."""
    client = TestClient(srv.app)
    r = client.get("/api/trading/unified-snapshot")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "account",
        "positions",
        "broker",
        "loop",
        "cycles",
        "decisions",
        "window",
        "streak",
        "errors",
    ):
        assert key in body, f"unified-snapshot missing key: {key}"


def test_unified_snapshot_broker_subset() -> None:
    """The broker block has the documented keys regardless of env state."""
    client = TestClient(srv.app)
    body = client.get("/api/trading/unified-snapshot").json()
    broker = body["broker"]
    assert broker is not None
    for key in ("keys_present", "reachable", "base_url"):
        assert key in broker


def test_unified_snapshot_decisions_limit_clamped() -> None:
    """``decisions_limit`` is clamped to [1, 50] so a hostile caller can't DoS."""
    client = TestClient(srv.app)
    # Below the floor \u2014 endpoint should still return 200 with a non-empty
    # list (if any decisions exist) or an empty list.
    r = client.get("/api/trading/unified-snapshot?decisions_limit=0")
    assert r.status_code == 200
    # Above the ceiling \u2014 clamped to 50, should not raise.
    r = client.get("/api/trading/unified-snapshot?decisions_limit=99999")
    assert r.status_code == 200


def test_unified_snapshot_best_effort_degradation(monkeypatch) -> None:
    """If one source raises, the response still returns 200 and other\n    sections are populated. We force ``compute_paper_streak`` to raise."""
    import packages.cockpit.web.server as srv_mod

    def boom(*a, **kw):
        raise RuntimeError("simulated streak failure")

    monkeypatch.setattr(srv_mod, "compute_paper_streak", boom)
    client = TestClient(srv.app)
    r = client.get("/api/trading/unified-snapshot")
    assert r.status_code == 200
    body = r.json()
    # streak should be None and errors should mention it.
    assert body["streak"] is None
    assert "streak" in body["errors"]
    # Other sections should still be populated.
    assert body["broker"] is not None


def test_trading_template_has_unified_panel() -> None:
    """trading.html embeds the Phase 36a unified panel + force-cycle button."""
    tpl = Path("packages/cockpit/web/templates/trading.html").read_text()
    # Force-cycle button moved onto the trading page.
    assert 'id="force-cycle-btn"' in tpl
    # Unified snapshot card hooks.
    assert 'id="u-equity"' in tpl
    assert 'id="u-bp"' in tpl
    assert 'id="u-positions"' in tpl
    assert 'id="u-window"' in tpl
    # Decisions table embedded.
    assert 'id="decisions-body"' in tpl
    # Poller wired.
    assert "refreshUnified" in tpl
    assert "/api/trading/unified-snapshot" in tpl


def test_shadow_template_reframed_as_research() -> None:
    """shadow.html is now framed as the research deep-dive surface."""
    tpl = Path("packages/cockpit/web/templates/shadow.html").read_text()
    # Title updated.
    assert "Research" in tpl
    # Page header explains the new role and links back to /trading.
    assert "research surface" in tpl or "research deep-dive" in tpl.lower()
    assert '/trading' in tpl


def test_nav_label_is_research_not_shadow() -> None:
    """Across the cockpit nav, the /shadow link is labeled Research now."""
    for template in (
        "index.html",
        "trading.html",
        "settings.html",
        "autopilot.html",
        "agents.html",
        "models.html",
        "health.html",
        "shadow.html",
    ):
        path = Path("packages/cockpit/web/templates") / template
        tpl = path.read_text()
        # The literal text ">Shadow</a>" should no longer appear as a
        # nav link (it may appear in comments or unrelated copy).
        assert '<a href="/shadow" class="nav-link">Shadow</a>' not in tpl, (
            f"Phase 36a regression: {template} still uses 'Shadow' as the "
            f"nav label; should be 'Research'."
        )
