"""Phase 35c — paper-trading is the cockpit's primary mode.

Verifies the two operator-visible defaults that this phase flipped:

1. ``POST /api/shadow/force-cycle`` defaults to ``dry_run=False`` so the
   cockpit's "Run one cycle now" button submits to Alpaca paper unless
   the operator explicitly asks for a dry-run preview.
2. ``GET /api/trading/broker-health`` reports key presence + Alpaca
   reachability so the UI can warn before the operator launches a loop.

We use ``inspect.signature`` to assert the route's default rather than
running a full cycle (which would require an event loop, broker keys,
market data, and a clean shadow ledger).
"""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv


def test_force_cycle_defaults_to_live_paper() -> None:
    """Phase 35c flipped the default to LIVE PAPER (``dry_run=False``)."""
    sig = inspect.signature(srv.api_shadow_force_cycle)
    assert sig.parameters["dry_run"].default is False, (
        "Phase 35c regression: /api/shadow/force-cycle must default to "
        "dry_run=False so the cockpit's primary mode is live Alpaca paper "
        "trading. Found default: "
        f"{sig.parameters['dry_run'].default!r}"
    )


def test_broker_health_reports_missing_keys(monkeypatch) -> None:
    """With no Alpaca keys in env, health endpoint reports keys_present=False."""
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    client = TestClient(srv.app)
    r = client.get("/api/trading/broker-health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["keys_present"] is False
    assert body["reachable"] is False
    assert "ALPACA_PAPER_KEY_ID" in (body["error"] or "")


def test_broker_health_reports_base_url(monkeypatch) -> None:
    """The base_url field reflects ALPACA_BASE_URL (or paper default)."""
    monkeypatch.delenv("ALPACA_PAPER_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_PAPER_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_BASE_URL", raising=False)
    client = TestClient(srv.app)
    body = client.get("/api/trading/broker-health").json()
    assert body["base_url"] == "https://paper-api.alpaca.markets"


def test_broker_health_shape() -> None:
    """Endpoint always returns the documented keys, even on failure."""
    client = TestClient(srv.app)
    body = client.get("/api/trading/broker-health").json()
    for key in ("ok", "keys_present", "reachable", "base_url", "error"):
        assert key in body, f"broker-health response missing key: {key}"


def test_shadow_template_defaults_to_live_paper() -> None:
    """The shadow.html button must not hardcode dry_run=true anymore.

    Phase 35c replaced the hardcoded URL with a reactive checkbox that
    posts the chosen mode. Regression: the literal substring
    ``dry_run=true`` (hardcoded in the URL) must no longer appear in the
    force-cycle handler.
    """
    from pathlib import Path

    tpl = Path("packages/cockpit/web/templates/shadow.html").read_text()
    # The old hardcoded URL was:
    #   /api/shadow/force-cycle?strategy=ensemble&dry_run=true
    # The new code interpolates from a checkbox: dry_run=${dryRun}
    assert "dry_run=true" not in tpl, (
        "Phase 35c regression: shadow.html still hardcodes dry_run=true. "
        "The force-cycle URL should interpolate the dry-run checkbox."
    )
    # Positive check: the new pattern is present.
    assert "force-cycle-dryrun" in tpl
    assert "dry_run=${dryRun}" in tpl


def test_trading_template_default_unchecked() -> None:
    """trading.html dry-run checkbox is unchecked by default (LIVE PAPER)."""
    from pathlib import Path

    tpl = Path("packages/cockpit/web/templates/trading.html").read_text()
    # The old line was: <input id="dry-run" type="checkbox" checked />
    # The new line drops the `checked` attribute.
    assert 'id="dry-run" type="checkbox" checked' not in tpl, (
        "Phase 35c regression: trading.html dry-run checkbox is checked "
        "by default. It should be unchecked so LIVE PAPER is the default."
    )
    # The mode badge should be present.
    assert "mode-badge" in tpl
    assert "LIVE PAPER" in tpl
