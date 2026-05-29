"""Cockpit /shadow page + /api/shadow/snapshot integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from packages.execution import robinhood as rh_mod
from packages.shadow import greenlight as gl_mod
from packages.shadow import notify as notify_mod


@pytest.fixture
def isolated_shadow(monkeypatch, tmp_path) -> tuple[Path, Path]:
    """Point shadow trades, status, and flip-event log at tmp_path."""
    trades = tmp_path / "shadow_trades.jsonl"
    status = tmp_path / "shadow_status.json"
    flips = tmp_path / "shadow_flips.jsonl"
    monkeypatch.setattr(rh_mod, "SHADOW_TRADES_PATH", trades)
    monkeypatch.setattr(gl_mod, "STATUS_PATH", status)
    monkeypatch.setattr(notify_mod, "FLIPS_PATH", flips)
    return trades, status


def test_shadow_page_renders() -> None:
    client = TestClient(srv.app)
    r = client.get("/shadow")
    assert r.status_code == 200
    body = r.text
    assert "Shadow Trading" in body
    assert "/api/shadow/snapshot" in body
    assert "14" in body  # default days required


def test_api_shadow_snapshot_empty(isolated_shadow) -> None:
    client = TestClient(srv.app)
    r = client.get("/api/shadow/snapshot")
    assert r.status_code == 200
    payload = r.json()
    assert payload["n_round_trips"] == 0
    assert payload["total_pnl"] == 0.0
    assert payload["greenlight"]["status"] == "shadow"
    assert payload["days_required"] == 14


def test_api_shadow_snapshot_with_round_trips(isolated_shadow) -> None:
    trades, _ = isolated_shadow
    import json

    trades.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            "ts": "2026-05-01T10:00:00Z",
            "side": "buy",
            "symbol": "SPY",
            "qty": 10,
            "limit_price": 100.0,
        },
        {
            "ts": "2026-05-02T10:00:00Z",
            "side": "sell",
            "symbol": "SPY",
            "qty": 10,
            "limit_price": 105.0,
        },
    ]
    trades.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    client = TestClient(srv.app)
    r = client.get("/api/shadow/snapshot")
    assert r.status_code == 200
    payload = r.json()
    assert payload["n_round_trips"] == 1
    assert payload["total_pnl"] == 50.0  # (105-100)*10
    assert len(payload["daily"]) == 1
    assert payload["daily"][0]["day"] == "2026-05-02"
    assert payload["daily"][0]["pnl"] == 50.0


def test_api_shadow_flip_events_empty(isolated_shadow) -> None:
    client = TestClient(srv.app)
    r = client.get("/api/shadow/flip-events")
    assert r.status_code == 200
    payload = r.json()
    assert payload["events"] == []
    assert payload["count"] == 0


def test_api_shadow_flip_events_records_after_greenlight(isolated_shadow) -> None:
    """Hitting /api/shadow/snapshot with 14 clean days flips the gate and logs it."""
    import json
    from datetime import date, timedelta

    trades, _ = isolated_shadow
    trades.parent.mkdir(parents=True, exist_ok=True)
    start = date(2026, 5, 1)
    lines: list[dict] = []
    for i in range(14):
        day = (start + timedelta(days=i)).isoformat()
        lines.append(
            {"ts": f"{day}T10:00:00Z", "side": "buy", "symbol": "SPY",
             "qty": 1, "limit_price": 100.0}
        )
        lines.append(
            {"ts": f"{day}T15:00:00Z", "side": "sell", "symbol": "SPY",
             "qty": 1, "limit_price": 101.0}
        )
    trades.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    client = TestClient(srv.app)
    snap = client.get("/api/shadow/snapshot").json()
    assert snap["greenlight"]["status"] == "ready"

    events = client.get("/api/shadow/flip-events").json()
    assert events["count"] == 1
    assert events["events"][0]["to"] == "ready"
    assert events["events"][0]["streak_days"] == 14


def test_api_promote_includes_shadow_gate(isolated_shadow) -> None:
    """/api/promote must include the shadow soak as a gating reason while still soaking."""
    client = TestClient(srv.app)
    r = client.get("/api/promote")
    assert r.status_code == 200
    payload = r.json()
    # Shadow not ready -> live_enabled must be False regardless of other gates.
    assert payload["live_enabled"] is False
    assert payload["progress"]["shadow_ready"] is False
    assert payload["progress"]["shadow_days_required"] == 14
    reasons = [str(r).lower() for r in payload["readiness"]["reasons"]]
    assert any("shadow soak" in r for r in reasons)


# ---------------------------------------------------------------------------
# Phase 11 — decision instrumentation endpoints.
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_decisions(monkeypatch, tmp_path) -> Path:
    """Point the decisions JSONL log at tmp_path."""
    from packages.paper import decisions as dec_mod

    p = tmp_path / "decisions.jsonl"
    monkeypatch.setattr(dec_mod, "DEFAULT_DECISIONS_PATH", p)
    return p


def _decision_row(
    *,
    ts: str = "2026-05-29T20:00:00+00:00",
    halted: bool = False,
    halt_reasons: list[str] | None = None,
    submitted: int = 0,
) -> dict:
    return {
        "ts": ts,
        "strategy": "ensemble",
        "dry_run": True,
        "halted": halted,
        "halt_reasons": halt_reasons or [],
        "pipeline": [
            {"name": "sweep_candidates", "count": 5, "sample_symbols": ["SPY", "QQQ"]},
            {"name": "corroborated", "count": 2, "sample_symbols": ["SPY"]},
            {"name": "agent_approved", "count": 1, "sample_symbols": ["SPY"]},
            {"name": "target_weighted", "count": 1, "sample_symbols": ["SPY"]},
            {"name": "orders_planned", "count": 1, "sample_symbols": ["SPY"]},
            {"name": "orders_submitted", "count": submitted, "sample_symbols": []},
        ],
        "planned_count": 1,
        "submitted_count": submitted,
        "error_count": 0,
        "account_equity": 100_000.0,
        "regime": "chop",
        "decision_id": "abc",
    }


def _write_decisions(path: Path, rows: list[dict]) -> None:
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")


def test_api_shadow_decisions_empty(isolated_decisions: Path) -> None:
    client = TestClient(srv.app)
    r = client.get("/api/shadow/decisions")
    assert r.status_code == 200
    payload = r.json()
    assert payload["decisions"] == []
    assert payload["count"] == 0
    assert payload["limit"] == 50


def test_api_shadow_decisions_returns_newest_first(isolated_decisions: Path) -> None:
    from datetime import UTC, datetime, timedelta

    base = datetime(2026, 5, 29, 12, 0, tzinfo=UTC)
    rows = [
        _decision_row(ts=(base + timedelta(minutes=i)).isoformat()) for i in range(3)
    ]
    # Tag each so we can verify ordering.
    for i, r in enumerate(rows):
        r["decision_id"] = f"id{i}"
    _write_decisions(isolated_decisions, rows)

    client = TestClient(srv.app)
    payload = client.get("/api/shadow/decisions").json()
    assert payload["count"] == 3
    assert [d["decision_id"] for d in payload["decisions"]] == ["id2", "id1", "id0"]


def test_api_shadow_decisions_caps_limit(isolated_decisions: Path) -> None:
    client = TestClient(srv.app)
    # 999 -> capped to 500.
    r = client.get("/api/shadow/decisions?limit=999")
    assert r.json()["limit"] == 500
    # 0 -> raised to 1.
    r = client.get("/api/shadow/decisions?limit=0")
    assert r.json()["limit"] == 1


def test_api_shadow_pipeline_empty_skeleton(isolated_decisions: Path) -> None:
    client = TestClient(srv.app)
    payload = client.get("/api/shadow/pipeline").json()
    assert payload == {
        "stages": [],
        "n_cycles": 0,
        "window_hours": 24,
        "halts": {},
    }


def test_api_shadow_pipeline_aggregates_canonical_stages(
    isolated_decisions: Path,
) -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    _write_decisions(isolated_decisions, [_decision_row(ts=now), _decision_row(ts=now)])
    client = TestClient(srv.app)
    payload = client.get("/api/shadow/pipeline").json()
    assert payload["n_cycles"] == 2
    names = [s["name"] for s in payload["stages"]]
    assert names == [
        "sweep_candidates",
        "corroborated",
        "agent_approved",
        "target_weighted",
        "orders_planned",
        "orders_submitted",
    ]
    sweep = next(s for s in payload["stages"] if s["name"] == "sweep_candidates")
    assert sweep["total"] == 10  # 5 * 2 cycles
    assert sweep["avg_per_cycle"] == 5.0


def test_api_shadow_pipeline_records_halt_tally(isolated_decisions: Path) -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    _write_decisions(
        isolated_decisions,
        [
            _decision_row(ts=now, halted=True, halt_reasons=["kill_switch:dd"]),
            _decision_row(ts=now, halted=True, halt_reasons=["cockpit_pause"]),
        ],
    )
    client = TestClient(srv.app)
    payload = client.get("/api/shadow/pipeline").json()
    assert payload["halts"].get("kill_switch") == 1
    assert payload["halts"].get("cockpit_pause") == 1


def test_api_shadow_window_empty_returns_target(isolated_decisions: Path) -> None:
    client = TestClient(srv.app)
    payload = client.get("/api/shadow/window").json()
    assert payload["target_days"] == 14
    assert payload["days_remaining"] == 14
    assert payload["days_with_activity"] == 0
    assert payload["grid"] == []


def test_api_shadow_window_dense_grid(isolated_decisions: Path) -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()
    _write_decisions(
        isolated_decisions,
        [_decision_row(ts=now, submitted=1)],
    )
    client = TestClient(srv.app)
    payload = client.get("/api/shadow/window").json()
    assert payload["days_with_activity"] == 1
    # 14-day window -> grid spans at least 14 days.
    assert len(payload["grid"]) >= 14
    today_str = datetime.now(UTC).date().isoformat()
    today_cell = next((c for c in payload["grid"] if c["day"] == today_str), None)
    assert today_cell is not None
    assert today_cell["cycles"] == 1
    assert today_cell["submitted"] == 1


def test_shadow_page_includes_phase11_panel_endpoints() -> None:
    """The /shadow template must reference the three new endpoints so
    its JS poller can render the new panels."""
    client = TestClient(srv.app)
    body = client.get("/shadow").text
    assert "/api/shadow/decisions" in body
    assert "/api/shadow/pipeline" in body
    assert "/api/shadow/window" in body


def test_api_promote_clears_shadow_gate_when_ready(isolated_shadow) -> None:
    """Pre-populating shadow_status.json with status=ready clears the soak gate."""
    _, status = isolated_shadow
    import json

    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(
        json.dumps(
            {
                "status": "ready",
                "streak_days": 14,
                "reasons": ["greenlit"],
                "last_evaluated_utc": "2026-05-28T19:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(srv.app)
    payload = client.get("/api/promote").json()
    assert payload["progress"]["shadow_ready"] is True
    reasons = [str(r).lower() for r in payload["readiness"]["reasons"]]
    assert not any("shadow soak" in r for r in reasons)


# ---------------------------------------------------------------------------
# Phase 12: /api/shadow/force-cycle -- the "Run one cycle now" button on the
# /shadow page POSTs here so the user can trigger a paper-trade cycle without
# waiting for the background loop.
# ---------------------------------------------------------------------------


def test_force_cycle_runs_paper_cycle_and_returns_summary(monkeypatch) -> None:
    """Happy path: the endpoint imports tools.paper_trade.run, awaits it,
    and returns the shaped summary the dashboard renders."""
    import sys
    import types

    called: dict[str, object] = {}

    async def fake_run(strategy_name: str, *, dry_run: bool = False, **_kw):
        called["strategy"] = strategy_name
        called["dry_run"] = dry_run
        return {
            "ts": "2026-05-29T15:00:00+00:00",
            "halted": False,
            "reasons": [],
            "orders_planned": 3,
            "orders_submitted": 0,
            "account_equity": 100_001.23,
        }

    # Inject a stub tools.paper_trade module so the endpoint's lazy
    # import sees our fake_run. Save+restore so we don't leak across
    # tests in the same session.
    fake_mod = types.ModuleType("tools.paper_trade")
    fake_mod.run = fake_run  # type: ignore[attr-defined]
    saved = sys.modules.get("tools.paper_trade")
    sys.modules["tools.paper_trade"] = fake_mod
    try:
        client = TestClient(srv.app)
        r = client.post("/api/shadow/force-cycle?strategy=ensemble&dry_run=true")
    finally:
        if saved is not None:
            sys.modules["tools.paper_trade"] = saved
        else:
            sys.modules.pop("tools.paper_trade", None)

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["strategy"] == "ensemble"
    assert body["dry_run"] is True
    assert body["halted"] is False
    assert body["planned"] == 3
    assert body["submitted"] == 0
    assert body["equity"] == pytest.approx(100_001.23)
    assert called == {"strategy": "ensemble", "dry_run": True}


def test_force_cycle_reports_halted_cycle(monkeypatch) -> None:
    """When the paper_trade cycle halts (e.g. kill-switch tripped), the
    endpoint must still return ok=True with halted=True + reasons so the
    dashboard can render the yellow status."""
    import sys
    import types

    async def fake_run(strategy_name: str, *, dry_run: bool = False, **_kw):
        return {
            "ts": "2026-05-29T15:01:00+00:00",
            "halted": True,
            "reasons": ["kill switch: SPY drawdown -3.2%"],
            "orders_planned": 0,
            "orders_submitted": 0,
            "account_equity": 99_000.0,
        }

    fake_mod = types.ModuleType("tools.paper_trade")
    fake_mod.run = fake_run  # type: ignore[attr-defined]
    saved = sys.modules.get("tools.paper_trade")
    sys.modules["tools.paper_trade"] = fake_mod
    try:
        client = TestClient(srv.app)
        r = client.post("/api/shadow/force-cycle")
    finally:
        if saved is not None:
            sys.modules["tools.paper_trade"] = saved
        else:
            sys.modules.pop("tools.paper_trade", None)

    body = r.json()
    assert r.status_code == 200
    assert body["ok"] is True
    assert body["halted"] is True
    assert "kill switch: SPY drawdown -3.2%" in body["reasons"]


def test_force_cycle_catches_paper_trade_exceptions(monkeypatch) -> None:
    """If the paper-trade cycle blows up, the endpoint MUST return
    ok=False with a readable error -- never a 500. The dashboard
    surfaces this inline next to the button."""
    import sys
    import types

    async def fake_run(strategy_name: str, *, dry_run: bool = False, **_kw):
        raise RuntimeError("alpaca down")

    fake_mod = types.ModuleType("tools.paper_trade")
    fake_mod.run = fake_run  # type: ignore[attr-defined]
    saved = sys.modules.get("tools.paper_trade")
    sys.modules["tools.paper_trade"] = fake_mod
    try:
        client = TestClient(srv.app)
        r = client.post("/api/shadow/force-cycle")
    finally:
        if saved is not None:
            sys.modules["tools.paper_trade"] = saved
        else:
            sys.modules.pop("tools.paper_trade", None)

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "RuntimeError" in body["error"]
    assert "alpaca down" in body["error"]


def test_force_cycle_single_flight_lock() -> None:
    """If a cycle is already in flight, a second request must bail with a
    friendly error rather than running a second concurrent cycle. We
    simulate this by acquiring the module lock directly."""
    import asyncio

    async def _acquire_then_call() -> tuple[int, dict]:
        async with srv._FORCE_CYCLE_LOCK:
            client = TestClient(srv.app)
            r = client.post("/api/shadow/force-cycle")
            return r.status_code, r.json()

    status, body = asyncio.run(_acquire_then_call())
    assert status == 200
    assert body["ok"] is False
    assert "already running" in body["error"].lower()


def test_shadow_page_contains_force_cycle_button() -> None:
    """The /shadow template must render the "Run one cycle now" button so
    the endpoint above is actually reachable from the dashboard."""
    client = TestClient(srv.app)
    body = client.get("/shadow").text
    assert "force-cycle-btn" in body
    assert "Run one cycle now" in body
    assert "/api/shadow/force-cycle" in body
