"""Phase 36b — auto-start paper loop + log visibility.

Covers two operator-facing behaviors:

1. **First-boot auto-start** of the paper-trade loop. When the operator
   has never touched Start/Stop (``paper_loop_user_touched=False``),
   the env opt-in ``COCKPIT_AUTO_START_LOOP`` is on (default), and
   Alpaca paper keys are present, the cockpit boots straight into
   ``ensemble`` LIVE PAPER. Once the operator clicks Start or Stop
   (``paper_loop_user_touched=True``), future boots use the resume
   path instead.

2. **Operator-visible logs** on the /trading page. The existing
   ``/api/jobs/paper_loop/log`` endpoint backs the on-page tail view
   and a Download full log button. The template embeds a refresh
   button + SSE stream so logs are always visible.

We test the pure ``_decide_paper_loop_autostart`` decision function
rather than the FastAPI startup hook so the tests don't need a real
subprocess or job manager.
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from packages.cockpit import state as st
from packages.cockpit.state import CockpitState
from packages.cockpit.web import server as srv


# ---------------------------------------------------------------------------
# Pure decision function
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> CockpitState:
    s = CockpitState()
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def test_autostart_fires_on_first_boot_with_keys() -> None:
    """No prior intent + untouched + keys present → action=start."""
    cstate = _make_state(paper_loop_intended=False, paper_loop_user_touched=False)
    env = {
        "COCKPIT_AUTO_RESUME_LOOP": "1",
        "COCKPIT_AUTO_START_LOOP": "1",
        "ALPACA_PAPER_KEY_ID": "PK_TEST",
        "ALPACA_PAPER_SECRET": "SK_TEST",
    }
    decision = srv._decide_paper_loop_autostart(
        cstate, job_is_running=False, env=env
    )
    assert decision["action"] == "start"
    assert decision["strategy"] == "ensemble"
    assert decision["dry_run"] is False  # LIVE PAPER, not dry-run


def test_autostart_skipped_when_user_has_touched() -> None:
    """Once operator has clicked Start or Stop, no first-boot auto-start."""
    cstate = _make_state(paper_loop_intended=False, paper_loop_user_touched=True)
    env = {
        "COCKPIT_AUTO_RESUME_LOOP": "1",
        "COCKPIT_AUTO_START_LOOP": "1",
        "ALPACA_PAPER_KEY_ID": "PK_TEST",
        "ALPACA_PAPER_SECRET": "SK_TEST",
    }
    decision = srv._decide_paper_loop_autostart(
        cstate, job_is_running=False, env=env
    )
    assert decision["action"] == "skip"
    assert "touched" in decision["reason"]


def test_autostart_skipped_when_alpaca_keys_missing() -> None:
    """Without paper API keys we must not spawn a loop that would halt."""
    cstate = _make_state(paper_loop_intended=False, paper_loop_user_touched=False)
    env = {
        "COCKPIT_AUTO_RESUME_LOOP": "1",
        "COCKPIT_AUTO_START_LOOP": "1",
        # no ALPACA_PAPER_KEY_ID / SECRET
    }
    decision = srv._decide_paper_loop_autostart(
        cstate, job_is_running=False, env=env
    )
    assert decision["action"] == "skip"
    assert "alpaca" in decision["reason"].lower()


def test_autostart_skipped_when_env_opt_out() -> None:
    """COCKPIT_AUTO_START_LOOP=0 disables the Phase 36b first-boot path."""
    cstate = _make_state(paper_loop_intended=False, paper_loop_user_touched=False)
    env = {
        "COCKPIT_AUTO_RESUME_LOOP": "1",
        "COCKPIT_AUTO_START_LOOP": "0",
        "ALPACA_PAPER_KEY_ID": "PK_TEST",
        "ALPACA_PAPER_SECRET": "SK_TEST",
    }
    decision = srv._decide_paper_loop_autostart(
        cstate, job_is_running=False, env=env
    )
    assert decision["action"] == "skip"
    assert "AUTO_START_LOOP" in decision["reason"]


def test_resume_path_still_works_when_user_touched_and_intended() -> None:
    """If operator previously had it running, resume regardless of auto-start env."""
    cstate = _make_state(
        paper_loop_intended=True,
        paper_loop_user_touched=True,
        paper_loop_strategy="ensemble",
        paper_loop_dry_run=False,
    )
    env = {
        "COCKPIT_AUTO_RESUME_LOOP": "1",
        # Auto-start gate off — should not block resume path.
        "COCKPIT_AUTO_START_LOOP": "0",
    }
    decision = srv._decide_paper_loop_autostart(
        cstate, job_is_running=False, env=env
    )
    assert decision["action"] == "resume"
    assert decision["strategy"] == "ensemble"
    assert decision["dry_run"] is False


def test_autostart_skipped_when_already_running() -> None:
    cstate = _make_state(paper_loop_intended=False, paper_loop_user_touched=False)
    env = {
        "COCKPIT_AUTO_RESUME_LOOP": "1",
        "COCKPIT_AUTO_START_LOOP": "1",
        "ALPACA_PAPER_KEY_ID": "PK_TEST",
        "ALPACA_PAPER_SECRET": "SK_TEST",
    }
    decision = srv._decide_paper_loop_autostart(
        cstate, job_is_running=True, env=env
    )
    assert decision["action"] == "skip"
    assert "running" in decision["reason"]


def test_autostart_skipped_when_cockpit_paused() -> None:
    cstate = _make_state(
        paused=True, paper_loop_intended=False, paper_loop_user_touched=False
    )
    env = {
        "COCKPIT_AUTO_RESUME_LOOP": "1",
        "COCKPIT_AUTO_START_LOOP": "1",
        "ALPACA_PAPER_KEY_ID": "PK_TEST",
        "ALPACA_PAPER_SECRET": "SK_TEST",
    }
    decision = srv._decide_paper_loop_autostart(
        cstate, job_is_running=False, env=env
    )
    assert decision["action"] == "skip"
    assert "paused" in decision["reason"]


def test_autostart_skipped_when_master_resume_off() -> None:
    """COCKPIT_AUTO_RESUME_LOOP=0 disables everything (resume AND auto-start)."""
    cstate = _make_state(paper_loop_intended=True, paper_loop_user_touched=True)
    env = {
        "COCKPIT_AUTO_RESUME_LOOP": "0",
        "COCKPIT_AUTO_START_LOOP": "1",
        "ALPACA_PAPER_KEY_ID": "PK_TEST",
        "ALPACA_PAPER_SECRET": "SK_TEST",
    }
    decision = srv._decide_paper_loop_autostart(
        cstate, job_is_running=False, env=env
    )
    assert decision["action"] == "skip"


# ---------------------------------------------------------------------------
# State plumbing
# ---------------------------------------------------------------------------


def test_state_has_paper_loop_user_touched_default_false() -> None:
    """Fresh CockpitState has the field and it defaults to False."""
    s = CockpitState()
    assert hasattr(s, "paper_loop_user_touched")
    assert s.paper_loop_user_touched is False


def test_load_state_round_trips_paper_loop_user_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving with the field set + loading back preserves the value."""
    path = tmp_path / "state.json"
    monkeypatch.setattr(st, "STATE_PATH", path)
    monkeypatch.setattr(st.load_state, "__defaults__", (path,))
    monkeypatch.setattr(st.save_state, "__defaults__", (path,))

    s = st.load_state()
    assert s.paper_loop_user_touched is False
    s.paper_loop_user_touched = True
    st.save_state(s)
    s2 = st.load_state()
    assert s2.paper_loop_user_touched is True


def test_load_state_defaults_field_for_legacy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old state.json files without the field load with default False."""
    path = tmp_path / "state.json"
    # Write a legacy file missing the new field entirely.
    path.write_text('{"paper_loop_intended": true}')
    monkeypatch.setattr(st, "STATE_PATH", path)
    monkeypatch.setattr(st.load_state, "__defaults__", (path,))
    s = st.load_state()
    assert s.paper_loop_user_touched is False
    assert s.paper_loop_intended is True


# ---------------------------------------------------------------------------
# Start / stop endpoints flip paper_loop_user_touched
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect cockpit state writes/reads to a temp file."""
    path = tmp_path / "state.json"
    monkeypatch.setattr(st, "STATE_PATH", path)
    monkeypatch.setattr(st.load_state, "__defaults__", (path,))
    monkeypatch.setattr(st.save_state, "__defaults__", (path,))
    return path


def test_trading_start_marks_user_touched(
    fake_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/trading/start must set paper_loop_user_touched=True."""
    # Stub job_mgr.start so we don't actually spawn a subprocess.
    fake_info = types.SimpleNamespace(
        to_dict=lambda: {"running": True, "pid": 12345}
    )
    monkeypatch.setattr(srv.job_mgr, "start", lambda *a, **kw: fake_info)

    client = TestClient(srv.app)
    r = client.post(
        "/api/trading/start", json={"strategy": "ensemble", "dry_run": False}
    )
    assert r.status_code == 200
    s = st.load_state()
    assert s.paper_loop_user_touched is True
    assert s.paper_loop_intended is True


def test_trading_stop_marks_user_touched(
    fake_state: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POST /api/trading/stop must set paper_loop_user_touched=True."""
    fake_info = types.SimpleNamespace(to_dict=lambda: {"running": False})
    monkeypatch.setattr(srv.job_mgr, "stop", lambda kind: fake_info)

    client = TestClient(srv.app)
    r = client.post("/api/trading/stop")
    assert r.status_code == 200
    s = st.load_state()
    assert s.paper_loop_user_touched is True
    assert s.paper_loop_intended is False


# ---------------------------------------------------------------------------
# Log visibility
# ---------------------------------------------------------------------------


def test_log_endpoint_shape() -> None:
    """The JSON tail endpoint returns the documented shape."""
    client = TestClient(srv.app)
    r = client.get("/api/jobs/paper_loop/log")
    assert r.status_code == 200
    body = r.json()
    assert "kind" in body and "tail" in body
    assert body["kind"] == "paper_loop"
    assert isinstance(body["tail"], str)


def test_log_endpoint_download_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """?download=1 returns the on-disk log as text/plain (or a placeholder)."""
    client = TestClient(srv.app)
    r = client.get("/api/jobs/paper_loop/log?download=1")
    # Either we get the real file or the (no log yet) placeholder; both
    # must be text/plain 200s.
    assert r.status_code == 200
    assert "text/plain" in r.headers.get("content-type", "")


def test_trading_template_has_logs_panel() -> None:
    """The /trading template embeds the Phase 36b logs UI scaffolding."""
    tpl = Path("packages/cockpit/web/templates/trading.html").read_text()
    # Download button + refresh button + on-disk path documented.
    assert "log-download" in tpl, "missing Download full log button"
    assert "log-refresh" in tpl, "missing Refresh tail button"
    assert "/api/jobs/paper_loop/log?download=1" in tpl
    assert "/api/jobs/paper_loop/stream" in tpl
    # Tail loader is wired into DOMContentLoaded.
    assert "loadLogTail" in tpl
    # Path hint visible to operator.
    assert "data/cockpit/logs/paper_loop.log" in tpl


# ---------------------------------------------------------------------------
# Endpoint signatures (cheap regression guards)
# ---------------------------------------------------------------------------


def test_trading_start_signature_unchanged() -> None:
    """Phase 36b doesn't alter the public shape of /api/trading/start."""
    sig = inspect.signature(srv.api_trading_start)
    assert "req" in sig.parameters
