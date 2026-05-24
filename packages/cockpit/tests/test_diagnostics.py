"""Tests for :mod:`packages.cockpit.diagnostics`.

The diagnostics layer is intentionally a flat collection of small pure
functions so each one is trivially testable: point the module's path
constants at a tmp_path, monkeypatch the few external probes (port
listener, Ollama HTTP, orphan scanner), and assert the resulting
``Check`` dataclass.

We also exercise the API surface that the Health page depends on:
``run_all``, ``summary`` rollup semantics, and the ``auto_heal``
dispatch table.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from packages.cockpit import diagnostics as diag

# ---------------------------------------------------------------------------
# Check dataclass + summary rollup
# ---------------------------------------------------------------------------


def test_check_to_dict_roundtrip() -> None:
    c = diag.Check(
        name="demo",
        title="Demo",
        status="warn",
        message="hello",
        fix_command="echo hi",
        auto_fixable=True,
        detail={"pid": 123},
    )
    d = c.to_dict()
    assert d["name"] == "demo"
    assert d["status"] == "warn"
    assert d["auto_fixable"] is True
    assert d["detail"] == {"pid": 123}
    assert d["fix_command"] == "echo hi"


def _mk(name: str, status: diag.Status) -> diag.Check:
    return diag.Check(name=name, title=name, status=status, message="x")


def test_summary_rollup_picks_worst_status() -> None:
    # error beats warn beats ok beats info
    s = diag.summary([_mk("a", "ok"), _mk("b", "warn"), _mk("c", "error")])
    assert s["status"] == "error"
    assert s["counts"] == {"ok": 1, "warn": 1, "error": 1, "info": 0}
    assert len(s["checks"]) == 3
    # ``now`` is an ISO-8601 timestamp
    datetime.fromisoformat(s["now"])  # type: ignore[arg-type]


def test_summary_rollup_all_ok() -> None:
    s = diag.summary([_mk("a", "ok"), _mk("b", "ok")])
    assert s["status"] == "ok"
    assert s["counts"]["ok"] == 2


def test_summary_rollup_only_info_is_info() -> None:
    s = diag.summary([_mk("a", "info"), _mk("b", "info")])
    assert s["status"] == "info"


def test_summary_rollup_empty_is_info() -> None:
    s = diag.summary([])
    assert s["status"] == "info"
    assert s["counts"] == {"ok": 0, "warn": 0, "error": 0, "info": 0}


# ---------------------------------------------------------------------------
# Individual checks (path-based)
# ---------------------------------------------------------------------------


def _redirect_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Repoint every module-level path constant into ``tmp_path``."""
    monkeypatch.setattr(diag, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(diag, "DATA_DIR", tmp_path / "data" / "cockpit")
    monkeypatch.setattr(diag, "PAPER_LOG", tmp_path / "data" / "cockpit" / "paper_loop.jsonl")
    monkeypatch.setattr(
        diag, "PRETRAIN_STATE", tmp_path / "data" / "cockpit" / "pretrain_state.json"
    )
    monkeypatch.setattr(diag, "ENV_FILE", tmp_path / ".env")
    return tmp_path


def test_check_venv_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    c = diag.check_venv()
    assert c.status == "error"
    assert c.fix_command  # must offer a copy-paste fix


def test_check_venv_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    # Create whichever path the platform looks for.
    import os
    venv = tmp_path / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    venv.mkdir(parents=True)
    (venv / ("python.exe" if os.name == "nt" else "python")).write_text("")
    c = diag.check_venv()
    assert c.status == "ok"


def test_check_env_file_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    c = diag.check_env_file()
    assert c.status == "error"
    assert c.name == "env_file"
    assert "Copy" in (c.fix_command or "")


def test_check_env_file_blank_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text(
        "ALPACA_PAPER_KEY_ID=\nALPACA_PAPER_SECRET=change_me\n",
        encoding="utf-8",
    )
    c = diag.check_env_file()
    assert c.status == "warn"


def test_check_env_file_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text(
        "ALPACA_PAPER_KEY_ID=PKREAL123\nALPACA_PAPER_SECRET=somerealsecret\n",
        encoding="utf-8",
    )
    c = diag.check_env_file()
    assert c.status == "ok"


def test_looks_blank_helper() -> None:
    text = "FOO=\nBAR=change_me\nBAZ=actualvalue\nQUUX=  ...  \n"
    assert diag._looks_blank(text, "FOO") is True
    assert diag._looks_blank(text, "BAR") is True
    assert diag._looks_blank(text, "QUUX") is True
    assert diag._looks_blank(text, "BAZ") is False
    assert diag._looks_blank(text, "MISSING") is True


# ---------------------------------------------------------------------------
# Port + orphan checks (monkeypatch the system probes)
# ---------------------------------------------------------------------------


def test_check_port_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "_our_cockpit_pid_on_port", lambda _p: None)
    monkeypatch.setattr(diag, "_foreign_pid_on_port", lambda _p: None)
    c = diag.check_port(8765)
    assert c.status == "ok"
    assert c.auto_fixable is False


def test_check_port_held_by_us(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "_our_cockpit_pid_on_port", lambda _p: 4242)
    c = diag.check_port(8765)
    assert c.status == "ok"
    assert "bound" in c.message.lower()


def test_check_port_held_by_foreign_is_auto_fixable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "_our_cockpit_pid_on_port", lambda _p: None)
    monkeypatch.setattr(diag, "_foreign_pid_on_port", lambda _p: 9999)
    c = diag.check_port(8765)
    assert c.status == "warn"
    assert c.auto_fixable is True
    assert c.detail == {"holder_pid": 9999}


def test_check_orphan_pythons_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "_list_orphan_repo_pythons", lambda: [])
    c = diag.check_orphan_pythons()
    assert c.status == "ok"
    assert c.auto_fixable is False


def test_check_orphan_pythons_some(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diag,
        "_list_orphan_repo_pythons",
        lambda: [
            {"pid": 1234, "cmd": "python.exe -m packages.cockpit"},
            {"pid": 5678, "cmd": "python.exe -m packages.data.pretrain"},
        ],
    )
    c = diag.check_orphan_pythons()
    assert c.status == "warn"
    assert c.auto_fixable is True
    assert "1234" in c.message and "5678" in c.message


# ---------------------------------------------------------------------------
# Ollama checks (CLI presence + HTTP probe)
# ---------------------------------------------------------------------------


def test_check_ollama_installed_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag.shutil, "which", lambda _name: "/usr/bin/ollama")
    c = diag.check_ollama_installed()
    assert c.status == "ok"


def test_check_ollama_installed_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag.shutil, "which", lambda _name: None)
    c = diag.check_ollama_installed()
    assert c.status == "error"
    assert c.fix_command and "ollama.com" in c.fix_command


def test_check_ollama_running_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag.shutil, "which", lambda _name: None)
    c = diag.check_ollama_running()
    assert c.status == "info"


def test_check_ollama_running_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag.shutil, "which", lambda _name: "/usr/bin/ollama")

    class _Resp:
        status_code = 200

        def json(self) -> dict:
            return {"models": [{"name": "llama3"}, {"name": "qwen"}]}

    monkeypatch.setattr(diag.httpx, "get", lambda _url, timeout=1.5: _Resp())
    c = diag.check_ollama_running()
    assert c.status == "ok"
    assert c.detail and c.detail.get("models") == ["llama3", "qwen"]


def test_check_ollama_running_down_is_auto_fixable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag.shutil, "which", lambda _name: "/usr/bin/ollama")

    def _boom(*_a, **_kw):
        raise diag.httpx.HTTPError("connection refused")

    monkeypatch.setattr(diag.httpx, "get", _boom)
    c = diag.check_ollama_running()
    assert c.status == "warn"
    assert c.auto_fixable is True


# ---------------------------------------------------------------------------
# Pretrain freshness
# ---------------------------------------------------------------------------


def test_check_last_pretrain_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    c = diag.check_last_pretrain()
    assert c.status == "warn"
    assert c.auto_fixable is True


def test_check_last_pretrain_fresh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    state_path = tmp_path / "data" / "cockpit" / "pretrain_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps({"finished_at": datetime.now(UTC).isoformat(timespec="seconds")}),
        encoding="utf-8",
    )
    c = diag.check_last_pretrain()
    assert c.status == "ok"


def test_check_last_pretrain_stale(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    state_path = tmp_path / "data" / "cockpit" / "pretrain_state.json"
    state_path.parent.mkdir(parents=True)
    old = datetime.now(UTC) - timedelta(hours=72)
    state_path.write_text(
        json.dumps({"finished_at": old.isoformat(timespec="seconds")}),
        encoding="utf-8",
    )
    c = diag.check_last_pretrain()
    assert c.status == "warn"
    assert c.auto_fixable is True
    assert c.detail and c.detail.get("age_hours", 0) >= 36


def test_check_last_pretrain_unreadable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    state_path = tmp_path / "data" / "cockpit" / "pretrain_state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("not json at all", encoding="utf-8")
    c = diag.check_last_pretrain()
    assert c.status == "warn"
    assert c.auto_fixable is True


# ---------------------------------------------------------------------------
# run_all + auto_heal dispatch
# ---------------------------------------------------------------------------


def test_run_all_returns_seven_checks_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(diag, "_our_cockpit_pid_on_port", lambda _p: None)
    monkeypatch.setattr(diag, "_foreign_pid_on_port", lambda _p: None)
    monkeypatch.setattr(diag, "_list_orphan_repo_pythons", lambda: [])
    monkeypatch.setattr(diag.shutil, "which", lambda _name: None)
    checks = diag.run_all()
    names = [c.name for c in checks]
    assert names == [
        "venv",
        "env_file",
        "port_8765_clear",
        "orphan_pythons",
        "ollama_installed",
        "ollama_running",
        "last_pretrain",
    ]


def test_auto_heal_unknown_check_returns_error() -> None:
    out = diag.auto_heal("not_a_real_check")
    assert out["ok"] is False
    assert "not_a_real_check" in str(out["message"])


def test_auto_heal_dispatches_to_registered_fixer(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def _fake_orphan() -> dict[str, object]:
        called.append("orphan")
        return {"ok": True, "message": "killed 0 procs"}

    monkeypatch.setattr(diag, "_heal_orphan_pythons", _fake_orphan)
    out = diag.auto_heal("orphan_pythons")
    assert called == ["orphan"]
    assert out["ok"] is True


def test_auto_heal_catches_internal_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> dict[str, object]:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(diag, "_heal_port", _boom)
    out = diag.auto_heal("port_8765_clear")
    assert out["ok"] is False
    assert "kaboom" in str(out["message"])


# ---------------------------------------------------------------------------
# Auto-heal action: port + orphans (process-side patched)
# ---------------------------------------------------------------------------


def test_heal_port_no_holder_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "_foreign_pid_on_port", lambda _p: None)
    out = diag._heal_port()
    assert out["ok"] is True
    assert "free" in str(out["message"]).lower()


def test_heal_port_refuses_to_kill_stranger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "_foreign_pid_on_port", lambda _p: 9999)
    monkeypatch.setattr(diag, "_is_our_repo_python", lambda _pid: False)
    out = diag._heal_port()
    assert out["ok"] is False
    assert "9999" in str(out["message"])


def test_heal_port_kills_our_orphan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "_foreign_pid_on_port", lambda _p: 9999)
    monkeypatch.setattr(diag, "_is_our_repo_python", lambda _pid: True)
    killed: list[int] = []
    monkeypatch.setattr(diag, "_kill_pid", lambda pid: killed.append(pid) or True)
    out = diag._heal_port()
    assert out["ok"] is True
    assert killed == [9999]
    assert out.get("pid") == 9999


def test_heal_orphan_pythons_kills_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        diag,
        "_list_orphan_repo_pythons",
        lambda: [{"pid": 11, "cmd": "x"}, {"pid": 22, "cmd": "y"}],
    )
    killed: list[int] = []
    monkeypatch.setattr(diag, "_kill_pid", lambda pid: killed.append(pid) or True)
    out = diag._heal_orphan_pythons()
    assert out["ok"] is True
    assert sorted(killed) == [11, 22]


def test_heal_orphan_pythons_nothing_to_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diag, "_list_orphan_repo_pythons", lambda: [])
    out = diag._heal_orphan_pythons()
    assert out["ok"] is True
