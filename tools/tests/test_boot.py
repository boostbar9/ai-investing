"""Tests for the one-click boot orchestrator.

These are unit tests — every step is stubbed so the suite has no network,
no Ollama, no real Alpaca dependency. Integration smoke is covered by the
launch scripts in CI.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from tools import boot
from tools.boot import (
    STEPS,
    BootContext,
    StepResult,
    _format_human,
    _split_host,
    run_boot,
    step_cockpit_port,
    step_data_dirs,
    step_env,
    step_models,
    step_ollama,
    step_venv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Iterator[Path]:
    """A throwaway 'repo root' with the scaffolding boot expects."""
    (tmp_path / "data").mkdir()
    yield tmp_path


@pytest.fixture
def ctx(tmp_repo: Path) -> BootContext:
    return BootContext(repo_root=tmp_repo, log=logging.getLogger("test"))


# ---------------------------------------------------------------------------
# step_env
# ---------------------------------------------------------------------------


def test_step_env_ok_when_keys_present(ctx: BootContext) -> None:
    (ctx.repo_root / ".env").write_text(
        "ALPACA_PAPER_KEY_ID=PKABC123\nALPACA_PAPER_SECRET=supersecret\n"
    )
    r = step_env(ctx)
    assert r.status == "ok"
    assert "keys present" in r.message


def test_step_env_degraded_when_keys_blank(ctx: BootContext) -> None:
    (ctx.repo_root / ".env").write_text("ALPACA_PAPER_KEY_ID=\nALPACA_PAPER_SECRET=\n")
    r = step_env(ctx)
    assert r.status == "degraded"
    assert "empty" in r.message


def test_step_env_seeds_from_example(ctx: BootContext) -> None:
    (ctx.repo_root / ".env.example").write_text("ALPACA_PAPER_KEY_ID=\n")
    r = step_env(ctx)
    assert r.status == "degraded"
    assert (ctx.repo_root / ".env").exists()
    assert "Created .env" in r.message


def test_step_env_fails_when_nothing_to_seed(ctx: BootContext) -> None:
    r = step_env(ctx)
    assert r.status == "failed"
    assert "install" in r.message


def test_step_env_blank_key_with_comment_treated_as_blank(ctx: BootContext) -> None:
    """A line like ``KEY=# todo`` is still a blank value."""
    (ctx.repo_root / ".env").write_text(
        "ALPACA_PAPER_KEY_ID=# todo\nALPACA_PAPER_SECRET=# todo\n"
    )
    r = step_env(ctx)
    assert r.status == "degraded"


# ---------------------------------------------------------------------------
# step_venv
# ---------------------------------------------------------------------------


def test_step_venv_ok_in_venv(ctx: BootContext) -> None:
    # Running pytest from .venv means sys.prefix != sys.base_prefix already.
    # That guarantees this path is exercised.
    r = step_venv(ctx)
    assert r.status == "ok"


def test_step_venv_failed_when_no_venv(ctx: BootContext) -> None:
    with patch("tools.boot.sys") as mock_sys:
        mock_sys.prefix = "/usr"
        mock_sys.base_prefix = "/usr"
        r = step_venv(ctx)
        assert r.status == "failed"


def test_step_venv_degraded_when_venv_exists_but_not_active(ctx: BootContext) -> None:
    (ctx.repo_root / ".venv" / "bin").mkdir(parents=True)
    (ctx.repo_root / ".venv" / "bin" / "python").touch()
    with patch("tools.boot.sys") as mock_sys:
        mock_sys.prefix = "/usr"
        mock_sys.base_prefix = "/usr"
        r = step_venv(ctx)
        assert r.status == "degraded"


# ---------------------------------------------------------------------------
# step_data_dirs
# ---------------------------------------------------------------------------


def test_step_data_dirs_creates_missing(ctx: BootContext) -> None:
    r = step_data_dirs(ctx)
    assert r.status == "ok"
    assert (ctx.repo_root / "data" / "audit").is_dir()
    assert (ctx.repo_root / "data" / "params").is_dir()
    assert len(r.detail["created"]) > 0


def test_step_data_dirs_idempotent(ctx: BootContext) -> None:
    step_data_dirs(ctx)
    r2 = step_data_dirs(ctx)
    assert r2.status == "ok"
    assert r2.detail["created"] == []
    assert "all data dirs present" in r2.message


# ---------------------------------------------------------------------------
# step_ollama
# ---------------------------------------------------------------------------


def test_step_ollama_skipped_when_disabled(ctx: BootContext, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_START", "0")
    r = step_ollama(ctx)
    assert r.status == "skipped"


def test_step_ollama_ok_when_daemon_responds(ctx: BootContext, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_START", "1")
    with patch("tools.check_ollama.ensure_daemon", return_value=True), patch(
        "tools.check_ollama.status_snapshot",
        return_value={"installed": ["qwen2.5:7b"]},
    ):
        r = step_ollama(ctx)
    assert r.status == "ok"
    assert "qwen2.5:7b" in r.detail["installed"]


def test_step_ollama_degraded_when_daemon_down(ctx: BootContext, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_START", "1")
    with patch("tools.check_ollama.ensure_daemon", return_value=False), patch(
        "tools.check_ollama.status_snapshot", return_value={}
    ):
        r = step_ollama(ctx)
    assert r.status == "degraded"
    assert "Local LLM features disabled" in r.message or "disabled" in r.message


# ---------------------------------------------------------------------------
# step_models
# ---------------------------------------------------------------------------


def test_step_models_skipped_when_pull_disabled(ctx: BootContext, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_PULL", "0")
    r = step_models(ctx)
    assert r.status == "skipped"


def test_step_models_pulls_missing(ctx: BootContext, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_PULL", "1")
    with (
        patch.object(boot, "_port_open", return_value=True),
        patch("packages.agents.model_profiles.active_profile"),
        patch(
            "packages.agents.model_profiles.all_models",
            return_value=["qwen2.5:7b", "deepseek-r1:14b"],
        ),
        patch("tools.check_ollama._list_installed", return_value=["qwen2.5:7b"]),
        patch("tools.check_ollama.pull_model", return_value=True) as pull,
    ):
        r = step_models(ctx)
    assert r.status == "ok"
    assert "deepseek-r1:14b" in r.detail["pulled"]
    assert pull.call_count == 1


def test_step_models_degraded_on_pull_failure(ctx: BootContext, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_PULL", "1")
    with patch.object(boot, "_port_open", return_value=True), patch(
        "packages.agents.model_profiles.active_profile"
    ), patch(
        "packages.agents.model_profiles.all_models", return_value=["foo:1b", "bar:2b"]
    ), patch("tools.check_ollama._list_installed", return_value=[]), patch(
        "tools.check_ollama.pull_model", side_effect=[True, False]
    ):
        r = step_models(ctx)
    assert r.status == "degraded"
    assert "bar:2b" in r.detail["failed"]


def test_step_models_skipped_when_ollama_down(ctx: BootContext, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_PULL", "1")
    with patch.object(boot, "_port_open", return_value=False):
        r = step_models(ctx)
    assert r.status == "skipped"


# ---------------------------------------------------------------------------
# step_cockpit_port
# ---------------------------------------------------------------------------


def test_step_cockpit_port_free(ctx: BootContext, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_PORT", "65500")
    with patch.object(boot, "_port_open", return_value=False):
        r = step_cockpit_port(ctx)
    assert r.status == "ok"
    assert r.detail["port"] == 65500


def test_step_cockpit_port_in_use(ctx: BootContext) -> None:
    with patch.object(boot, "_port_open", return_value=True):
        r = step_cockpit_port(ctx)
    assert r.status == "degraded"
    assert "already in use" in r.message


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def test_run_boot_honors_skip(ctx: BootContext) -> None:
    # Skip everything network-ish so the test is fast and hermetic.
    summary = run_boot(skip={"ollama", "models", "doctor"}, ctx=ctx)
    by_name = {r.name: r for r in summary.results}
    assert by_name["ollama"].status == "skipped"
    assert by_name["models"].status == "skipped"
    assert by_name["doctor"].status == "skipped"
    # data_dirs always runs unless skipped
    assert by_name["data_dirs"].status == "ok"


def test_run_boot_only_runs_subset(ctx: BootContext) -> None:
    summary = run_boot(only={"data_dirs"}, ctx=ctx)
    by_name = {r.name: r for r in summary.results}
    assert by_name["data_dirs"].status == "ok"
    # Every other step should be skipped.
    others = [r.status for n, r in by_name.items() if n != "data_dirs"]
    assert all(s == "skipped" for s in others)


def test_run_boot_overall_failed_propagates(ctx: BootContext) -> None:
    """If any step is failed, overall is failed."""
    summary = run_boot(skip={"ollama", "models", "doctor"}, ctx=ctx)
    # step_env will fail because we provided neither .env nor .env.example.
    assert summary.overall == "failed"


def test_run_boot_overall_degraded_when_mixed(ctx: BootContext) -> None:
    (ctx.repo_root / ".env.example").write_text("ALPACA_PAPER_KEY_ID=\n")
    summary = run_boot(skip={"ollama", "models", "doctor"}, ctx=ctx)
    assert summary.overall == "degraded"


def test_run_boot_on_step_callback_called(ctx: BootContext) -> None:
    seen: list[str] = []
    run_boot(only={"data_dirs"}, ctx=ctx, on_step=lambda r: seen.append(r.name))
    assert "data_dirs" in seen


def test_run_boot_summary_to_dict_is_json_serializable(ctx: BootContext) -> None:
    summary = run_boot(only={"data_dirs"}, ctx=ctx)
    payload = summary.to_dict()
    text = json.dumps(payload)  # must not raise
    assert "data_dirs" in text


def test_run_boot_step_exceptions_become_failed(ctx: BootContext) -> None:
    """Last-line safety: a step that explodes shouldn't crash the driver."""

    def boom(_ctx: BootContext) -> StepResult:
        raise RuntimeError("kaboom")

    with patch.object(boot, "STEPS", [("boom", boom)]):
        summary = run_boot(ctx=ctx)
    assert summary.results[0].status == "failed"
    assert "kaboom" in summary.results[0].message


def test_step_registry_is_non_empty_and_unique() -> None:
    names = [n for n, _ in STEPS]
    assert len(names) > 0
    assert len(names) == len(set(names)), "duplicate step name"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def test_format_human_renders_all_statuses(ctx: BootContext) -> None:
    summary = run_boot(skip={"ollama", "models", "doctor"}, ctx=ctx)
    out = _format_human(summary)
    assert "ai-investing boot" in out
    assert "overall:" in out


def test_split_host_handles_full_url() -> None:
    assert _split_host("http://127.0.0.1:11434") == ("127.0.0.1", 11434)


def test_split_host_handles_bare_host() -> None:
    assert _split_host("127.0.0.1:11434") == ("127.0.0.1", 11434)


def test_split_host_handles_https() -> None:
    assert _split_host("https://ollama.local:9999") == ("ollama.local", 9999)


def test_main_cli_returns_zero_when_degraded(ctx: BootContext, tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env.example").write_text("ALPACA_PAPER_KEY_ID=\n")
    with patch.object(boot, "REPO_ROOT", tmp_path), patch.dict(
        boot.__dict__, {"REPO_ROOT": tmp_path}
    ):
        rc = boot.main(["--skip", "ollama", "--skip", "models", "--skip", "doctor", "--quiet"])
    # Even though env is degraded, overall is degraded → exit 0.
    assert rc == 0


def test_main_cli_json_mode(ctx: BootContext, capsys) -> None:
    rc = boot.main(["--only", "data_dirs", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["overall"] == "ok"
    assert any(s["name"] == "data_dirs" for s in payload["steps"])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_run_boot_persists_summary_to_disk(ctx: BootContext) -> None:
    summary = run_boot(only={"data_dirs"}, ctx=ctx)
    summary_path = ctx.repo_root / "data" / "cockpit" / "boot.json"
    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text())
    assert loaded["overall"] == "ok"
    assert loaded["duration_s"] == pytest.approx(summary.duration_s, rel=1e-6)


def test_run_boot_persist_false_skips_disk_write(ctx: BootContext) -> None:
    run_boot(only={"data_dirs"}, ctx=ctx, persist=False)
    summary_path = ctx.repo_root / "data" / "cockpit" / "boot.json"
    assert not summary_path.exists()


def test_persist_summary_writes_atomically(ctx: BootContext) -> None:
    """Confirm the temp-file-then-rename pattern — readers never see a
    half-written JSON during the write."""
    from tools.boot import _persist_summary  # type: ignore[attr-defined]

    summary = run_boot(only={"data_dirs"}, ctx=ctx, persist=False)
    _persist_summary(ctx.repo_root, summary)
    target = ctx.repo_root / "data" / "cockpit" / "boot.json"
    # Temp file should be gone after a clean write.
    assert not target.with_suffix(".json.tmp").exists()
    assert target.exists()
