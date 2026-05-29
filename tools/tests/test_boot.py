"""Tests for the one-click boot orchestrator.

These are unit tests — every step is stubbed so the suite has no network,
no Ollama, no real Alpaca dependency. Integration smoke is covered by the
launch scripts in CI.
"""
from __future__ import annotations

import json
import logging
import sys
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


def test_step_ollama_ok_fast_path_when_daemon_already_up(
    ctx: BootContext, monkeypatch
) -> None:
    """If ``_daemon_alive`` returns True we skip the spawn path entirely
    -- which is what protects us from the Python 3.12.0 Windows
    subprocess.Popen abort. Verify no Popen-related code runs."""
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_START", "1")
    with patch("tools.check_ollama._daemon_alive", return_value=True), patch(
        "tools.check_ollama.status_snapshot",
        return_value={"installed": ["qwen2.5:7b"]},
    ), patch("tools.check_ollama.ensure_daemon") as ensure:
        r = step_ollama(ctx)
    assert r.status == "ok"
    assert "already running" in r.message
    assert "qwen2.5:7b" in r.detail["installed"]
    ensure.assert_not_called()  # critical: we did NOT spawn anything


def test_step_ollama_ok_when_ensure_daemon_succeeds(
    ctx: BootContext, monkeypatch
) -> None:
    """Slow path: daemon not initially up, ensure_daemon spawns it."""
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_START", "1")
    with patch("tools.check_ollama._daemon_alive", return_value=False), patch(
        "tools.check_ollama.ensure_daemon", return_value=True
    ), patch(
        "tools.check_ollama.status_snapshot",
        return_value={"installed": ["qwen2.5:7b"]},
    ):
        r = step_ollama(ctx)
    assert r.status == "ok"
    assert r.message.startswith("daemon responding")


def test_step_ollama_degraded_when_daemon_down(
    ctx: BootContext, monkeypatch
) -> None:
    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_START", "1")
    with patch("tools.check_ollama._daemon_alive", return_value=False), patch(
        "tools.check_ollama.ensure_daemon", return_value=False
    ), patch("tools.check_ollama.status_snapshot", return_value={}):
        r = step_ollama(ctx)
    assert r.status == "degraded"
    assert "disabled" in r.message.lower()


def test_step_ollama_auto_skip_when_crash_sentinel_present(
    ctx: BootContext, monkeypatch
) -> None:
    """The CORE crash-loop protection: if the previous boot left a
    sentinel file behind (because Python aborted mid-step before the
    finally block could clean it up), the next boot must auto-skip
    this step instead of crashing again."""
    from tools.boot import _crash_sentinel_path

    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_START", "1")
    sentinel = _crash_sentinel_path(ctx.repo_root, "ollama")
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("prev-crash", encoding="utf-8")

    # ensure_daemon must NOT be called even if it would have succeeded --
    # we never want to retry the step that just killed the interpreter.
    with patch("tools.check_ollama._daemon_alive") as alive, patch(
        "tools.check_ollama.ensure_daemon"
    ) as ensure:
        r = step_ollama(ctx)
    assert r.status == "skipped"
    assert "previous boot crashed" in r.message
    alive.assert_not_called()
    ensure.assert_not_called()
    # The sentinel must be cleaned up so the *next* boot can try again.
    assert not sentinel.exists()


def test_step_ollama_writes_and_clears_sentinel_on_slow_path(
    ctx: BootContext, monkeypatch
) -> None:
    """On the slow path (need to spawn) we drop a sentinel before
    calling ensure_daemon and remove it afterwards. Verify the file
    is gone after a successful run."""
    from tools.boot import _crash_sentinel_path

    monkeypatch.setenv("COCKPIT_OLLAMA_AUTO_START", "1")
    sentinel = _crash_sentinel_path(ctx.repo_root, "ollama")
    assert not sentinel.exists()

    seen_sentinels: list[bool] = []

    def _fake_ensure(host: str, *, verbose: bool = True) -> bool:
        # While ensure_daemon is running, the sentinel MUST exist so
        # that a hard abort here would be caught on the next boot.
        seen_sentinels.append(sentinel.exists())
        return True

    with patch("tools.check_ollama._daemon_alive", return_value=False), patch(
        "tools.check_ollama.ensure_daemon", side_effect=_fake_ensure
    ), patch("tools.check_ollama.status_snapshot", return_value={"installed": []}):
        r = step_ollama(ctx)

    assert r.status == "ok"
    assert seen_sentinels == [True], "sentinel must exist while ensure_daemon runs"
    assert not sentinel.exists(), "sentinel must be cleaned up after success"


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


def test_main_cli_quiet_still_prints_human_summary(ctx: BootContext, capsys) -> None:
    """Regression: --quiet only silences INFO logs. The human summary with
    [ok]/[!!]/[XX] markers must ALWAYS land on stdout so the launcher's
    "Fix the [XX] step above" message has something to point at."""
    rc = boot.main(["--only", "data_dirs", "--quiet"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== ai-investing boot ===" in out
    assert "[ok]" in out
    assert "data_dirs" in out
    assert "overall:" in out


def test_main_cli_returns_two_when_step_failed(ctx: BootContext, capsys, monkeypatch) -> None:
    """Exit code 2 is the launcher contract for 'one or more steps failed'."""
    def _boom(_ctx):
        from tools.boot import StepResult
        return StepResult("data_dirs", "failed", "simulated")

    monkeypatch.setattr(
        boot,
        "STEPS",
        [("data_dirs", _boom)],
    )
    rc = boot.main(["--only", "data_dirs"])
    assert rc == 2
    out = capsys.readouterr().out
    # The [XX] marker for the failed step must be visible.
    assert "[XX]" in out
    assert "data_dirs" in out


def test_run_boot_traces_each_step_to_stdout(ctx: BootContext, capsys, monkeypatch) -> None:
    """>>> step X starting / <<< step X ok markers go to STDOUT (not stderr)
    so Windows PowerShell 5.1 doesn't wrap each line in a NativeCommandError
    record. The launcher tees stdout to data/cockpit/boot_launcher.log so a
    hard exit during a step still leaves a 'got this far' marker on disk."""
    monkeypatch.setenv("COCKPIT_BOOT_TRACE", "1")
    run_boot(only={"data_dirs"}, ctx=ctx, persist=False)
    cap = capsys.readouterr()
    assert ">>> step data_dirs starting" in cap.out
    assert "<<< step data_dirs ok" in cap.out
    # Critical: stderr stays clean so PowerShell 5.1 doesn't editorialize.
    assert ">>> step" not in cap.err


def test_run_boot_trace_disabled_when_env_zero(ctx: BootContext, capsys, monkeypatch) -> None:
    monkeypatch.setenv("COCKPIT_BOOT_TRACE", "0")
    run_boot(only={"data_dirs"}, ctx=ctx, persist=False)
    cap = capsys.readouterr()
    assert ">>> step" not in cap.out
    assert ">>> step" not in cap.err


def test_run_boot_catches_systemexit_in_step(ctx: BootContext, monkeypatch) -> None:
    """A step that calls sys.exit() must NOT take down the orchestrator.
    Instead the step is recorded as failed and run_boot returns normally."""
    def _suicide(_ctx):
        raise SystemExit(7)

    monkeypatch.setattr(boot, "STEPS", [("data_dirs", _suicide)])
    summary = run_boot(only={"data_dirs"}, ctx=ctx, persist=False)
    assert summary.overall == "failed"
    assert summary.results[0].status == "failed"
    assert "sys.exit(7)" in summary.results[0].message


def test_run_boot_catches_base_exception_in_step(ctx: BootContext, monkeypatch) -> None:
    """Even an unusual BaseException subclass (e.g. GeneratorExit) must be
    contained so a misbehaving step can't silently kill boot."""
    def _weird(_ctx):
        raise GeneratorExit("unexpected")

    monkeypatch.setattr(boot, "STEPS", [("data_dirs", _weird)])
    summary = run_boot(only={"data_dirs"}, ctx=ctx, persist=False)
    assert summary.overall == "failed"
    assert "GeneratorExit" in summary.results[0].message


def test_run_boot_propagates_keyboard_interrupt(ctx: BootContext, monkeypatch) -> None:
    """Ctrl+C must still quit. Catching it would leave users unable to
    abort a hung step."""
    def _hang(_ctx):
        raise KeyboardInterrupt()

    monkeypatch.setattr(boot, "STEPS", [("data_dirs", _hang)])
    with pytest.raises(KeyboardInterrupt):
        run_boot(only={"data_dirs"}, ctx=ctx, persist=False)


class _FakeVersionInfo(tuple):
    """Quack like ``sys.version_info`` — indexable AND has named attrs.

    ``sys.version_info`` itself can't be instantiated directly
    (TypeError: cannot create 'sys.version_info' instances), so we ship a
    duck-typed stand-in that supports both ``v[:2]`` slicing and
    ``v.major / v.minor / v.micro`` attribute access — which is the full
    surface boot.py uses.
    """

    def __new__(cls, major: int, minor: int, micro: int,
                releaselevel: str = "final", serial: int = 0):
        return super().__new__(cls, (major, minor, micro, releaselevel, serial))

    @property
    def major(self) -> int: return self[0]
    @property
    def minor(self) -> int: return self[1]
    @property
    def micro(self) -> int: return self[2]
    @property
    def releaselevel(self) -> str: return self[3]
    @property
    def serial(self) -> int: return self[4]


def test_main_cli_warns_on_old_python_3_12(capsys, monkeypatch) -> None:
    """Python 3.12.0..3.12.5 had Windows stability bugs; the banner nudges
    users to upgrade without blocking the launch."""
    monkeypatch.setattr(sys, "version_info", _FakeVersionInfo(3, 12, 0))
    boot.main(["--only", "data_dirs"])
    out = capsys.readouterr().out
    assert "3.12.0 has known Windows stability bugs" in out
    assert "3.12.6+" in out


def test_main_cli_no_python_warning_on_current(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "version_info", _FakeVersionInfo(3, 12, 8))
    boot.main(["--only", "data_dirs"])
    out = capsys.readouterr().out
    assert "known Windows stability bugs" not in out


def test_main_cli_prints_starting_banner(ctx: BootContext, capsys) -> None:
    """Even before any step runs we must emit a 'tools.boot starting' marker
    plus python/cwd/PYTHONPATH diagnostics. That way a hard crash mid-step
    still leaves the user with proof the orchestrator was reached at all,
    and the launcher's tail-the-log fallback has something to show."""
    boot.main(["--only", "data_dirs"])
    out = capsys.readouterr().out
    assert "=== tools.boot starting ===" in out
    assert "python" in out
    assert "cwd" in out
    assert "PYTHONPATH" in out


def test_main_cli_json_mode_suppresses_banner(ctx: BootContext, capsys) -> None:
    """--json mode is consumed by scripts; the diagnostic banner would
    corrupt the JSON payload, so it must be suppressed."""
    rc = boot.main(["--only", "data_dirs", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "=== tools.boot starting ===" not in out
    # Output must be parseable JSON (no banner garbage prefix).
    json.loads(out)


def test_main_cli_returns_three_on_orchestrator_crash(
    ctx: BootContext, capsys, monkeypatch
) -> None:
    """If run_boot itself raises (not a step), main() must catch it, print
    a traceback, and exit 3 so the launcher can give a distinct hint."""
    def _explode(**_kwargs):
        raise RuntimeError("simulated orchestrator crash")

    monkeypatch.setattr(boot, "run_boot", _explode)
    rc = boot.main([])
    assert rc == 3
    out = capsys.readouterr().out
    assert "CRASHED" in out
    assert "simulated orchestrator crash" in out
    # Traceback delimiters must be present so users can paste a complete block.
    assert "--- traceback ---" in out
    assert "--- end traceback ---" in out


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


# ---------------------------------------------------------------------------
# step_research_sweep -- Phase 1D wiring
# ---------------------------------------------------------------------------


def test_research_sweep_step_is_last(ctx: BootContext) -> None:
    """The boot orchestrator must launch the cockpit *before* kicking off
    the research sweep. The sweep is fire-and-forget; if it crashed during
    boot, an earlier sweep step would block the cockpit. Lock the order
    in so a future refactor doesn't accidentally reshuffle."""
    names = [name for name, _ in STEPS]
    assert names[-1] == "research_sweep"
    assert "cockpit_port" in names
    assert names.index("cockpit_port") < names.index("research_sweep")


def test_research_sweep_step_is_degraded_tolerant(
    ctx: BootContext, monkeypatch
) -> None:
    """If the kick-off raises (e.g. broken import), the step must degrade
    rather than fail -- the cockpit should still launch."""
    # Force the import to raise by injecting a bad sys.modules entry.
    import sys

    from tools.boot import step_research_sweep

    real = sys.modules.get("packages.agents.research_sweep")

    class _Boom:
        def __getattr__(self, _name):
            raise RuntimeError("simulated failure")

    sys.modules["packages.agents.research_sweep"] = _Boom()
    try:
        result = step_research_sweep(ctx)
    finally:
        if real is not None:
            sys.modules["packages.agents.research_sweep"] = real
        else:
            sys.modules.pop("packages.agents.research_sweep", None)
    assert result.status == "degraded"
