"""Canonical one-click boot orchestrator for ai-investing.

This module is the single entry point for "click one button → everything
starts." It coalesces the work that used to be scattered across
``scripts/launch.ps1``, ``scripts/start.bat``, the cockpit's in-process
startup hooks, and the tray launcher.

Design goals (no exceptions):

1. **Deterministic ordering**. Steps run in a fixed dependency-aware order
   so failures point at the real culprit, not a downstream symptom.
2. **Hermetic & testable**. Every step is a pure function that takes a
   ``BootContext`` and returns a ``StepResult``. Steps can be skipped,
   re-run, or stubbed in tests without touching real binaries.
3. **Recoverable**. A failed step records why and (when safe) continues so
   the cockpit still launches with a degraded badge instead of a black
   screen.
4. **Observable**. Each step emits a structured event the cockpit
   ``/api/boot`` endpoint can stream. Useful for the first-boot wizard
   we ship in §1C and the self-healing layer in §4.

Public surface:

    from tools.boot import run_boot, BootContext
    result = run_boot()             # honors $COCKPIT_BOOT_* env knobs
    result = run_boot(skip={"ollama"})  # CI / smoke test escape hatch

CLI:

    PYTHONPATH=. python -m tools.boot --json
    PYTHONPATH=. python -m tools.boot --skip ollama --skip docker
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
_LOG = logging.getLogger("boot")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

StepStatus = Literal["ok", "skipped", "degraded", "failed"]


@dataclass
class StepResult:
    name: str
    status: StepStatus
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


@dataclass
class BootContext:
    """Shared mutable state passed between boot steps."""

    repo_root: Path = REPO_ROOT
    skip: set[str] = field(default_factory=set)
    only: set[str] | None = None  # if set, run only these steps
    interactive: bool = True
    log: logging.Logger = field(default_factory=lambda: _LOG)
    state: dict[str, Any] = field(default_factory=dict)  # cross-step scratch

    def env_on(self, key: str, default: str = "1") -> bool:
        """True if env var is set to a truthy value (1/true/True/yes)."""
        return os.environ.get(key, default).lower() in ("1", "true", "yes")

    def should_run(self, step: str) -> bool:
        if self.only is not None:
            return step in self.only
        return step not in self.skip


# ---------------------------------------------------------------------------
# Individual boot steps
# ---------------------------------------------------------------------------


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """True if something is listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def step_env(ctx: BootContext) -> StepResult:
    """Validate .env exists and seed from .env.example if missing.

    Non-blocking: a missing .env is degraded, not failed, so a user can
    still see the cockpit and follow the wizard.
    """
    env_file = ctx.repo_root / ".env"
    example = ctx.repo_root / ".env.example"
    if env_file.exists():
        text = env_file.read_text(encoding="utf-8", errors="ignore")
        has_keys = (
            "ALPACA_PAPER_KEY_ID" in text
            and "ALPACA_PAPER_SECRET" in text
            and not _line_blank(text, "ALPACA_PAPER_KEY_ID")
            and not _line_blank(text, "ALPACA_PAPER_SECRET")
        )
        if has_keys:
            return StepResult("env", "ok", "Alpaca paper keys present")
        return StepResult(
            "env",
            "degraded",
            ".env exists but Alpaca paper keys look empty",
            {"path": str(env_file)},
        )
    if example.exists():
        shutil.copy2(example, env_file)
        return StepResult(
            "env",
            "degraded",
            "Created .env from .env.example — add your Alpaca paper keys",
            {"path": str(env_file)},
        )
    return StepResult(
        "env",
        "failed",
        ".env and .env.example both missing — run scripts/install.ps1",
    )


def _line_blank(text: str, key: str) -> bool:
    """True if env file has KEY=  (no value) for the given key."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(f"{key}="):
            value = s.split("=", 1)[1].strip()
            return value == "" or value.startswith("#")
    return True


def step_venv(ctx: BootContext) -> StepResult:
    """Verify .venv exists. We don't create it here — install.ps1 owns that.

    If the venv is missing, the launcher script (which already activates
    .venv) couldn't have invoked Python in the first place, so we'd never
    reach this step. Treat absence as a misconfiguration to flag loudly.
    """
    if sys.prefix == sys.base_prefix:
        # Not in a venv at all.
        venv_py = ctx.repo_root / ".venv" / "Scripts" / "python.exe"
        if not venv_py.exists():
            venv_py = ctx.repo_root / ".venv" / "bin" / "python"
        if not venv_py.exists():
            return StepResult(
                "venv",
                "failed",
                ".venv not found — run scripts/install.ps1 (Windows) or scripts/install.sh",
            )
        return StepResult(
            "venv",
            "degraded",
            "Python not running from .venv — restart via scripts/launch.cmd",
        )
    return StepResult("venv", "ok", f"venv active at {sys.prefix}")


def step_ollama(ctx: BootContext) -> StepResult:
    """Start the Ollama daemon and confirm it responds.

    Reuses ``tools.check_ollama.ensure_daemon`` so the resolution rules
    (Adrenalin bundle on Devin's box, then PATH) stay in one place.
    """
    if not ctx.env_on("COCKPIT_OLLAMA_AUTO_START"):
        return StepResult("ollama", "skipped", "COCKPIT_OLLAMA_AUTO_START=0")
    try:
        from tools.check_ollama import (  # local import — keeps boot fast
            DEFAULT_HOST,
            ensure_daemon,
            status_snapshot,
        )
    except Exception as exc:  # pragma: no cover — defensive
        return StepResult("ollama", "failed", f"could not import check_ollama: {exc}")

    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    ok = ensure_daemon(host, verbose=False)
    snap = status_snapshot(host=host)
    if ok:
        return StepResult(
            "ollama",
            "ok",
            f"daemon responding at {host}",
            {"installed": snap.get("installed", [])},
        )
    return StepResult(
        "ollama",
        "degraded",
        "Ollama not responding — local LLM features disabled until you start it",
        snap,
    )


def step_models(ctx: BootContext) -> StepResult:
    """Pull any missing models for the active hardware profile.

    Only runs if Ollama is up. ``COCKPIT_OLLAMA_AUTO_PULL=0`` skips this.
    Missing models don't fail the boot — the LLM router falls back to a
    smaller available chain.
    """
    if not ctx.env_on("COCKPIT_OLLAMA_AUTO_PULL"):
        return StepResult("models", "skipped", "COCKPIT_OLLAMA_AUTO_PULL=0")
    try:
        from packages.agents.model_profiles import active_profile, all_models
        from tools.check_ollama import (
            DEFAULT_HOST,
            _list_installed,
            _matches,
            pull_model,
        )
    except Exception as exc:
        return StepResult("models", "failed", f"could not import deps: {exc}")

    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    if not _port_open(*_split_host(host)):
        return StepResult("models", "skipped", "Ollama not running; skipping pulls")

    profile = active_profile()
    required = all_models(profile)
    try:
        installed = _list_installed(host)
    except Exception as exc:
        return StepResult("models", "degraded", f"could not list models: {exc}")

    missing = [m for m in required if not _matches(m, installed)]
    if not missing:
        return StepResult(
            "models",
            "ok",
            f"all {len(required)} model(s) ready",
            {"required": required},
        )

    pulled: list[str] = []
    failed: list[str] = []
    for model in missing:
        ctx.log.info("Pulling missing model %s …", model)
        try:
            if pull_model(host, model, verbose=False):
                pulled.append(model)
            else:
                failed.append(model)
        except Exception as exc:  # pragma: no cover — network flakes
            ctx.log.warning("pull %s failed: %s", model, exc)
            failed.append(model)

    if failed:
        return StepResult(
            "models",
            "degraded",
            f"pulled {len(pulled)}, failed {len(failed)}",
            {"pulled": pulled, "failed": failed},
        )
    return StepResult("models", "ok", f"pulled {len(pulled)} missing model(s)", {"pulled": pulled})


def step_data_dirs(ctx: BootContext) -> StepResult:
    """Create the data/ subdirectories the runtime writes to.

    These are listed in .gitignore so a fresh clone has no data/ at all,
    which breaks the first paper run. Idempotent.
    """
    targets = [
        "data/audit",
        "data/cockpit",
        "data/db",
        "data/paper_log",
        "data/parquet/sentiment",
        "data/params",
        "data/raw",
        "data/processed",
        "data/cache",
    ]
    created: list[str] = []
    for rel in targets:
        path = ctx.repo_root / rel
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(rel)
    msg = f"created {len(created)} dir(s)" if created else "all data dirs present"
    return StepResult("data_dirs", "ok", msg, {"created": created})


def step_doctor(ctx: BootContext) -> StepResult:
    """Run tools.doctor in --json mode for a fast health snapshot.

    The doctor exits 0 even when sources are missing keys (by design) so
    we read its JSON to decide what to surface, not its exit code.
    """
    venv_py = sys.executable
    try:
        out = subprocess.run(
            [venv_py, "-m", "tools.doctor", "--json"],
            cwd=str(ctx.repo_root),
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "PYTHONPATH": str(ctx.repo_root)},
        )
    except subprocess.TimeoutExpired:
        return StepResult("doctor", "degraded", "doctor timed out after 15s")
    except Exception as exc:  # pragma: no cover
        return StepResult("doctor", "degraded", f"doctor crashed: {exc}")

    if out.returncode != 0:
        return StepResult(
            "doctor",
            "degraded",
            f"doctor exited {out.returncode}",
            {"stderr": out.stderr[:500]},
        )
    try:
        payload = json.loads(out.stdout)
    except json.JSONDecodeError:
        return StepResult("doctor", "degraded", "doctor produced non-JSON output")
    enabled = [k for k, v in payload.get("sources", {}).items() if v.get("ok")]
    return StepResult(
        "doctor",
        "ok",
        f"{len(enabled)} data source(s) ready",
        {"enabled": enabled, "raw": payload},
    )


def step_cockpit_port(ctx: BootContext) -> StepResult:
    """Confirm the cockpit port is free (or owned by us). Don't start the
    cockpit here — the launcher script does that as its foreground process
    so logs show in the terminal.
    """
    port = int(os.environ.get("COCKPIT_PORT", "8765"))
    if _port_open("127.0.0.1", port):
        # Someone is on the port. Could be a stale cockpit from a prior
        # session — flag, don't kill (that's the watchdog's job).
        return StepResult(
            "cockpit_port",
            "degraded",
            f"port {port} already in use — kill the other cockpit or change COCKPIT_PORT",
            {"port": port},
        )
    return StepResult("cockpit_port", "ok", f"port {port} free", {"port": port})


def step_research_sweep(ctx: BootContext) -> StepResult:
    """Kick off the boot-time research sweep in the background.

    Returns immediately (the sweep runs on its own thread / loop). We only
    fail the step if the import itself blows up -- the sweep's own error
    handling takes over once it's spawned, and the dashboard surfaces any
    failure via the ``status`` heartbeat file.

    This is intentionally the *last* step so the cockpit comes up first
    and the user sees something useful while the agents gather data.
    """
    try:
        from packages.agents.research_sweep import (
            kick_off_background,
            save_status,
        )

        # Pre-mark running so the dashboard tile shows 'running' even
        # before the background task gets scheduled.
        save_status("running", detail="boot-time sweep starting")
        kick_off_background()
        return StepResult(
            "research_sweep",
            "ok",
            "boot-time sweep kicked off in background",
        )
    except Exception as exc:
        # Never block the launch on this -- it's a 'nice to have' that
        # the user can re-run manually from the dashboard.
        return StepResult(
            "research_sweep",
            "degraded",
            f"could not kick off sweep: {exc.__class__.__name__}",
            {"error": str(exc)},
        )


def _split_host(host: str) -> tuple[str, int]:
    """``http://127.0.0.1:11434`` → ``('127.0.0.1', 11434)``."""
    stripped = host.replace("http://", "").replace("https://", "")
    if ":" in stripped:
        h, p = stripped.split(":", 1)
        return h, int(p)
    return stripped, 80


# ---------------------------------------------------------------------------
# Step registry — ordered list. Keep dependencies obvious.
# ---------------------------------------------------------------------------

Step = Callable[[BootContext], StepResult]

STEPS: list[tuple[str, Step]] = [
    ("env", step_env),
    ("venv", step_venv),
    ("data_dirs", step_data_dirs),
    ("ollama", step_ollama),
    ("models", step_models),
    ("doctor", step_doctor),
    ("cockpit_port", step_cockpit_port),
    # research_sweep is last on purpose: it's a fire-and-forget background
    # task that benefits from everything above being green first.
    ("research_sweep", step_research_sweep),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class BootSummary:
    started_at: float
    finished_at: float
    results: list[StepResult]

    @property
    def duration_s(self) -> float:
        return self.finished_at - self.started_at

    @property
    def overall(self) -> StepStatus:
        statuses = [r.status for r in self.results]
        if "failed" in statuses:
            return "failed"
        if "degraded" in statuses:
            return "degraded"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "steps": [asdict(r) for r in self.results],
        }


def run_boot(
    *,
    skip: Iterable[str] | None = None,
    only: Iterable[str] | None = None,
    ctx: BootContext | None = None,
    on_step: Callable[[StepResult], None] | None = None,
    persist: bool = True,
) -> BootSummary:
    """Run every boot step in registry order.

    Args:
        skip: step names to skip (e.g. for CI hermetic runs).
        only: if set, run only these steps.
        ctx: pre-built context (mostly for tests).
        on_step: callback invoked after each step completes — used by the
            cockpit /api/boot/stream endpoint to push progress live.
        persist: write summary to ``data/cockpit/boot.json``. The cockpit
            and tray read this so a restart inherits the prior state
            without re-running expensive steps.
    """
    if ctx is None:
        ctx = BootContext()
    if skip:
        ctx.skip = set(skip)
    if only is not None:
        ctx.only = set(only)

    started = time.time()
    results: list[StepResult] = []
    for name, fn in STEPS:
        if not ctx.should_run(name):
            r = StepResult(name, "skipped", "step skipped by caller")
            results.append(r)
            if on_step:
                on_step(r)
            continue
        t0 = time.time()
        try:
            r = fn(ctx)
        except Exception as exc:  # pragma: no cover — last-line safety net
            r = StepResult(name, "failed", f"unexpected error: {exc}")
        r.duration_s = round(time.time() - t0, 3)
        results.append(r)
        if on_step:
            on_step(r)
    summary = BootSummary(started_at=started, finished_at=time.time(), results=results)
    if persist:
        _persist_summary(ctx.repo_root, summary)
    return summary


def _persist_summary(repo_root: Path, summary: BootSummary) -> None:
    """Atomic write of the boot summary so concurrent readers don't tear."""
    out_dir = repo_root / "data" / "cockpit"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "boot.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(target)
    except OSError:  # pragma: no cover — disk full / readonly fs
        pass


def _format_human(summary: BootSummary) -> str:
    """Pretty terminal output."""
    lines = ["", "=== ai-investing boot ===", ""]
    icons = {"ok": "[ok]", "skipped": "[--]", "degraded": "[!!]", "failed": "[XX]"}
    for r in summary.results:
        icon = icons.get(r.status, "[??]")
        lines.append(f"  {icon} {r.name:<14} {r.message} ({r.duration_s}s)")
    lines.append("")
    lines.append(f"  overall: {summary.overall}  ({summary.duration_s:.1f}s total)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ai-investing one-click boot")
    ap.add_argument("--json", action="store_true", help="emit JSON summary")
    ap.add_argument("--skip", action="append", default=[], help="skip a step (repeatable)")
    ap.add_argument("--only", action="append", default=[], help="run only these (repeatable)")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "suppress per-step INFO log lines (the human summary is always "
            "printed so launchers can show which step failed)"
        ),
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Print a banner BEFORE running anything so even a hard segfault inside
    # a step leaves the user with a visible marker ("got at least this far").
    # The launcher tees stdout to data/cockpit/boot_launcher.log so this is
    # the bottom of the truth chain.
    if not args.json:
        print("", flush=True)
        print("=== tools.boot starting ===", flush=True)
        print(f"  python  : {sys.version.split()[0]} ({sys.executable})", flush=True)
        print(f"  cwd     : {os.getcwd()}", flush=True)
        print(f"  PYTHONPATH={os.environ.get('PYTHONPATH', '<unset>')}", flush=True)
        if args.skip:
            print(f"  skip    : {','.join(args.skip)}", flush=True)
        if args.only:
            print(f"  only    : {','.join(args.only)}", flush=True)
        print("", flush=True)

    # Run with a defensive outer try/except so an unhandled error inside a
    # step (or in run_boot itself) NEVER leaves the launcher staring at an
    # empty terminal with a meaningless exit code. We print a self-contained
    # error block + traceback so the user can paste it into a bug report
    # and we can debug from a single screenshot.
    try:
        summary = run_boot(
            skip=args.skip,
            only=args.only or None,
        )
    except BaseException as exc:  # last-line safety net for the launcher
        import traceback

        sys.stdout.flush()
        print("", flush=True)
        print("=== ai-investing boot CRASHED ===", flush=True)
        print(f"  {type(exc).__name__}: {exc}", flush=True)
        print("", flush=True)
        print("--- traceback ---", flush=True)
        traceback.print_exc()
        print("--- end traceback ---", flush=True)
        print("", flush=True)
        print(
            "This is a bug in tools/boot.py. Please open an issue with the "
            "traceback above.",
            flush=True,
        )
        return 3

    if args.json:
        print(json.dumps(summary.to_dict(), indent=2), flush=True)
    else:
        # ALWAYS print the human summary, even with --quiet, so failed steps
        # show their [XX] marker before the launcher's red "Fix the [XX] step
        # above" message.
        print(_format_human(summary), flush=True)

    # Exit code contract for launchers:
    #   0 -- ok or degraded; cockpit may still come up
    #   2 -- one or more required steps failed; do not start the cockpit
    #   3 -- orchestrator itself crashed (see traceback above)
    return 0 if summary.overall in ("ok", "degraded") else 2


if __name__ == "__main__":
    sys.exit(main())
