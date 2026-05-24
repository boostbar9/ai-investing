"""Pre-flight + auto-setup for Ollama models.

Usage:
    .venv/bin/python tools/check_ollama.py                # just report
    .venv/bin/python tools/check_ollama.py --pull-missing # pull what's missing
    .venv/bin/python tools/check_ollama.py --auto         # one-shot: start daemon, pull, verify

Exit codes:
    0 — every required model present and daemon responds.
    1 — daemon unreachable AND we couldn't start it.
    2 — daemon up but at least one required model is missing.

What ``--auto`` does that ``--pull-missing`` didn't:
    * If the Ollama HTTP API is unreachable, try ``ollama serve`` in the
      background and wait up to 15s for it to come up. So you don't have
      to remember to start the daemon manually.
    * Pulls missing models via the HTTP ``POST /api/pull`` endpoint with
      live progress, so it works inside Docker / WSL2 / anywhere the
      ``ollama`` CLI binary isn't installed (fallback chain: CLI -> HTTP).
    * Re-verifies after each pull so a half-broken cache is caught.

Side effects:
    * ``--pull-missing`` and ``--auto`` write to the Ollama model cache.
    * ``--auto`` can spawn ``ollama serve`` if no daemon is up.
    * Plain ``check_ollama.py`` (no flags) is read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

# Allow running as ``python tools/check_ollama.py`` from the repo root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from packages.agents.model_profiles import active_profile, all_models  # noqa: E402

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# How long to wait for ``ollama serve`` to start responding before we give up.
DAEMON_STARTUP_TIMEOUT_S = 15.0
# How often to print pull progress (every N MB downloaded). Ollama's stream
# is chatty — we don't want to flood the terminal with one line per chunk.
PROGRESS_REPORT_EVERY_MB = 50


# ---------------------------------------------------------------------------
# Daemon reachability + autostart
# ---------------------------------------------------------------------------


def _daemon_alive(host: str, timeout: float = 2.0) -> bool:
    """Cheap probe — does the HTTP API answer?"""
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=timeout)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _start_daemon_background() -> subprocess.Popen | None:
    """Spawn ``ollama serve`` detached so this script can exit cleanly.

    Returns the Popen handle (caller does NOT need to wait) or None if the
    ``ollama`` CLI isn't on PATH. The daemon will keep running after this
    script exits because we don't tie its lifetime to ours.
    """
    if not shutil.which("ollama"):
        return None
    try:
        # stdout/stderr to DEVNULL so a slow daemon doesn't block us.
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from this script's process group
        )
        return proc
    except (OSError, FileNotFoundError):
        return None


def _wait_for_daemon(host: str, timeout_s: float = DAEMON_STARTUP_TIMEOUT_S) -> bool:
    """Poll the HTTP API every 500ms until alive or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _daemon_alive(host, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def ensure_daemon(host: str, *, verbose: bool = True) -> bool:
    """Return True iff the daemon is reachable, starting it if necessary."""
    if _daemon_alive(host):
        return True
    if verbose:
        print(f"  Ollama not reachable at {host}; trying to start it...")
    proc = _start_daemon_background()
    if proc is None:
        if verbose:
            print("  [ERR] 'ollama' CLI not on PATH — install Ollama or start the daemon manually.")
        return False
    if _wait_for_daemon(host):
        if verbose:
            print(f"  daemon up after starting (pid={proc.pid}).")
        return True
    if verbose:
        print("  [ERR] daemon did not respond within timeout — check Ollama logs.")
    return False


# ---------------------------------------------------------------------------
# Model inventory
# ---------------------------------------------------------------------------


def _list_installed(host: str, timeout: float = 5.0) -> list[str]:
    """Hit ``GET /api/tags`` and return installed model tags."""
    req = urllib.request.Request(f"{host}/api/tags")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", [])]


def _matches(required: str, installed: list[str]) -> bool:
    """An installed tag matches if the full name OR family prefix matches.

    Ollama tags look like ``deepseek-r1:32b``. Sometimes operators have a
    differently-quantized variant like ``deepseek-r1:32b-q4_K_M`` — still
    fine for our purposes.
    """
    if required in installed:
        return True
    base = required.split(":", 1)[0]
    target_tag = required.split(":", 1)[1] if ":" in required else ""
    for tag in installed:
        if not tag.startswith(base + ":"):
            continue
        installed_tag = tag.split(":", 1)[1] if ":" in tag else ""
        if not target_tag:
            return True
        if installed_tag.startswith(target_tag) or target_tag.startswith(installed_tag):
            return True
    return False


# ---------------------------------------------------------------------------
# Pulling: HTTP-first with CLI fallback
# ---------------------------------------------------------------------------


def _pull_via_http(host: str, model: str, *, verbose: bool = True) -> bool:
    """Pull a model via ``POST /api/pull`` and stream progress.

    Returns True on success. Works without the ``ollama`` CLI binary on PATH
    so this is usable from inside Docker, WSL2, or a stripped-down image.
    """
    payload = json.dumps({"name": model, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/pull",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_report_mb = -PROGRESS_REPORT_EVERY_MB  # ensure first chunk prints
    final_status = ""
    try:
        # No timeout on the read itself — 20GB pulls can run for an hour.
        # We do set a connect-side timeout so a wedged daemon fails loudly.
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in msg:
                    if verbose:
                        print(f"    [ERR] {msg['error']}")
                    return False
                status = msg.get("status", "")
                final_status = status or final_status
                total = msg.get("total")
                completed = msg.get("completed")
                if verbose and total and completed:
                    mb_done = completed / (1024 * 1024)
                    if mb_done - last_report_mb >= PROGRESS_REPORT_EVERY_MB:
                        pct = 100.0 * completed / total
                        print(
                            f"    {status}: {mb_done:6.0f} MB / {total / (1024*1024):6.0f} MB ({pct:5.1f}%)"
                        )
                        last_report_mb = mb_done
                elif verbose and status and status != final_status:
                    # status changed but no progress numbers — print once.
                    print(f"    {status}")
        return final_status == "success" or "success" in final_status.lower()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        if verbose:
            print(f"    [ERR] HTTP pull failed: {e}")
        return False


def _pull_via_cli(model: str, *, verbose: bool = True) -> bool:
    """Fallback: shell out to ``ollama pull``. Inherits the user's terminal
    so they see the native progress bar. Used when HTTP pull fails."""
    if not shutil.which("ollama"):
        return False
    if verbose:
        print(f"    falling back to: ollama pull {model}")
    rc = subprocess.run(["ollama", "pull", model]).returncode
    return rc == 0


def pull_model(host: str, model: str, *, verbose: bool = True) -> bool:
    """Pull a model. HTTP first (works headless), CLI fallback if needed."""
    if verbose:
        print(f"  pulling {model} ...")
    if _pull_via_http(host, model, verbose=verbose):
        return True
    if verbose:
        print(f"    HTTP pull did not finish cleanly; trying CLI fallback for {model}")
    return _pull_via_cli(model, verbose=verbose)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Ollama HTTP host (default $OLLAMA_HOST or 127.0.0.1:11434)",
    )
    ap.add_argument(
        "--pull-missing",
        action="store_true",
        help="Pull any missing models (HTTP API; CLI as fallback).",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Start the daemon if needed, then pull any missing models. Idempotent.",
    )
    ap.add_argument(
        "--profile",
        default=None,
        help="Override hardware profile (e.g. rx_7900_xt).",
    )
    args = ap.parse_args()

    profile = active_profile(env_value=args.profile)
    required = all_models(profile)

    print(f"profile        : {profile.name} — {profile.description}")
    print(f"ollama host    : {args.host}")
    print(f"models required: {len(required)}")
    print()

    # --auto implies --pull-missing AND will try to start the daemon.
    want_pull = args.pull_missing or args.auto
    want_autostart = args.auto

    if want_autostart and not ensure_daemon(args.host):
        return 1

    try:
        installed = _list_installed(args.host)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  [ERR] cannot reach Ollama at {args.host}: {e}")
        print("        is the daemon running?  `ollama serve` or rerun with --auto.")
        return 1

    print(f"models installed: {len(installed)}")
    for tag in installed:
        print(f"   - {tag}")
    print()

    missing: list[str] = []
    for req in required:
        ok = _matches(req, installed)
        mark = "OK " if ok else "MISS"
        print(f"  [{mark}] {req}")
        if not ok:
            missing.append(req)
    print()

    if not missing:
        print("All required models present. You're good.")
        return 0

    print(f"Missing {len(missing)} model(s). To pull them manually:")
    for m in missing:
        print(f"    ollama pull {m}")

    if not want_pull:
        # Read-only mode — exit with code 2 so CI/setup scripts can detect it.
        return 2

    print()
    pulled_all = True
    for m in missing:
        ok = pull_model(args.host, m)
        if not ok:
            print(f"  [ERR] failed to pull {m}")
            pulled_all = False

    if not pulled_all:
        return 2

    # Re-verify against a fresh /api/tags listing so a half-broken cache is
    # caught (e.g. pull succeeded but tag is named slightly differently).
    print()
    print("re-verifying after pull ...")
    try:
        installed = _list_installed(args.host)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  [ERR] post-pull verify failed: {e}")
        return 2

    still_missing = [r for r in required if not _matches(r, installed)]
    if still_missing:
        print(f"  [ERR] still missing after pull: {still_missing}")
        return 2

    print("All required models present. You're good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
