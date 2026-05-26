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
# Binary resolution
# ---------------------------------------------------------------------------
#
# Windows installers can leave multiple ollama.exe on PATH. The two we see in
# the wild on AMD machines are:
#
#   1. C:\Users\<user>\AppData\Local\Programs\Ollama\ollama.exe
#      Standard ollama.com installer. Ships generic ROCm libs that
#      *sometimes* fail on RDNA3 (no gfx1100 tensor files in the
#      consolidated v0.30.0 7z).
#
#   2. C:\Users\<user>\AppData\Local\AMD\AI_Bundle\Ollama\ollama.exe
#      Bundled by the January 2026+ Adrenalin driver "AI tab" install.
#      Ships AMD-blessed ROCm libs that are known-good for the
#      RX 7700+ (gfx1100/gfx1103/gfx1201). On these machines we want to
#      *prefer* this binary because the standard one will pick CPU.
#
# Resolution order (first match wins):
#   1. ``$COCKPIT_OLLAMA_BIN`` env var if set (explicit user override)
#   2. The Adrenalin AI_Bundle path if it exists on disk
#   3. Whatever ``shutil.which("ollama")`` returns (legacy behavior)
#
# The flavor ("adrenalin" / "standard" / "unknown") is surfaced to the UI so
# the user can see which binary is actually running.


def _adrenalin_candidate() -> str | None:
    """Return the Adrenalin-installed ollama.exe path, or None.

    Lives under %LOCALAPPDATA%\\AMD\\AI_Bundle\\Ollama on Windows. Returns
    None on non-Windows systems or when the file isn't present.
    """
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    candidate = os.path.join(local, "AMD", "AI_Bundle", "Ollama", "ollama.exe")
    return candidate if os.path.isfile(candidate) else None


def _standard_candidate() -> str | None:
    """Return the ollama.com installer path under %LOCALAPPDATA%, or None."""
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return None
    candidate = os.path.join(local, "Programs", "Ollama", "ollama.exe")
    return candidate if os.path.isfile(candidate) else None


def resolve_ollama_binary() -> tuple[str | None, str]:
    """Pick which ollama executable the cockpit should invoke.

    Returns a ``(path, flavor)`` pair where ``flavor`` is one of:

      * ``"override"``   - the user pinned the path via env var
      * ``"adrenalin"``  - AMD AI_Bundle binary detected on disk
      * ``"standard"``   - ollama.com installer binary detected on disk
      * ``"path"``       - whatever appears first on PATH (Linux/Mac/unknown)
      * ``"missing"``    - nothing found anywhere (path is None)

    The cockpit prefers the Adrenalin binary because it ships AMD's blessed
    ROCm libs for the RX 7700+. When both are present the standard binary
    is ignored unless the user overrides via ``COCKPIT_OLLAMA_BIN``.
    """
    override = os.environ.get("COCKPIT_OLLAMA_BIN")
    if override and os.path.isfile(override):
        return override, "override"

    adrenalin = _adrenalin_candidate()
    if adrenalin:
        return adrenalin, "adrenalin"

    standard = _standard_candidate()
    if standard:
        return standard, "standard"

    on_path = shutil.which("ollama")
    if on_path:
        # Some users have a non-Windows install or a custom location that
        # doesn't match our LOCALAPPDATA probes. Trust PATH as a fallback.
        return on_path, "path"

    return None, "missing"


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
    binary, _flavor = resolve_ollama_binary()
    if binary is None:
        return None
    try:
        # stdout/stderr to DEVNULL so a slow daemon doesn't block us.
        proc = subprocess.Popen(
            [binary, "serve"],
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


def _backend_snapshot(host: str, timeout: float = 2.0) -> dict:
    """Detect whether Ollama is using GPU or CPU.

    Calls ``/api/ps`` which reports any *currently loaded* model along
    with a ``size_vram`` field. If size_vram > 0 the model is on the GPU.
    If no model is loaded we can't tell from /api/ps alone, so we return
    ``backend="unknown"`` and the caller surfaces an idle state to the UI.

    Returns a dict with keys:
      * backend:  'gpu' | 'cpu' | 'unknown'
      * loaded:   list of currently-loaded models (with size_vram + size)
      * vram_used_bytes: total VRAM in use
      * gpu_fraction:  fraction of the loaded model on GPU (0.0–1.0)
    """
    out: dict = {
        "backend": "unknown",
        "loaded": [],
        "vram_used_bytes": 0,
        "gpu_fraction": 0.0,
    }
    try:
        req = urllib.request.Request(f"{host}/api/ps")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return out
    models = data.get("models") or []
    total_vram = 0
    total_size = 0
    for m in models:
        vram = int(m.get("size_vram") or 0)
        size = int(m.get("size") or 0)
        total_vram += vram
        total_size += size
        out["loaded"].append({
            "name": m.get("name") or m.get("model") or "?",
            "size_vram": vram,
            "size": size,
            "on_gpu": vram > 0,
        })
    out["vram_used_bytes"] = total_vram
    if total_size > 0:
        out["gpu_fraction"] = round(total_vram / total_size, 3)
    if models:
        if total_vram > 0:
            out["backend"] = "gpu" if out["gpu_fraction"] >= 0.5 else "partial-gpu"
        else:
            out["backend"] = "cpu"
    return out


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
    binary, _flavor = resolve_ollama_binary()
    if binary is None:
        return False
    if verbose:
        print(f"    falling back to: ollama pull {model}")
    rc = subprocess.run([binary, "pull", model]).returncode
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
# Status snapshot (used by the cockpit GUI)
# ---------------------------------------------------------------------------


def status_snapshot(host: str | None = None, profile_name: str | None = None) -> dict:
    """Read-only inventory for the cockpit's Ollama panel.

    Pure function: never spawns the daemon, never pulls, never writes.
    Safe to call on every page poll. Returns a JSON-serializable dict the
    frontend can render directly.
    """
    h = host or DEFAULT_HOST
    profile = active_profile(env_value=profile_name)
    required = all_models(profile)

    binary, flavor = resolve_ollama_binary()
    alive = _daemon_alive(h)
    if not alive:
        return {
            "host": h,
            "daemon_alive": False,
            "profile": {"name": profile.name, "description": profile.description},
            "required": required,
            "installed": [],
            "missing": required,  # we can't verify, so report worst case
            "ready": False,
            "cli_on_path": bool(shutil.which("ollama")),
            "ollama_binary": binary,
            "ollama_flavor": flavor,
            "backend": {"backend": "unknown", "loaded": [], "vram_used_bytes": 0, "gpu_fraction": 0.0},
        }

    try:
        installed = _list_installed(h)
    except (urllib.error.URLError, TimeoutError, OSError):
        installed = []
    missing = [r for r in required if not _matches(r, installed)]
    backend = _backend_snapshot(h)
    return {
        "host": h,
        "daemon_alive": True,
        "profile": {"name": profile.name, "description": profile.description},
        "required": required,
        "installed": installed,
        "missing": missing,
        "ready": len(missing) == 0,
        "cli_on_path": bool(shutil.which("ollama")),
        "ollama_binary": binary,
        "ollama_flavor": flavor,
        "backend": backend,
    }


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
