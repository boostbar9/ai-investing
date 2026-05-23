"""ai-investing system-tray launcher.

A tiny tray app for Windows (and macOS/Linux via pystray) that gives Devin a
one-click experience over the local stack:

* Start / Stop the full Docker stack (api, cockpit, worker, postgres, dragonfly).
* Open the cockpit in the default browser.
* Check GitHub for new commits on ``main`` (auto-check every 15 min).
* "Update from GitHub" → ``git pull`` + rebuild + restart, all from the menu.
* Show overall status (green dot = running, red = stopped, yellow = updating).

Run:
    python -m tools.tray.launcher

Requires:
    pip install pystray Pillow

This file is import-safe: pystray/Pillow are imported lazily inside ``main`` so
unit tests can import the pure functions (``check_for_updates``,
``stack_status``) without GUI deps.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import httpx  # already a project dep; avoids adding `requests` just for the tray

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "infra" / "docker" / "docker-compose.yml"
COCKPIT_URL = "http://localhost:3000"
GITHUB_API = "https://api.github.com/repos/boostbar9/ai-investing/commits/main"
CHECK_INTERVAL_SECONDS = 15 * 60  # 15 minutes
STATE_FILE = REPO_ROOT / ".tray-state.json"

log = logging.getLogger("tray")


# ---------------------------------------------------------------------------
# Pure helpers (testable, no GUI)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateStatus:
    local_sha: str
    remote_sha: str
    update_available: bool
    error: str | None = None


def local_head_sha() -> str:
    """Return the current local ``HEAD`` commit SHA (first 40 hex chars)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def remote_head_sha(timeout: float = 10.0) -> str:
    """Return the current GitHub ``main`` SHA. Empty string on failure."""
    try:
        r = httpx.get(GITHUB_API, timeout=timeout, headers={"Accept": "application/vnd.github+json"})
        if r.status_code != 200:
            return ""
        sha = r.json().get("sha", "")
        return sha if isinstance(sha, str) else ""
    except httpx.HTTPError:
        return ""


def check_for_updates() -> UpdateStatus:
    """Compare local HEAD against GitHub's ``main``. Pure function — easy to test."""
    local = local_head_sha()
    if not local:
        return UpdateStatus("", "", False, error="not a git repo")
    remote = remote_head_sha()
    if not remote:
        return UpdateStatus(local, "", False, error="could not reach GitHub")
    return UpdateStatus(local, remote, update_available=(local != remote))


def stack_status() -> str:
    """Return ``"running"``, ``"stopped"``, or ``"degraded"`` based on docker compose ps."""
    if not COMPOSE_FILE.exists():
        return "stopped"
    try:
        out = subprocess.check_output(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "ps", "--format", "json"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "stopped"
    # docker compose v2 prints either a JSON array or a stream of JSON objects.
    try:
        text = out.decode().strip()
        if not text:
            return "stopped"
        if text.startswith("["):
            rows = json.loads(text)
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        return "stopped"
    if not rows:
        return "stopped"
    states = {(r.get("State") or "").lower() for r in rows}
    if states <= {"running"}:
        return "running"
    if "running" in states:
        return "degraded"
    return "stopped"


# ---------------------------------------------------------------------------
# Stack control
# ---------------------------------------------------------------------------


def _compose(*args: str) -> int:
    cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    log.info("running: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=REPO_ROOT)


def start_stack() -> bool:
    return _compose("up", "-d") == 0


def stop_stack() -> bool:
    return _compose("down") == 0


def open_cockpit() -> None:
    webbrowser.open(COCKPIT_URL)


def update_from_github() -> tuple[bool, str]:
    """Pull main, rebuild images, recreate containers. Returns (ok, message)."""
    log.info("update: git pull")
    if subprocess.call(["git", "pull", "--ff-only", "origin", "main"], cwd=REPO_ROOT) != 0:
        return False, "git pull failed (uncommitted changes? bad network?)"
    log.info("update: docker compose pull + up")
    if _compose("pull") != 0:
        return False, "docker compose pull failed"
    if _compose("up", "-d", "--build") != 0:
        return False, "docker compose up --build failed"
    return True, "updated to " + (local_head_sha()[:8] or "?")


# ---------------------------------------------------------------------------
# GUI (lazy-imported)
# ---------------------------------------------------------------------------


def _make_icon_image(color: str):
    """Build a 64x64 colored-dot PIL image. Lazy-imports Pillow."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((6, 6, 58, 58), fill=color)
    return img


_STATUS_COLORS = {
    "running": "#22c55e",   # green
    "stopped": "#ef4444",   # red
    "degraded": "#eab308",  # yellow
    "updating": "#3b82f6",  # blue
}


def main() -> int:  # pragma: no cover — GUI loop
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import pystray
    from pystray import MenuItem as Item

    state = {"status": stack_status(), "update": check_for_updates()}

    def _refresh_icon(icon: pystray.Icon) -> None:
        color = _STATUS_COLORS.get(state["status"], "#9ca3af")
        icon.icon = _make_icon_image(color)
        title = f"ai-investing — {state['status']}"
        if state["update"].update_available:
            title += " (update available)"
        icon.title = title

    def _on_start(icon, _item) -> None:
        threading.Thread(target=lambda: (start_stack(), _poll_now(icon)), daemon=True).start()

    def _on_stop(icon, _item) -> None:
        threading.Thread(target=lambda: (stop_stack(), _poll_now(icon)), daemon=True).start()

    def _on_open(_icon, _item) -> None:
        open_cockpit()

    def _on_update(icon, _item) -> None:
        def _run() -> None:
            state["status"] = "updating"
            _refresh_icon(icon)
            ok, msg = update_from_github()
            log.info("update result: %s — %s", ok, msg)
            state["update"] = check_for_updates()
            state["status"] = stack_status()
            _refresh_icon(icon)

        threading.Thread(target=_run, daemon=True).start()

    def _on_quit(icon, _item) -> None:
        icon.stop()

    def _poll_now(icon) -> None:
        state["status"] = stack_status()
        state["update"] = check_for_updates()
        _refresh_icon(icon)

    def _background_poll(icon) -> None:
        while True:
            time.sleep(CHECK_INTERVAL_SECONDS)
            _poll_now(icon)

    menu = pystray.Menu(
        Item("Open cockpit", _on_open, default=True),
        pystray.Menu.SEPARATOR,
        Item("Start stack", _on_start),
        Item("Stop stack", _on_stop),
        Item("Update from GitHub", _on_update),
        pystray.Menu.SEPARATOR,
        Item("Quit", _on_quit),
    )
    icon = pystray.Icon(
        "ai-investing",
        _make_icon_image(_STATUS_COLORS.get(state["status"], "#9ca3af")),
        f"ai-investing — {state['status']}",
        menu,
    )
    threading.Thread(target=_background_poll, args=(icon,), daemon=True).start()
    icon.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# Keep STATE_FILE referenced so future versions can persist user choices
# across restarts without an import-time side effect.
_ = STATE_FILE
_ = sys
_ = os
