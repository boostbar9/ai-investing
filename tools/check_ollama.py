"""Pre-flight: verify Ollama is reachable and all models in the active profile are pulled.

Usage:
    .venv/bin/python tools/check_ollama.py
    .venv/bin/python tools/check_ollama.py --pull-missing

Exit codes:
    0 \u2014 every required model is present and the daemon responds.
    1 \u2014 daemon unreachable.
    2 \u2014 daemon up but at least one required model is missing.

The script is intentionally side-effect-free unless ``--pull-missing`` is
given. It prints a tidy report so the operator can copy/paste the
``ollama pull`` commands needed to close the gap.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# Allow running as ``python tools/check_ollama.py`` from the repo root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from packages.agents.model_profiles import active_profile, all_models  # noqa: E402

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def _list_installed(host: str, timeout: float = 5.0) -> list[str]:
    """Hit ``GET /api/tags`` and return the list of installed model names."""
    req = urllib.request.Request(f"{host}/api/tags")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", [])]


def _matches(required: str, installed: list[str]) -> bool:
    """An installed tag matches if the full name OR the family prefix matches.

    Ollama tags look like ``deepseek-r1:32b``. Sometimes operators have a
    differently-quantized variant like ``deepseek-r1:32b-q4_K_M`` \u2014 still
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=DEFAULT_HOST, help="Ollama HTTP host (default $OLLAMA_HOST or 127.0.0.1:11434)")
    ap.add_argument("--pull-missing", action="store_true", help="Run ``ollama pull`` for each missing model (requires CLI on PATH).")
    ap.add_argument("--profile", default=None, help="Override hardware profile (e.g. rx_7900_xt).")
    args = ap.parse_args()

    profile = active_profile(env_value=args.profile)
    required = all_models(profile)

    print(f"profile        : {profile.name} \u2014 {profile.description}")
    print(f"ollama host    : {args.host}")
    print(f"models required: {len(required)}")
    print()

    try:
        installed = _list_installed(args.host)
    except urllib.error.URLError as e:
        print(f"  [ERR] cannot reach Ollama at {args.host}: {e}")
        print("        is the daemon running?  `ollama serve` or restart the app.")
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

    if args.pull_missing:
        import shutil
        import subprocess

        if not shutil.which("ollama"):
            print("\n  [ERR] --pull-missing requested but 'ollama' CLI is not on PATH.")
            return 2
        print()
        for m in missing:
            print(f"  pulling {m} ...")
            rc = subprocess.run(["ollama", "pull", m]).returncode
            if rc != 0:
                print(f"  [ERR] failed to pull {m} (rc={rc})")
                return 2
        print("\nAll missing models pulled.")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
