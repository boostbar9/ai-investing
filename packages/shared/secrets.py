"""Secrets layer with Windows Credential Manager as primary, .env as fallback.

The cockpit's Settings page needs to read/write API keys without exposing them
in plain text on the filesystem when possible. On Windows we use the OS
Credential Manager via ``keyring``. On macOS/Linux we fall back to a plain
``.env`` file in the repo root (still gitignored).

Read order (first hit wins):

1. process env vars (so ``ALPACA_PAPER_KEY_ID=...`` on the shell wins)
2. OS keystore (keyring) under service name ``ai-investing``
3. ``.env`` file in the repo root

Writes go to whichever backend is available:

* On Windows: keyring (Windows Credential Manager)
* Elsewhere: ``.env``

The ``.env`` is also kept in sync on Windows so existing tools that read
``os.environ`` after ``load_dotenv()`` continue to work — we treat the keystore
as authoritative, but mirror to .env so subprocesses see the values too.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Final

log = logging.getLogger("secrets")

SERVICE_NAME: Final[str] = "ai-investing"

# Provider definitions: human label + list of env keys they own.
# The order here drives the Settings UI.
PROVIDERS: Final[dict[str, dict[str, list[str] | str]]] = {
    "alpaca_paper": {
        "label": "Alpaca (paper)",
        "keys": ["ALPACA_PAPER_KEY_ID", "ALPACA_PAPER_SECRET"],
    },
    "alpaca_live": {
        "label": "Alpaca (live - DANGER)",
        "keys": ["ALPACA_LIVE_KEY_ID", "ALPACA_LIVE_SECRET"],
    },
    "fred": {
        "label": "FRED (macro data)",
        "keys": ["FRED_API_KEY"],
    },
    "polygon": {
        "label": "Polygon.io",
        "keys": ["POLYGON_API_KEY"],
    },
    "alphavantage": {
        "label": "AlphaVantage",
        "keys": ["ALPHAVANTAGE_API_KEY"],
    },
    "finnhub": {
        "label": "Finnhub",
        "keys": ["FINNHUB_API_KEY"],
    },
    "notifications": {
        "label": "Notifications (greenlight webhook)",
        "keys": ["SHADOW_FLIP_WEBHOOK_URL"],
    },
}

# Map env key -> provider id, for reverse lookups.
KEY_TO_PROVIDER: Final[dict[str, str]] = {
    k: pid
    for pid, p in PROVIDERS.items()
    for k in (p["keys"] if isinstance(p["keys"], list) else [])
}

ALL_KEYS: Final[list[str]] = list(KEY_TO_PROVIDER.keys())


def _is_windows() -> bool:
    return platform.system().lower().startswith("win") or sys.platform.startswith("win")


def _try_keyring():  # type: ignore[no-untyped-def]
    """Import keyring lazily; return module or None if unavailable."""
    try:
        import keyring  # type: ignore[import-not-found]

        return keyring
    except Exception:
        return None


def _repo_root() -> Path:
    # packages/shared/secrets.py -> repo root 2 levels up.
    return Path(__file__).resolve().parents[2]


def _env_path() -> Path:
    return _repo_root() / ".env"


# --------------------------------------------------------------------------
# .env helpers (minimal, stdlib only)
# --------------------------------------------------------------------------


def _read_env_file() -> dict[str, str]:
    p = _env_path()
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Strip matching quotes.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def _write_env_file(updates: dict[str, str]) -> None:
    """Merge ``updates`` into .env, preserving comments and ordering.

    Keys with empty values are removed entirely.
    """
    p = _env_path()
    existing_lines: list[str] = []
    seen_keys: set[str] = set()
    if p.exists():
        existing_lines = p.read_text(encoding="utf-8").splitlines()

    out_lines: list[str] = []
    for raw in existing_lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            out_lines.append(raw)
            continue
        body = stripped
        if body.startswith("export "):
            body = body[len("export ") :].strip()
        if "=" not in body:
            out_lines.append(raw)
            continue
        key = body.split("=", 1)[0].strip()
        if key in updates:
            new_val = updates[key]
            if new_val == "":
                # Drop the line entirely.
                seen_keys.add(key)
                continue
            out_lines.append(f"{key}={_quote_if_needed(new_val)}")
            seen_keys.add(key)
        else:
            out_lines.append(raw)

    # Append any new keys that weren't already in the file.
    appended = False
    for k, v in updates.items():
        if k in seen_keys or v == "":
            continue
        if not appended:
            if out_lines and out_lines[-1].strip() != "":
                out_lines.append("")
            out_lines.append("# Added by cockpit settings page")
            appended = True
        out_lines.append(f"{k}={_quote_if_needed(v)}")

    p.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    # Restrict perms on POSIX so secrets aren't world-readable.
    if not _is_windows():
        with contextlib.suppress(OSError):
            p.chmod(0o600)


def _quote_if_needed(v: str) -> str:
    if any(c in v for c in (" ", "#", "'", '"')):
        return '"' + v.replace('"', '\\"') + '"'
    return v


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def backend() -> str:
    """Return the active write backend: 'keyring' or 'dotenv'."""
    if _is_windows() and _try_keyring() is not None:
        return "keyring"
    return "dotenv"


def get_secret(key: str) -> str:
    """Read a secret. Checks env, then keystore, then .env."""
    # 1. Process env wins (lets shell exports override).
    v = os.environ.get(key)
    if v:
        return v
    # 2. Keystore (Windows primary).
    kr = _try_keyring()
    if kr is not None:
        try:
            stored = kr.get_password(SERVICE_NAME, key)
            if stored:
                return stored
        except Exception as e:
            log.debug("keyring read failed for %s: %s", key, e)
    # 3. .env fallback.
    return _read_env_file().get(key, "")


def get_all_secrets() -> dict[str, str]:
    """Return a dict of every known key -> value (empty string if unset)."""
    return {k: get_secret(k) for k in ALL_KEYS}


def set_secrets(updates: dict[str, str]) -> dict[str, str]:
    """Persist a set of key -> value pairs.

    Empty string deletes. Unknown keys are silently ignored. Returns the
    refreshed all-secrets dict so the UI can re-render immediately.
    """
    sanitized = {k: v for k, v in updates.items() if k in ALL_KEYS}
    if not sanitized:
        return get_all_secrets()

    kr = _try_keyring() if _is_windows() else None

    if kr is not None:
        for k, v in sanitized.items():
            try:
                if v == "":
                    with contextlib.suppress(Exception):
                        kr.delete_password(SERVICE_NAME, k)
                else:
                    kr.set_password(SERVICE_NAME, k, v)
            except Exception as e:
                log.warning("keyring write failed for %s: %s", k, e)

    # Mirror to .env regardless of keyring availability so that subprocesses
    # which only call load_dotenv() (e.g. paper_trade.py) still see the values.
    _write_env_file(sanitized)

    # Update the live process env so the rest of the request handles it.
    for k, v in sanitized.items():
        if v == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    return get_all_secrets()


def mask(value: str) -> str:
    """Return a masked rendition safe for the UI: keeps last 4 chars."""
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


def provider_status() -> list[dict[str, object]]:
    """Build a UI-facing list of providers + which of their keys are set."""
    out: list[dict[str, object]] = []
    for pid, meta in PROVIDERS.items():
        keys = meta["keys"] if isinstance(meta["keys"], list) else []
        key_states = []
        all_set = True
        for k in keys:
            v = get_secret(k)
            key_states.append({"key": k, "is_set": bool(v), "masked": mask(v)})
            if not v:
                all_set = False
        out.append(
            {
                "id": pid,
                "label": meta["label"],
                "keys": key_states,
                "configured": all_set,
            }
        )
    return out


def hydrate_environment() -> int:
    """Pull every known secret from the keystore into ``os.environ``.

    Returns the number of variables populated. Safe to call at app startup so
    subprocesses launched after this point inherit credentials even when the
    user has only configured them via the GUI (not .env).
    """
    n = 0
    for k in ALL_KEYS:
        if os.environ.get(k):
            continue
        v = get_secret(k)
        if v:
            os.environ[k] = v
            n += 1
    return n
