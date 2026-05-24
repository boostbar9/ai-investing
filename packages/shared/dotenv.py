"""Minimal stdlib-only ``.env`` loader.

Loads ``KEY=VALUE`` pairs from a ``.env`` file into ``os.environ`` if not
already set. Mirrors the small subset of ``python-dotenv`` we actually use
so we don't pull in another dependency.

Format rules:
    * Blank lines and lines starting with ``#`` are ignored.
    * Inline comments after a value (preceded by space then ``#``) are stripped.
    * Values may be wrapped in single or double quotes; the quotes are removed.
    * No variable interpolation (``$VAR`` is treated literally).
    * Existing environment variables are NOT overwritten (so explicit shell
      exports always win over the file).

Failing-soft is intentional: a missing or unreadable ``.env`` should never
crash the runner.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATH = Path(".env")


def load_dotenv(path: Path | str = DEFAULT_PATH, *, override: bool = False) -> dict[str, str]:
    """Load env vars from ``path`` into ``os.environ``.

    Returns the dict of values that were actually applied (i.e. set into
    ``os.environ`` on this call). Variables present in ``os.environ`` are
    preserved unless ``override=True``.
    """
    p = Path(path)
    if not p.exists():
        return {}

    applied: dict[str, str] = {}
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Permit "export KEY=VALUE" so .env files copied from shell scripts work.
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not _is_valid_key(key):
            continue
        value = _parse_value(value)

        if not override and key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value

    return applied


def _is_valid_key(key: str) -> bool:
    """POSIX-style env var name: starts with a letter/underscore, then
    letters/digits/underscores."""
    if not key:
        return False
    if not (key[0].isalpha() or key[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in key[1:])


def _parse_value(raw: str) -> str:
    """Strip surrounding quotes and trailing inline comments."""
    v = raw.strip()
    # Quoted: keep everything between the matching quotes; ignore comments.
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        return v[1:-1]
    # Unquoted: strip an inline comment (must be preceded by whitespace).
    if " #" in v:
        v = v.split(" #", 1)[0].rstrip()
    elif "\t#" in v:
        v = v.split("\t#", 1)[0].rstrip()
    return v.strip()
