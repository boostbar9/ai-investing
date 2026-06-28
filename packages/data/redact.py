"""Secret redaction for logs and log-exposing endpoints.

Several adapters pass credentials in query strings (Finnhub uses
``?token=<key>``) or ``Authorization`` headers. When those URLs/headers
land in a log line — or are surfaced through ``/api/remote/weblog`` — the
secret leaks in plaintext. This module masks them.

We change ONLY what gets logged, never how requests are actually made.

The redaction is deliberately conservative: it targets a small allowlist
of known-sensitive parameter/header names so innocent params like
``symbol=AAPL`` are left readable.
"""
from __future__ import annotations

import logging
import re

__all__ = ["redact", "RedactionFilter", "install_redaction"]

# Sensitive query-string parameter names (case-insensitive). Finnhub's
# ``token`` is the live offender; the rest are common credential params.
_SECRET_PARAMS = (
    "token",
    "apikey",
    "api_key",
    "apiKey",
    "key",
    "access_token",
    "auth",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "pwd",
    "signature",
    "sig",
)

_MASK = "***"

# ``?token=abc&x=1`` / ``&apikey=abc`` -> keep the name + ``=``, mask value.
# Value is any run of non-delimiter chars (stops at & # whitespace quote).
_PARAM_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(p) for p in _SECRET_PARAMS) + r")=([^&#\s\"'<>]+)"
)

# ``Authorization: Bearer xyz`` and ``X-Cockpit-Token: xyz`` style headers.
_HEADER_RE = re.compile(
    r"(?i)(authorization|x-cockpit-token|x-api-key)(\s*[:=]\s*)(bearer\s+)?(\S+)"
)


def redact(text: str) -> str:
    """Return ``text`` with secret query params and auth headers masked.

    Non-string / empty input is returned unchanged. Idempotent: running it
    twice produces the same output.
    """
    if not text:
        return text
    out = _PARAM_RE.sub(lambda m: f"{m.group(1)}={_MASK}", text)
    out = _HEADER_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3) or ''}{_MASK}", out
    )
    return out


class RedactionFilter(logging.Filter):
    """Logging filter that masks secrets in the rendered message.

    Attach to handlers (not loggers) so it runs for every record a handler
    emits, including records that merely propagated up to the root logger.
    The filter renders ``record.getMessage()``, redacts it, and — only when
    something changed — replaces ``record.msg`` and clears ``record.args``
    so downstream formatting sees the masked text.

    Returns ``True`` always (a filter that drops nothing, only rewrites).
    """

    # Marker so installation is idempotent across uvicorn --reload.
    _cockpit_redact = True

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        red = redact(msg)
        if red != msg:
            record.msg = red
            record.args = ()
        return True


def install_redaction(logger: logging.Logger | None = None) -> None:
    """Attach a :class:`RedactionFilter` to every handler on ``logger``.

    Defaults to the root logger so console + rotating-file output are both
    covered. Idempotent: a handler that already carries a redaction filter
    is skipped.
    """
    root = logger or logging.getLogger()
    for handler in root.handlers:
        if any(getattr(f, "_cockpit_redact", False) for f in handler.filters):
            continue
        handler.addFilter(RedactionFilter())
