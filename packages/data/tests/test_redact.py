"""Tests for secret redaction in logs / log-exposing endpoints."""

from __future__ import annotations

import logging

from packages.data.redact import RedactionFilter, install_redaction, redact


def test_redacts_finnhub_token_in_url():
    url = "https://finnhub.io/api/v1/quote?symbol=AAPL&token=abc123secret"
    out = redact(url)
    assert "abc123secret" not in out
    assert "token=***" in out
    # Innocent params survive.
    assert "symbol=AAPL" in out


def test_redacts_various_secret_params():
    text = (
        "?api_key=KKK&apikey=LLL&access_token=MMM&key=NNN&secret=OOO"
        "&password=PPP&sig=QQQ"
    )
    out = redact(text)
    for leaked in ("KKK", "LLL", "MMM", "NNN", "OOO", "PPP", "QQQ"):
        assert leaked not in out
    assert out.count("***") == 7


def test_redacts_authorization_header():
    text = "Authorization: Bearer sk-supersecrettoken"
    out = redact(text)
    assert "sk-supersecrettoken" not in out
    assert "Bearer" in out
    assert "***" in out


def test_redacts_cockpit_token_header():
    assert "tok-123" not in redact("X-Cockpit-Token: tok-123")
    assert "key-9" not in redact("X-Api-Key: key-9")


def test_redact_is_idempotent():
    once = redact("token=abc&x=1")
    twice = redact(once)
    assert once == twice == "token=***&x=1"


def test_redact_empty_and_non_secret_passthrough():
    assert redact("") == ""
    assert redact("nothing sensitive here symbol=SPY") == (
        "nothing sensitive here symbol=SPY"
    )


def test_redaction_filter_masks_log_record():
    flt = RedactionFilter()
    rec = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="GET ?token=%s done",
        args=("leakme",),
        exc_info=None,
    )
    assert flt.filter(rec) is True
    assert "leakme" not in rec.getMessage()
    assert "token=***" in rec.getMessage()


def test_install_redaction_is_idempotent():
    logger = logging.getLogger("test_redact_install")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    install_redaction(logger)
    install_redaction(logger)
    handler = logger.handlers[0]
    redaction_filters = [
        f for f in handler.filters if getattr(f, "_cockpit_redact", False)
    ]
    assert len(redaction_filters) == 1
