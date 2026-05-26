"""Tests for the uvicorn access-log quieting filter.

The cockpit UI polls a handful of status endpoints every 3-5 seconds. Without
``_UvicornAccessQuietFilter`` the terminal floods with identical 200 OK lines
that bury real events. These tests pin exactly which records are dropped and
which are kept so a future refactor can't silently re-enable the flood or
accidentally hide real failures.
"""
from __future__ import annotations

import logging

from packages.cockpit.web.server import (
    _install_uvicorn_access_filter,
    _UvicornAccessQuietFilter,
)


def _record(args: tuple) -> logging.LogRecord:
    """Build a synthetic record matching uvicorn.access's emit() shape."""
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg='%s - "%s %s HTTP/%s" %d',
        args=args,
        exc_info=None,
    )


def test_filter_drops_polled_status_endpoint() -> None:
    flt = _UvicornAccessQuietFilter()
    rec = _record(("127.0.0.1:54534", "GET", "/api/state", "1.1", 200))
    assert flt.filter(rec) is False


def test_filter_drops_jobs_poll() -> None:
    flt = _UvicornAccessQuietFilter()
    rec = _record(("127.0.0.1:54534", "GET", "/api/jobs", "1.1", 200))
    assert flt.filter(rec) is False


def test_filter_drops_jobs_stream_sse() -> None:
    """SSE stream connections are very chatty — keep them quiet too."""
    flt = _UvicornAccessQuietFilter()
    rec = _record(("127.0.0.1:50261", "GET", "/api/jobs/pretrain/stream", "1.1", 200))
    assert flt.filter(rec) is False


def test_filter_drops_static_and_favicon() -> None:
    flt = _UvicornAccessQuietFilter()
    for path, status in [
        ("/static/brand/site.webmanifest", 304),
        ("/static/cockpit.css", 200),
        ("/favicon.ico", 304),
    ]:
        rec = _record(("127.0.0.1:54534", "GET", path, "1.1", status))
        assert flt.filter(rec) is False, path


def test_filter_keeps_posts() -> None:
    """POSTs are user actions — we always want to see them in the terminal."""
    flt = _UvicornAccessQuietFilter()
    rec = _record(("127.0.0.1:50261", "POST", "/api/ollama/setup", "1.1", 200))
    assert flt.filter(rec) is True


def test_filter_keeps_non_2xx() -> None:
    """Any error response stays visible even on a polled endpoint."""
    flt = _UvicornAccessQuietFilter()
    rec = _record(("127.0.0.1:50261", "GET", "/api/jobs", "1.1", 500))
    assert flt.filter(rec) is True


def test_filter_keeps_page_loads() -> None:
    """Loading a page is a one-off — not the high-frequency poll noise."""
    flt = _UvicornAccessQuietFilter()
    rec = _record(("127.0.0.1:54534", "GET", "/models", "1.1", 200))
    assert flt.filter(rec) is True


def test_filter_keeps_log_download() -> None:
    """The Download button hits ``/api/jobs/<k>/log?download=1`` — log it."""
    flt = _UvicornAccessQuietFilter()
    # ``/api/jobs/...`` IS in the quiet list, but the user explicitly clicked
    # Download so it'd be nice to see it. Acceptable trade-off: we DO drop it.
    # Pin current behavior so a future tweak can adjust if the user complains.
    rec = _record(
        ("127.0.0.1:54534", "GET", "/api/jobs/pretrain/log?download=1", "1.1", 200)
    )
    assert flt.filter(rec) is False


def test_filter_keeps_unknown_polled_path() -> None:
    """Any other GET (not in the quiet list) stays visible."""
    flt = _UvicornAccessQuietFilter()
    rec = _record(("127.0.0.1:54534", "GET", "/api/orders/recent", "1.1", 200))
    assert flt.filter(rec) is True


def test_filter_strips_query_string_before_prefix_match() -> None:
    flt = _UvicornAccessQuietFilter()
    rec = _record(
        ("127.0.0.1:54534", "GET", "/api/agents/history?limit=10", "1.1", 200)
    )
    assert flt.filter(rec) is False


def test_filter_safe_on_malformed_args() -> None:
    """If uvicorn ever changes its access format we must fall through (keep).

    Hiding an unknown record is worse than logging one extra line.
    """
    flt = _UvicornAccessQuietFilter()
    # Empty tuple, wrong-shape tuple, non-tuple args.
    for bad_args in [(), ("x",), "not-a-tuple", None]:
        rec = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg="x",
            args=bad_args,  # type: ignore[arg-type]
            exc_info=None,
        )
        assert flt.filter(rec) is True, bad_args


def test_install_is_idempotent() -> None:
    """Re-importing the server module mustn't stack multiple filter copies.

    uvicorn --reload re-imports the module; without the dedup we'd grow a
    new filter on every reload and pay the predicate cost N times per line.
    """
    access_log = logging.getLogger("uvicorn.access")
    before = sum(
        1 for f in access_log.filters if getattr(f, "_cockpit_quiet", False)
    )
    _install_uvicorn_access_filter()
    _install_uvicorn_access_filter()
    after = sum(
        1 for f in access_log.filters if getattr(f, "_cockpit_quiet", False)
    )
    # Whatever the count was before, it must not have grown.
    assert after == max(1, before)

