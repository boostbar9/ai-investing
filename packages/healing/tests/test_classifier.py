"""Classifier rule coverage."""

from __future__ import annotations

import pytest

from packages.healing.classifier import (
    PATCHABLE_CATEGORIES,
    ErrorCategory,
    classify,
    is_patchable,
)
from packages.healing.error_capture import ErrorEvent


def _ev(exc_type: str, msg: str = "", tb: str = "") -> ErrorEvent:
    return ErrorEvent(
        ts="2026-01-01T00:00:00+00:00",
        where="test",
        exc_type=exc_type,
        exc_message=msg,
        traceback=tb,
    )


@pytest.mark.parametrize(
    "exc,msg,tb,expected",
    [
        ("NotImplementedError", "foo.bar", "", ErrorCategory.MISSING_STUB),
        (
            "ModuleNotFoundError",
            "No module named 'packages.foo'",
            "",
            ErrorCategory.IMPORT_ERROR,
        ),
        ("ImportError", "cannot import name 'X' from 'packages.bar'", "", ErrorCategory.IMPORT_ERROR),
        # Third-party import errors should NOT match
        ("ModuleNotFoundError", "No module named 'numpy'", "", ErrorCategory.UNKNOWN),
        (
            "AttributeError",
            "'Foo' object has no attribute 'bar'",
            'File "/x/packages/foo/bar.py" line 1',
            ErrorCategory.ATTRIBUTE_ERROR,
        ),
        ("AttributeError", "'dict' object has no attribute 'x'", "", ErrorCategory.UNKNOWN),
        ("TypeError", "anything", "", ErrorCategory.TYPE_ERROR),
        ("ValueError", "anything", "", ErrorCategory.VALUE_ERROR),
        ("ConnectError", "tls handshake failed", "", ErrorCategory.NETWORK),
        ("RuntimeError", "httpx.ReadTimeout in flight", "", ErrorCategory.NETWORK),
        ("RuntimeError", "completely unrelated", "", ErrorCategory.UNKNOWN),
    ],
)
def test_classify(exc, msg, tb, expected) -> None:
    assert classify(_ev(exc, msg, tb)) == expected


def test_is_patchable_set() -> None:
    assert is_patchable(ErrorCategory.MISSING_STUB)
    assert is_patchable(ErrorCategory.ATTRIBUTE_ERROR)
    assert is_patchable(ErrorCategory.IMPORT_ERROR)
    assert not is_patchable(ErrorCategory.NETWORK)
    assert not is_patchable(ErrorCategory.UNKNOWN)
    assert frozenset(
        {
            ErrorCategory.MISSING_STUB,
            ErrorCategory.ATTRIBUTE_ERROR,
            ErrorCategory.IMPORT_ERROR,
        }
    ) == PATCHABLE_CATEGORIES
