"""Error classification.

Maps an ``ErrorEvent`` to an ``ErrorCategory`` using lightweight rules
over the exception type and message. We deliberately keep this rule-based
rather than ML-driven -- the categories are small in number, the cost
of mis-classification is "the synth produces a no-op patch", and the
operator reviews everything before merge anyway.

Categories drive ``stub_synth``:

* ``missing_stub``     -- ``NotImplementedError`` raised by an unimplemented
                          method or function. Highest-confidence target.
* ``attribute_error``  -- ``AttributeError`` on a project module/class.
                          Synth proposes a stub method.
* ``import_error``     -- ``ImportError`` / ``ModuleNotFoundError`` for a
                          first-party module. Synth proposes a stub module.
* ``type_error``       -- ``TypeError``. Often signature mismatch; synth
                          surfaces but does not patch automatically.
* ``value_error``      -- ``ValueError``. Same -- surface only.
* ``network``          -- transient HTTP/TLS errors (httpx, urllib3, ssl).
* ``unknown``          -- anything else.
"""
from __future__ import annotations

from enum import StrEnum

from packages.healing.error_capture import ErrorEvent


class ErrorCategory(StrEnum):
    MISSING_STUB = "missing_stub"
    ATTRIBUTE_ERROR = "attribute_error"
    IMPORT_ERROR = "import_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    NETWORK = "network"
    UNKNOWN = "unknown"


_NETWORK_HINTS = (
    "httpx.",
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "RemoteProtocolError",
    "SSLError",
    "ssl.SSLError",
    "ConnectionResetError",
    "urllib3",
)


def _looks_first_party(message: str) -> bool:
    """Heuristic: does the message refer to a packages.* module?"""
    return "packages." in message or "ai_investing" in message


def classify(event: ErrorEvent) -> ErrorCategory:
    exc = event.exc_type
    msg = event.exc_message or ""
    tb = event.traceback or ""

    if exc == "NotImplementedError":
        return ErrorCategory.MISSING_STUB

    if exc in {"ImportError", "ModuleNotFoundError"} and _looks_first_party(msg):
        return ErrorCategory.IMPORT_ERROR

    if exc == "AttributeError":
        # AttributeError("'Foo' object has no attribute 'bar'") -- treat as
        # missing-stub-y when it points at a packages.* class via tb.
        if "packages." in tb or "packages/" in tb or "packages\\" in tb or _looks_first_party(msg):
            return ErrorCategory.ATTRIBUTE_ERROR
        return ErrorCategory.UNKNOWN

    if exc == "TypeError":
        return ErrorCategory.TYPE_ERROR

    if exc == "ValueError":
        return ErrorCategory.VALUE_ERROR

    for hint in _NETWORK_HINTS:
        if hint in exc or hint in tb or hint in msg:
            return ErrorCategory.NETWORK

    return ErrorCategory.UNKNOWN


PATCHABLE_CATEGORIES = frozenset(
    {
        ErrorCategory.MISSING_STUB,
        ErrorCategory.ATTRIBUTE_ERROR,
        ErrorCategory.IMPORT_ERROR,
    }
)


def is_patchable(category: ErrorCategory) -> bool:
    return category in PATCHABLE_CATEGORIES
