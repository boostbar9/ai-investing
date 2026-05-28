"""Stub synthesis tests."""

from __future__ import annotations

from packages.healing.classifier import ErrorCategory
from packages.healing.error_capture import ErrorEvent
from packages.healing.stub_synth import synthesize_stub


def _ev(exc_type: str, msg: str = "", tb: str = "") -> ErrorEvent:
    return ErrorEvent(
        ts="2026-01-01T00:00:00+00:00",
        where="t",
        exc_type=exc_type,
        exc_message=msg,
        traceback=tb,
    )


def test_synth_returns_none_for_non_patchable() -> None:
    ev = _ev("RuntimeError", "boom")
    assert synthesize_stub(ev, ErrorCategory.UNKNOWN) is None
    assert synthesize_stub(ev, ErrorCategory.NETWORK) is None
    assert synthesize_stub(ev, ErrorCategory.TYPE_ERROR) is None


def test_synth_attribute_error_emits_method() -> None:
    tb = 'Traceback ...\n  File "/repo/packages/foo/bar.py", line 12, in something\n    self.missing()\n'
    ev = _ev("AttributeError", "'Bar' object has no attribute 'missing'", tb)
    patch = synthesize_stub(ev, ErrorCategory.ATTRIBUTE_ERROR)
    assert patch is not None
    assert patch.mode == "append_method"
    assert patch.symbol == "missing"
    assert patch.target_path == "packages/foo/bar.py"
    assert "def missing(self" in patch.snippet
    assert "NotImplementedError" in patch.snippet


def test_synth_attribute_error_without_traceback_returns_none() -> None:
    ev = _ev("AttributeError", "'X' object has no attribute 'y'", "")
    assert synthesize_stub(ev, ErrorCategory.ATTRIBUTE_ERROR) is None


def test_synth_import_error_first_party() -> None:
    ev = _ev("ModuleNotFoundError", "No module named 'packages.healing.missing'")
    patch = synthesize_stub(ev, ErrorCategory.IMPORT_ERROR)
    assert patch is not None
    assert patch.mode == "new_file"
    assert patch.target_path == "packages/healing/missing.py"
    assert "Auto-stubbed module" in patch.snippet
    assert patch.symbol == "packages.healing.missing"


def test_synth_import_error_third_party_returns_none() -> None:
    ev = _ev("ModuleNotFoundError", "No module named 'pandas'")
    assert synthesize_stub(ev, ErrorCategory.IMPORT_ERROR) is None


def test_synth_not_implemented_surfaces_symbol() -> None:
    tb = 'Traceback ...\n  File "/repo/packages/healing/foo.py", line 1, in <module>\n    raise NotImplementedError("do_thing")\n'
    ev = _ev("NotImplementedError", "do_thing", tb)
    patch = synthesize_stub(ev, ErrorCategory.MISSING_STUB)
    assert patch is not None
    assert patch.symbol == "do_thing"
    assert "NotImplementedError" in patch.snippet
    assert patch.target_path == "packages/healing/foo.py"


def test_synth_not_implemented_empty_message() -> None:
    ev = _ev("NotImplementedError", "")
    patch = synthesize_stub(ev, ErrorCategory.MISSING_STUB)
    assert patch is not None
    assert patch.symbol == "<unknown>"
    # No snippet when symbol is unknown.
    assert patch.snippet == ""


def test_synth_attribute_error_normalises_windows_paths() -> None:
    tb = 'File "C:\\repo\\packages\\foo\\bar.py", line 1\n'
    ev = _ev("AttributeError", "'X' object has no attribute 'baz'", tb)
    patch = synthesize_stub(ev, ErrorCategory.ATTRIBUTE_ERROR)
    assert patch is not None
    assert patch.target_path == "packages/foo/bar.py"
