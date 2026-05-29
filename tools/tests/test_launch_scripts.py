"""Static guarantees for the cross-platform launch scripts.

These tests exist because of a real production failure: ``launch.ps1`` was
saved as UTF-8 *without* BOM and contained em-dashes (U+2014) in user-facing
strings. Windows PowerShell 5.1 (the default shell on Windows 10/11) opens
``.ps1`` files in the ANSI codepage unless a UTF-8 BOM is present, so the
3-byte em-dash got decoded as the mojibake string ``\u00e2\u20ac\u201d`` -- which
PowerShell then tried to tokenize, producing a parse-time error before
``launch.ps1`` could run a single line.

We can't unit-test "PowerShell parses this" from CI on Linux, but we can
enforce the two static invariants that prevent the failure mode entirely:

1. ``launch.ps1`` must start with a UTF-8 BOM (``EF BB BF``).
2. Neither launch script may contain non-ASCII characters; ASCII-only is
   the only safe lowest-common-denominator across Windows code pages,
   Git for Windows bash, macOS Terminal, and Linux shells.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

UTF8_BOM = b"\xef\xbb\xbf"


@pytest.mark.parametrize(
    "script_name",
    ["launch.ps1", "repair.ps1", "install-repair-shortcut.ps1"],
)
def test_powershell_scripts_start_with_utf8_bom(script_name: str) -> None:
    """PowerShell 5.1 only treats a .ps1 file as UTF-8 when the BOM is
    present. Without it, the file is decoded using the active Windows
    code page (usually Windows-1252), which mangles every multi-byte
    character into mojibake and breaks the parser. The BOM is the
    single byte sequence that fixes this for every PS version >= 5.1."""
    path = SCRIPTS_DIR / script_name
    assert path.exists(), f"{path} missing"
    head = path.read_bytes()[:3]
    assert head == UTF8_BOM, (
        f"{script_name} must start with a UTF-8 BOM (EF BB BF) so Windows "
        f"PowerShell 5.1 decodes it as UTF-8 instead of the ANSI code "
        f"page. Got bytes: {head!r}."
    )


@pytest.mark.parametrize(
    "script_name",
    [
        "launch.ps1",
        "launch.sh",
        "launch.cmd",
        "repair.ps1",
        "install-repair-shortcut.ps1",
    ],
)
def test_launch_scripts_are_ascii_only(script_name: str) -> None:
    """No em-dashes, en-dashes, smart-quotes, or other non-ASCII bytes
    in the launch scripts. They get re-encoded by every tool in the
    chain (Git, editors, CI artifact uploaders) and one bad round-trip
    is enough to brick the launcher. ASCII-only is the only reliably
    portable choice -- if you want a dash, use ``--``; if you want a
    quote, use ``'`` or ``\"``."""
    path = SCRIPTS_DIR / script_name
    assert path.exists(), f"{path} missing"
    raw = path.read_bytes()
    # Strip the BOM before scanning so launch.ps1 isn't flagged on
    # its own required prefix bytes.
    if raw.startswith(UTF8_BOM):
        raw = raw[len(UTF8_BOM):]
    offending: list[tuple[int, int, int]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        for col, byte in enumerate(line, start=1):
            if byte >= 0x80:
                offending.append((line_no, col, byte))
                break  # one finding per line is enough
    assert not offending, (
        f"{script_name} contains non-ASCII bytes at "
        f"{[(ln, col, hex(b)) for ln, col, b in offending[:5]]}. "
        f"Use ASCII equivalents (`--` for em/en-dash, plain quotes for "
        f"smart quotes)."
    )


def test_launch_ps1_em_dash_regression() -> None:
    """Explicit regression guard for the U+2014 (em-dash) bytes that
    caused the original parser explosion. If anyone re-introduces an
    em-dash anywhere in launch.ps1, this test names the exact byte
    sequence in the failure message so the fix is obvious."""
    raw = (SCRIPTS_DIR / "launch.ps1").read_bytes()
    # UTF-8 encoding of U+2014 EM DASH
    em_dash_utf8 = "\u2014".encode("utf-8")  # b'\xe2\x80\x94'
    # UTF-8 encoding of U+2013 EN DASH (the other common offender)
    en_dash_utf8 = "\u2013".encode("utf-8")  # b'\xe2\x80\x93'
    assert em_dash_utf8 not in raw, (
        "launch.ps1 contains a U+2014 EM DASH. Replace it with `--`. "
        "Windows PowerShell 5.1 in the default Windows-1252 code page "
        "decodes this byte sequence as the mojibake string that breaks "
        "the parser."
    )
    assert en_dash_utf8 not in raw, (
        "launch.ps1 contains a U+2013 EN DASH. Replace it with `-`."
    )
