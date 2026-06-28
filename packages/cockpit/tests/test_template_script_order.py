"""Regression guard for the load-order bug that left dashboards stuck on
"Loading..." after the visibility-aware polling refactor.

If ``cockpit.js`` is loaded with the ``defer`` attribute but inline page
scripts later in the body reference ``Cockpit.poll`` / ``Cockpit.flashSaved``
at parse time (not inside a ``DOMContentLoaded`` handler), the inline script
runs *before* the deferred bundle has defined the ``Cockpit`` object,
throwing ``ReferenceError`` and breaking every fetch on the page.

This test enforces the simple invariant we settled on: any template that
references ``Cockpit.*`` from an inline ``<script>`` block must load
``cockpit.js`` synchronously (no ``defer``) so the global is defined
before the inline script runs.
"""

from __future__ import annotations

import re
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parents[1] / "web" / "templates"


def _templates():
    # Skip ``_``-prefixed partials (e.g. _nav.html): they are never served
    # standalone, only included into full pages that load the bundle, so a
    # ``Cockpit.*`` reference in an onclick handler there is always safe.
    return sorted(p for p in TEMPLATES.glob("*.html") if not p.name.startswith("_"))


def test_no_template_loads_cockpit_js_with_defer():
    """cockpit.js must load synchronously so inline scripts can use it."""
    offenders = []
    for path in _templates():
        text = path.read_text()
        if re.search(r'src="/static/cockpit\.js"\s+defer', text):
            offenders.append(path.name)
    assert not offenders, (
        "These templates load cockpit.js with defer, which causes inline "
        "scripts that reference Cockpit.poll / Cockpit.flashSaved at parse "
        "time to throw ReferenceError and leave dashboards stuck on "
        f"'Loading...': {offenders}"
    )


def test_every_template_that_uses_cockpit_loads_the_bundle():
    """Sanity check: any template using Cockpit.* must load cockpit.js."""
    missing = []
    for path in _templates():
        text = path.read_text()
        if "Cockpit." in text and "/static/cockpit.js" not in text:
            missing.append(path.name)
    assert not missing, (
        f"Templates use Cockpit.* but don't include cockpit.js: {missing}"
    )
