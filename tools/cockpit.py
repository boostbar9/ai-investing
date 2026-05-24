"""Launch the local cockpit web GUI.

Usage::

    PYTHONPATH=. python tools/cockpit.py
    PYTHONPATH=. python tools/cockpit.py --port 9000 --host 0.0.0.0

The cockpit serves on http://127.0.0.1:8765 by default and polls
``data/paper_log/runs.jsonl`` for live state.
"""

from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from threading import Timer

log = logging.getLogger("cockpit-launcher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Local ai-investing cockpit web GUI")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=8765, help="Bind port (default 8765)")
    ap.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    ap.add_argument("--reload", action="store_true", help="Enable auto-reload (dev)")
    args = ap.parse_args(argv)

    try:
        import uvicorn  # imported lazily so --help doesn't require it
    except ImportError:
        log.error("uvicorn is not installed. Run: pip install -e .[dev]")
        return 1

    url = f"http://{args.host if args.host != '0.0.0.0' else '127.0.0.1'}:{args.port}"
    log.info("Cockpit starting at %s", url)
    log.info("Press Ctrl+C to stop.")

    if not args.no_browser:
        Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(
            "packages.cockpit.web.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        log.info("Cockpit stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
