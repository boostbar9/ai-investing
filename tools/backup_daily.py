"""Daily backup tool (§17, task 9).

Zips the artifacts a future debugger or restoration step needs:

    - ``data/``                     (paper logs, audit log, snapshot, SQLite DB)
    - ``logs/`` if present          (cockpit tray + server logs)
    - ``data/cockpit/state.json``   (already inside data/ but explicitly included)

Output: ``backups/YYYY-MM-DD.zip`` (relative to the current working
directory by default, override with ``--out``).

Retention: pass ``--keep N`` to delete older backups beyond the N most
recent. Default 30. ``--keep 0`` disables retention.

This script is intentionally dependency-free (zipfile from stdlib) so it
can run on a fresh Windows machine after a clean checkout. Call it from
the cockpit's autopilot scheduler or run it manually before risky
changes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = Path("backups")
DEFAULT_SOURCES = [Path("data"), Path("logs")]


def build_backup(
    *,
    sources: list[Path] | None = None,
    out_dir: Path = DEFAULT_BACKUP_DIR,
    date_str: str | None = None,
) -> Path:
    """Create a single zip with today's date in the filename. Returns the path written."""
    sources = sources or DEFAULT_SOURCES
    date_str = date_str or datetime.now(UTC).strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.zip"
    # Tempfile-then-replace so a crash mid-write doesn't leave a corrupt zip.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in sources:
            if not src.exists():
                log.info("skip %s (missing)", src)
                continue
            if src.is_file():
                zf.write(src, arcname=str(src))
                continue
            for p in src.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p))
    tmp_path.replace(out_path)
    return out_path


def prune_old(out_dir: Path = DEFAULT_BACKUP_DIR, keep: int = 30) -> list[Path]:
    """Delete *.zip files in ``out_dir`` beyond the N most recent. Returns deleted paths."""
    if keep <= 0:
        return []
    if not out_dir.exists():
        return []
    zips = sorted(out_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    to_delete = zips[keep:]
    deleted: list[Path] = []
    for p in to_delete:
        try:
            p.unlink()
            deleted.append(p)
        except OSError as e:  # pragma: no cover - I/O failure path
            log.warning("could not delete %s: %s", p, e)
    return deleted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily backup of state, logs, and DB.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_BACKUP_DIR,
        help="Output directory (default: ./backups).",
    )
    parser.add_argument(
        "--source", action="append", type=Path,
        help="Add a directory to back up (repeatable). Default: data/ and logs/.",
    )
    parser.add_argument(
        "--keep", type=int, default=30,
        help="Keep N most recent backups (default 30; 0 disables retention).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    sources = args.source if args.source else DEFAULT_SOURCES
    out_path = build_backup(sources=sources, out_dir=args.out)
    deleted = prune_old(out_dir=args.out, keep=args.keep)
    print(f"wrote {out_path}")
    if deleted:
        print(f"pruned {len(deleted)} older backups")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
