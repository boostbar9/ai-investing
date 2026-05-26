"""Tests for the daily backup tool (§17, task 9)."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from tools.backup_daily import build_backup, prune_old


def _make_tree(root: Path) -> None:
    (root / "data" / "cockpit").mkdir(parents=True)
    (root / "data" / "cockpit" / "state.json").write_text('{"paused": false}')
    (root / "data" / "paper_log").mkdir()
    (root / "data" / "paper_log" / "runs.jsonl").write_text('{"x": 1}\n')
    (root / "logs").mkdir()
    (root / "logs" / "tray.log").write_text("hello\n")


def test_build_backup_packs_data_and_logs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _make_tree(tmp_path)
    out = build_backup(
        sources=[Path("data"), Path("logs")],
        out_dir=Path("backups"),
        date_str="2026-05-26",
    )
    assert out.exists()
    assert out.name == "2026-05-26.zip"
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert any(n.endswith("state.json") for n in names)
    assert any(n.endswith("runs.jsonl") for n in names)
    assert any(n.endswith("tray.log") for n in names)


def test_build_backup_skips_missing_sources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.txt").write_text("y")
    out = build_backup(
        sources=[Path("data"), Path("does-not-exist")],
        out_dir=Path("backups"),
        date_str="2026-05-26",
    )
    assert out.exists()  # missing source did not abort the run


def test_prune_old_keeps_n_most_recent(tmp_path: Path) -> None:
    out_dir = tmp_path / "backups"
    out_dir.mkdir()
    paths = []
    for i in range(5):
        p = out_dir / f"2026-05-{i+1:02d}.zip"
        p.write_bytes(b"x")
        # Stagger mtime so sort is deterministic.
        ts = time.time() - (5 - i)
        import os
        os.utime(p, (ts, ts))
        paths.append(p)
    deleted = prune_old(out_dir=out_dir, keep=2)
    remaining = sorted(out_dir.glob("*.zip"))
    assert len(remaining) == 2
    assert len(deleted) == 3
    # Newest two survive.
    survivor_names = {p.name for p in remaining}
    assert "2026-05-05.zip" in survivor_names
    assert "2026-05-04.zip" in survivor_names


def test_prune_disabled_when_keep_zero(tmp_path: Path) -> None:
    out_dir = tmp_path / "backups"
    out_dir.mkdir()
    for i in range(3):
        (out_dir / f"f{i}.zip").write_bytes(b"x")
    deleted = prune_old(out_dir=out_dir, keep=0)
    assert deleted == []
    assert len(list(out_dir.glob("*.zip"))) == 3
