"""Tests for the stdlib .env loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from packages.shared.dotenv import load_dotenv


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot os.environ so each test starts clean for the keys we touch."""
    for key in ("DV_TEST_KEY_1", "DV_TEST_KEY_2", "DV_TEST_KEY_3", "DV_TEST_QUOTED", "DV_TEST_COMMENT"):
        monkeypatch.delenv(key, raising=False)


def test_missing_file_returns_empty_dict(tmp_path: Path) -> None:
    result = load_dotenv(tmp_path / "nope.env")
    assert result == {}


def test_loads_basic_key_value(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("DV_TEST_KEY_1=hello\nDV_TEST_KEY_2=world\n")
    applied = load_dotenv(p)
    assert applied == {"DV_TEST_KEY_1": "hello", "DV_TEST_KEY_2": "world"}
    assert os.environ["DV_TEST_KEY_1"] == "hello"


def test_ignores_blank_lines_and_comments(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("\n# a comment\nDV_TEST_KEY_1=alpha\n\n# another\nDV_TEST_KEY_2=beta\n")
    applied = load_dotenv(p)
    assert applied == {"DV_TEST_KEY_1": "alpha", "DV_TEST_KEY_2": "beta"}


def test_strips_surrounding_quotes(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text('DV_TEST_QUOTED="quoted value with spaces"\n')
    load_dotenv(p)
    assert os.environ["DV_TEST_QUOTED"] == "quoted value with spaces"


def test_strips_inline_comments_on_unquoted_value(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("DV_TEST_COMMENT=abc123 # this is a note\n")
    load_dotenv(p)
    assert os.environ["DV_TEST_COMMENT"] == "abc123"


def test_inline_hash_inside_quotes_is_preserved(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text('DV_TEST_QUOTED="value#with#hashes"\n')
    load_dotenv(p)
    assert os.environ["DV_TEST_QUOTED"] == "value#with#hashes"


def test_does_not_overwrite_existing_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DV_TEST_KEY_1", "from-shell")
    p = tmp_path / ".env"
    p.write_text("DV_TEST_KEY_1=from-file\n")
    applied = load_dotenv(p)
    assert applied == {}  # nothing was applied
    assert os.environ["DV_TEST_KEY_1"] == "from-shell"


def test_override_replaces_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DV_TEST_KEY_1", "from-shell")
    p = tmp_path / ".env"
    p.write_text("DV_TEST_KEY_1=from-file\n")
    applied = load_dotenv(p, override=True)
    assert applied == {"DV_TEST_KEY_1": "from-file"}
    assert os.environ["DV_TEST_KEY_1"] == "from-file"


def test_accepts_export_prefix(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("export DV_TEST_KEY_1=fromshell\n")
    load_dotenv(p)
    assert os.environ["DV_TEST_KEY_1"] == "fromshell"


def test_skips_malformed_lines(tmp_path: Path) -> None:
    p = tmp_path / ".env"
    p.write_text("not a real line\nDV_TEST_KEY_1=ok\n=no-key\n123KEY=bad-key\n")
    applied = load_dotenv(p)
    # Only the valid one should land.
    assert applied == {"DV_TEST_KEY_1": "ok"}
