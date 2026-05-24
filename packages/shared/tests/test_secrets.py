"""Tests for the secrets layer.

We exercise the .env path explicitly (the keyring path is not unit-tested
because we can't safely write to the host's Credential Manager from CI).
"""

from __future__ import annotations

import os

import pytest

from packages.shared import secrets


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Point the module at a temp .env and clear relevant env vars."""
    # Redirect _env_path / _repo_root to a temp directory.
    monkeypatch.setattr(secrets, "_repo_root", lambda: tmp_path)
    # Force the dotenv path on this test run regardless of host OS.
    monkeypatch.setattr(secrets, "_is_windows", lambda: False)
    # Clear all known keys from the live env so reads come from .env only.
    for k in secrets.ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield


def test_backend_is_dotenv_on_non_windows():
    assert secrets.backend() == "dotenv"


def test_get_returns_empty_when_unset():
    assert secrets.get_secret("ALPACA_PAPER_KEY_ID") == ""
    assert secrets.get_all_secrets()["ALPACA_PAPER_KEY_ID"] == ""


def test_set_then_get_round_trip():
    secrets.set_secrets({"ALPACA_PAPER_KEY_ID": "PKABC123", "ALPACA_PAPER_SECRET": "shh"})
    assert secrets.get_secret("ALPACA_PAPER_KEY_ID") == "PKABC123"
    assert secrets.get_secret("ALPACA_PAPER_SECRET") == "shh"


def test_empty_string_deletes_key():
    secrets.set_secrets({"FRED_API_KEY": "abc"})
    assert secrets.get_secret("FRED_API_KEY") == "abc"
    secrets.set_secrets({"FRED_API_KEY": ""})
    assert secrets.get_secret("FRED_API_KEY") == ""


def test_unknown_keys_are_ignored():
    secrets.set_secrets({"NOT_A_REAL_KEY": "x"})
    # Should not raise and should not show up in any provider's status.
    all_set = {p["id"] for p in secrets.provider_status() if p["configured"]}
    assert "alpaca_paper" not in all_set  # nothing else was set either


def test_process_env_overrides_stored(monkeypatch):
    secrets.set_secrets({"POLYGON_API_KEY": "from-env-file"})
    monkeypatch.setenv("POLYGON_API_KEY", "from-shell")
    assert secrets.get_secret("POLYGON_API_KEY") == "from-shell"


def test_mask_short_and_long():
    assert secrets.mask("") == ""
    assert secrets.mask("abcd") == "****"
    assert secrets.mask("abcdef") == "**cdef"
    assert secrets.mask("PKOS4GYGFSBAZP7HX7IZOGD7OK") == "**" * 0 + "*" * 22 + "D7OK"


def test_provider_status_marks_configured():
    secrets.set_secrets({"ALPACA_PAPER_KEY_ID": "k", "ALPACA_PAPER_SECRET": "s"})
    statuses = {p["id"]: p for p in secrets.provider_status()}
    assert statuses["alpaca_paper"]["configured"] is True
    # FRED only has one key; not setting it should leave it unconfigured.
    assert statuses["fred"]["configured"] is False


def test_dotenv_quotes_values_with_spaces(tmp_path):
    secrets.set_secrets({"FINNHUB_API_KEY": "hello world"})
    contents = (tmp_path / ".env").read_text(encoding="utf-8")
    assert 'FINNHUB_API_KEY="hello world"' in contents


def test_existing_comments_preserved(tmp_path):
    (tmp_path / ".env").write_text(
        "# my comment\nALPACA_PAPER_KEY_ID=old\n# trailing\n",
        encoding="utf-8",
    )
    secrets.set_secrets({"ALPACA_PAPER_KEY_ID": "new"})
    contents = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "# my comment" in contents
    assert "# trailing" in contents
    assert "ALPACA_PAPER_KEY_ID=new" in contents
    assert "old" not in contents


def test_hydrate_environment_sets_unset_vars(monkeypatch):
    secrets.set_secrets({"FRED_API_KEY": "stored"})
    # set_secrets writes to os.environ; clear it so hydrate has work to do.
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    n = secrets.hydrate_environment()
    assert n >= 1
    assert os.environ.get("FRED_API_KEY") == "stored"
