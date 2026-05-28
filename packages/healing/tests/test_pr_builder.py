"""PR builder tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.healing import pr_builder as pb
from packages.healing.error_capture import ErrorEvent
from packages.healing.pr_builder import (
    BuildResult,
    build_diff,
    build_patch,
    open_draft_pr,
    write_patch_file,
)
from packages.healing.stub_synth import StubPatch


@pytest.fixture
def isolated_patches(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "patches"
    monkeypatch.setattr(pb, "PATCHES_DIR", p)
    return p


def _ev() -> ErrorEvent:
    return ErrorEvent(
        ts="2026-05-28T19:00:00+00:00",
        where="test",
        exc_type="AttributeError",
        exc_message="'Foo' object has no attribute 'bar'",
        traceback="",
    )


def _stub_new_file() -> StubPatch:
    return StubPatch(
        target_path="packages/healing/_pretend.py",
        mode="new_file",
        symbol="packages.healing._pretend",
        snippet='"""new"""\n\n\ndef x():\n    raise NotImplementedError("x")\n',
        rationale="r",
    )


def test_build_diff_new_file(tmp_path) -> None:
    stub = _stub_new_file()
    diff = build_diff(stub, repo_root=tmp_path)
    assert "+++ b/packages/healing/_pretend.py" in diff
    assert "+def x():" in diff
    assert "--- a/packages/healing/_pretend.py" in diff


def test_build_diff_append_method(tmp_path) -> None:
    target = tmp_path / "packages" / "x" / "y.py"
    target.parent.mkdir(parents=True)
    target.write_text("class C:\n    def existing(self):\n        return 1\n")
    stub = StubPatch(
        target_path="packages/x/y.py",
        mode="append_method",
        symbol="missing",
        snippet="\n    def missing(self, *a, **kw):\n        raise NotImplementedError\n",
        rationale="r",
    )
    diff = build_diff(stub, repo_root=tmp_path)
    assert "+    def missing(self" in diff
    # original content not removed
    assert "-class C:" not in diff


def test_write_patch_file_creates_dir(isolated_patches: Path) -> None:
    p = write_patch_file("diff content", "label-1")
    assert p.exists()
    assert p.read_text() == "diff content"
    assert p.parent == isolated_patches


def test_build_patch_dry_run_by_default(
    isolated_patches: Path, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(pb, "AUTO_PR_ENABLED", False)
    stub = _stub_new_file()
    res = build_patch(stub, _ev(), repo_root=tmp_path)
    assert isinstance(res, BuildResult)
    assert res.dry_run is True
    assert res.pr_url is None
    assert res.patch_path is not None and res.patch_path.exists()
    assert res.diff.startswith("---")


def test_build_patch_invokes_creator_when_enabled(
    isolated_patches: Path, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(pb, "AUTO_PR_ENABLED", True)
    called = {}

    def fake_creator(*, title, body, branch, diff):
        called["title"] = title
        called["body"] = body
        called["branch"] = branch
        called["diff"] = diff
        return "https://github.com/x/y/pull/1"

    stub = _stub_new_file()
    res = build_patch(stub, _ev(), repo_root=tmp_path, creator=fake_creator)
    assert res.dry_run is False
    assert res.pr_url == "https://github.com/x/y/pull/1"
    assert "AttributeError" in called["title"]
    assert called["branch"].startswith("healing/")
    assert "Auto-stub" in called["body"]


def test_build_patch_disabled_ignores_creator(
    isolated_patches: Path, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pb, "AUTO_PR_ENABLED", False)
    creator_called = False

    def fake_creator(*, title, body, branch, diff):
        nonlocal creator_called
        creator_called = True
        return "u"

    res = build_patch(_stub_new_file(), _ev(), repo_root=tmp_path, creator=fake_creator)
    assert res.dry_run is True
    assert not creator_called


def test_open_draft_pr_forces_creator(
    isolated_patches: Path, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(pb, "AUTO_PR_ENABLED", False)
    calls = []

    def fake_creator(*, title, body, branch, diff):
        calls.append((title, branch))
        return "https://github.com/x/y/pull/9"

    res = open_draft_pr(_stub_new_file(), _ev(), fake_creator, repo_root=tmp_path)
    assert res.dry_run is False
    assert res.pr_url.endswith("/9")
    assert len(calls) == 1
