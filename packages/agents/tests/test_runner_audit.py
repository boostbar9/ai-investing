"""Verifies the per-decision audit log is populated by `_run` (§17, task 7)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from packages.agents.llm_router import LLMRouter
from packages.agents.runners import build_research_runner
from packages.shared.schemas import ResearchInput


def _router_with(responses):
    class _T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            body = json.loads(request.content)
            model = body["model"]
            for key, val in responses.items():
                if model.startswith(key):
                    if val is None:
                        return httpx.Response(500, json={"error": "boom"})
                    return httpx.Response(200, json={"response": json.dumps(val)})
            return httpx.Response(500, json={"error": "no-stub"})

    client = httpx.AsyncClient(transport=_T(), base_url="http://x")
    return LLMRouter(host="http://x", client=client)


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _patch_audit_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    """Override the default ``path=`` argument on log_decision so the runner
    (which never passes path explicitly) writes into our tmp_path."""
    import packages.persistence.audit as audit_mod
    real = audit_mod.log_decision

    def _patched(*args, **kwargs):
        kwargs.setdefault("path", path)
        return real(*args, **kwargs)

    # Patch the symbol the runner imported (re-exported into runners module).
    import packages.agents.runners as runners_mod
    monkeypatch.setattr(runners_mod, "log_decision", _patched)


@pytest.mark.asyncio
async def test_audit_logs_happy_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audit_path = tmp_path / "decisions.jsonl"
    _patch_audit_path(monkeypatch, audit_path)

    did = uuid4()
    router = _router_with(
        {
            "deepseek-r1": {
                "decision_id": "00000000-0000-0000-0000-000000000000",
                "thesis": "ok",
                "sentiment": 0.1,
                "citations": [],
            }
        }
    )
    run = build_research_runner(router)
    await run(ResearchInput(decision_id=did, symbols=["SPY"]))
    await router.aclose()

    records = _read_audit(audit_path)
    assert len(records) == 1
    r = records[0]
    assert r["agent"] == "research"
    assert r["validation_ok"] is True
    assert r["decision_id"] == str(did)
    assert r["raw_response"]["thesis"] == "ok"


@pytest.mark.asyncio
async def test_audit_logs_validation_error_then_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path = tmp_path / "decisions.jsonl"
    _patch_audit_path(monkeypatch, audit_path)

    did = uuid4()
    # Both attempts return out-of-range sentiment -> two validation errors.
    bad = {
        "decision_id": "00000000-0000-0000-0000-000000000000",
        "thesis": "bad",
        "sentiment": 99.0,  # invalid (range [-1,1])
        "citations": [],
    }
    router = _router_with({"deepseek-r1": bad})
    run = build_research_runner(router)
    await run(ResearchInput(decision_id=did, symbols=["SPY"]))
    await router.aclose()

    records = _read_audit(audit_path)
    # Should record both attempts: attempt 1 fails, attempt 2 also fails.
    assert len(records) >= 2
    assert all(r["validation_ok"] is False for r in records)
    assert records[0]["attempt"] == 1
    assert records[-1]["attempt"] == 2
