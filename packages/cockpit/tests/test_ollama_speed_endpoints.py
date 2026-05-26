"""Tests for the LLM-speed-fix endpoints: backend detection, warmup, GPU fix.

These cover the routes added to keep the user out of cold-start hell:

  * ``tools.check_ollama._backend_snapshot``      \u2014 GPU vs CPU classification
  * ``GET  /api/ollama/warmup_status``            \u2014 poll-friendly snapshot
  * ``POST /api/ollama/warmup``                   \u2014 graceful when daemon down
  * ``POST /api/ollama/gpu_fix`` on non-win32     \u2014 returns 400 (script is PS1)
  * ``GET  /api/ollama/gpu_fix``                  \u2014 job status snapshot shape

We hit the FastAPI routes via TestClient so the shapes match what the
cockpit JS actually consumes.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from packages.cockpit.web import server as srv
from tools import check_ollama

# ---------------------------------------------------------------------------
# _backend_snapshot
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _patch_ps(monkeypatch, payload: dict) -> None:
    import json

    def _fake_urlopen(_req, timeout=2.0):
        return _FakeResp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(check_ollama.urllib.request, "urlopen", _fake_urlopen)


def test_backend_snapshot_classifies_gpu(monkeypatch) -> None:
    _patch_ps(
        monkeypatch,
        {"models": [{"name": "llama3:8b", "size": 1000, "size_vram": 950}]},
    )
    out = check_ollama._backend_snapshot("http://127.0.0.1:11434")
    assert out["backend"] == "gpu"
    assert out["vram_used_bytes"] == 950
    assert out["gpu_fraction"] == 0.95
    assert out["loaded"][0]["on_gpu"] is True


def test_backend_snapshot_classifies_cpu_when_vram_is_zero(monkeypatch) -> None:
    _patch_ps(
        monkeypatch,
        {"models": [{"name": "llama3:8b", "size": 1000, "size_vram": 0}]},
    )
    out = check_ollama._backend_snapshot("http://127.0.0.1:11434")
    assert out["backend"] == "cpu"
    assert out["vram_used_bytes"] == 0


def test_backend_snapshot_unknown_when_no_models_loaded(monkeypatch) -> None:
    _patch_ps(monkeypatch, {"models": []})
    out = check_ollama._backend_snapshot("http://127.0.0.1:11434")
    assert out["backend"] == "unknown"
    assert out["loaded"] == []


def test_backend_snapshot_partial_gpu_when_fraction_below_half(monkeypatch) -> None:
    _patch_ps(
        monkeypatch,
        {"models": [{"name": "llama3:8b", "size": 1000, "size_vram": 200}]},
    )
    out = check_ollama._backend_snapshot("http://127.0.0.1:11434")
    assert out["backend"] == "partial-gpu"
    assert out["gpu_fraction"] == 0.2


def test_backend_snapshot_swallows_network_errors(monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(check_ollama.urllib.request, "urlopen", _boom)
    out = check_ollama._backend_snapshot("http://127.0.0.1:11434")
    # No exception leaks; we just return the unknown default.
    assert out["backend"] == "unknown"
    assert out["loaded"] == []


# ---------------------------------------------------------------------------
# /api/ollama/warmup_status + /api/ollama/warmup
# ---------------------------------------------------------------------------


def test_warmup_status_returns_shape() -> None:
    client = TestClient(srv.app)
    r = client.get("/api/ollama/warmup_status")
    assert r.status_code == 200
    body = r.json()
    # Stable contract for the cockpit panel; missing keys would break the UI.
    for key in ("started_at", "finished_at", "in_progress", "results"):
        assert key in body
    assert isinstance(body["results"], list)


def test_warmup_returns_error_when_daemon_down(monkeypatch) -> None:
    """When status_snapshot reports daemon_alive=False, /api/ollama/warmup
    must return ok=False fast instead of hanging the request on retries."""

    def _fake_snapshot(*_a, **_k):
        return {
            "daemon_alive": False,
            "host": "http://127.0.0.1:11434",
            "installed": [],
        }

    monkeypatch.setattr(check_ollama, "status_snapshot", _fake_snapshot)

    client = TestClient(srv.app)
    r = client.post("/api/ollama/warmup")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "daemon" in body["error"].lower()
    assert body["results"] == []


# ---------------------------------------------------------------------------
# /api/ollama/gpu_fix (POST + GET)
# ---------------------------------------------------------------------------


def test_gpu_fix_returns_400_on_non_windows(monkeypatch) -> None:
    monkeypatch.setattr(srv, "sys", srv.sys)  # ensure attr access is safe
    monkeypatch.setattr(srv.sys, "platform", "linux", raising=False)
    client = TestClient(srv.app)
    r = client.post("/api/ollama/gpu_fix")
    assert r.status_code == 400
    assert "Windows-only" in r.json()["detail"]


def test_gpu_fix_status_returns_idle_shape() -> None:
    client = TestClient(srv.app)
    r = client.get("/api/ollama/gpu_fix")
    assert r.status_code == 200
    body = r.json()
    # JobManager.status().to_dict() shape \u2014 the cockpit JS reads .running and
    # .pid to decide whether to stream the SSE log channel.
    assert "running" in body
    assert "kind" in body
    assert body["kind"] == srv.OLLAMA_GPU_FIX_KIND


def test_warmup_status_and_gpu_fix_paths_are_quiet() -> None:
    """The two new poll endpoints must be in the uvicorn quiet list so the
    UI's 1\u20133s polling doesn't flood the log buffer (which we already saw
    paint the screen with garbage in an earlier session)."""
    assert "/api/ollama/warmup_status" in srv._QUIET_PATH_PREFIXES
    assert "/api/ollama/gpu_fix" in srv._QUIET_PATH_PREFIXES
