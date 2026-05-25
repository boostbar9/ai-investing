import json

import httpx
import pytest

from packages.agents.llm_router import LLMError, LLMRouter


class _FakeTransport(httpx.AsyncBaseTransport):
    """Fakes Ollama. First model errors; second returns valid JSON."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        self.calls.append(model)
        if "deepseek-r1:70b" in model:
            return httpx.Response(500, json={"error": "oom"})
        return httpx.Response(200, json={"response": json.dumps({"thesis": "ok", "sentiment": 0.4, "citations": []})})


@pytest.mark.asyncio
async def test_router_falls_back():
    transport = _FakeTransport()
    client = httpx.AsyncClient(transport=transport, base_url="http://x")
    router = LLMRouter(host="http://x", client=client)
    out = await router.generate_json("research", "tell me about SPY", decision_id="abc")
    assert out["sentiment"] == 0.4
    # First call hit primary (deepseek-r1:70b) and failed; second hit backup (qwen2.5:72b)
    assert transport.calls[0].startswith("deepseek-r1")
    assert transport.calls[1].startswith("qwen2.5")
    await router.aclose()


@pytest.mark.asyncio
async def test_router_all_fail():
    class Boom(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

    client = httpx.AsyncClient(transport=Boom(), base_url="http://x")
    router = LLMRouter(host="http://x", client=client)
    with pytest.raises(LLMError):
        await router.generate_json("research", "x", decision_id="abc")
    await router.aclose()


class _TagsAwareTransport(httpx.AsyncBaseTransport):
    """Pretends to be Ollama. Reports the given installed-models inventory
    on ``GET /api/tags`` and 404s any ``POST /api/generate`` whose model is
    not in that list. Tracks ordered call history so tests can assert the
    walk order the router picked.
    """

    def __init__(self, installed: list[str], succeed_with: str | None = None) -> None:
        self.installed = installed
        self.succeed_with = succeed_with
        self.calls: list[tuple[str, str]] = []  # (verb, model_or_path)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/api/tags"):
            self.calls.append(("GET", "/api/tags"))
            return httpx.Response(
                200,
                json={"models": [{"name": n} for n in self.installed]},
            )
        body = json.loads(request.content)
        model = body["model"]
        self.calls.append(("POST", model))
        if self.succeed_with and model == self.succeed_with:
            return httpx.Response(
                200,
                json={"response": json.dumps({"ok": True, "served_by": model})},
            )
        if model not in self.installed:
            return httpx.Response(404, json={"error": f"model '{model}' not found"})
        return httpx.Response(
            200,
            json={"response": json.dumps({"ok": True, "served_by": model})},
        )


@pytest.mark.asyncio
async def test_router_skips_uninstalled_models_in_chain():
    """When ``/api/tags`` is reachable, the router must not even attempt
    declared models that aren't pulled -- saves three 404 round-trips and
    a stack trace per agent run."""
    transport = _TagsAwareTransport(
        installed=["qwen3:14b"],  # backup-tier only for research/risk on RX 7900 XT
        succeed_with="qwen3:14b",
    )
    client = httpx.AsyncClient(transport=transport, base_url="http://x")
    # Pin to a known profile so the test isn't sensitive to host env vars.
    import os
    os.environ["HARDWARE_PROFILE"] = "rx_7900_xt"
    router = LLMRouter(host="http://x", client=client)
    out = await router.generate_json("research", "x", decision_id="abc")
    assert out["served_by"] == "qwen3:14b"
    # First call must be the inventory probe.
    assert transport.calls[0] == ("GET", "/api/tags")
    # No generate calls should have targeted deepseek-r1:32b (not installed).
    gens = [m for v, m in transport.calls if v == "POST"]
    assert "deepseek-r1:32b" not in gens
    assert "deepseek-r1:14b" not in gens
    assert gens[0] == "qwen3:14b"
    await router.aclose()


@pytest.mark.asyncio
async def test_router_uses_emergency_fallback_when_chain_unpulled():
    """Operator only has the tiny universal fallback installed; the router
    must still serve a response instead of throwing LLMError."""
    transport = _TagsAwareTransport(
        installed=["qwen2.5:7b-instruct-q4_K_M"],
        succeed_with="qwen2.5:7b-instruct-q4_K_M",
    )
    client = httpx.AsyncClient(transport=transport, base_url="http://x")
    import os
    os.environ["HARDWARE_PROFILE"] = "rx_7900_xt"
    router = LLMRouter(host="http://x", client=client)
    out = await router.generate_json("strategy", "x", decision_id="abc")
    assert out["served_by"] == "qwen2.5:7b-instruct-q4_K_M"
    await router.aclose()


@pytest.mark.asyncio
async def test_router_caches_installed_inventory():
    """Multiple agent runs in quick succession should not re-fetch /api/tags."""
    transport = _TagsAwareTransport(
        installed=["qwen2.5:7b-instruct-q4_K_M"],
        succeed_with="qwen2.5:7b-instruct-q4_K_M",
    )
    client = httpx.AsyncClient(transport=transport, base_url="http://x")
    import os
    os.environ["HARDWARE_PROFILE"] = "rx_7900_xt"
    router = LLMRouter(host="http://x", client=client)
    for _ in range(3):
        await router.generate_json("research", "x", decision_id="abc")
    tag_calls = [c for c in transport.calls if c[0] == "GET"]
    assert len(tag_calls) == 1  # cache served the next two runs
    await router.aclose()


@pytest.mark.asyncio
async def test_router_recovers_when_tags_endpoint_broken():
    """If ``/api/tags`` is unreachable but a declared model works, the
    router must still succeed (degrade to Tier 3 unfiltered walk)."""

    class TagsBroken(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path.endswith("/api/tags"):
                return httpx.Response(500)
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={"response": json.dumps({"ok": True, "served_by": body["model"]})},
            )

    client = httpx.AsyncClient(transport=TagsBroken(), base_url="http://x")
    import os
    os.environ["HARDWARE_PROFILE"] = "rx_7900_xt"
    router = LLMRouter(host="http://x", client=client)
    out = await router.generate_json("research", "x", decision_id="abc")
    # Tier 3 walks declared in order, so primary wins.
    assert out["served_by"] == "deepseek-r1:32b"
    await router.aclose()
