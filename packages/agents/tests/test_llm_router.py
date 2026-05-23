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
