"""FastAPI entrypoint for the ai-investing cockpit + bot backend.

Phase 0 stub: only exposes /health and /version. Real routes land in Phase 3.
"""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="ai-investing API", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
async def version() -> dict[str, str]:
    return {"spec": "v3.1", "phase": "0-foundation"}
