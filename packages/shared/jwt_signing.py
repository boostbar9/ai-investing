"""5-minute internal JWT signing for inter-service calls (v3.1 §13)."""
from __future__ import annotations

import os
import time

import jwt

DEFAULT_TTL = int(os.getenv("INTERNAL_JWT_TTL_SECONDS", "300"))


def sign(claims: dict, secret: str | None = None, ttl: int = DEFAULT_TTL) -> str:
    secret = secret or os.environ["INTERNAL_JWT_SECRET"]
    now = int(time.time())
    payload = {**claims, "iat": now, "exp": now + ttl}
    return jwt.encode(payload, secret, algorithm="HS256")


def verify(token: str, secret: str | None = None) -> dict:
    secret = secret or os.environ["INTERNAL_JWT_SECRET"]
    return jwt.decode(token, secret, algorithms=["HS256"])
