import time

import pytest

from packages.shared.rate_limit import TokenBucket


@pytest.mark.asyncio
async def test_token_bucket_throttles():
    # 5 tokens/sec, capacity 5 — burst of 5 then forced sleep
    b = TokenBucket(rate_per_second=5.0, capacity=5)
    start = time.monotonic()
    for _ in range(7):
        await b.acquire()
    elapsed = time.monotonic() - start
    # After burst of 5 we needed 2 more at 5/s -> ~0.4s minimum
    assert elapsed >= 0.3
