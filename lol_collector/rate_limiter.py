"""
Dual-window token-bucket rate limiter.

Each API key has TWO rate limits enforced simultaneously:
  - Short window:  N requests per 1 second
  - Long window:   M requests per 120 seconds

The limiter tracks both and blocks until both windows have capacity.
Also respects Riot's Retry-After header for 429 responses.
"""

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class RateWindow:
    """Sliding-window token bucket for a single time window."""
    max_tokens: int
    window_seconds: float
    _timestamps: list = field(default_factory=list, repr=False)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        # bisect would be faster but list is small enough
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.pop(0)

    @property
    def available(self) -> int:
        self._prune(time.monotonic())
        return self.max_tokens - len(self._timestamps)

    def wait_time(self) -> float:
        """Seconds until at least one token is available."""
        now = time.monotonic()
        self._prune(now)
        if len(self._timestamps) < self.max_tokens:
            return 0.0
        # oldest request will expire at timestamps[0] + window_seconds
        return self._timestamps[0] + self.window_seconds - now

    def consume(self) -> None:
        self._timestamps.append(time.monotonic())


class KeyRateLimiter:
    """
    Rate limiter for a single API key with dual windows.

    Usage:
        limiter = KeyRateLimiter(per_second=20, per_two_minutes=100)
        async with limiter:
            # make request
    """

    def __init__(self, key_id: str, per_second: int, per_two_minutes: int):
        self.key_id = key_id
        self._short = RateWindow(max_tokens=per_second, window_seconds=1.0)
        self._long = RateWindow(max_tokens=per_two_minutes, window_seconds=120.0)
        self._lock = asyncio.Lock()
        self._retry_after: float = 0.0  # monotonic time when 429 cooldown ends

    @property
    def available(self) -> int:
        return min(self._short.available, self._long.available)

    def set_retry_after(self, seconds: float) -> None:
        """Called when we get a 429 with Retry-After header."""
        self._retry_after = time.monotonic() + seconds

    async def acquire(self) -> None:
        """Wait until both windows have capacity, then consume a token."""
        while True:
            async with self._lock:
                now = time.monotonic()

                # Respect 429 cooldown
                if now < self._retry_after:
                    wait = self._retry_after - now
                    # release lock while waiting
                else:
                    short_wait = self._short.wait_time()
                    long_wait = self._long.wait_time()
                    wait = max(short_wait, long_wait)

                    if wait <= 0:
                        self._short.consume()
                        self._long.consume()
                        return

            # Sleep outside the lock so other coroutines can check too
            await asyncio.sleep(wait + 0.01)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        pass


class MultiKeyLimiter:
    """
    Manages rate limiters across multiple API keys.

    Picks the key with the most available capacity to spread load.
    """

    def __init__(self):
        self._limiters: dict[str, KeyRateLimiter] = {}
        self._keys: dict[str, str] = {}  # key_id -> actual API key string

    def add_key(self, key_id: str, api_key: str, per_second: int, per_two_minutes: int) -> None:
        self._limiters[key_id] = KeyRateLimiter(key_id, per_second, per_two_minutes)
        self._keys[key_id] = api_key

    def get_api_key(self, key_id: str) -> str:
        return self._keys[key_id]

    @property
    def key_ids(self) -> list[str]:
        return list(self._limiters.keys())

    def get_limiter(self, key_id: str) -> KeyRateLimiter:
        return self._limiters[key_id]

    async def acquire_best(self) -> tuple[str, KeyRateLimiter]:
        """
        Pick the key with the most available capacity and acquire it.
        Returns (key_id, limiter).
        """
        while True:
            # Sort by available capacity, descending
            ranked = sorted(
                self._limiters.items(),
                key=lambda kv: kv[1].available,
                reverse=True,
            )

            best_id, best_limiter = ranked[0]

            if best_limiter.available > 0:
                await best_limiter.acquire()
                return best_id, best_limiter

            # All keys exhausted — find minimum wait across all keys
            min_wait = float("inf")
            for _, lim in ranked:
                w = max(lim._short.wait_time(), lim._long.wait_time())
                if lim._retry_after > time.monotonic():
                    w = max(w, lim._retry_after - time.monotonic())
                min_wait = min(min_wait, w)

            await asyncio.sleep(max(min_wait, 0.05))

    def report_429(self, key_id: str, retry_after: float) -> None:
        """Report a 429 for a specific key."""
        self._limiters[key_id].set_retry_after(retry_after)
