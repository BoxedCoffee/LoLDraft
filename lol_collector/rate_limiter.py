"""Dual-window token-bucket rate limiter."""

import asyncio
import time
from dataclasses import dataclass, field


SAFETY_MARGIN = 0.80


@dataclass
class RateWindow:
    max_tokens: int
    window_seconds: float
    _timestamps: list = field(default_factory=list, repr=False)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.pop(0)

    @property
    def available(self) -> int:
        self._prune(time.monotonic())
        return self.max_tokens - len(self._timestamps)

    def wait_time(self) -> float:
        now = time.monotonic()
        self._prune(now)
        if len(self._timestamps) < self.max_tokens:
            return 0.0
        return self._timestamps[0] + self.window_seconds - now

    def consume(self) -> None:
        self._timestamps.append(time.monotonic())


class KeyRateLimiter:
    def __init__(self, key_id: str, per_second: int, per_two_minutes: int):
        self.key_id = key_id
        safe_ps = max(1, int(per_second * SAFETY_MARGIN))
        safe_p2m = max(1, int(per_two_minutes * SAFETY_MARGIN))
        self._short = RateWindow(max_tokens=safe_ps, window_seconds=1.0)
        self._long = RateWindow(max_tokens=safe_p2m, window_seconds=120.0)
        self._lock = asyncio.Lock()
        self._retry_after: float = 0.0

    @property
    def available(self) -> int:
        return min(self._short.available, self._long.available)

    def set_retry_after(self, seconds: float) -> None:
        self._retry_after = time.monotonic() + seconds

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                if now < self._retry_after:
                    wait = self._retry_after - now
                else:
                    short_wait = self._short.wait_time()
                    long_wait = self._long.wait_time()
                    wait = max(short_wait, long_wait)

                    if wait <= 0:
                        self._short.consume()
                        self._long.consume()
                        return

            await asyncio.sleep(wait + 0.05)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        pass


class MultiKeyLimiter:
    def __init__(self, max_total_concurrent: int = 8):
        self._limiters: dict[str, KeyRateLimiter] = {}
        self._keys: dict[str, str] = {}
        self._global_sem = asyncio.Semaphore(max(1, int(max_total_concurrent)))

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

    async def acquire_preferred(self, key_id: str) -> KeyRateLimiter:
        await self._global_sem.acquire()
        try:
            limiter = self._limiters[key_id]
            await limiter.acquire()
            return limiter
        except Exception:
            self._global_sem.release()
            raise

    async def acquire_best(self) -> tuple[str, KeyRateLimiter]:
        await self._global_sem.acquire()
        try:
            while True:
                ranked = sorted(
                    self._limiters.items(),
                    key=lambda kv: kv[1].available,
                    reverse=True,
                )

                best_id, best_limiter = ranked[0]
                if best_limiter.available > 0:
                    await best_limiter.acquire()
                    return best_id, best_limiter

                min_wait = float("inf")
                now = time.monotonic()
                for _, lim in ranked:
                    w = max(lim._short.wait_time(), lim._long.wait_time())
                    if lim._retry_after > now:
                        w = max(w, lim._retry_after - now)
                    min_wait = min(min_wait, w)

                await asyncio.sleep(max(min_wait, 0.1))
        except Exception:
            self._global_sem.release()
            raise

    def release(self) -> None:
        self._global_sem.release()

    def report_429(self, key_id: str, retry_after: float) -> None:
        self._limiters[key_id].set_retry_after(retry_after)
