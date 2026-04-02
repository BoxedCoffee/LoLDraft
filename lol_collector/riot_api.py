"""Async Riot Games API client."""

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any, Optional

import aiohttp

from rate_limiter import MultiKeyLimiter

logger = logging.getLogger("riot_api")

# ── Riot API base URLs ────────────────────────────────────────
# Platform endpoints (summoner, league)
PLATFORM_URL = "https://{platform}.api.riotgames.com"
# Routing endpoints (match-v5)
ROUTING_URL = "https://{routing}.api.riotgames.com"


@dataclass
class RetryConfig:
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 120.0
    backoff_multiplier: float = 2.0
    retry_on_status: tuple = (429, 500, 502, 503, 504)


class RiotAPIError(Exception):
    def __init__(self, status: int, message: str, url: str):
        self.status = status
        self.url = url
        super().__init__(f"HTTP {status} from {url}: {message}")


class RiotClient:
    def __init__(
        self,
        limiter: MultiKeyLimiter,
        retry_config: RetryConfig = RetryConfig(),
        key_region_map: Optional[dict[str, list[str]]] = None,
    ):
        self._limiter = limiter
        self._retry = retry_config
        self._session: Optional[aiohttp.ClientSession] = None
        self._key_region_map = key_region_map or {}
        self._request_count = 0
        self._error_count = 0

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    @property
    def stats(self) -> dict:
        return {"requests": self._request_count, "errors": self._error_count}

    def _pick_key_for_region(self, platform: str) -> Optional[str]:
        candidates = []
        for key_id, regions in self._key_region_map.items():
            if platform in regions:
                candidates.append(key_id)
        return candidates[0] if len(candidates) == 1 else None

    async def _request(self, url: str, preferred_key: Optional[str] = None) -> Any:
        last_exc = None

        for attempt in range(self._retry.max_retries + 1):
            key_id: Optional[str] = None
            try:
                if preferred_key:
                    await self._limiter.acquire_preferred(preferred_key)
                    key_id = preferred_key
                else:
                    key_id, _ = await self._limiter.acquire_best()

                api_key = self._limiter.get_api_key(key_id)
                headers = {"X-Riot-Token": api_key}

                self._request_count += 1
                async with self._session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        return await resp.json()

                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", "5"))
                        self._limiter.report_429(key_id, retry_after)
                        logger.warning(
                            "429 on key %s - backing off %.0fs (attempt %d/%d)",
                            key_id[:8],
                            retry_after,
                            attempt + 1,
                            self._retry.max_retries,
                        )
                        self._error_count += 1
                        await asyncio.sleep(retry_after)
                        continue

                    if resp.status == 404:
                        return None

                    if resp.status == 403:
                        body = await resp.text()
                        logger.error("403 on key %s - may be expired: %s", key_id[:8], body[:200])
                        raise RiotAPIError(403, body[:200], url)

                    if resp.status in self._retry.retry_on_status:
                        self._error_count += 1
                        body = await resp.text()
                        last_exc = RiotAPIError(resp.status, body[:200], url)
                        logger.warning(
                            "HTTP %d on %s (attempt %d/%d)",
                            resp.status,
                            url,
                            attempt + 1,
                            self._retry.max_retries,
                        )
                    else:
                        body = await resp.text()
                        raise RiotAPIError(resp.status, body[:200], url)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self._error_count += 1
                last_exc = e
                logger.warning(
                    "Connection error on %s (attempt %d/%d): %s",
                    url,
                    attempt + 1,
                    self._retry.max_retries,
                    str(e)[:100],
                )
            finally:
                if key_id is not None:
                    self._limiter.release()

            # Exponential backoff with jitter
            delay = min(
                self._retry.base_delay * (self._retry.backoff_multiplier ** attempt),
                self._retry.max_delay,
            )
            jitter = random.uniform(0, delay * 0.3)
            await asyncio.sleep(delay + jitter)

        # Exhausted retries
        raise last_exc or RiotAPIError(0, "Exhausted retries", url)

    # ── League endpoints (platform) ───────────────────────────

    async def get_league_entries(
        self, platform: str, tier: str, division: str = "I", page: int = 1
    ) -> list[dict]:
        """Get league entries for a tier/division. Returns list of summoner entries."""
        url = (
            f"{PLATFORM_URL.format(platform=platform)}"
            f"/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/{division}"
            f"?page={page}"
        )
        key = self._pick_key_for_region(platform)
        result = await self._request(url, preferred_key=key)
        return result or []

    async def get_apex_league(
        self, platform: str, tier: str
    ) -> Optional[dict]:
        """Get challenger/grandmaster/master league. Returns full league object."""
        tier_lower = tier.lower()
        if tier_lower not in ("challenger", "grandmaster", "master"):
            raise ValueError(f"Apex tier must be challenger/grandmaster/master, got {tier}")
        url = (
            f"{PLATFORM_URL.format(platform=platform)}"
            f"/lol/league/v4/{tier_lower}leagues/by-queue/RANKED_SOLO_5x5"
        )
        key = self._pick_key_for_region(platform)
        return await self._request(url, preferred_key=key)

    async def get_summoner_by_id(
        self, platform: str, summoner_id: str
    ) -> Optional[dict]:
        url = (
            f"{PLATFORM_URL.format(platform=platform)}"
            f"/lol/summoner/v4/summoners/{summoner_id}"
        )
        key = self._pick_key_for_region(platform)
        return await self._request(url, preferred_key=key)

    # ── Match endpoints (routing) ─────────────────────────────

    async def get_match_ids(
        self,
        routing: str,
        puuid: str,
        queue: int = 420,
        count: int = 100,
        start: int = 0,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> list[str]:
        """Get match IDs for a player. Returns list of match ID strings."""
        params = f"queue={queue}&count={min(count, 100)}&start={start}"
        if start_time:
            params += f"&startTime={start_time}"
        if end_time:
            params += f"&endTime={end_time}"
        url = (
            f"{ROUTING_URL.format(routing=routing)}"
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids?{params}"
        )
        result = await self._request(url)
        return result or []

    async def get_match(self, routing: str, match_id: str) -> Optional[dict]:
        """Get full match data."""
        url = (
            f"{ROUTING_URL.format(routing=routing)}"
            f"/lol/match/v5/matches/{match_id}"
        )
        return await self._request(url)

    async def get_timeline(self, routing: str, match_id: str) -> Optional[dict]:
        """Get match timeline (per-minute events + frames)."""
        url = (
            f"{ROUTING_URL.format(routing=routing)}"
            f"/lol/match/v5/matches/{match_id}/timeline"
        )
        return await self._request(url)

    # ── Account endpoint (routing) ────────────────────────────

    async def get_account_by_puuid(
        self, routing: str, puuid: str
    ) -> Optional[dict]:
        url = (
            f"{ROUTING_URL.format(routing=routing)}"
            f"/riot/account/v1/accounts/by-puuid/{puuid}"
        )
        return await self._request(url)
