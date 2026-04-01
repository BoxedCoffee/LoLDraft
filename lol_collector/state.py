"""
SQLite-backed collection state for incremental checkpointing.

Tracks:
  - Discovered players (puuid, region, tier)
  - Discovered match IDs (deduplicated across players)
  - Collection status per match (metadata fetched? timeline fetched?)
  - Collection statistics

All operations are async via aiosqlite.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional

import aiosqlite

logger = logging.getLogger("state")


class MatchStatus(IntEnum):
    DISCOVERED = 0       # Match ID known, nothing fetched
    METADATA_DONE = 1    # Match metadata fetched and stored
    TIMELINE_DONE = 2    # Timeline fetched and stored
    COMPLETE = 3         # Fully processed (metadata + timeline + exported)
    SKIPPED = -1         # Filtered out (wrong patch, too short, etc.)
    FAILED = -2          # Permanently failed after max retries


SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    puuid           TEXT PRIMARY KEY,
    platform        TEXT NOT NULL,
    routing         TEXT NOT NULL,
    summoner_id     TEXT,
    tier            TEXT,
    lp              INTEGER DEFAULT 0,
    matches_pulled  INTEGER DEFAULT 0,
    discovered_at   TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT PRIMARY KEY,
    platform        TEXT NOT NULL,
    routing         TEXT NOT NULL,
    status          INTEGER NOT NULL DEFAULT 0,
    game_version    TEXT,
    game_duration   INTEGER,
    queue_id        INTEGER,
    discovered_at   TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    ended_at        TEXT,
    config_hash     TEXT,
    players_found   INTEGER DEFAULT 0,
    matches_found   INTEGER DEFAULT 0,
    timelines_done  INTEGER DEFAULT 0,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_matches_status ON matches(status);
CREATE INDEX IF NOT EXISTS idx_players_platform ON players(platform);
CREATE INDEX IF NOT EXISTS idx_matches_platform ON matches(platform);
"""


class StateDB:
    """Async SQLite state manager."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA busy_timeout=30000")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        logger.info("State DB connected: %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.commit()
            await self._db.close()

    # ── Players ───────────────────────────────────────────────

    async def upsert_player(
        self,
        puuid: str,
        platform: str,
        routing: str,
        summoner_id: str = None,
        tier: str = None,
        lp: int = 0,
    ) -> bool:
        """Insert or update a player. Returns True if newly inserted."""
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            cursor = await self._db.execute(
                "SELECT puuid FROM players WHERE puuid = ?", (puuid,)
            )
            existing = await cursor.fetchone()

            if existing:
                await self._db.execute(
                    """UPDATE players SET tier=COALESCE(?,tier), lp=?,
                       summoner_id=COALESCE(?,summoner_id), updated_at=?
                       WHERE puuid=?""",
                    (tier, lp, summoner_id, now, puuid),
                )
                return False

            await self._db.execute(
                """INSERT INTO players
                   (puuid, platform, routing, summoner_id, tier, lp,
                    discovered_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (puuid, platform, routing, summoner_id, tier, lp, now, now),
            )
            return True

    async def get_players_without_matches(
        self, platform: str = None, limit: int = 500
    ) -> list[dict]:
        """Get players whose match histories haven't been pulled yet."""
        if platform:
            cursor = await self._db.execute(
                """SELECT puuid, platform, routing FROM players
                   WHERE matches_pulled = 0 AND platform = ?
                   ORDER BY lp DESC LIMIT ?""",
                (platform, limit),
            )
        else:
            cursor = await self._db.execute(
                """SELECT puuid, platform, routing FROM players
                   WHERE matches_pulled = 0
                   ORDER BY lp DESC LIMIT ?""",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def mark_player_matches_pulled(self, puuid: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            await self._db.execute(
                "UPDATE players SET matches_pulled=1, updated_at=? WHERE puuid=?",
                (now, puuid),
            )

    async def player_count(self, platform: str = None) -> int:
        if platform:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM players WHERE platform=?", (platform,)
            )
        else:
            cursor = await self._db.execute("SELECT COUNT(*) FROM players")
        row = await cursor.fetchone()
        return row[0]

    # ── Matches ───────────────────────────────────────────────

    async def upsert_match(
        self,
        match_id: str,
        platform: str,
        routing: str,
        status: MatchStatus = MatchStatus.DISCOVERED,
        game_version: str = None,
        game_duration: int = None,
        queue_id: int = None,
    ) -> bool:
        """Insert or update a match. Returns True if newly inserted."""
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            cursor = await self._db.execute(
                "SELECT match_id FROM matches WHERE match_id = ?", (match_id,)
            )
            existing = await cursor.fetchone()

            if existing:
                updates = ["updated_at=?"]
                params = [now]
                if game_version:
                    updates.append("game_version=?")
                    params.append(game_version)
                if game_duration is not None:
                    updates.append("game_duration=?")
                    params.append(game_duration)
                if queue_id is not None:
                    updates.append("queue_id=?")
                    params.append(queue_id)
                if status != MatchStatus.DISCOVERED:
                    updates.append("status=?")
                    params.append(int(status))
                params.append(match_id)
                await self._db.execute(
                    f"UPDATE matches SET {','.join(updates)} WHERE match_id=?",
                    params,
                )
                return False

            await self._db.execute(
                """INSERT INTO matches
                   (match_id, platform, routing, status, game_version,
                    game_duration, queue_id, discovered_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    match_id,
                    platform,
                    routing,
                    int(status),
                    game_version,
                    game_duration,
                    queue_id,
                    now,
                    now,
                ),
            )
            return True

    async def update_match_status(
        self, match_id: str, status: MatchStatus, **kwargs
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        sets = ["status=?", "updated_at=?"]
        params = [int(status), now]
        for k, v in kwargs.items():
            sets.append(f"{k}=?")
            params.append(v)
        params.append(match_id)
        async with self._write_lock:
            await self._db.execute(
                f"UPDATE matches SET {','.join(sets)} WHERE match_id=?", params
            )

    async def get_matches_by_status(
        self,
        status: MatchStatus,
        platform: str = None,
        limit: int = 1000,
    ) -> list[dict]:
        if platform:
            cursor = await self._db.execute(
                """SELECT match_id, platform, routing FROM matches
                   WHERE status=? AND platform=? LIMIT ?""",
                (int(status), platform, limit),
            )
        else:
            cursor = await self._db.execute(
                """SELECT match_id, platform, routing FROM matches
                   WHERE status=? LIMIT ?""",
                (int(status), limit),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def match_count(self, status: MatchStatus = None) -> int:
        if status is not None:
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM matches WHERE status=?", (int(status),)
            )
        else:
            cursor = await self._db.execute("SELECT COUNT(*) FROM matches")
        row = await cursor.fetchone()
        return row[0]

    # ── Runs ──────────────────────────────────────────────────

    async def start_run(self, config_hash: str = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        async with self._write_lock:
            cursor = await self._db.execute(
                "INSERT INTO collection_runs (started_at, config_hash) VALUES (?,?)",
                (now, config_hash),
            )
            await self._db.commit()
            return cursor.lastrowid

    async def end_run(self, run_id: int, **stats) -> None:
        now = datetime.now(timezone.utc).isoformat()
        sets = ["ended_at=?"]
        params = [now]
        for k, v in stats.items():
            sets.append(f"{k}=?")
            params.append(v)
        params.append(run_id)
        async with self._write_lock:
            await self._db.execute(
                f"UPDATE collection_runs SET {','.join(sets)} WHERE id=?", params
            )
            await self._db.commit()

    # ── Stats ─────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        stats = {}
        for label, status in [
            ("discovered", MatchStatus.DISCOVERED),
            ("metadata_done", MatchStatus.METADATA_DONE),
            ("timeline_done", MatchStatus.TIMELINE_DONE),
            ("complete", MatchStatus.COMPLETE),
            ("skipped", MatchStatus.SKIPPED),
            ("failed", MatchStatus.FAILED),
        ]:
            stats[label] = await self.match_count(status)
        stats["total_matches"] = await self.match_count()
        stats["total_players"] = await self.player_count()
        return stats

    async def commit(self) -> None:
        async with self._write_lock:
            await self._db.commit()
