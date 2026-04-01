"""
Parquet exporter for match metadata and timeline data.

Outputs:
  matches/  — one Parquet file per flush batch
    Columns: match_id, platform, game_version, patch, game_duration,
             game_creation, queue_id, winner,
             [per-participant: champion_id, role, team_id, puuid, kills,
              deaths, assists, gold_earned, total_damage, cs, vision_score,
              win]

  timelines/ — one Parquet file per flush batch
    Columns: match_id, timestamp_min, participant_id,
             position_x, position_y, current_gold, total_gold,
             xp, level, cs, jungle_cs
    + per-frame event columns for objectives
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("exporter")


def extract_patch(game_version: str) -> str:
    """Extract major.minor patch from game version string like '14.10.123.456'."""
    parts = game_version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return game_version


# ── Match metadata schema ─────────────────────────────────────

MATCH_SCHEMA = pa.schema([
    ("match_id", pa.string()),
    ("platform", pa.string()),
    ("game_version", pa.string()),
    ("patch", pa.string()),
    ("game_duration", pa.int32()),
    ("game_creation", pa.int64()),
    ("queue_id", pa.int16()),
    # Per-participant (10 rows per match)
    ("participant_id", pa.int8()),
    ("team_id", pa.int16()),
    ("champion_id", pa.int16()),
    ("champion_name", pa.string()),
    ("role", pa.string()),
    ("individual_position", pa.string()),
    ("puuid", pa.string()),
    ("win", pa.bool_()),
    ("kills", pa.int16()),
    ("deaths", pa.int16()),
    ("assists", pa.int16()),
    ("gold_earned", pa.int32()),
    ("total_damage_to_champions", pa.int32()),
    ("total_cs", pa.int16()),
    ("vision_score", pa.int16()),
    ("summoner1_id", pa.int16()),
    ("summoner2_id", pa.int16()),
])

# ── Timeline frame schema ─────────────────────────────────────

TIMELINE_SCHEMA = pa.schema([
    ("match_id", pa.string()),
    ("timestamp_ms", pa.int64()),
    ("timestamp_min", pa.float32()),
    ("participant_id", pa.int8()),
    ("position_x", pa.int32()),
    ("position_y", pa.int32()),
    ("current_gold", pa.int32()),
    ("total_gold", pa.int32()),
    ("xp", pa.int32()),
    ("level", pa.int8()),
    ("minions_killed", pa.int16()),
    ("jungle_minions_killed", pa.int16()),
    ("time_enemy_spent_controlled", pa.int32()),
    ("damage_stats_total", pa.int32()),
])

# ── Event schema (objectives, kills) ─────────────────────────

EVENT_SCHEMA = pa.schema([
    ("match_id", pa.string()),
    ("timestamp_ms", pa.int64()),
    ("timestamp_min", pa.float32()),
    ("event_type", pa.string()),       # CHAMPION_KILL, ELITE_MONSTER_KILL, BUILDING_KILL, etc.
    ("killer_id", pa.int8()),
    ("victim_id", pa.int8()),          # 0 for non-kill events
    ("assisting_ids", pa.string()),    # JSON array
    ("position_x", pa.int32()),
    ("position_y", pa.int32()),
    ("monster_type", pa.string()),     # DRAGON, BARON, RIFTHERALD, etc.
    ("monster_subtype", pa.string()),  # FIRE_DRAGON, etc.
    ("building_type", pa.string()),    # TOWER_BUILDING, INHIBITOR_BUILDING
    ("lane_type", pa.string()),        # TOP_LANE, MID_LANE, BOT_LANE
    ("team_id", pa.int16()),
])


class ParquetExporter:
    """Buffered Parquet writer with periodic flushing."""

    def __init__(self, matches_dir: str, timelines_dir: str, flush_interval: int = 500):
        self._matches_dir = Path(matches_dir)
        self._timelines_dir = Path(timelines_dir)
        self._events_dir = Path(timelines_dir) / "events"
        self._flush_interval = flush_interval

        self._match_buffer: list[dict] = []
        self._timeline_buffer: list[dict] = []
        self._event_buffer: list[dict] = []
        self._flush_count = 0

        # Create directories
        for d in [self._matches_dir, self._timelines_dir, self._events_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def match_buffer_size(self) -> int:
        return len(self._match_buffer)

    @property
    def timeline_buffer_size(self) -> int:
        return len(self._timeline_buffer)

    def add_match(self, match_data: dict) -> bool:
        """
        Parse a Riot match-v5 response and buffer it.
        Returns True if the match was valid and buffered.
        """
        try:
            info = match_data.get("info", {})
            metadata = match_data.get("metadata", {})

            match_id = metadata.get("matchId", "")
            platform = metadata.get("platformId", "").lower()
            game_version = info.get("gameVersion", "")
            patch = extract_patch(game_version)

            for p_data in info.get("participants", []):
                row = {
                    "match_id": match_id,
                    "platform": platform,
                    "game_version": game_version,
                    "patch": patch,
                    "game_duration": info.get("gameDuration", 0),
                    "game_creation": info.get("gameCreation", 0),
                    "queue_id": info.get("queueId", 0),
                    "participant_id": p_data.get("participantId", 0),
                    "team_id": p_data.get("teamId", 0),
                    "champion_id": p_data.get("championId", 0),
                    "champion_name": p_data.get("championName", ""),
                    "role": p_data.get("teamPosition", ""),
                    "individual_position": p_data.get("individualPosition", ""),
                    "puuid": p_data.get("puuid", ""),
                    "win": p_data.get("win", False),
                    "kills": p_data.get("kills", 0),
                    "deaths": p_data.get("deaths", 0),
                    "assists": p_data.get("assists", 0),
                    "gold_earned": p_data.get("goldEarned", 0),
                    "total_damage_to_champions": p_data.get(
                        "totalDamageDealtToChampions", 0
                    ),
                    "total_cs": (
                        p_data.get("totalMinionsKilled", 0)
                        + p_data.get("neutralMinionsKilled", 0)
                    ),
                    "vision_score": p_data.get("visionScore", 0),
                    "summoner1_id": p_data.get("summoner1Id", 0),
                    "summoner2_id": p_data.get("summoner2Id", 0),
                }
                self._match_buffer.append(row)

            return True

        except Exception as e:
            logger.error("Failed to parse match: %s", e)
            return False

    def add_timeline(self, match_id: str, timeline_data: dict) -> bool:
        """
        Parse a Riot match-v5 timeline response and buffer it.
        Returns True if valid and buffered.
        """
        try:
            info = timeline_data.get("info", {})
            frames = info.get("frames", [])

            for frame in frames:
                timestamp_ms = frame.get("timestamp", 0)
                timestamp_min = timestamp_ms / 60000.0

                # Participant frames
                p_frames = frame.get("participantFrames", {})
                for pid_str, pf in p_frames.items():
                    pos = pf.get("position", {})
                    dmg = pf.get("damageStats", {})
                    row = {
                        "match_id": match_id,
                        "timestamp_ms": timestamp_ms,
                        "timestamp_min": round(timestamp_min, 2),
                        "participant_id": int(pid_str),
                        "position_x": pos.get("x", 0),
                        "position_y": pos.get("y", 0),
                        "current_gold": pf.get("currentGold", 0),
                        "total_gold": pf.get("totalGold", 0),
                        "xp": pf.get("xp", 0),
                        "level": pf.get("level", 1),
                        "minions_killed": pf.get("minionsKilled", 0),
                        "jungle_minions_killed": pf.get(
                            "jungleMinionsKilled", 0
                        ),
                        "time_enemy_spent_controlled": pf.get(
                            "timeEnemySpentControlled", 0
                        ),
                        "damage_stats_total": dmg.get(
                            "totalDamageDoneToChampions", 0
                        ),
                    }
                    self._timeline_buffer.append(row)

                # Events (kills, objectives, buildings)
                for event in frame.get("events", []):
                    etype = event.get("type", "")
                    if etype not in (
                        "CHAMPION_KILL",
                        "ELITE_MONSTER_KILL",
                        "BUILDING_KILL",
                        "TURRET_PLATE_DESTROYED",
                    ):
                        continue

                    pos = event.get("position", {})
                    erow = {
                        "match_id": match_id,
                        "timestamp_ms": event.get("timestamp", 0),
                        "timestamp_min": round(
                            event.get("timestamp", 0) / 60000.0, 2
                        ),
                        "event_type": etype,
                        "killer_id": event.get("killerId", 0),
                        "victim_id": event.get("victimId", 0),
                        "assisting_ids": json.dumps(
                            event.get("assistingParticipantIds", [])
                        ),
                        "position_x": pos.get("x", 0),
                        "position_y": pos.get("y", 0),
                        "monster_type": event.get("monsterType", ""),
                        "monster_subtype": event.get("monsterSubType", ""),
                        "building_type": event.get("buildingType", ""),
                        "lane_type": event.get("laneType", ""),
                        "team_id": event.get("teamId", 0),
                    }
                    self._event_buffer.append(erow)

            return True

        except Exception as e:
            logger.error("Failed to parse timeline for %s: %s", match_id, e)
            return False

    def should_flush(self) -> bool:
        """Check if any buffer has reached the flush threshold."""
        # Count matches (10 rows per match)
        match_games = len(self._match_buffer) // 10
        return match_games >= self._flush_interval or len(self._timeline_buffer) > 50000

    def flush(self) -> dict:
        """Write buffered data to Parquet files. Returns counts."""
        counts = {"matches": 0, "timeline_frames": 0, "events": 0}

        if self._match_buffer:
            df = pd.DataFrame(self._match_buffer)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self._matches_dir / f"matches_{ts}_{self._flush_count:04d}.parquet"
            table = pa.Table.from_pandas(df, schema=MATCH_SCHEMA)
            pq.write_table(table, path, compression="snappy")
            counts["matches"] = len(df) // 10
            self._match_buffer.clear()
            logger.info("Flushed %d matches to %s", counts["matches"], path.name)

        if self._timeline_buffer:
            df = pd.DataFrame(self._timeline_buffer)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self._timelines_dir / f"frames_{ts}_{self._flush_count:04d}.parquet"
            table = pa.Table.from_pandas(df, schema=TIMELINE_SCHEMA)
            pq.write_table(table, path, compression="snappy")
            counts["timeline_frames"] = len(df)
            self._timeline_buffer.clear()
            logger.info(
                "Flushed %d timeline frames to %s",
                counts["timeline_frames"],
                path.name,
            )

        if self._event_buffer:
            df = pd.DataFrame(self._event_buffer)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = self._events_dir / f"events_{ts}_{self._flush_count:04d}.parquet"
            table = pa.Table.from_pandas(df, schema=EVENT_SCHEMA)
            pq.write_table(table, path, compression="snappy")
            counts["events"] = len(df)
            self._event_buffer.clear()

        self._flush_count += 1
        return counts

    def flush_remaining(self) -> dict:
        """Flush any remaining buffered data."""
        if self._match_buffer or self._timeline_buffer or self._event_buffer:
            return self.flush()
        return {"matches": 0, "timeline_frames": 0, "events": 0}
