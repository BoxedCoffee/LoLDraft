"""
Collection pipeline orchestrator.

Stages:
  1. Discover players — pull Master/GM/Challenger leagues per region
  2. Discover matches — pull match histories per player, deduplicate
  3. Fetch metadata  — pull match details, filter by patch/duration
  4. Fetch timelines — pull timeline data for valid matches
  5. Export          — flush to Parquet

Each stage is resumable — checkpointed via StateDB.
"""

import asyncio
import logging
from typing import Optional

from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from riot_api import RiotClient
from state import StateDB, MatchStatus
from export import ParquetExporter

logger = logging.getLogger("pipeline")


class CollectionPipeline:
    def __init__(
        self,
        client: RiotClient,
        state: StateDB,
        exporter: ParquetExporter,
        regions: list[dict],
        config: dict,
    ):
        self.client = client
        self.state = state
        self.exporter = exporter
        self.regions = regions  # [{"platform": "na1", "routing": "americas"}, ...]
        self.cfg = config
        self._stop = False

    def request_stop(self):
        """Signal graceful shutdown."""
        self._stop = True
        logger.info("Graceful shutdown requested...")

    # ── Stage 1: Discover Players ─────────────────────────────

    async def discover_players(self) -> int:
        """
        Pull high-elo player lists from all configured regions.
        Returns total new players discovered.
        """
        total_new = 0

        for region in self.regions:
            if self._stop:
                break

            platform = region["platform"]
            routing = region["routing"]
            logger.info("Discovering players in %s...", platform.upper())

            for tier in ["CHALLENGER", "GRANDMASTER", "MASTER"]:
                if self._stop:
                    break

                try:
                    league = await self.client.get_apex_league(platform, tier)
                    if not league:
                        logger.warning("No %s league data for %s", tier, platform)
                        continue

                    entries = league.get("entries", [])
                    logger.info(
                        "  %s %s: %d players", platform.upper(), tier, len(entries)
                    )

                    for entry in entries:
                        summoner_id = entry.get("summonerId", "")
                        lp = entry.get("leaguePoints", 0)

                        if not summoner_id:
                            continue

                        summoner = await self.client.get_summoner_by_id(
                            platform, summoner_id
                        )
                        if not summoner:
                            continue

                        puuid = summoner.get("puuid", "")
                        if not puuid:
                            continue

                        is_new = await self.state.upsert_player(
                            puuid=puuid,
                            platform=platform,
                            routing=routing,
                            summoner_id=summoner_id,
                            tier=tier,
                            lp=lp,
                        )
                        if is_new:
                            total_new += 1

                    await self.state.commit()

                except Exception as e:
                    logger.error("Error discovering %s players in %s: %s", tier, platform, e)
                    continue

        logger.info("Player discovery complete: %d new players", total_new)
        return total_new

    # ── Stage 2: Discover Match IDs ───────────────────────────

    async def discover_matches(
        self,
        batch_size: int = 50,
        concurrency: int = 5,
    ) -> int:
        """
        Pull match histories for all undiscovered players.
        Returns total new match IDs found.
        """
        total_new = 0
        sem = asyncio.Semaphore(concurrency)

        for region in self.regions:
            if self._stop:
                break

            platform = region["platform"]
            routing = region["routing"]

            while True:
                if self._stop:
                    break

                players = await self.state.get_players_without_matches(
                    platform=platform, limit=batch_size
                )
                if not players:
                    break

                logger.info(
                    "Pulling match histories for %d players in %s...",
                    len(players),
                    platform.upper(),
                )

                async def _pull_one(player: dict) -> int:
                    async with sem:
                        if self._stop:
                            return 0
                        try:
                            match_ids = await self.client.get_match_ids(
                                routing=player["routing"],
                                puuid=player["puuid"],
                                queue=self.cfg["collection"]["queue_id"],
                                count=self.cfg["collection"]["matches_per_player"],
                            )
                            new = 0
                            for mid in match_ids:
                                is_new = await self.state.upsert_match(
                                    match_id=mid,
                                    platform=platform,
                                    routing=routing,
                                )
                                if is_new:
                                    new += 1

                            await self.state.mark_player_matches_pulled(
                                player["puuid"]
                            )
                            return new

                        except Exception as e:
                            logger.error(
                                "Error pulling matches for %s: %s",
                                player["puuid"][:12],
                                e,
                            )
                            return 0

                tasks = [_pull_one(p) for p in players]
                results = await asyncio.gather(*tasks)
                batch_new = sum(results)
                total_new += batch_new
                await self.state.commit()

                logger.info(
                    "  Batch done: %d new match IDs (total: %d)",
                    batch_new,
                    total_new,
                )

        logger.info("Match discovery complete: %d new matches", total_new)
        return total_new

    # ── Stage 3: Fetch Match Metadata ─────────────────────────

    async def fetch_metadata(
        self,
        batch_size: int = 200,
        concurrency: int = 10,
    ) -> int:
        """
        Fetch match details for all DISCOVERED matches.
        Filters by patch and duration. Returns count of valid matches.
        """
        total_valid = 0
        total_skipped = 0
        sem = asyncio.Semaphore(concurrency)

        patch_filter = set(self.cfg["collection"].get("patch_filter", []))
        min_duration = self.cfg["collection"].get("min_game_duration", 900)

        while True:
            if self._stop:
                break

            matches = await self.state.get_matches_by_status(
                MatchStatus.DISCOVERED, limit=batch_size
            )
            if not matches:
                break

            pbar = tqdm(
                total=len(matches),
                desc="Fetching metadata",
                unit="match",
                leave=False,
            )

            async def _fetch_one(match: dict) -> tuple[bool, bool]:
                """Returns (valid, skipped)."""
                async with sem:
                    if self._stop:
                        return False, False
                    try:
                        data = await self.client.get_match(
                            match["routing"], match["match_id"]
                        )
                        pbar.update(1)

                        if data is None:
                            await self.state.update_match_status(
                                match["match_id"], MatchStatus.SKIPPED
                            )
                            return False, True

                        info = data.get("info", {})
                        game_version = info.get("gameVersion", "")
                        patch = ".".join(game_version.split(".")[:2])
                        duration = info.get("gameDuration", 0)
                        queue_id = info.get("queueId", 0)

                        # Filter: wrong queue
                        if queue_id != self.cfg["collection"]["queue_id"]:
                            await self.state.update_match_status(
                                match["match_id"], MatchStatus.SKIPPED
                            )
                            return False, True

                        # Filter: patch
                        if patch_filter and patch not in patch_filter:
                            await self.state.update_match_status(
                                match["match_id"], MatchStatus.SKIPPED
                            )
                            return False, True

                        # Filter: too short (remake)
                        if duration < min_duration:
                            await self.state.update_match_status(
                                match["match_id"], MatchStatus.SKIPPED
                            )
                            return False, True

                        # Valid — buffer for export
                        self.exporter.add_match(data)

                        await self.state.update_match_status(
                            match["match_id"],
                            MatchStatus.METADATA_DONE,
                            game_version=game_version,
                            game_duration=duration,
                            queue_id=queue_id,
                        )
                        return True, False

                    except Exception as e:
                        logger.error(
                            "Error fetching %s: %s",
                            match["match_id"],
                            e,
                        )
                        await self.state.update_match_status(
                            match["match_id"], MatchStatus.FAILED
                        )
                        return False, False

            tasks = [_fetch_one(m) for m in matches]
            results = await asyncio.gather(*tasks)
            pbar.close()

            batch_valid = sum(1 for v, _ in results if v)
            batch_skipped = sum(1 for _, s in results if s)
            total_valid += batch_valid
            total_skipped += batch_skipped

            # Flush if buffer is full
            if self.exporter.should_flush():
                self.exporter.flush()

            await self.state.commit()

            logger.info(
                "  Metadata batch: %d valid, %d skipped (total valid: %d)",
                batch_valid,
                batch_skipped,
                total_valid,
            )

        # Final flush
        self.exporter.flush_remaining()
        logger.info(
            "Metadata fetch complete: %d valid, %d skipped",
            total_valid,
            total_skipped,
        )
        return total_valid

    # ── Stage 4: Fetch Timelines ──────────────────────────────

    async def fetch_timelines(
        self,
        batch_size: int = 200,
        concurrency: int = 10,
    ) -> int:
        """
        Fetch timelines for all matches with METADATA_DONE status.
        Returns count of successful timeline fetches.
        """
        total_done = 0
        target = self.cfg["collection"].get("target_games", 200000)
        sem = asyncio.Semaphore(concurrency)

        # Check how many we already have
        existing = await self.state.match_count(MatchStatus.TIMELINE_DONE)
        existing += await self.state.match_count(MatchStatus.COMPLETE)
        logger.info(
            "Timeline fetch: %d already done, target %d",
            existing,
            target,
        )

        if existing >= target:
            logger.info("Target already met!")
            return 0

        while True:
            if self._stop:
                break

            if existing + total_done >= target:
                logger.info("Target of %d games reached!", target)
                break

            matches = await self.state.get_matches_by_status(
                MatchStatus.METADATA_DONE, limit=batch_size
            )
            if not matches:
                break

            pbar = tqdm(
                total=len(matches),
                desc=f"Fetching timelines ({existing + total_done}/{target})",
                unit="tl",
                leave=False,
            )

            async def _fetch_one(match: dict) -> bool:
                async with sem:
                    if self._stop:
                        return False
                    try:
                        data = await self.client.get_timeline(
                            match["routing"], match["match_id"]
                        )
                        pbar.update(1)

                        if data is None:
                            await self.state.update_match_status(
                                match["match_id"], MatchStatus.FAILED
                            )
                            return False

                        success = self.exporter.add_timeline(
                            match["match_id"], data
                        )
                        if success:
                            await self.state.update_match_status(
                                match["match_id"], MatchStatus.COMPLETE
                            )
                            return True
                        else:
                            await self.state.update_match_status(
                                match["match_id"], MatchStatus.FAILED
                            )
                            return False

                    except Exception as e:
                        logger.error(
                            "Error fetching timeline %s: %s",
                            match["match_id"],
                            e,
                        )
                        await self.state.update_match_status(
                            match["match_id"], MatchStatus.FAILED
                        )
                        return False

            tasks = [_fetch_one(m) for m in matches]
            results = await asyncio.gather(*tasks)
            pbar.close()

            batch_done = sum(1 for r in results if r)
            total_done += batch_done

            if self.exporter.should_flush():
                self.exporter.flush()

            await self.state.commit()

            logger.info(
                "  Timeline batch: %d done (total: %d/%d)",
                batch_done,
                existing + total_done,
                target,
            )

        self.exporter.flush_remaining()
        logger.info("Timeline fetch complete: %d new timelines", total_done)
        return total_done

    # ── Full Pipeline Run ─────────────────────────────────────

    async def run_full(self) -> dict:
        """Run all stages in sequence. Returns summary stats."""
        stats = {}

        logger.info("=" * 60)
        logger.info("STAGE 1: Discovering players...")
        logger.info("=" * 60)
        stats["new_players"] = await self.discover_players()
        if self._stop:
            return stats

        logger.info("=" * 60)
        logger.info("STAGE 2: Discovering match IDs...")
        logger.info("=" * 60)
        stats["new_matches"] = await self.discover_matches()
        if self._stop:
            return stats

        logger.info("=" * 60)
        logger.info("STAGE 3: Fetching match metadata...")
        logger.info("=" * 60)
        stats["valid_matches"] = await self.fetch_metadata()
        if self._stop:
            return stats

        logger.info("=" * 60)
        logger.info("STAGE 4: Fetching timelines...")
        logger.info("=" * 60)
        stats["timelines"] = await self.fetch_timelines()

        # Final stats from DB
        db_stats = await self.state.get_stats()
        stats["db"] = db_stats

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("  Players:    %d", db_stats["total_players"])
        logger.info("  Matches:    %d total", db_stats["total_matches"])
        logger.info("  Complete:   %d (with timelines)", db_stats["complete"])
        logger.info("  Skipped:    %d", db_stats["skipped"])
        logger.info("  Failed:     %d", db_stats["failed"])
        logger.info("  API stats:  %s", self.client.stats)
        logger.info("=" * 60)

        return stats
