#!/usr/bin/env python3
"""
LoL Macro Strategy Agent — Data Collector

Usage:
    python collect.py                  # Full pipeline run
    python collect.py --stage players  # Only discover players
    python collect.py --stage matches  # Only discover match IDs
    python collect.py --stage metadata # Only fetch match details
    python collect.py --stage timelines# Only fetch timelines
    python collect.py --stats          # Print collection stats
    python collect.py --config my.yaml # Use custom config file

Ctrl+C triggers graceful shutdown — all progress is checkpointed.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
from pathlib import Path

import yaml

from rate_limiter import MultiKeyLimiter
from riot_api import RiotClient, RetryConfig
from state import StateDB
from export import ParquetExporter
from pipeline import CollectionPipeline

logger = logging.getLogger("collector")


def load_config(path: str) -> dict:
    """Load and validate config YAML."""
    with open(path, encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    # Validate required fields
    keys = cfg.get("api_keys", [])
    if not keys:
        print("ERROR: No API keys configured in config.yaml")
        print("  Add at least one key under 'api_keys'")
        sys.exit(1)

    for i, k in enumerate(keys):
        key = k.get("key", "")
        if not key or "REPLACE_ME" in key or key.startswith("RGAPI-xxxx"):
            print(f"ERROR: API key {i} is still the placeholder value")
            print("  Replace it with your actual Riot API key from:")
            print("  https://developer.riotgames.com/")
            sys.exit(1)

    regions = cfg.get("regions", [])
    if not regions:
        print("ERROR: No regions configured")
        sys.exit(1)

    # Ensure storage dirs exist
    storage = cfg.get("storage", {})
    for dir_key in ["data_dir", "matches_dir", "timelines_dir"]:
        d = storage.get(dir_key)
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

    return cfg


def setup_logging(cfg: dict) -> None:
    """Configure logging from config."""
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO"))

    handlers = []

    if log_cfg.get("console", True):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        handlers.append(console)

    log_file = log_cfg.get("log_file")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
            )
        )
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers)

    # Quiet noisy libs
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)


def build_limiter(cfg: dict) -> MultiKeyLimiter:
    """Build multi-key rate limiter from config."""
    concurrency = cfg.get("concurrency", {}).get("max_concurrent_per_key", 10)
    total_concurrency = concurrency * len(cfg.get("api_keys", []))
    max_in_flight = cfg.get("concurrency", {}).get("max_in_flight", min(8, total_concurrency or 1))
    limiter = MultiKeyLimiter(max_total_concurrent=max_in_flight)
    rate_defaults = cfg.get("rate_limits", {})

    for i, key_cfg in enumerate(cfg["api_keys"]):
        key_id = f"key_{i}"
        api_key = key_cfg["key"]
        key_type = key_cfg.get("key_type", "development")

        if key_type == "production":
            custom_limits = key_cfg.get("rate_limits", {})
            per_s = custom_limits.get(
                "per_second",
                rate_defaults.get("production", {}).get("per_second", 50),
            )
            per_2m = custom_limits.get(
                "per_two_minutes",
                rate_defaults.get("production", {}).get("per_two_minutes", 3000),
            )
        else:
            per_s = rate_defaults.get("development", {}).get("per_second", 20)
            per_2m = rate_defaults.get("development", {}).get("per_two_minutes", 100)

        limiter.add_key(key_id, api_key, per_s, per_2m)
        logger.info(
            "Registered key %s (%s): %d/s, %d/2min",
            key_id,
            key_type,
            per_s,
            per_2m,
        )

    return limiter


def build_key_region_map(cfg: dict) -> dict[str, list[str]]:
    """Build key→region affinity map."""
    mapping = {}
    for i, key_cfg in enumerate(cfg["api_keys"]):
        regions = key_cfg.get("regions")
        if regions:
            mapping[f"key_{i}"] = regions
    return mapping


async def print_stats(cfg: dict) -> None:
    """Print current collection statistics."""
    state = StateDB(cfg["storage"]["state_db"])
    await state.connect()

    stats = await state.get_stats()
    await state.close()

    print("\n" + "=" * 50)
    print("  Collection Statistics")
    print("=" * 50)
    print(f"  Players discovered:  {stats['total_players']:>8,}")
    print(f"  Matches discovered:  {stats['total_matches']:>8,}")
    print(f"  {'-' * 32}")
    print(f"  Awaiting metadata:   {stats['discovered']:>8,}")
    print(f"  Metadata done:       {stats['metadata_done']:>8,}")
    print(f"  Timelines done:      {stats['timeline_done']:>8,}")
    print(f"  Complete:            {stats['complete']:>8,}")
    print(f"  Skipped:             {stats['skipped']:>8,}")
    print(f"  Failed:              {stats['failed']:>8,}")
    print("=" * 50)

    target = cfg["collection"].get("target_games", 200000)
    complete = stats["complete"]
    pct = (complete / target * 100) if target > 0 else 0
    print(f"  Progress: {complete:,} / {target:,} ({pct:.1f}%)")
    print("=" * 50 + "\n")


async def run(cfg: dict, stage: str = None) -> None:
    """Main async entry point."""
    limiter = build_limiter(cfg)
    key_region_map = build_key_region_map(cfg)

    retry_cfg = RetryConfig(
        max_retries=cfg.get("retry", {}).get("max_retries", 5),
        base_delay=cfg.get("retry", {}).get("base_delay", 1.0),
        max_delay=cfg.get("retry", {}).get("max_delay", 120.0),
        backoff_multiplier=cfg.get("retry", {}).get("backoff_multiplier", 2.0),
        retry_on_status=tuple(
            cfg.get("retry", {}).get("retry_on_status", [429, 500, 502, 503, 504])
        ),
    )

    state = StateDB(cfg["storage"]["state_db"])
    await state.connect()

    exporter = ParquetExporter(
        matches_dir=cfg["storage"]["matches_dir"],
        timelines_dir=cfg["storage"]["timelines_dir"],
        flush_interval=cfg["storage"].get("flush_interval", 500),
    )

    concurrency = cfg.get("concurrency", {}).get("max_concurrent_per_key", 10)
    total_concurrency = concurrency * len(cfg["api_keys"])

    async with RiotClient(limiter, retry_cfg, key_region_map) as client:
        pipeline = CollectionPipeline(
            client=client,
            state=state,
            exporter=exporter,
            regions=cfg["regions"],
            config=cfg,
        )

        # Wire up graceful shutdown
        loop = asyncio.get_event_loop()

        def _signal_handler():
            pipeline.request_stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Config hash for run tracking
        config_hash = hashlib.md5(
            json.dumps(cfg, sort_keys=True, default=str).encode()
        ).hexdigest()[:8]

        run_id = await state.start_run(config_hash)
        logger.info("Starting collection run #%d (config: %s)", run_id, config_hash)
        logger.info(
            "Keys: %d | Regions: %s | Concurrency: %d total",
            len(cfg["api_keys"]),
            [r["platform"] for r in cfg["regions"]],
            total_concurrency,
        )

        try:
            if stage:
                # Run specific stage
                if stage == "players":
                    await pipeline.discover_players()
                elif stage == "matches":
                    await pipeline.discover_matches(concurrency=total_concurrency)
                elif stage == "metadata":
                    await pipeline.fetch_metadata(concurrency=total_concurrency)
                elif stage == "timelines":
                    await pipeline.fetch_timelines(concurrency=total_concurrency)
                else:
                    logger.error("Unknown stage: %s", stage)
            else:
                # Full pipeline
                result = await pipeline.run_full()

            db_stats = await state.get_stats()
            await state.end_run(
                run_id,
                players_found=db_stats["total_players"],
                matches_found=db_stats["total_matches"],
                timelines_done=db_stats["complete"],
            )

        except Exception as e:
            logger.exception("Pipeline error: %s", e)
            await state.end_run(run_id, notes=f"Error: {e}")
            raise

        finally:
            exporter.flush_remaining()
            await state.close()


def main():
    parser = argparse.ArgumentParser(
        description="LoL Macro Strategy Agent — Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect.py                     Full pipeline
  python collect.py --stage players     Discover high-elo players only
  python collect.py --stage timelines   Fetch timelines for known matches
  python collect.py --stats             Show collection progress
  python collect.py --config prod.yaml  Use production config

Stages: players, matches, metadata, timelines
        """,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=["players", "matches", "metadata", "timelines"],
        help="Run a specific pipeline stage instead of the full pipeline",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print collection statistics and exit",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.stats:
        asyncio.run(print_stats(cfg))
    else:
        asyncio.run(run(cfg, stage=args.stage))


if __name__ == "__main__":
    main()
