# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project overview
This repo contains two related efforts:

1) **`lol_collector/`**  a Python, async, multi-key, **resumable Riot API data-collection pipeline** that exports League of Legends match data to **Parquet**, with **SQLite** checkpointing (`collection_state.db`). This is the Phase 1.1 data collection stage for a draft-conditioned macro strategy agent.

2) **Draft encoder (Phase 1.2, planned)**  a model that embeds a 10-champion draft (+ roles) into a 256-dim vector using the Kaggle LoL Match Interval Snapshots dataset. See `lol_collector/PLAN_DRAFT_ENCODER.md`.

## Common commands (collector)
Run from `lol_collector/`.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the pipeline
```bash
# Full 4-stage run (players -> matches -> metadata -> timelines)
python collect.py

# Run one stage
python collect.py --stage players
python collect.py --stage matches
python collect.py --stage metadata
python collect.py --stage timelines

# Show progress from the SQLite state DB
python collect.py --stats

# Use an alternate config file
python collect.py --config production.yaml
```

### Outputs / artifacts
Collector artifacts live under `lol_collector/data/`:
- `collection_state.db` (SQLite checkpoint; source of truth for resumability)
- Parquet under `data/matches/` and `data/timelines/` (including `data/timelines/events/`)

See `lol_collector/README.md` for the exact directory layout.

## High-level architecture (collector)
The collector is a 4-stage, independently resumable pipeline:

1. **Discover players**: pulls Challenger/GM/Master leagues per region and resolves entries to PUUIDs.
2. **Discover matches**: pulls match ID lists per player (match-v5), deduplicates, stores match IDs.
3. **Fetch metadata**: fetches match details and filters by queue/patch/duration; exports valid match rows.
4. **Fetch timelines**: fetches timeline frames + events for metadata-complete matches.

Key modules (under `lol_collector/`):
- `collect.py`: CLI entrypoint (config, limiter+client, opens state DB, runs stages).
- `pipeline.py`: orchestrates stages + concurrency + graceful stop.
- `riot_api.py`: async Riot API client (aiohttp) with retries/backoff and 429 Retry-After handling.
- `rate_limiter.py`: per-key dual-window buckets (1s + 120s) + global concurrency; balances across keys.
- `state.py`: SQLite checkpointing (players, matches, status transitions, run summaries).
- `export.py`: buffered Parquet writer for match metadata, timeline frames, and selected events.

### Data flow
- `collect.py` builds `MultiKeyLimiter` + `RiotClient` + `StateDB` + `ParquetExporter`, then constructs `CollectionPipeline`.
- `CollectionPipeline` fetches from Riot, persists progress/status to SQLite, and exports payloads to Parquet.

## Configuration / safety
- **Do not commit Riot API keys.** Keep `lol_collector/config.yaml` local and redact logs/snippets.
- Avoid running multiple collector instances against the same `collection_state.db` (SQLite lock contention). If you need parallelism, use separate state DB paths/output dirs.

## Notes
- No dedicated test suite is present currently.
