# LoL Data Collector — Draft-Conditioned Macro Strategy Agent

Robust, multi-key, resumable data collection pipeline for high-elo League of Legends match data. Built for the Phase 1.1 data collection stage of the macro strategy agent project.

## What It Collects

- **Match metadata**: champion picks, roles, win/loss, patch version, game duration, summoner spells, KDA, gold, damage, CS, vision score — 10 rows per match (one per participant)
- **Timeline frames**: per-minute position, gold, XP, level, CS for every player — the raw material for trajectory modeling in Phase 2
- **Events**: kills, dragon/baron/herald takes, tower kills — structured and timestamped for objective sequence prediction

All stored in **Parquet** format (columnar, compressed, fast to query with pandas/polars).

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
│  config.yaml│────▶│  collect.py (CLI) │────▶│  pipeline.py  │
└─────────────┘     └──────────────────┘     │  (orchestrator)│
                                              └──────┬────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────┐
                    │                                │                │
            ┌───────▼───────┐              ┌─────────▼──────┐  ┌─────▼──────┐
            │  riot_api.py  │              │   state.py     │  │  export.py │
            │  (async HTTP) │              │  (SQLite ckpt) │  │  (Parquet) │
            └───────┬───────┘              └────────────────┘  └────────────┘
                    │
            ┌───────▼──────────┐
            │ rate_limiter.py  │
            │ (per-key dual    │
            │  window buckets) │
            └──────────────────┘
```

### Pipeline Stages

| Stage | What it does | API calls per item |
|-------|-------------|-------------------|
| **1. Discover players** | Pulls Challenger/GM/Master leagues per region | 1 league per tier |
| **2. Discover matches** | Pulls match histories per player, deduplicates | 1 per player |
| **3. Fetch metadata** | Pulls match details, filters by patch/duration | 1 per match |
| **4. Fetch timelines** | Pulls timeline data for valid matches | 1 per match |

Each stage is **independently resumable** — state is checkpointed in SQLite after every batch. Kill the process at any point and restart; it picks up where it left off.

## Setup

```bash
# 1. Clone and install
cd lol_collector
pip install -r requirements.txt

# 2. Get your API key(s) from https://developer.riotgames.com/
#    Development keys expire every 24 hours.
#    Production keys are permanent — apply early (free for personal projects).

# 3. Create config.yaml — add your key(s) and choose regions
cp config.example.yaml config.yaml
nano config.yaml
```

### Single Key Setup (Development)

```yaml
api_keys:
  - key: "RGAPI-your-actual-key-here"
    key_type: "development"
```

### Multi-Key Setup (Split Workload)

```yaml
api_keys:
  # Key 1 handles NA
  - key: "RGAPI-aaaa-bbbb-cccc"
    key_type: "production"
    rate_limits:
      per_second: 50
      per_two_minutes: 3000
    regions: ["na1"]

  # Key 2 handles EUW + KR
  - key: "RGAPI-dddd-eeee-ffff"
    key_type: "production"
    rate_limits:
      per_second: 50
      per_two_minutes: 3000
    regions: ["euw1", "kr"]

  # Key 3 (dev key, no region restriction — fills in wherever needed)
  - key: "RGAPI-gggg-hhhh-iiii"
    key_type: "development"
```

When a key has `regions` specified, it is **preferred** for those regions. Keys without region restrictions serve as overflow capacity. The rate limiter picks the key with the most available headroom.

## Usage

```bash
# Full pipeline — runs all 4 stages in order
python collect.py

# Run specific stages (useful for debugging or parallel terminals)
python collect.py --stage players     # Just discover high-elo players
python collect.py --stage matches     # Just pull match histories
python collect.py --stage metadata    # Just fetch match details
python collect.py --stage timelines   # Just fetch timelines

# Check progress
python collect.py --stats

# Use a different config
python collect.py --config production.yaml
```

### Graceful Shutdown

Press **Ctrl+C** at any time. The pipeline finishes the current batch, flushes buffered data to Parquet, and exits cleanly. All progress is saved — restart and it resumes.

## Output Structure

```
data/
├── collection_state.db          # SQLite checkpoint (don't delete this!)
├── collector.log                # Full log
├── matches/
│   ├── matches_20260401_143022_0000.parquet
│   ├── matches_20260401_151544_0001.parquet
│   └── ...
└── timelines/
    ├── frames_20260401_160033_0000.parquet
    ├── frames_20260401_163211_0001.parquet
    ├── events/
    │   ├── events_20260401_160033_0000.parquet
    │   └── ...
    └── ...
```

### Reading the Data

```python
import pandas as pd

# Load all match metadata
matches = pd.read_parquet("data/matches/")
print(f"{matches.match_id.nunique()} unique games")
print(matches.groupby('patch').match_id.nunique())

# Load all timeline frames
frames = pd.read_parquet("data/timelines/", ignore_directories=["events"])
print(f"{len(frames)} total frames")

# Load objective events
events = pd.read_parquet("data/timelines/events/")
dragons = events[events.monster_type == "DRAGON"]

# Example: gold differential at minute 15 per team
min15 = frames[frames.timestamp_min.between(14.5, 15.5)]
gold_by_team = min15.groupby(['match_id', 'participant_id']).total_gold.first()
```

## Rate Limiting Details

Each API key has **two simultaneous rate windows**:
- **Short window**: N requests per 1 second
- **Long window**: M requests per 120 seconds

Both must have capacity for a request to proceed. The limiter also respects Riot's `Retry-After` header on 429 responses — when a key gets throttled, it's automatically cooled down.

**Multi-key load balancing**: when multiple keys are available, the limiter picks the one with the most headroom. This naturally spreads load across keys.

### Development vs Production Keys

| | Development | Production |
|---|---|---|
| Rate limit | 20/s, 100/2min | Custom (typically 50/s, 3000/2min) |
| Expiry | 24 hours | Permanent |
| Application | Instant | Days–weeks |

**Recommendation**: apply for a production key on day 1. Use a dev key to start testing the pipeline immediately, then switch to production once approved.

## Estimated Collection Times

Rough estimates assuming 150,000 target games:

| Setup | Players | Match IDs | Metadata | Timelines | Total |
|-------|---------|-----------|----------|-----------|-------|
| 1 dev key (100/2min) | ~2 hrs | ~8 hrs | ~50 hrs | ~50 hrs | ~5 days |
| 1 prod key (3000/min) | ~20 min | ~1 hr | ~2 hrs | ~2 hrs | ~5 hrs |
| 3 prod keys | ~10 min | ~25 min | ~45 min | ~45 min | ~2 hrs |

These are conservative estimates. Actual speed depends on API response times and retry rates.

## Troubleshooting

**403 Forbidden**: Your API key has expired (dev keys last 24 hours) or is invalid. Generate a new one at developer.riotgames.com.

**429 Too Many Requests**: The rate limiter handles this automatically, but if you see persistent 429s, your rate limit config may be set higher than your actual key allows. Check your Riot developer dashboard for your key's actual limits.

**Empty player lists**: Riot occasionally returns empty responses for apex tier leagues. Re-run `--stage players` — the data is usually available on retry.

**Pipeline stuck on "Discovering match IDs"**: Some players have privacy settings that block match history. The pipeline skips these automatically; if it seems stuck, it's just working through players with blocked histories.

**SQLite locked**: Don't run multiple instances of collect.py against the same state database simultaneously. If you want parallel collection, use separate configs with separate `state_db` paths and separate output directories, then merge Parquet files afterward.
