# Phase 1.2 — Draft Encoder: Plan

## Overview

Build the draft encoder from the revised roadmap using the Kaggle
"LoL Match Interval Snapshots" dataset (5M games, 3 tables).
The encoder takes 10 champion IDs + roles and outputs a 256-dim
vector capturing the draft's strategic identity.

## Data Sources

### Kaggle Tables (already available)
1. **matches** — match_id, date, patch, queue_id, platform, rank, team compositions (champion IDs)
2. **participants** — match_id, participant_id, team_id, champion, role, individual_position
3. **snapshots** — match_id, player_id, minute (5-min intervals), gold, xp, cs, kills, deaths, assists, items, objectives, diffs

### Collector Parquet (ongoing, used later for Phase 2a)
- Timeline frames with per-minute position data (x, y)
- Used for trajectory modeling, NOT needed for the encoder

## Pipeline Architecture

```
kaggle CSVs
    │
    ▼
┌──────────────────────┐
│  prepare_dataset.py  │  ← Load, join, filter, encode, split, save
└──────────┬───────────┘
           │
           ▼
    data/processed/
    ├── drafts.parquet        (match_id, 10 champion IDs, 10 roles, patch, win)
    ├── gold_curves.parquet   (match_id, team, gold_diff @ min 5,10,15,20,25,30)
    ├── objectives.parquet    (match_id, team, first_dragon, first_herald, first_baron, ...)
    └── splits.json           (train/val/test match_id lists — 80/10/10)
           │
           ▼
┌──────────────────────┐
│  draft_encoder/      │
│    model.py          │  ← Champion embeddings + team MLP + output head
│    dataset.py        │  ← PyTorch Dataset reading processed parquets
│    train.py          │  ← Multi-task training loop (Hydra config, W&B)
│    evaluate.py       │  ← Accuracy, t-SNE, nearest neighbours
│    config/           │  ← Hydra YAML configs
└──────────────────────┘
```

## Step 1: prepare_dataset.py

### Input
- `kaggle_data/matches.csv`
- `kaggle_data/participants.csv`
- `kaggle_data/snapshots.csv`

### Processing
1. Load all three tables
2. Filter: queue_id == 420 (ranked solo), patch whitelist (optional)
3. Join participants → matches on match_id to get champion + role + team + win
4. Pivot participants to one row per match: 10 champion columns + 10 role columns
   (blue_top_champ, blue_jng_champ, ... red_sup_champ)
5. Build champion → integer ID mapping (save as champion_vocab.json)
6. Extract gold curves from snapshots: per-team gold_diff at each 5-min mark
7. Extract objective sequences: who got first dragon, first herald, first baron
   (derivable from the team_dragons, team_barons, team_heralds columns at each
   timestamp — first team to increment from 0→1 at the earliest minute)
8. Derive win labels (team with higher gold at final snapshot, or from match table if available)
9. Train/val/test split by match_id (80/10/10), stratified by patch
10. Save to parquet + splits.json

### Output Schema: drafts.parquet
```
match_id         str
patch            str
blue_top         int    (champion ID)
blue_jng         int
blue_mid         int
blue_bot         int
blue_sup         int
red_top          int
red_jng          int
red_mid          int
red_bot          int
red_sup          int
blue_win         bool
```

### Output Schema: gold_curves.parquet
```
match_id         str
gold_diff_5      float  (blue - red total gold at min 5)
gold_diff_10     float
gold_diff_15     float
gold_diff_20     float
gold_diff_25     float
gold_diff_30     float
```

### Output Schema: objectives.parquet
```
match_id              str
first_dragon_team     int    (100=blue, 200=red, 0=none)
first_dragon_minute   float
first_herald_team     int
first_herald_minute   float
first_baron_team      int
first_baron_minute    float
first_tower_team      int
first_tower_minute    float
```

## Step 2: draft_encoder/model.py

### Architecture (from roadmap)
- Champion ID → 64-dim learned embedding (168+ champions)
- Role-aware: separate embedding per (champion, role) via positional encoding
- Team MLP: 5 champion embeddings → concat (320-dim) → 3-layer MLP (512 hidden, ReLU, residual) → 256-dim team vector
- Draft vector: concat blue + red team vectors (512-dim) → projection → 256-dim draft embedding
- Patch embedding: learned 16-dim embedding per patch, concatenated before output heads

### Output Heads (multi-task)
1. **Win prediction**: draft_embed → MLP → sigmoid (BCE loss)
2. **Gold curve regression**: draft_embed → MLP → 6 values (MSE loss)
3. **Objective prediction**: draft_embed → MLP → logits per objective (CE loss)

### Loss Balancing
- Uncertainty-weighted multi-task loss (Kendall et al. 2018)
- Each task has a learned log-variance parameter σ²
- Total loss = Σ (1/2σ²_i) * L_i + log(σ_i)
- This automatically balances sharp (win) vs noisy (gold curve) losses

## Step 3: draft_encoder/train.py

### Training Loop
- Hydra config for all hyperparameters
- W&B logging (loss curves per task, learning rate, gradient norms)
- AdamW optimizer, cosine LR schedule with warmup
- Early stopping on val loss (patience=10)
- Checkpoint best model by val loss

### Key Hyperparameters (defaults)
- Champion embed dim: 64
- Team MLP hidden: 512, 3 layers, residual connections
- Draft embed dim: 256
- Patch embed dim: 16
- Batch size: 512
- Learning rate: 3e-4
- Weight decay: 0.01
- Warmup: 1000 steps
- Max epochs: 100

## Step 4: draft_encoder/evaluate.py

### Quantitative
- Win prediction accuracy (target >63%, go/no-go >60%)
- Gold curve MAE per timestep vs mean baseline
- Objective prediction accuracy vs majority class baseline

### Qualitative
- t-SNE / UMAP of 256-dim embeddings colored by comp archetype
- Nearest-neighbour queries for known compositions
- Embedding interpolation between comp archetypes

### Benchmark
- Compare win prediction against simple champion-winrate baseline
- Compare against existing Kaggle draft predictors if available

## Data Augmentation
- Blue/red side swap (doubles dataset, draft meaning is side-invariant)
- Role-flex augmentation for known flex picks (Jayce top/mid, etc.)

## File Structure After Implementation
```
lol_collector/           (existing — data collection)
draft_encoder/
├── README.md
├── requirements.txt
├── prepare_dataset.py
├── champion_vocab.json  (generated)
├── model.py
├── dataset.py
├── train.py
├── evaluate.py
├── config/
│   ├── default.yaml
│   └── experiment/
│       ├── baseline.yaml
│       └── no_patch_embed.yaml
└── data/processed/      (generated, gitignored)
```
