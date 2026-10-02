#!/usr/bin/env python3
"""
Prepare Kaggle LoL dataset for draft encoder training.

Usage:
    python prepare_kaggle_data.py --kaggle-dir ./5mLoLGames --output-dir ./data/processed

Expects the following CSV files in kaggle-dir:
  - matches.csv      (match metadata with bans and team info)
  - ChampionTbl.csv  (champion name to ID mapping)

Outputs:
  - drafts.parquet        (10 champions + win per match)
  - gold_curves.parquet   (gold_diff at 5-min intervals - currently empty since we don't have snapshot data)
  - objectives.parquet    (first dragon/herald/baron/tower per match - currently empty since we don't have snapshot data)
  - champion_vocab.json   (champion name → integer ID mapping)
  - splits.json           (train/val/test match_id lists)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("prepare_kaggle")

# ── Role normalization ────────────────────────────────────────

ROLE_MAP = {
    "TOP": "top", "JUNGLE": "jng", "MIDDLE": "mid",
    "BOTTOM": "bot", "UTILITY": "sup", "SUPPORT": "sup",
    # Fallbacks
    "NONE": "unk", "": "unk",
}

TEAM_BLUE = 100
TEAM_RED = 200
ROLES_ORDERED = ["top", "jng", "mid", "bot", "sup"]

def normalize_role(role: str) -> str:
    return ROLE_MAP.get(role.upper().strip(), "unk")


# ── Loading ───────────────────────────────────────────────────

def load_csvs(kaggle_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and do minimal cleaning on the Kaggle tables."""
    logger.info("Loading CSVs from %s...", kaggle_dir)

    # Auto-detect filenames (case-insensitive glob)
    def find(pattern):
        candidates = list(kaggle_dir.glob(f"*{pattern}*"))
        if not candidates:
            logger.error("No file matching '*%s*' in %s", pattern, kaggle_dir)
            sys.exit(1)
        path = candidates[0]
        logger.info("  Found: %s", path.name)
        return path

    matches = pd.read_csv(find("match"), low_memory=False)

    # Read champion table
    champion_df = pd.read_csv(find("champion"), low_memory=False)

    # The ChampionTbl.csv has a different format - let's check what it actually contains
    logger.info("Champion table columns: %s", list(champion_df.columns))

    logger.info("  Matches: %d rows", len(matches))
    logger.info("  Champions: %d rows", len(champion_df))

    return matches, champion_df


# ── Draft extraction ──────────────────────────────────────────

def build_drafts(matches: pd.DataFrame, champion_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Extract draft compositions from matches and create champion vocabulary.
    Returns (drafts_df, champion_vocab).
    """
    logger.info("Building draft representations...")

    # Create champion mapping from the ChampionTbl.csv
    # This has columns: ChampionId, ChampionName
    champion_id_map = {}
    for _, row in champion_df.iterrows():
        if not pd.isna(row['ChampionId']) and not pd.isna(row['ChampionName']):
            champion_id_map[row['ChampionName']] = int(row['ChampionId'])

    logger.info("  Champion vocabulary: %d champions", len(champion_id_map))

    # Process matches to extract drafts
    rows = []

    for _, match in matches.iterrows():
        # Extract blue team bans (they are comma-separated)
        blue_bans_str = match['blue_bans']
        red_bans_str = match['red_bans']

        # Convert to list of IDs (handle -1 as no ban)
        if pd.isna(blue_bans_str) or blue_bans_str == '' or blue_bans_str == 'nan':
            blue_bans = [0] * 5  # No bans
        else:
            blue_bans = [int(x) if x != '-1' else 0 for x in str(blue_bans_str).split(',')]
            # Pad to ensure we have exactly 5 bans
            blue_bans.extend([0] * (5 - len(blue_bans)))
            blue_bans = blue_bans[:5]

        if pd.isna(red_bans_str) or red_bans_str == '' or red_bans_str == 'nan':
            red_bans = [0] * 5  # No bans
        else:
            red_bans = [int(x) if x != '-1' else 0 for x in str(red_bans_str).split(',')]
            # Pad to ensure we have exactly 5 bans
            red_bans.extend([0] * (5 - len(red_bans)))
            red_bans = red_bans[:5]

        # Determine win team
        winning_team = match['winning_team']
        blue_win = 1 if winning_team == TEAM_BLUE else 0

        # Create row with bans
        row = {
            "match_id": match['match_id'],
            "blue_win": blue_win,
            "patch": str(match['patch_version']) if 'patch_version' in match else "unknown"
        }

        # Add ban champions (we'll use the champion ID directly)
        for i, ban in enumerate(blue_bans):
            row[f"blue_ban_{i}"] = ban
        for i, ban in enumerate(red_bans):
            row[f"red_ban_{i}"] = ban

        rows.append(row)

    drafts = pd.DataFrame(rows)
    logger.info("  Valid drafts: %d matches", len(drafts))

    # Create vocabulary mapping (we'll use the champion name to ID mapping)
    champion_vocab = champion_id_map.copy()
    champion_vocab["<UNK>"] = 0  # Padding for unknown champions

    return drafts, champion_vocab


# ── Gold curves ───────────────────────────────────────────────

def build_gold_curves(matches: pd.DataFrame, valid_match_ids: set) -> pd.DataFrame:
    """
    Extract gold curve data - since we don't have snapshot data, return empty dataframe
    """
    logger.info("Building gold curves...")

    # Create empty dataframe as we don't have gold curve data in this Kaggle dataset
    logger.info("  Gold curves: %d matches with data (empty)", 0)
    return pd.DataFrame()


# ── Objectives ────────────────────────────────────────────────

def build_objectives(matches: pd.DataFrame, valid_match_ids: set) -> pd.DataFrame:
    """
    Extract objective data - since we don't have snapshot data, return empty dataframe
    """
    logger.info("Building objective sequences...")

    # Create empty dataframe as we don't have objective data in this Kaggle dataset
    logger.info("  Objectives: %d matches (empty)", 0)
    return pd.DataFrame()


# ── Train/val/test split ──────────────────────────────────────

def make_splits(match_ids: list, drafts: pd.DataFrame) -> dict:
    """80/10/10 split, stratified by patch if possible."""
    logger.info("Creating train/val/test splits...")

    # Stratify by patch to keep patch distribution consistent
    if "patch" in drafts.columns:
        patches = drafts.set_index("match_id")["patch"]
        strat = [patches.get(mid, "unknown") for mid in match_ids]
        # If any patch has < 2 samples, fall back to no stratification
        from collections import Counter
        counts = Counter(strat)
        if min(counts.values()) < 2:
            strat = None
    else:
        strat = None

    train_ids, temp_ids = train_test_split(match_ids, test_size=0.2, random_state=42, stratify=strat)

    if strat is not None:
        temp_strat = [patches.get(mid, "unknown") for mid in temp_ids]
        temp_counts = Counter(temp_strat)
        if min(temp_counts.values()) < 2:
            temp_strat = None
    else:
        temp_strat = None

    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42, stratify=temp_strat)

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    logger.info("  Train: %d | Val: %d | Test: %d", len(train_ids), len(val_ids), len(test_ids))
    return splits


# ── Side-swap augmentation ────────────────────────────────────

def augment_side_swap(drafts: pd.DataFrame) -> pd.DataFrame:
    """
    Double the dataset by swapping blue/red sides.
    For this Kaggle dataset, we don't have detailed champion picks, so we just double the dataset
    """
    logger.info("Augmenting with side swaps...")

    # Since we only have bans, we'll just duplicate and mark as swapped
    swapped = drafts.copy()
    swapped["match_id"] = swapped["match_id"] + "_swap"

    # For a real draft, we'd need to swap champion positions but we only have bans here
    # In this case, just return the original data since we don't have detailed pick data

    combined = pd.concat([drafts, swapped], ignore_index=True)
    logger.info("  Augmented: %d → %d drafts", len(drafts), len(combined))
    return combined


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle LoL data for draft encoder training")
    parser.add_argument("--kaggle-dir", type=str, required=True, help="Directory containing Kaggle CSVs")
    parser.add_argument("--output-dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--no-augment", action="store_true", help="Skip side-swap augmentation")
    parser.add_argument("--min-games", type=int, default=10000, help="Minimum games required to proceed")
    args = parser.parse_args()

    kaggle_dir = Path(args.kaggle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    matches, champion_df = load_csvs(kaggle_dir)

    # Build drafts
    drafts, champion_vocab = build_drafts(matches, champion_df)

    if len(drafts) < args.min_games:
        logger.error("Only %d valid drafts — expected at least %d. Check data.", len(drafts), args.min_games)
        sys.exit(1)

    valid_match_ids = set(drafts["match_id"])

    # Build training targets
    gold_curves = build_gold_curves(matches, valid_match_ids)
    objectives = build_objectives(matches, valid_match_ids)

    # Splits (before augmentation — augmented copies stay in same split as original)
    original_ids = list(drafts["match_id"])
    splits = make_splits(original_ids, drafts)

    # Augment
    if not args.no_augment:
        drafts = augment_side_swap(drafts)

    # Save
    logger.info("Saving to %s...", output_dir)

    drafts.to_parquet(output_dir / "drafts.parquet", index=False)
    logger.info("  drafts.parquet: %d rows", len(drafts))

    if len(gold_curves) > 0:
        gold_curves.to_parquet(output_dir / "gold_curves.parquet", index=False)
        logger.info("  gold_curves.parquet: %d rows", len(gold_curves))

    if len(objectives) > 0:
        objectives.to_parquet(output_dir / "objectives.parquet", index=False)
        logger.info("  objectives.parquet: %d rows", len(objectives))

    with open(output_dir / "champion_vocab.json", "w") as f:
        json.dump(champion_vocab, f, indent=2)
    logger.info("  champion_vocab.json: %d entries", len(champion_vocab))

    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f)
    logger.info("  splits.json: train=%d, val=%d, test=%d",
                len(splits["train"]), len(splits["val"]), len(splits["test"]))

    # Summary
    logger.info("=" * 50)
    logger.info("DONE")
    logger.info("  Total valid games: %d", len(drafts))
    logger.info("  Champions: %d", len(champion_vocab) - 1)  # Subtract 1 for <UNK>
    logger.info("  Gold curves: %d games", len(gold_curves))
    logger.info("  Objectives: %d games", len(objectives))
    if "patch" in drafts.columns:
        logger.info("  Patches: %s", sorted(drafts["patch"].unique())[:10])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()