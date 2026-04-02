#!/usr/bin/env python3
"""
Prepare Kaggle LoL dataset for draft encoder training.

Usage:
    python prepare_dataset.py --kaggle-dir ./kaggle_data --output-dir ./data/processed

Expects three CSV files in kaggle-dir:
  - matches.csv      (match metadata)
  - participants.csv  (champion picks + roles)
  - snapshots.csv     (5-min interval stats)

Outputs:
  - drafts.parquet        (10 champions + roles + win per match)
  - gold_curves.parquet   (gold_diff at 5-min intervals)
  - objectives.parquet    (first dragon/herald/baron/tower per match)
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
logger = logging.getLogger("prepare")

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

def load_csvs(kaggle_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and do minimal cleaning on the three Kaggle tables."""
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
    participants = pd.read_csv(find("particip"), low_memory=False)
    snapshots = pd.read_csv(find("snapshot"), low_memory=False)

    logger.info("  Matches: %d rows", len(matches))
    logger.info("  Participants: %d rows", len(participants))
    logger.info("  Snapshots: %d rows", len(snapshots))

    return matches, participants, snapshots


# ── Draft extraction ──────────────────────────────────────────

def build_drafts(participants: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Pivot participant rows into one row per match with 10 champion + role columns.
    Returns (drafts_df, champion_vocab).
    """
    logger.info("Building draft representations...")

    # Normalize roles
    role_col = "individual_position" if "individual_position" in participants.columns else "role"
    participants = participants.copy()
    participants["norm_role"] = participants[role_col].fillna("").apply(normalize_role)

    # Filter to rows with valid roles
    valid = participants[participants["norm_role"].isin(ROLES_ORDERED)].copy()
    logger.info("  Participants with valid roles: %d / %d (%.1f%%)",
                len(valid), len(participants), 100 * len(valid) / len(participants))

    # Deduplicate: if multiple players have the same role on the same team,
    # keep the first (this handles rare data issues)
    valid = valid.drop_duplicates(subset=["match_id", "team_id", "norm_role"], keep="first")

    # Build champion vocab
    all_champions = sorted(valid["champion"].dropna().unique())
    champion_vocab = {name: idx + 1 for idx, name in enumerate(all_champions)}  # 0 = padding/unknown
    champion_vocab["<UNK>"] = 0
    logger.info("  Champion vocabulary: %d champions", len(champion_vocab) - 1)

    valid["champion_id"] = valid["champion"].map(champion_vocab).fillna(0).astype(int)

    # Pivot to one row per match
    rows = []
    grouped = valid.groupby("match_id")

    for match_id, group in grouped:
        blue = group[group["team_id"] == TEAM_BLUE].set_index("norm_role")
        red = group[group["team_id"] == TEAM_RED].set_index("norm_role")

        # Skip matches where not all 10 roles are filled
        if len(blue) < 5 or len(red) < 5:
            continue
        if not all(r in blue.index for r in ROLES_ORDERED):
            continue
        if not all(r in red.index for r in ROLES_ORDERED):
            continue

        row = {"match_id": match_id}
        for role in ROLES_ORDERED:
            row[f"blue_{role}"] = blue.loc[role, "champion_id"]
            row[f"red_{role}"] = red.loc[role, "champion_id"]

        # Win label: check if blue team won
        blue_wins = blue["win"].values if "win" in blue.columns else None
        if blue_wins is not None and len(blue_wins) > 0:
            row["blue_win"] = bool(blue_wins[0])
        else:
            row["blue_win"] = None

        rows.append(row)

    drafts = pd.DataFrame(rows)
    logger.info("  Valid drafts: %d matches", len(drafts))

    # Join patch from matches table
    if "patch" in matches.columns:
        patch_map = matches.set_index("match_id")["patch"]
        drafts["patch"] = drafts["match_id"].map(patch_map).fillna("unknown")
    elif "game_version" in matches.columns:
        def extract_patch(v):
            parts = str(v).split(".")
            return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else "unknown"
        matches["patch"] = matches["game_version"].apply(extract_patch)
        patch_map = matches.set_index("match_id")["patch"]
        drafts["patch"] = drafts["match_id"].map(patch_map).fillna("unknown")
    else:
        drafts["patch"] = "unknown"

    return drafts, champion_vocab


# ── Gold curves ───────────────────────────────────────────────

def build_gold_curves(snapshots: pd.DataFrame, valid_match_ids: set) -> pd.DataFrame:
    """
    Extract per-match gold differential at 5-min intervals.
    """
    logger.info("Building gold curves...")

    # Filter to valid matches
    snaps = snapshots[snapshots["match_id"].isin(valid_match_ids)].copy()

    # We need team_gold_diff — if it exists, use it directly
    if "team_gold_diff" in snaps.columns:
        # team_gold_diff is blue perspective (positive = blue ahead)
        # Get one row per (match_id, minute) — should be consistent across players on same team
        # Use the blue-side player (participant_id 1-5 typically, but safer to use team_id)
        if "team_id" in snaps.columns:
            blue_snaps = snaps[snaps["team_id"] == TEAM_BLUE]
        else:
            # Assume participant_id 1-5 = blue
            blue_snaps = snaps[snaps["participant_id"].between(1, 5)]

        # Take first player per team per minute (they all have same team_gold_diff)
        minute_diffs = (
            blue_snaps.groupby(["match_id", "minute"])["team_gold_diff"]
            .first()
            .reset_index()
        )
    elif "gold_diff" in snaps.columns and "total_gold" in snaps.columns:
        # Compute team gold diff from individual totals
        team_gold = (
            snaps.groupby(["match_id", "minute", "team_id"])["total_gold"]
            .sum()
            .reset_index()
        )
        blue_gold = team_gold[team_gold["team_id"] == TEAM_BLUE].set_index(["match_id", "minute"])["total_gold"]
        red_gold = team_gold[team_gold["team_id"] == TEAM_RED].set_index(["match_id", "minute"])["total_gold"]
        diff = (blue_gold - red_gold).reset_index()
        diff.columns = ["match_id", "minute", "team_gold_diff"]
        minute_diffs = diff
    else:
        logger.warning("No gold diff columns found — skipping gold curves")
        return pd.DataFrame()

    # Pivot minutes to columns
    target_minutes = [5, 10, 15, 20, 25, 30]
    minute_diffs = minute_diffs[minute_diffs["minute"].isin(target_minutes)]

    pivoted = minute_diffs.pivot_table(
        index="match_id", columns="minute", values="team_gold_diff", aggfunc="first"
    )
    pivoted.columns = [f"gold_diff_{int(m)}" for m in pivoted.columns]
    pivoted = pivoted.reset_index()

    logger.info("  Gold curves: %d matches with data", len(pivoted))
    return pivoted


# ── Objectives ────────────────────────────────────────────────

def build_objectives(snapshots: pd.DataFrame, valid_match_ids: set) -> pd.DataFrame:
    """
    Extract first-objective events from snapshot progression.
    """
    logger.info("Building objective sequences...")

    snaps = snapshots[snapshots["match_id"].isin(valid_match_ids)].copy()

    # Need one row per (match_id, minute, team_id) with objective counts
    obj_cols = [c for c in snaps.columns if c.startswith("team_") and c not in
                ("team_id", "team_kills", "team_gold_diff")]

    if not obj_cols:
        logger.warning("No objective columns found — skipping")
        return pd.DataFrame()

    # Get per-team objective counts at each minute
    # Use first player per team per minute as representative
    if "team_id" in snaps.columns:
        team_col = "team_id"
    else:
        # Derive team from participant_id
        snaps["team_id"] = snaps["participant_id"].apply(lambda x: TEAM_BLUE if x <= 5 else TEAM_RED)
        team_col = "team_id"

    team_snaps = (
        snaps.groupby(["match_id", "minute", team_col])[obj_cols]
        .first()
        .reset_index()
    )

    results = []
    for match_id, match_group in team_snaps.groupby("match_id"):
        row = {"match_id": match_id}

        for obj_name, col_name in [
            ("dragon", "team_dragons"),
            ("herald", "team_heralds"),
            ("baron", "team_barons"),
            ("tower", "team_towers"),
        ]:
            if col_name not in match_group.columns:
                continue

            first_team = 0
            first_minute = 0.0

            for team_id in [TEAM_BLUE, TEAM_RED]:
                team_data = match_group[match_group[team_col] == team_id].sort_values("minute")
                for _, r in team_data.iterrows():
                    val = r[col_name]
                    if pd.notna(val) and val > 0:
                        if first_team == 0 or r["minute"] < first_minute:
                            first_team = team_id
                            first_minute = r["minute"]
                        break

            row[f"first_{obj_name}_team"] = first_team
            row[f"first_{obj_name}_minute"] = first_minute

        results.append(row)

    objectives = pd.DataFrame(results)
    logger.info("  Objectives: %d matches", len(objectives))
    return objectives


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
    Draft meaning should be side-invariant.
    """
    logger.info("Augmenting with side swaps...")

    swapped = drafts.copy()
    swapped["match_id"] = swapped["match_id"] + "_swap"

    # Swap champion columns
    for role in ROLES_ORDERED:
        swapped[f"blue_{role}"], swapped[f"red_{role}"] = (
            drafts[f"red_{role}"].values,
            drafts[f"blue_{role}"].values,
        )

    # Invert win label
    if "blue_win" in swapped.columns:
        swapped["blue_win"] = ~swapped["blue_win"]

    combined = pd.concat([drafts, swapped], ignore_index=True)
    logger.info("  Augmented: %d → %d drafts", len(drafts), len(combined))
    return combined


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle LoL data for draft encoder training")
    parser.add_argument("--kaggle-dir", type=str, required=True, help="Directory containing Kaggle CSVs")
    parser.add_argument("--output-dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--queue-filter", type=int, default=420, help="Queue ID filter (420=ranked solo)")
    parser.add_argument("--no-augment", action="store_true", help="Skip side-swap augmentation")
    parser.add_argument("--min-games", type=int, default=100000, help="Minimum games required to proceed")
    args = parser.parse_args()

    kaggle_dir = Path(args.kaggle_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    matches, participants, snapshots = load_csvs(kaggle_dir)

    # Filter by queue
    if "queue_id" in matches.columns:
        before = len(matches)
        matches = matches[matches["queue_id"] == args.queue_filter]
        logger.info("Queue filter (%d): %d → %d matches", args.queue_filter, before, len(matches))
        valid_ids = set(matches["match_id"])
        participants = participants[participants["match_id"].isin(valid_ids)]
        snapshots = snapshots[snapshots["match_id"].isin(valid_ids)]

    # Build drafts
    drafts, champion_vocab = build_drafts(participants, matches)

    if len(drafts) < args.min_games:
        logger.error("Only %d valid drafts — expected at least %d. Check data.", len(drafts), args.min_games)
        sys.exit(1)

    valid_match_ids = set(drafts["match_id"])

    # Build training targets
    gold_curves = build_gold_curves(snapshots, valid_match_ids)
    objectives = build_objectives(snapshots, valid_match_ids)

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
    logger.info("  Champions: %d", len(champion_vocab) - 1)
    logger.info("  Gold curves: %d games", len(gold_curves))
    logger.info("  Objectives: %d games", len(objectives))
    if "patch" in drafts.columns:
        logger.info("  Patches: %s", sorted(drafts["patch"].unique())[:10])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
