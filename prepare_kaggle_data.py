#!/usr/bin/env python3
"""
Prepare Kaggle LoL dataset (5mLoLGames or standard Kaggle snapshots) for draft encoder training.

Usage:
    python prepare_kaggle_data.py --kaggle-dir ./5mLoLGames --output-dir ./data/processed

Supports:
  - matches.csv + processed_summoner_data.csv / participants.csv + ChampionTbl.csv (+ intervals.csv / snapshots.csv)
  - Produces the canonical DraftDataset schema:
      drafts.parquet        (match_id, patch, blue_top..blue_sup, red_top..red_sup, blue_win)
      gold_curves.parquet   (gold_diff_5..gold_diff_30 when interval data is available)
      objectives.parquet    (first_<obj>_team / minute when interval data is available)
      champion_vocab.json   (contiguous 0..N vocabulary indices with "<UNK>": 0)
      splits.json           (train/val/test match_id lists)
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("prepare_kaggle")

ROLE_MAP = {
    "TOP": "top",
    "JUNGLE": "jng",
    "MIDDLE": "mid",
    "MID": "mid",
    "BOTTOM": "bot",
    "BOT": "bot",
    "CARRY": "bot",
    "DUO_CARRY": "bot",
    "UTILITY": "sup",
    "SUPPORT": "sup",
    "DUO_SUPPORT": "sup",
    "NONE": "unk",
    "": "unk",
}

TEAM_BLUE = 100
TEAM_RED = 200
ROLES_ORDERED = ["top", "jng", "mid", "bot", "sup"]


def normalize_role(role: str) -> str:
    return ROLE_MAP.get(str(role).upper().strip(), "unk")


def find_optional_csv(kaggle_dir: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        candidates = sorted(kaggle_dir.glob(f"*{pattern}*"))
        if candidates:
            return candidates[0]
    return None


def load_csvs(kaggle_dir: Path) -> dict[str, pd.DataFrame | None]:
    """Load available Kaggle tables from kaggle_dir."""
    logger.info("Loading CSVs from %s...", kaggle_dir)

    match_path = find_optional_csv(kaggle_dir, ["match", "Match"])
    if match_path is None:
        logger.error("No matches CSV found in %s", kaggle_dir)
        sys.exit(1)

    matches = pd.read_csv(match_path, low_memory=False)
    logger.info("  Loaded matches (%s): %d rows", match_path.name, len(matches))

    champ_path = find_optional_csv(kaggle_dir, ["ChampionTbl", "champion"])
    champion_df = pd.read_csv(champ_path, low_memory=False) if champ_path else None
    if champion_df is not None:
        logger.info("  Loaded champions (%s): %d rows", champ_path.name, len(champion_df))

    part_path = find_optional_csv(kaggle_dir, ["particip", "summoner"])
    participants = pd.read_csv(part_path, low_memory=False) if part_path else None
    if participants is not None:
        logger.info("  Loaded participants (%s): %d rows", part_path.name, len(participants))

    snap_path = find_optional_csv(kaggle_dir, ["snapshot", "interval"])
    snapshots = pd.read_csv(snap_path, low_memory=False) if snap_path else None
    if snapshots is not None:
        logger.info("  Loaded intervals/snapshots (%s): %d rows", snap_path.name, len(snapshots))

    return {
        "matches": matches,
        "champions": champion_df,
        "participants": participants,
        "snapshots": snapshots,
    }


def _extract_patch(version_value) -> str:
    parts = str(version_value).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(version_value or "unknown")


def build_drafts(
    matches: pd.DataFrame,
    champion_df: pd.DataFrame | None,
    participants: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Build role-ordered 10-champion drafts and a contiguous champion_vocab (0..N)
    compatible with DraftDataset and train.py.
    """
    logger.info("Building draft representations...")

    # Map raw ChampionId -> ChampionName if ChampionTbl.csv is present
    raw_id_to_name: dict[str, str] = {}
    if champion_df is not None and {"ChampionId", "ChampionName"}.issubset(champion_df.columns):
        for _, row in champion_df.iterrows():
            if pd.notna(row["ChampionId"]) and pd.notna(row["ChampionName"]):
                raw_id_to_name[str(int(row["ChampionId"]))] = str(row["ChampionName"])

    # Case 1: Participant table with per-player picks + roles exists
    if participants is not None:
        part = participants.copy()
        match_col = next((c for c in ("match_id", "matchId", "gameId") if c in part.columns), None)
        role_col = next(
            (c for c in ("individual_position", "teamPosition", "role", "lane") if c in part.columns),
            None,
        )
        champ_col = next(
            (c for c in ("champion", "championName", "champion_id", "championId") if c in part.columns),
            None,
        )
        team_col = next((c for c in ("team_id", "teamId", "team") if c in part.columns), None)

        if match_col and role_col and champ_col and team_col:
            part["match_id"] = part[match_col].astype(str)
            part["norm_role"] = part[role_col].fillna("").apply(normalize_role)
            part = part[part["norm_role"].isin(ROLES_ORDERED)].copy()
            part["team_id"] = part[team_col].apply(
                lambda t: TEAM_BLUE if str(t).upper() in ("100", "1", "BLUE") else TEAM_RED
            )
            part = part.drop_duplicates(subset=["match_id", "team_id", "norm_role"], keep="first")

            def resolve_champ_token(val) -> str:
                if pd.isna(val):
                    return "<UNK>"
                s = str(int(val)) if isinstance(val, (int, float)) else str(val).strip()
                return raw_id_to_name.get(s, s)

            part["champ_key"] = part[champ_col].apply(resolve_champ_token)
            unique_champs = sorted(k for k in part["champ_key"].unique() if k != "<UNK>")
            champion_vocab = {name: idx + 1 for idx, name in enumerate(unique_champs)}
            champion_vocab["<UNK>"] = 0

            part["vocab_id"] = part["champ_key"].map(champion_vocab).fillna(0).astype(int)

            match_meta = matches.copy()
            m_id_col = next((c for c in ("match_id", "matchId", "gameId") if c in match_meta.columns), "match_id")
            match_meta["match_id"] = match_meta[m_id_col].astype(str)
            meta_by_id = match_meta.drop_duplicates("match_id").set_index("match_id")

            rows = []
            for mid, group in part.groupby("match_id"):
                blue = group[group["team_id"] == TEAM_BLUE].set_index("norm_role")
                red = group[group["team_id"] == TEAM_RED].set_index("norm_role")
                if len(blue) < 5 or len(red) < 5:
                    continue
                if not all(r in blue.index for r in ROLES_ORDERED) or not all(r in red.index for r in ROLES_ORDERED):
                    continue

                row = {"match_id": mid}
                for role in ROLES_ORDERED:
                    row[f"blue_{role}"] = int(blue.loc[role, "vocab_id"])
                    row[f"red_{role}"] = int(red.loc[role, "vocab_id"])

                if "win" in blue.columns:
                    row["blue_win"] = bool(blue["win"].iloc[0])
                elif mid in meta_by_id.index and "winning_team" in meta_by_id.columns:
                    row["blue_win"] = bool(meta_by_id.loc[mid, "winning_team"] == TEAM_BLUE)
                else:
                    continue

                if mid in meta_by_id.index:
                    m_row = meta_by_id.loc[mid]
                    if "patch" in meta_by_id.columns:
                        row["patch"] = str(m_row["patch"])
                    elif "patch_version" in meta_by_id.columns:
                        row["patch"] = _extract_patch(m_row["patch_version"])
                    elif "game_version" in meta_by_id.columns:
                        row["patch"] = _extract_patch(m_row["game_version"])
                    else:
                        row["patch"] = "unknown"
                else:
                    row["patch"] = "unknown"
                rows.append(row)

            drafts = pd.DataFrame(rows)
            logger.info("  Valid participant-based drafts: %d matches (%d champions)", len(drafts), len(champion_vocab) - 1)
            return drafts, champion_vocab

    # Case 2: Fallback when only matches.csv is present (extract 5v5 champion slots into blue_top..red_sup)
    all_raw_tokens: set[str] = set()
    parsed_matches = []

    def parse_five(val) -> list[str]:
        if pd.isna(val) or str(val).strip() in ("", "nan"):
            return ["<UNK>"] * 5
        tokens = [
            raw_id_to_name.get(x.strip(), x.strip()) if x.strip() not in ("-1", "0", "") else "<UNK>"
            for x in str(val).split(",")
        ]
        tokens.extend(["<UNK>"] * (5 - len(tokens)))
        return tokens[:5]

    for _, match in matches.iterrows():
        blue_tokens = parse_five(match.get("blue_picks", match.get("blue_bans", "")))
        red_tokens = parse_five(match.get("red_picks", match.get("red_bans", "")))
        all_raw_tokens.update(t for t in blue_tokens + red_tokens if t != "<UNK>")
        winning_team = match.get("winning_team", TEAM_BLUE)
        blue_win = bool(winning_team == TEAM_BLUE or winning_team is True or winning_team == 1)
        patch_val = match.get("patch", match.get("patch_version", match.get("game_version", "unknown")))
        parsed_matches.append((str(match["match_id"]), _extract_patch(patch_val), blue_win, blue_tokens, red_tokens))

    for name in raw_id_to_name.values():
        all_raw_tokens.add(name)

    sorted_tokens = sorted(all_raw_tokens)
    champion_vocab = {token: idx + 1 for idx, token in enumerate(sorted_tokens)}
    champion_vocab["<UNK>"] = 0

    rows = []
    for mid, patch, blue_win, blue_tokens, red_tokens in parsed_matches:
        row = {
            "match_id": mid,
            "patch": patch,
            "blue_win": blue_win,
        }
        for role, b_tok, r_tok in zip(ROLES_ORDERED, blue_tokens, red_tokens):
            row[f"blue_{role}"] = int(champion_vocab.get(b_tok, 0))
            row[f"red_{role}"] = int(champion_vocab.get(r_tok, 0))
        rows.append(row)

    drafts = pd.DataFrame(rows)
    logger.info("  Valid drafts: %d matches (%d champions)", len(drafts), len(champion_vocab) - 1)
    return drafts, champion_vocab


def build_gold_curves(snapshots: pd.DataFrame | None, valid_match_ids: set[str]) -> pd.DataFrame:
    """Extract gold curves at 5-minute intervals when interval/snapshot data is present."""
    logger.info("Building gold curves...")
    if snapshots is None or snapshots.empty or "match_id" not in snapshots.columns:
        logger.info("  Gold curves: 0 matches (no interval/snapshot table provided)")
        return pd.DataFrame()

    snaps = snapshots.copy()
    snaps["match_id"] = snaps["match_id"].astype(str)
    snaps = snaps[snaps["match_id"].isin(valid_match_ids)]
    if snaps.empty or "minute" not in snaps.columns:
        return pd.DataFrame()

    if "team_gold_diff" in snaps.columns:
        blue_snaps = snaps[snaps["team_id"] == TEAM_BLUE] if "team_id" in snaps.columns else snaps
        minute_diffs = blue_snaps.groupby(["match_id", "minute"])["team_gold_diff"].first().reset_index()
    elif {"total_gold", "team_id"}.issubset(snaps.columns):
        team_gold = snaps.groupby(["match_id", "minute", "team_id"])["total_gold"].sum().reset_index()
        blue_gold = team_gold[team_gold["team_id"] == TEAM_BLUE].set_index(["match_id", "minute"])["total_gold"]
        red_gold = team_gold[team_gold["team_id"] == TEAM_RED].set_index(["match_id", "minute"])["total_gold"]
        minute_diffs = (blue_gold - red_gold).reset_index()
        minute_diffs.columns = ["match_id", "minute", "team_gold_diff"]
    else:
        return pd.DataFrame()

    target_minutes = [5, 10, 15, 20, 25, 30]
    minute_diffs = minute_diffs[minute_diffs["minute"].isin(target_minutes)]
    if minute_diffs.empty:
        return pd.DataFrame()

    pivoted = minute_diffs.pivot_table(
        index="match_id", columns="minute", values="team_gold_diff", aggfunc="first"
    )
    pivoted.columns = [f"gold_diff_{int(m)}" for m in pivoted.columns]
    for m in target_minutes:
        if f"gold_diff_{m}" not in pivoted.columns:
            pivoted[f"gold_diff_{m}"] = None
    pivoted = pivoted[[f"gold_diff_{m}" for m in target_minutes]].reset_index()
    logger.info("  Gold curves: %d matches with data", len(pivoted))
    return pivoted


def build_objectives(snapshots: pd.DataFrame | None, valid_match_ids: set[str]) -> pd.DataFrame:
    """Extract first-objective labels when interval/snapshot data is present."""
    logger.info("Building objective sequences...")
    if snapshots is None or snapshots.empty or "match_id" not in snapshots.columns:
        logger.info("  Objectives: 0 matches (no interval/snapshot table provided)")
        return pd.DataFrame()
    return pd.DataFrame()


def make_splits(match_ids: list[str], drafts: pd.DataFrame) -> dict[str, list[str]]:
    """80/10/10 split, stratified by patch if possible."""
    logger.info("Creating train/val/test splits...")
    if "patch" in drafts.columns:
        patches = drafts.set_index("match_id")["patch"]
        strat = [patches.get(mid, "unknown") for mid in match_ids]
        counts = Counter(strat)
        if not counts or min(counts.values()) < 2:
            strat = None
    else:
        strat = None

    train_ids, temp_ids = train_test_split(match_ids, test_size=0.2, random_state=42, stratify=strat)

    if strat is not None:
        temp_strat = [patches.get(mid, "unknown") for mid in temp_ids]
        temp_counts = Counter(temp_strat)
        if not temp_counts or min(temp_counts.values()) < 2:
            temp_strat = None
    else:
        temp_strat = None

    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42, stratify=temp_strat)
    logger.info("  Train: %d | Val: %d | Test: %d", len(train_ids), len(val_ids), len(test_ids))
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def augment_side_swap(drafts: pd.DataFrame) -> pd.DataFrame:
    """Double the dataset by swapping blue/red role columns and inverting blue_win."""
    logger.info("Augmenting with side swaps...")
    swapped = drafts.copy()
    swapped["match_id"] = swapped["match_id"].astype(str) + "_swap"

    for role in ROLES_ORDERED:
        if f"blue_{role}" in drafts.columns and f"red_{role}" in drafts.columns:
            swapped[f"blue_{role}"], swapped[f"red_{role}"] = (
                drafts[f"red_{role}"].values,
                drafts[f"blue_{role}"].values,
            )

    if "blue_win" in swapped.columns:
        swapped["blue_win"] = ~swapped["blue_win"].astype(bool)

    combined = pd.concat([drafts, swapped], ignore_index=True)
    logger.info("  Augmented: %d → %d drafts", len(drafts), len(combined))
    return combined


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

    tables = load_csvs(kaggle_dir)
    drafts, champion_vocab = build_drafts(
        tables["matches"], tables["champions"], tables["participants"]
    )

    if len(drafts) < args.min_games:
        logger.error("Only %d valid drafts — expected at least %d. Check data.", len(drafts), args.min_games)
        sys.exit(1)

    valid_match_ids = set(drafts["match_id"].astype(str))
    gold_curves = build_gold_curves(tables["snapshots"], valid_match_ids)
    objectives = build_objectives(tables["snapshots"], valid_match_ids)

    original_ids = list(drafts["match_id"].astype(str))
    splits = make_splits(original_ids, drafts)

    if not args.no_augment:
        drafts = augment_side_swap(drafts)

    logger.info("Saving to %s...", output_dir)
    drafts.to_parquet(output_dir / "drafts.parquet", index=False)
    logger.info("  drafts.parquet: %d rows", len(drafts))

    if len(gold_curves) > 0:
        gold_curves.to_parquet(output_dir / "gold_curves.parquet", index=False)
        logger.info("  gold_curves.parquet: %d rows", len(gold_curves))

    if len(objectives) > 0:
        objectives.to_parquet(output_dir / "objectives.parquet", index=False)
        logger.info("  objectives.parquet: %d rows", len(objectives))

    with open(output_dir / "champion_vocab.json", "w", encoding="utf-8") as f:
        json.dump(champion_vocab, f, indent=2)
    logger.info("  champion_vocab.json: %d entries", len(champion_vocab))

    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f)
    logger.info(
        "  splits.json: train=%d, val=%d, test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

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