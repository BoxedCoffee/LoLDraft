#!/usr/bin/env python3
"""Prepare Riot-collected Parquet data for draft encoder training.

Usage:
  python prepare_riot_dataset.py --state-db ../data/collection_state.db --output-dir ./data/processed_riot
  python prepare_riot_dataset.py --max-games 50000 --seed 42

Inputs (defaults assume running from lol_collector/draft_encoder):
  - Matches parquet:   ../data/matches/
  - Timeline frames:   ../data/timelines/frames_*.parquet
  - Timeline events:   ../data/timelines/events/events_*.parquet

Outputs (same schema as prepare_dataset.py):
  - drafts.parquet
  - gold_curves.parquet
  - objectives.parquet
  - champion_vocab.json
  - splits.json
"""

import argparse
import json
import logging
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("prepare_riot")

ROLE_MAP = {
    "TOP": "top",
    "JUNGLE": "jng",
    "MIDDLE": "mid",
    "MID": "mid",
    "BOTTOM": "bot",
    "BOT": "bot",
    "UTILITY": "sup",
    "SUPPORT": "sup",
    "NONE": "unk",
    "": "unk",
}

TEAM_BLUE = 100
TEAM_RED = 200
ROLES_ORDERED = ["top", "jng", "mid", "bot", "sup"]
TARGET_MINUTES = [5, 10, 15, 20, 25, 30]


def normalize_role(role: str) -> str:
    return ROLE_MAP.get(str(role).upper().strip(), "unk")


def load_complete_match_ids(state_db: Path) -> list[str]:
    con = sqlite3.connect(state_db)
    try:
        cur = con.cursor()
        cur.execute("SELECT match_id FROM matches WHERE status=3")
        return [r[0] for r in cur.fetchall()]
    finally:
        con.close()


def iter_parquet_files(dir_path: Path, pattern: str) -> list[Path]:
    files = sorted(dir_path.glob(pattern))
    return [p for p in files if p.is_file()]


def build_drafts(
    matches_dir: Path,
    match_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    logger.info("Building drafts from %s...", matches_dir)

    champion_ids: set[int] = set()
    rows: list[dict] = []

    dataset = ds.dataset(matches_dir, format="parquet", exclude_invalid_files=True)
    match_id_list = list(match_ids)
    expr = ds.field("match_id").isin(match_id_list)
    table = dataset.to_table(
        columns=["match_id", "team_id", "champion_id", "role", "win", "patch"],
        filter=expr,
    )
    df = table.to_pandas()
    logger.info("  Loaded %d participant rows", len(df))

    if df.empty:
        return pd.DataFrame(), {"<UNK>": 0}

    df["norm_role"] = df["role"].apply(normalize_role)
    df = df[df["norm_role"].isin(ROLES_ORDERED)]
    df = df.drop_duplicates(subset=["match_id", "team_id", "norm_role"], keep="first")
    champion_ids.update(int(x) for x in df["champion_id"].dropna().unique())

    for match_id, group in df.groupby("match_id"):
        blue = group[group["team_id"] == TEAM_BLUE].set_index("norm_role")
        red = group[group["team_id"] == TEAM_RED].set_index("norm_role")

        if len(blue) < 5 or len(red) < 5:
            continue
        if not all(r in blue.index for r in ROLES_ORDERED):
            continue
        if not all(r in red.index for r in ROLES_ORDERED):
            continue

        row = {
            "match_id": match_id,
            "patch": str(group["patch"].iloc[0]) if "patch" in group.columns else "unknown",
        }
        for role in ROLES_ORDERED:
            row[f"blue_{role}"] = int(blue.loc[role, "champion_id"])
            row[f"red_{role}"] = int(red.loc[role, "champion_id"])
        row["blue_win"] = bool(blue["win"].iloc[0])
        rows.append(row)

    drafts = pd.DataFrame(rows)
    logger.info("  Valid drafts: %d", len(drafts))

    champion_list = sorted(champion_ids)
    champion_vocab = {str(cid): idx + 1 for idx, cid in enumerate(champion_list)}
    champion_vocab["<UNK>"] = 0
    logger.info("  Champion vocab: %d", len(champion_vocab) - 1)

    for col in [f"blue_{r}" for r in ROLES_ORDERED] + [f"red_{r}" for r in ROLES_ORDERED]:
        drafts[col] = drafts[col].map(lambda x: champion_vocab.get(str(int(x)), 0)).astype(int)

    return drafts, champion_vocab


def make_splits(match_ids: list[str], drafts: pd.DataFrame, seed: int) -> dict:
    if "patch" in drafts.columns:
        patches = drafts.set_index("match_id")["patch"]
        strat = [patches.get(mid, "unknown") for mid in match_ids]
        from collections import Counter

        counts = Counter(strat)
        if not counts or min(counts.values()) < 2:
            strat = None
    else:
        strat = None

    train_ids, temp_ids = train_test_split(
        match_ids,
        test_size=0.2,
        random_state=seed,
        stratify=strat,
    )

    if strat is not None:
        patches = drafts.set_index("match_id")["patch"]
        temp_strat = [patches.get(mid, "unknown") for mid in temp_ids]
        from collections import Counter

        counts = Counter(temp_strat)
        if not counts or min(counts.values()) < 2:
            temp_strat = None
    else:
        temp_strat = None

    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.5,
        random_state=seed,
        stratify=temp_strat,
    )
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def augment_side_swap(drafts: pd.DataFrame) -> pd.DataFrame:
    swapped = drafts.copy()
    swapped["match_id"] = swapped["match_id"] + "_swap"

    for role in ROLES_ORDERED:
        swapped[f"blue_{role}"], swapped[f"red_{role}"] = (
            drafts[f"red_{role}"].values,
            drafts[f"blue_{role}"].values,
        )

    swapped["blue_win"] = ~swapped["blue_win"]
    return pd.concat([drafts, swapped], ignore_index=True)


def build_gold_curves(timelines_dir: Path, match_ids: set[str]) -> pd.DataFrame:
    frame_files = iter_parquet_files(timelines_dir, "frames_*.parquet")
    dataset = ds.dataset(frame_files, format="parquet", exclude_invalid_files=True)
    match_id_list = list(match_ids)
    expr = ds.field("match_id").isin(match_id_list)
    scanner = dataset.scanner(
        columns=["match_id", "timestamp_ms", "participant_id", "total_gold"],
        filter=expr,
    )

    team_gold: dict[tuple[str, int, int], int] = defaultdict(int)

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue
        df["minute"] = (df["timestamp_ms"] // 60000).astype(int)
        df = df[df["minute"].isin(TARGET_MINUTES)]
        if df.empty:
            continue
        df["team_id"] = df["participant_id"].apply(lambda x: TEAM_BLUE if int(x) <= 5 else TEAM_RED)
        grouped = df.groupby(["match_id", "minute", "team_id"])["total_gold"].sum()
        for (match_id, minute, team_id), total in grouped.items():
            team_gold[(match_id, int(minute), int(team_id))] += int(total)

    rows = []
    for match_id in match_ids:
        row = {"match_id": match_id}
        has_any = False
        for minute in TARGET_MINUTES:
            blue = team_gold.get((match_id, minute, TEAM_BLUE))
            red = team_gold.get((match_id, minute, TEAM_RED))
            if blue is None or red is None:
                row[f"gold_diff_{minute}"] = None
            else:
                row[f"gold_diff_{minute}"] = float(blue - red)
                has_any = True
        if has_any:
            rows.append(row)

    out = pd.DataFrame(rows)
    logger.info("  Gold curves: %d matches", len(out))
    return out


def build_objectives(events_dir: Path, match_ids: set[str]) -> pd.DataFrame:
    dataset = ds.dataset(events_dir, format="parquet", exclude_invalid_files=True)
    match_id_list = list(match_ids)
    expr = ds.field("match_id").isin(match_id_list)
    scanner = dataset.scanner(
        columns=["match_id", "timestamp_min", "event_type", "monster_type", "building_type", "team_id"],
        filter=expr,
    )
    logger.info("Building objectives...")

    first_team: dict[str, dict[str, int]] = {t: {} for t in ["dragon", "herald", "baron", "tower"]}
    first_min: dict[str, dict[str, float]] = {t: {} for t in ["dragon", "herald", "baron", "tower"]}

    def _update(obj_type: str, mid: str, minute: float, team: int):
        prev = first_min[obj_type].get(mid)
        if prev is None or minute < prev:
            first_min[obj_type][mid] = float(minute)
            first_team[obj_type][mid] = int(team)

    for batch in scanner.to_batches():
        df = batch.to_pandas()
        if df.empty:
            continue

        df = df[pd.notna(df["timestamp_min"]) & pd.notna(df["team_id"])]
        if df.empty:
            continue

        elite = df[df["event_type"] == "ELITE_MONSTER_KILL"]
        if not elite.empty:
            for monster, obj_type in [
                ("DRAGON", "dragon"),
                ("RIFTHERALD", "herald"),
                ("BARON_NASHOR", "baron"),
            ]:
                sub = elite[elite["monster_type"] == monster]
                if sub.empty:
                    continue
                sub = sub.sort_values(["match_id", "timestamp_min"])
                first = sub.groupby("match_id").first().reset_index()
                for _, r in first.iterrows():
                    _update(obj_type, r["match_id"], float(r["timestamp_min"]), int(r["team_id"]))

        towers = df[(df["event_type"] == "BUILDING_KILL") & (df["building_type"] == "TOWER_BUILDING")]
        if not towers.empty:
            towers = towers.sort_values(["match_id", "timestamp_min"])
            first = towers.groupby("match_id").first().reset_index()
            for _, r in first.iterrows():
                _update("tower", r["match_id"], float(r["timestamp_min"]), int(r["team_id"]))

    rows = []
    for mid in match_ids:
        row = {"match_id": mid}
        for obj_type in ["dragon", "herald", "baron", "tower"]:
            row[f"first_{obj_type}_team"] = first_team[obj_type].get(mid, 0)
            row[f"first_{obj_type}_minute"] = first_min[obj_type].get(mid, 0.0)
        rows.append(row)

    out = pd.DataFrame(rows)
    logger.info("  Objectives: %d matches", len(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Riot Parquet data for draft encoder")
    parser.add_argument("--state-db", type=str, default="../data/collection_state.db")
    parser.add_argument("--matches-dir", type=str, default="../data/matches")
    parser.add_argument("--timelines-dir", type=str, default="../data/timelines")
    parser.add_argument("--output-dir", type=str, default="./data/processed_riot")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--patch", action="append", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    state_db = Path(args.state_db)
    matches_dir = Path(args.matches_dir)
    timelines_dir = Path(args.timelines_dir)
    events_dir = timelines_dir / "events"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_complete = load_complete_match_ids(state_db)
    logger.info("Complete matches in state DB: %d", len(all_complete))

    if args.max_games is not None and args.max_games < len(all_complete):
        selected = set(random.sample(all_complete, args.max_games))
    else:
        selected = set(all_complete)

    drafts, champion_vocab = build_drafts(matches_dir, selected)

    if drafts.empty:
        logger.error("No drafts produced. Check input paths/state DB.")
        return

    if args.patch:
        patch_set = set(args.patch)
        drafts = drafts[drafts["patch"].isin(patch_set)].reset_index(drop=True)

    if drafts.empty:
        logger.error("No drafts left after patch filtering.")
        return

    base_ids = list(drafts["match_id"].unique())
    splits = make_splits(base_ids, drafts, seed=args.seed)

    if not args.no_augment:
        drafts = augment_side_swap(drafts)

    valid_ids = set(base_ids)
    gold_curves = build_gold_curves(timelines_dir, valid_ids)
    objectives = build_objectives(events_dir, valid_ids)

    drafts.to_parquet(output_dir / "drafts.parquet", index=False)
    if len(gold_curves) > 0:
        gold_curves.to_parquet(output_dir / "gold_curves.parquet", index=False)
    if len(objectives) > 0:
        objectives.to_parquet(output_dir / "objectives.parquet", index=False)

    with open(output_dir / "champion_vocab.json", "w", encoding="utf-8") as f:
        json.dump(champion_vocab, f)
    with open(output_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f)

    logger.info("Saved dataset to %s", output_dir)


if __name__ == "__main__":
    main()
