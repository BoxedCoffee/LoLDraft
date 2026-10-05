#!/usr/bin/env python3
"""Build match-level League draft analysis artifacts from processed Riot data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ["top", "jng", "mid", "bot", "sup"]
OBJECTIVES = ["dragon", "herald", "baron", "tower"]


def load_match_table(data_dir: Path) -> pd.DataFrame:
    drafts = pd.read_parquet(data_dir / "drafts.parquet")
    drafts["base_match_id"] = drafts["match_id"].str.replace(
        r"_swap$", "", regex=True
    )
    drafts = drafts.loc[~drafts["match_id"].str.endswith("_swap")].copy()
    drafts = drafts.drop(columns=["match_id"]).drop_duplicates("base_match_id").rename(
        columns={"base_match_id": "match_id"}
    )
    gold = pd.read_parquet(data_dir / "gold_curves.parquet")
    objectives = pd.read_parquet(data_dir / "objectives.parquet")
    table = drafts.merge(gold, on="match_id", how="left").merge(
        objectives, on="match_id", how="left"
    )
    table["blue_win"] = table["blue_win"].astype(bool)
    for objective in OBJECTIVES:
        team = table[f"first_{objective}_team"]
        available = team.isin([100, 200])
        table[f"blue_first_{objective}"] = team.eq(100).where(available)
        table[f"red_first_{objective}"] = team.eq(200).where(available)
    table["blue_ahead_10"] = table["gold_diff_10"] > 0
    table["blue_ahead_20"] = table["gold_diff_20"] > 0
    table["blue_comeback_10"] = (~table["blue_ahead_10"]) & table["blue_win"]
    table["red_comeback_10"] = (table["blue_ahead_10"]) & (~table["blue_win"])
    table["blue_lead_conversion_10"] = table["blue_ahead_10"] & table["blue_win"]
    table["red_lead_conversion_10"] = (~table["blue_ahead_10"]) & (~table["blue_win"])
    return table


def rate(series: pd.Series) -> float | None:
    return None if series.empty else float(series.mean())


def summarize(table: pd.DataFrame) -> dict:
    summary = {
        "matches": int(len(table)),
        "patches": int(table["patch"].nunique()),
        "blue_win_rate": rate(table["blue_win"]),
        "gold_mean": {
            str(minute): float(table[f"gold_diff_{minute}"].mean())
            for minute in (5, 10, 15, 20, 25, 30)
            if f"gold_diff_{minute}" in table
        },
        "objective_blue_rates": {
            objective: rate(table[f"blue_first_{objective}"])
            for objective in OBJECTIVES
        },
        "objective_coverage": {
            objective: int(
                table[f"first_{objective}_team"].isin([100, 200]).sum()
            )
            for objective in OBJECTIVES
        },
        "blue_comeback_rate_when_behind_10": rate(
            table.loc[~table["blue_ahead_10"], "blue_win"]
        ),
        "blue_lead_conversion_rate_at_10": rate(
            table.loc[table["blue_ahead_10"], "blue_win"]
        ),
    }
    return summary


def build_champion_stats(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for role in ROLES:
        for side in ("blue", "red"):
            column = f"{side}_{role}"
            frame = table[[column, "blue_win", "gold_diff_10"]].rename(
                columns={column: "champion"}
            )
            frame["champion"] = frame["champion"].astype(str)
            frame["champion_win"] = np.where(
                side == "blue", frame["blue_win"], ~frame["blue_win"]
            )
            frame["role"] = role
            rows.append(frame[["champion", "role", "champion_win", "gold_diff_10"]])
    stats = pd.concat(rows, ignore_index=True)
    return (
        stats.groupby(["champion", "role"], as_index=False)
        .agg(
            matches=("champion_win", "size"),
            win_rate=("champion_win", "mean"),
            mean_gold_diff_10=("gold_diff_10", "mean"),
        )
        .sort_values(["matches", "win_rate"], ascending=[False, False])
    )


def build_patch_summary(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby("patch", as_index=False)
        .agg(
            matches=("match_id", "size"),
            blue_win_rate=("blue_win", "mean"),
            mean_gold_diff_10=("gold_diff_10", "mean"),
            blue_first_dragon=("blue_first_dragon", "mean"),
            blue_first_tower=("blue_first_tower", "mean"),
            blue_comeback_rate=("blue_comeback_10", "mean"),
        )
        .sort_values("patch")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = load_match_table(args.data_dir)
    table.to_parquet(args.output_dir / "match_analysis.parquet", index=False)
    build_champion_stats(table).to_parquet(
        args.output_dir / "champion_role_stats.parquet", index=False
    )
    build_patch_summary(table).to_csv(args.output_dir / "patch_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summarize(table), indent=2), encoding="utf-8"
    )
    print(json.dumps(summarize(table), indent=2))


if __name__ == "__main__":
    main()
