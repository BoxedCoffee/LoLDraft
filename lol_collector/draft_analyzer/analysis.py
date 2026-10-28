#!/usr/bin/env python3
"""Build match-level League draft analysis artifacts and evidence-based pick/ban recommendations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ["top", "jng", "mid", "bot", "sup"]
ROLE_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
OBJECTIVES = ["dragon", "herald", "baron", "tower"]

# Full-roster 172-champion strategic identity rules (casefolded display names)
ARCHETYPE_RULES: dict[str, set[str]] = {
    "Engage": {
        "aatrox", "alistar", "ambessa", "amumu", "annie", "bard", "blitzcrank", "braum",
        "briar", "camille", "cho'gath", "diana", "elise", "fiddlesticks", "galio",
        "gnar", "gragas", "hecarim", "irelia", "jarvan iv", "k'sante", "kennen", "kled",
        "lee sin", "leona", "lissandra", "malphite", "maokai", "mordekaiser", "nautilus",
        "neeko", "nocturne", "nunu & willump", "olaf", "orianna", "ornn", "pantheon",
        "poppy", "pyke", "qiyana", "rakan", "rammus", "rek'sai", "rell", "renata glasc",
        "renekton", "rengar", "rumble", "samira", "sejuani", "sett", "shen", "shyvana",
        "singed", "sion", "skarner", "swain", "sylas", "tahm kench", "taric", "thresh",
        "vi", "viego", "volibear", "warwick", "wukong", "xin zhao", "yasuo", "yone",
        "zaahen", "zac",
    },
    "Poke": {
        "ahri", "akshan", "anivia", "ashe", "aurora", "brand", "caitlyn", "corki",
        "draven", "ezreal", "gangplank", "graves", "heimerdinger", "hwei", "ivern",
        "janna", "jayce", "jhin", "kalista", "karma", "karthus", "kha'zix", "leblanc",
        "lucian", "lulu", "lux", "malzahar", "mel", "milio", "miss fortune", "morgana",
        "naafiri", "nami", "nidalee", "quinn", "seraphine", "shaco", "sivir", "soraka",
        "syndra", "taliyah", "talon", "teemo", "twisted fate", "varus", "vel'koz",
        "vex", "viktor", "xerath", "yunara", "yuumi", "zed", "ziggs", "zilean", "zoe",
        "zyra",
    },
    "Scaling": {
        "aphelios", "aurelion sol", "azir", "bel'veth", "cassiopeia", "gwen",
        "jinx", "kai'sa", "karthus", "kassadin", "katarina", "kayle", "kayn", "kindred",
        "kog'maw", "lillia", "master yi", "nasus", "nilah", "ornn", "ryze", "senna",
        "sivir", "smolder", "sona", "swain", "tristana", "twitch", "vayne", "veigar",
        "viktor", "vladimir", "xayah", "yasuo", "yone", "zeri",
    },
    "Split push": {
        "akali", "camille", "darius", "dr. mundo", "ekko", "evelynn", "fiora", "fizz",
        "gangplank", "garen", "gwen", "illaoi", "irelia", "jax", "jayce", "k'sante",
        "kayle", "master yi", "mordekaiser", "nasus", "olaf", "quinn", "renekton",
        "riven", "ryze", "shen", "singed", "sion", "tristana", "trundle", "tryndamere",
        "twisted fate", "udyr", "urgot", "vayne", "volibear", "yasuo", "yone", "yorick",
        "zed",
    },
}


def champion_archetype_tags(champion_name: str) -> str:
    """Return comma-separated strategic archetype tags for a single champion."""
    folded = str(champion_name).strip().casefold()
    tags = [label for label, pool in ARCHETYPE_RULES.items() if folded in pool]
    return ", ".join(tags) if tags else "Flexible"


def wilson_lower_bound(wins: float, n: int, z: float = 1.96) -> float:
    """Compute the lower bound of the Wilson 95% score interval."""
    if n <= 0:
        return 0.0
    center = (wins + z * z / 2.0) / (n + z * z)
    margin = (
        z
        * math.sqrt(max(0.0, wins * (n - wins) / n + z * z / 4.0))
        / (n + z * z)
    )
    return float(center - margin)


def recommend_role_candidates(
    scoped_table: pd.DataFrame,
    similar_table: pd.DataFrame,
    stats: pd.DataFrame,
    blue_picks: dict[str, str],
    red_picks: dict[str, str],
    target_side: str = "blue",
    target_role: str = "top",
    min_role_matches: int = 15,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Rank candidate champions for an open (target_side, target_role) slot using:
      1) Smoothed role win rate on the scoped dataset
      2) Direct lane matchup win rate & 10m gold diff (if the opposing laner is known)
      3) Performance in comparable drafts (similar_table)
    """
    side = target_side.lower()
    opp_side = "red" if side == "blue" else "blue"
    own_picks = blue_picks if side == "blue" else red_picks
    opp_picks = red_picks if side == "blue" else blue_picks

    excluded = {
        str(c).strip()
        for c in [*blue_picks.values(), *red_picks.values()]
        if str(c).strip()
    }
    role_stats = stats[
        (stats["role"] == target_role)
        & (~stats["champion"].isin(excluded))
        & (stats["matches"] >= min_role_matches)
    ].copy()

    if role_stats.empty:
        role_stats = stats[
            (stats["role"] == target_role) & (~stats["champion"].isin(excluded))
        ].copy()
    if role_stats.empty:
        return pd.DataFrame()

    prior = float(scoped_table["blue_win"].mean()) if not scoped_table.empty else 0.5
    if side == "red":
        prior = 1.0 - prior

    opp_laner = str(opp_picks.get(target_role) or "").strip()

    # Pre-aggregate lane matchup performance if opposing laner is selected
    lane_lookup: dict[str, tuple[int, float, float]] = {}
    if opp_laner and not scoped_table.empty:
        lane_mask = scoped_table[f"{opp_side}_{target_role}"].eq(opp_laner)
        lane_frame = scoped_table.loc[lane_mask]
        if not lane_frame.empty:
            win_col = (
                lane_frame["blue_win"]
                if side == "blue"
                else ~lane_frame["blue_win"]
            )
            gold_col = (
                lane_frame["gold_diff_10"]
                if side == "blue"
                else -lane_frame["gold_diff_10"]
            )
            grouped = (
                pd.DataFrame(
                    {
                        "champion": lane_frame[f"{side}_{target_role}"],
                        "win": win_col.astype(float),
                        "gold_10": gold_col.astype(float),
                    }
                )
                .groupby("champion", as_index=False)
                .agg(
                    n=("win", "size"),
                    wr=("win", "mean"),
                    g10=("gold_10", "mean"),
                )
            )
            for row in grouped.itertuples(index=False):
                lane_lookup[str(row.champion)] = (
                    int(row.n),
                    float(row.wr),
                    float(row.g10),
                )

    # Pre-aggregate comparable-draft performance for this role
    comp_lookup: dict[str, tuple[int, float]] = {}
    if not similar_table.empty:
        win_col = (
            similar_table["blue_win"]
            if side == "blue"
            else ~similar_table["blue_win"]
        )
        comp_grouped = (
            pd.DataFrame(
                {
                    "champion": similar_table[f"{side}_{target_role}"],
                    "win": win_col.astype(float),
                }
            )
            .groupby("champion", as_index=False)
            .agg(n=("win", "size"), wr=("win", "mean"))
        )
        for row in comp_grouped.itertuples(index=False):
            comp_lookup[str(row.champion)] = (int(row.n), float(row.wr))

    rows = []
    strength = 50.0
    for r in role_stats.itertuples(index=False):
        champ = str(r.champion)
        n_role = int(r.matches)
        wr_role = float(r.win_rate)
        smoothed_role_wr = (n_role * wr_role + strength * prior) / (n_role + strength)

        comp_n, comp_wr = comp_lookup.get(champ, (0, smoothed_role_wr))
        smoothed_comp_wr = (comp_n * comp_wr + 20.0 * smoothed_role_wr) / (comp_n + 20.0)

        lane_n, lane_wr, lane_g10 = lane_lookup.get(champ, (0, smoothed_role_wr, float(r.mean_gold_diff_10)))
        smoothed_lane_wr = (lane_n * lane_wr + 15.0 * smoothed_role_wr) / (lane_n + 15.0)

        # Composite evidence score combining baseline role strength, comp synergy, and lane counter
        if opp_laner:
            composite_score = 0.40 * smoothed_role_wr + 0.35 * smoothed_lane_wr + 0.25 * smoothed_comp_wr
        else:
            composite_score = 0.55 * smoothed_role_wr + 0.45 * smoothed_comp_wr

        rows.append(
            {
                "Role": ROLE_LABELS.get(target_role, target_role),
                "Champion": champ,
                "Archetype tags": champion_archetype_tags(champ),
                "Composite score": round(composite_score * 100, 1),
                "Role win rate": f"{wr_role * 100:.1f}%",
                "Role matches": n_role,
                "Comp win rate": f"{comp_wr * 100:.1f}%" if comp_n > 0 else "—",
                "Comp matches": comp_n,
                "Lane vs opponent": (
                    f"{lane_wr * 100:.1f}% ({lane_n})" if (opp_laner and lane_n > 0) else "—"
                ),
                "Avg gold @10m": round(lane_g10 if (opp_laner and lane_n > 0) else float(r.mean_gold_diff_10), 1),
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df
    return (
        result_df.sort_values(["Composite score", "Role matches"], ascending=[False, False])
        .head(top_k)
        .reset_index(drop=True)
    )


def recommend_bans_from_evidence(
    scoped_table: pd.DataFrame,
    similar_table: pd.DataFrame,
    stats: pd.DataFrame,
    blue_picks: dict[str, str],
    red_picks: dict[str, str],
    target_side: str = "blue",
    min_role_matches: int = 25,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Recommend high-value bans for `target_side` by identifying enemy champions in
    the opponent's remaining open roles that have high win rates & pick presence.
    """
    side = target_side.lower()
    enemy_side = "red" if side == "blue" else "blue"
    enemy_picks = red_picks if side == "blue" else blue_picks

    open_enemy_roles = [r for r in ROLES if not str(enemy_picks.get(r) or "").strip()]
    if not open_enemy_roles:
        open_enemy_roles = list(ROLES)

    banned_or_picked = {
        str(c).strip()
        for c in [*blue_picks.values(), *red_picks.values()]
        if str(c).strip()
    }

    candidates = []
    for role in open_enemy_roles:
        cand_df = recommend_role_candidates(
            scoped_table=scoped_table,
            similar_table=similar_table,
            stats=stats,
            blue_picks=blue_picks,
            red_picks=red_picks,
            target_side=enemy_side,
            target_role=role,
            min_role_matches=min_role_matches,
            top_k=top_k,
        )
        if not cand_df.empty:
            candidates.append(cand_df)

    if not candidates:
        return pd.DataFrame()

    combined = pd.concat(candidates, ignore_index=True)
    combined = combined[~combined["Champion"].isin(banned_or_picked)]
    combined = (
        combined.sort_values(["Composite score", "Role matches"], ascending=[False, False])
        .drop_duplicates(subset=["Champion"], keep="first")
        .head(top_k)
        .rename(
            columns={
                "Role": "Enemy open role",
                "Composite score": "Threat score",
                "Role win rate": "Enemy role win rate",
                "Comp win rate": "Enemy win rate vs similar",
            }
        )
        .reset_index(drop=True)
    )
    return combined[
        [
            "Enemy open role",
            "Champion",
            "Archetype tags",
            "Threat score",
            "Enemy role win rate",
            "Role matches",
            "Enemy win rate vs similar",
            "Lane vs opponent",
        ]
    ]


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
            frame["gold_diff_10"] = np.where(
                side == "blue", frame["gold_diff_10"], -frame["gold_diff_10"]
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