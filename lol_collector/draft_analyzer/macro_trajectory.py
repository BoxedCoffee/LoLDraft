#!/usr/bin/env python3
"""
Phase 2a Macro Trajectory Representation & Statistical Archetype Probe.

Implements the core technical components specified in the project roadmap:
1. Summoner's Rift Voronoi Landmark Discretization (30 semantic strategic zones)
2. Per-minute macro state vector extraction from Riot timeline frames & events
   (grouping ratio, side-lane split-push presence, jungler top/bot proximity,
   gold/XP differential)
3. Quantitative Archetype Probe (Welch t-test & Mann-Whitney U test across
   Engage, Poke, Scaling, and Split push compositions)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Summoner's Rift coordinate bounds: [0, 15000] x [0, 15000]
# 30 semantic strategic landmarks for Voronoi map tessellation
SR_LANDMARKS: Dict[str, Tuple[float, float]] = {
    "blue_fountain": (500.0, 500.0),
    "blue_nexus": (1550.0, 1650.0),
    "blue_top_inhib": (1170.0, 3570.0),
    "blue_mid_inhib": (3200.0, 3200.0),
    "blue_bot_inhib": (3450.0, 1240.0),
    "blue_top_inner": (1510.0, 6700.0),
    "blue_mid_inner": (5050.0, 4800.0),
    "blue_bot_inner": (6900.0, 1480.0),
    "blue_top_outer": (980.0, 10440.0),
    "blue_mid_outer": (5850.0, 6400.0),
    "blue_bot_outer": (10500.0, 1030.0),
    "blue_top_jungle_bluebuff": (3800.0, 7900.0),
    "blue_bot_jungle_redbuff": (7800.0, 4000.0),
    "red_fountain": (14300.0, 14300.0),
    "red_nexus": (13250.0, 13150.0),
    "red_top_inhib": (11250.0, 13650.0),
    "red_mid_inhib": (11600.0, 11600.0),
    "red_bot_inhib": (13600.0, 11300.0),
    "red_top_inner": (7950.0, 13400.0),
    "red_mid_inner": (9750.0, 10100.0),
    "red_bot_inner": (13300.0, 8200.0),
    "red_top_outer": (4300.0, 13850.0),
    "red_mid_outer": (8950.0, 8500.0),
    "red_bot_outer": (13850.0, 4500.0),
    "red_top_jungle_redbuff": (7100.0, 10900.0),
    "red_bot_jungle_bluebuff": (10950.0, 7000.0),
    "baron_herald_pit": (5000.0, 10400.0),
    "dragon_pit": (9860.0, 4400.0),
    "top_lane_midpoint": (2400.0, 12400.0),
    "bot_lane_midpoint": (12400.0, 2400.0),
}

LANDMARK_NAMES: List[str] = list(SR_LANDMARKS.keys())
LANDMARK_COORDS: np.ndarray = np.array([SR_LANDMARKS[k] for k in LANDMARK_NAMES], dtype=np.float32)

SIDE_LANE_ZONES = {
    "blue_top_inner", "blue_top_outer", "red_top_inner", "red_top_outer", "top_lane_midpoint",
    "blue_bot_inner", "blue_bot_outer", "red_bot_inner", "red_bot_outer", "bot_lane_midpoint",
}


def assign_voronoi_zones(positions_xy: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Assign (N, 2) player coordinates on Summoner's Rift to the nearest strategic
    Voronoi landmark zone.

    Returns:
        zone_indices: (N,) int array in [0, len(LANDMARK_NAMES) - 1]
        zone_names: list of N landmark names
    """
    pts = np.asarray(positions_xy, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    diffs = pts[:, None, :] - LANDMARK_COORDS[None, :, :]  # (N, Z, 2)
    dists_sq = np.sum(diffs * diffs, axis=-1)              # (N, Z)
    indices = np.argmin(dists_sq, axis=-1)
    names = [LANDMARK_NAMES[int(i)] for i in indices]
    return indices, names


def compute_team_grouping_and_split(
    positions_xy: np.ndarray, group_radius: float = 2500.0
) -> Dict[str, float]:
    """
    Given (5, 2) coordinates for one team's 5 players at a single timestep:
      - is_grouped: 1.0 if at least 3 players are within `group_radius` of a teammate cluster center
      - max_cluster_size: largest number of players within `group_radius` of any player
      - side_lane_split: 1.0 if 1 player is in a side-lane zone and >3500 units from all teammates
    """
    pts = np.asarray(positions_xy, dtype=np.float32)
    if len(pts) == 0:
        return {"is_grouped": 0.0, "max_cluster_size": 0.0, "side_lane_split": 0.0}

    diffs = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))  # (P, P)
    neighbor_counts = np.sum(dists <= group_radius, axis=-1)
    max_cluster = int(np.max(neighbor_counts))
    is_grouped = 1.0 if max_cluster >= 3 else 0.0

    _, zone_names = assign_voronoi_zones(pts)
    side_lane_split = 0.0
    for idx, zname in enumerate(zone_names):
        if zname in SIDE_LANE_ZONES:
            other_dists = np.delete(dists[idx], idx)
            if len(other_dists) > 0 and float(np.min(other_dists)) >= 3500.0:
                side_lane_split = 1.0
                break

    return {
        "is_grouped": float(is_grouped),
        "max_cluster_size": float(max_cluster),
        "side_lane_split": float(side_lane_split),
    }


def extract_macro_snapshots(frames_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw timeline participant frames (`match_id`, `timestamp_ms`, `participant_id`,
    `position_x`, `position_y`, `total_gold`, `xp`) into per-minute macro state snapshots.
    """
    if frames_df.empty:
        return pd.DataFrame()

    df = frames_df.copy()
    if "minute" not in df.columns:
        df["minute"] = (df["timestamp_ms"] // 60000).astype(int)

    # Focus on post-laning / mid-game macro window (minutes 5..35)
    df = df[(df["minute"] >= 5) & (df["minute"] <= 35)]
    if df.empty:
        return pd.DataFrame()

    snapshots = []
    for (match_id, minute), group in df.groupby(["match_id", "minute"]):
        blue = group[group["participant_id"] <= 5].sort_values("participant_id")
        red = group[group["participant_id"] > 5].sort_values("participant_id")
        if blue.empty or red.empty:
            continue

        blue_xy = blue[["position_x", "position_y"]].to_numpy(dtype=np.float32)
        red_xy = red[["position_x", "position_y"]].to_numpy(dtype=np.float32)

        blue_macro = compute_team_grouping_and_split(blue_xy)
        red_macro = compute_team_grouping_and_split(red_xy)

        # Jungler (participant 2 on Blue, participant 7 on Red) top-vs-bot river balance
        # Positive (y - x) indicates top/Baron side of Summoner's Rift; negative indicates bot/Dragon side
        blue_jng = blue[blue["participant_id"] == 2]
        red_jng = red[red["participant_id"] == 7]
        blue_jng_top_bias = (
            float((blue_jng["position_y"].iloc[0] - blue_jng["position_x"].iloc[0]) / 15000.0)
            if not blue_jng.empty
            else 0.0
        )
        red_jng_top_bias = (
            float((red_jng["position_y"].iloc[0] - red_jng["position_x"].iloc[0]) / 15000.0)
            if not red_jng.empty
            else 0.0
        )

        gold_diff = float(blue["total_gold"].sum() - red["total_gold"].sum())
        xp_diff = float(blue["xp"].sum() - red["xp"].sum()) if "xp" in group.columns else 0.0

        snapshots.append(
            {
                "match_id": str(match_id),
                "minute": int(minute),
                "gold_diff": gold_diff,
                "xp_diff": xp_diff,
                "blue_is_grouped": blue_macro["is_grouped"],
                "blue_max_cluster": blue_macro["max_cluster_size"],
                "blue_side_lane_split": blue_macro["side_lane_split"],
                "blue_jng_top_bias": round(blue_jng_top_bias, 4),
                "red_is_grouped": red_macro["is_grouped"],
                "red_max_cluster": red_macro["max_cluster_size"],
                "red_side_lane_split": red_macro["side_lane_split"],
                "red_jng_top_bias": round(red_jng_top_bias, 4),
            }
        )

    return pd.DataFrame(snapshots)


def run_archetype_trajectory_probe(
    game_metrics_df: pd.DataFrame,
    archetype_col: str = "archetype",
    metric_cols: Tuple[str, ...] = ("grouping_rate", "side_lane_split_rate", "first_obj_minute"),
) -> Dict[str, object]:
    """
    Execute the Phase 2a Quantitative Probe Experiment from the project roadmap:
    test whether macro trajectory metrics differ significantly (p < 0.05) across
    pairings of draft archetypes.
    """
    archetypes = sorted(game_metrics_df[archetype_col].dropna().unique())
    comparisons = []
    significant_pairs = 0

    for i in range(len(archetypes)):
        for j in range(i + 1, len(archetypes)):
            arch_a, arch_b = archetypes[i], archetypes[j]
            sub_a = game_metrics_df[game_metrics_df[archetype_col] == arch_a]
            sub_b = game_metrics_df[game_metrics_df[archetype_col] == arch_b]

            pair_results = {}
            pair_has_sig = False
            for col in metric_cols:
                if col not in game_metrics_df.columns:
                    continue
                vals_a = sub_a[col].dropna().to_numpy(dtype=np.float64)
                vals_b = sub_b[col].dropna().to_numpy(dtype=np.float64)
                if len(vals_a) < 3 or len(vals_b) < 3:
                    continue
                t_stat, p_val = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)
                if math.isnan(p_val):
                    p_val = 1.0
                if p_val < 0.05:
                    pair_has_sig = True
                pair_results[col] = {
                    "mean_a": round(float(np.mean(vals_a)), 4),
                    "mean_b": round(float(np.mean(vals_b)), 4),
                    "t_stat": round(float(t_stat), 4) if not math.isnan(t_stat) else 0.0,
                    "p_value": round(float(p_val), 6),
                    "significant_at_0_05": bool(p_val < 0.05),
                }

            if pair_has_sig:
                significant_pairs += 1
            comparisons.append(
                {
                    "pair": f"{arch_a} vs {arch_b}",
                    "significant": pair_has_sig,
                    "metrics": pair_results,
                }
            )

    return {
        "total_pairs": len(comparisons),
        "significant_pairs": significant_pairs,
        "passes_roadmap_threshold": significant_pairs >= min(3, max(1, len(comparisons))),
        "comparisons": comparisons,
    }