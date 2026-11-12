#!/usr/bin/env python3
"""
LoL Draft Studio — Zero-Refresh Client-Side SPA + Vectorized int16 Analytics & Neural Engine.

Works seamlessly both via:
  1. `streamlit run lol_collector/draft_analyzer/app.py`
  2. `python lol_collector/draft_analyzer/app.py` (standalone server at http://127.0.0.1:8765)

Why this eliminates page refreshes & UI lag:
  - All slot selection, champion picking, search, role/archetype filtering, preset loading,
    and tab switching run 100% client-side in the browser DOM (0 ms, zero Streamlit rerun).
  - Live analytics & neural predictions are served over a persistent in-memory localhost JSON
    endpoint (`/api/analyze`) backed by a vectorized `np.int16` draft matrix (751,339 matches
    queried in ~10 ms) and the trained Wide & Deep `DraftModel` (`checkpoints/best.pt`).
"""

from __future__ import annotations

import json
import math
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
ENCODER_DIR = APP_DIR.parent / "draft_encoder"
for _p in (ENCODER_DIR, APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis import (
    ARCHETYPE_RULES,
    OBJECTIVES,
    ROLE_LABELS,
    ROLES,
    champion_archetype_tags,
)

try:
    import torch
    from model import DraftModel
    HAS_TORCH = True
except ImportError:
    torch = None
    DraftModel = None
    HAS_TORCH = False

DATA_DIR = ENCODER_DIR / "data" / "processed_riot"
OUTPUT_DIR = APP_DIR / "analysis_output"
CHECKPOINT_PATH = ENCODER_DIR / "checkpoints" / "best.pt"
OBJECTIVE_LABELS = {"dragon": "Dragon", "herald": "Herald", "baron": "Baron", "tower": "Tower"}
DDRAGON_VERSION = "14.20.1"

ROLE_ICONS = {
    "top": "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-top.png",
    "jng": "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-jungle.png",
    "mid": "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-middle.png",
    "bot": "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-bottom.png",
    "sup": "https://raw.communitydragon.org/latest/plugins/rcp-fe-lol-clash/global/default/assets/images/position-selector/positions/icon-position-utility.png",
}

DEMO_PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "Dive vs. Poke": {
        "blue": {"top": "Ornn", "jng": "Jarvan IV", "mid": "Orianna", "bot": "Kai'Sa", "sup": "Leona"},
        "red": {"top": "Jayce", "jng": "Nidalee", "mid": "Viktor", "bot": "Ezreal", "sup": "Karma"},
    },
    "Scaling vs. Split": {
        "blue": {"top": "K'Sante", "jng": "Sejuani", "mid": "Azir", "bot": "Jinx", "sup": "Lulu"},
        "red": {"top": "Fiora", "jng": "Viego", "mid": "Twisted Fate", "bot": "Caitlyn", "sup": "Lux"},
    },
    "Teamfight vs. Pick": {
        "blue": {"top": "Malphite", "jng": "Amumu", "mid": "Syndra", "bot": "Miss Fortune", "sup": "Rell"},
        "red": {"top": "Camille", "jng": "Lee Sin", "mid": "Ahri", "bot": "Jhin", "sup": "Thresh"},
    },
    "Mid-Draft (7/10)": {
        "blue": {"top": "Aatrox", "jng": "Lee Sin", "mid": "Ahri", "bot": "Jinx", "sup": ""},
        "red": {"top": "Renekton", "jng": "Vi", "mid": "", "bot": "Caitlyn", "sup": ""},
    },
}


def build_champion_metadata(encoder_data_dir: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """Resolve vocabulary IDs, Riot IDs, and Data Dragon metadata (square icon, splash art, title)."""
    id_to_name: dict[str, str] = {}
    riot_id_to_name: dict[str, str] = {}
    champ_meta: dict[str, dict] = {}

    full_mapping_path = encoder_data_dir / "full_champion_mapping.json"
    if full_mapping_path.exists():
        full_mapping = json.loads(full_mapping_path.read_text(encoding="utf-8"))
        riot_id_to_name.update({str(riot_id): name for name, riot_id in full_mapping.items()})

    champion_data_path = encoder_data_dir / "champion_data.json"
    if champion_data_path.exists():
        champion_data = json.loads(champion_data_path.read_text(encoding="utf-8"))
        for key, entry in champion_data.items():
            name = entry.get("name")
            if not name:
                continue
            ddragon_id = entry.get("ddragon_id") or name.replace(" ", "").replace("'", "").replace(".", "")
            square_url = entry.get("image") or f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/champion/{ddragon_id}.png"
            splash_url = f"https://ddragon.leagueoflegends.com/cdn/img/champion/splash/{ddragon_id}_0.jpg"
            arch_str = champion_archetype_tags(name)
            champ_meta[name] = {
                "name": name,
                "title": entry.get("title", ""),
                "riot_id": entry.get("riot_id"),
                "ddragon_id": ddragon_id,
                "image": square_url,
                "splash": splash_url,
                "archetypes": arch_str,
                "arch_list": [t.strip() for t in arch_str.split(",") if t.strip()],
            }
            id_to_name[str(key)] = name
            if entry.get("riot_id") is not None:
                riot_to_name_key = str(entry["riot_id"])
                riot_id_to_name[riot_to_name_key] = name

    vocab_path = encoder_data_dir / "processed_riot" / "champion_vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        for riot_id, vocab_id in vocab.items():
            if riot_id != "<UNK>" and str(riot_id) in riot_id_to_name:
                id_to_name[str(vocab_id)] = riot_id_to_name[str(riot_id)]

    for riot_id, name in riot_id_to_name.items():
        id_to_name.setdefault(riot_id, name)

    return id_to_name, champ_meta


def wilson_ci(wins: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return 0.5, 0.5
    z = 1.96
    center = (wins + z * z / 2.0) / (n + z * z)
    margin = z * math.sqrt(max(0.0, wins * (n - wins) / n + z * z / 4.0)) / (n + z * z)
    return float(center - margin), float(center + margin)


class FastDraftEngine:
    """
    In-memory vectorized int16 analytics + Wide & Deep PyTorch neural inference engine.
    Queries 751,339 historical matches in ~10 ms using integer array comparisons instead
    of Pandas string scans.
    """

    def __init__(self):
        table = pd.read_parquet(OUTPUT_DIR / "match_analysis.parquet")
        stats = pd.read_parquet(OUTPUT_DIR / "champion_role_stats.parquet")
        encoder_data_dir = APP_DIR.parent / "draft_encoder" / "data"
        self.id_to_name, self.champ_meta = build_champion_metadata(encoder_data_dir)

        if self.id_to_name:
            for role in ROLES:
                for side in ("blue", "red"):
                    col = f"{side}_{role}"
                    table[col] = table[col].map(lambda v: self.id_to_name.get(str(v), str(v)))
            stats["champion"] = stats["champion"].map(lambda v: self.id_to_name.get(str(v), str(v)))

        self.stats_df = stats
        self.total_matches = len(table)

        # Build compact champion <-> int16 index for instant numpy comparisons
        all_names = sorted(
            set(self.champ_meta.keys()) | set(stats["champion"].dropna().astype(str).unique()),
            key=str.casefold,
        )
        self.all_champions = all_names
        self.name_to_idx: dict[str, int] = {name: i + 1 for i, name in enumerate(all_names)}
        self.idx_to_name: dict[int, str] = {i + 1: name for i, name in enumerate(all_names)}

        # Encode (N, 10) int16 draft matrix: cols 0..4 = blue_top..sup, cols 5..9 = red_top..sup
        n_rows = len(table)
        self.draft_mat = np.zeros((n_rows, 10), dtype=np.int16)
        for r_idx, role in enumerate(ROLES):
            self.draft_mat[:, r_idx] = (
                table[f"blue_{role}"].map(self.name_to_idx).fillna(0).to_numpy(dtype=np.int16)
            )
            self.draft_mat[:, r_idx + 5] = (
                table[f"red_{role}"].map(self.name_to_idx).fillna(0).to_numpy(dtype=np.int16)
            )

        # Pre-extract contiguous metric arrays for sub-10ms aggregation
        self.patches = table["patch"].fillna("").astype(str).to_numpy()
        self.unique_patches = sorted(
            [p for p in pd.unique(self.patches) if p], reverse=True
        )
        self.blue_win = table["blue_win"].to_numpy(dtype=np.bool_)
        self.blue_ahead_10 = table["blue_ahead_10"].to_numpy(dtype=np.bool_)

        self.gold_Arrays: dict[int, np.ndarray] = {}
        for m in (5, 10, 15, 20, 25, 30):
            col = f"gold_diff_{m}"
            self.gold_Arrays[m] = (
                table[col].fillna(0.0).to_numpy(dtype=np.float32)
                if col in table
                else np.zeros(n_rows, dtype=np.float32)
            )

        self.obj_team: dict[str, np.ndarray] = {}
        for obj in OBJECTIVES:
            col = f"first_{obj}_team"
            self.obj_team[obj] = (
                table[col].fillna(0).to_numpy(dtype=np.int16)
                if col in table
                else np.zeros(n_rows, dtype=np.int16)
            )

        # Pre-build archetype membership boolean lookup per champion index (num_champs+1, 4)
        self.arch_labels = list(ARCHETYPE_RULES.keys())
        self.champ_arch_mask = np.zeros((len(all_names) + 1, len(self.arch_labels)), dtype=np.int8)
        for c_name, c_idx in self.name_to_idx.items():
            folded = c_name.casefold()
            for a_idx, a_label in enumerate(self.arch_labels):
                if folded in ARCHETYPE_RULES[a_label]:
                    self.champ_arch_mask[c_idx, a_idx] = 1

        # Pre-build role stats dictionary for instant client & server lookup
        self.role_stats_map: dict[str, list[dict[str, Any]]] = {r: [] for r in ROLES}
        self.champ_role_lookup: dict[tuple[str, str], dict[str, Any]] = {}
        for row in stats.itertuples(index=False):
            r = str(row.role)
            c = str(row.champion)
            entry = {
                "champion": c,
                "role": r,
                "matches": int(row.matches),
                "win_rate": round(float(row.win_rate), 4),
                "mean_gold_diff_10": round(float(row.mean_gold_diff_10), 1),
            }
            if r in self.role_stats_map:
                self.role_stats_map[r].append(entry)
            self.champ_role_lookup[(c, r)] = entry

        # Load Neural DraftModel if available
        self.predictor = self._load_neural_model()

    def _load_neural_model(self) -> dict[str, Any] | None:
        if not HAS_TORCH or not CHECKPOINT_PATH.exists():
            return None
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
            champion_vocab = ckpt["champion_vocab"]
            patch_vocab = ckpt.get("patch_vocab") or {}
            cfg = ckpt.get("config", {})
            model_cfg = cfg.get("model", {})
            gold_scale = float(cfg.get("training", {}).get("gold_scale", 5000.0))

            model = DraftModel(
                num_champions=len(champion_vocab),
                champion_dim=model_cfg.get("champion_dim", 64),
                draft_dim=model_cfg.get("draft_dim", 256),
                num_patches=len(patch_vocab),
                patch_dim=model_cfg.get("patch_dim", 16),
                team_hidden=model_cfg.get("team_hidden", 512),
                num_team_layers=model_cfg.get("num_team_layers", 3),
                num_gold_points=6,
                num_objectives=4,
                dropout=0.0,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            full_map_path = ENCODER_DIR / "data" / "full_champion_mapping.json"
            name_to_vocab_id: dict[str, int] = {}
            if full_map_path.exists():
                full_map = json.loads(full_map_path.read_text(encoding="utf-8"))
                for name, riot_id in full_map.items():
                    if str(riot_id) in champion_vocab:
                        name_to_vocab_id[name] = int(champion_vocab[str(riot_id)])

            return {
                "model": model,
                "patch_vocab": patch_vocab,
                "name_to_vocab_id": name_to_vocab_id,
                "gold_scale": gold_scale,
            }
        except Exception:
            return None

    def run_neural(self, blue: dict[str, str], red: dict[str, str], patch: str) -> dict[str, Any] | None:
        if self.predictor is None:
            return None
        name_to_id = self.predictor["name_to_vocab_id"]
        blue_ids = [name_to_id.get(blue.get(r, ""), 0) for r in ROLES]
        red_ids = [name_to_id.get(red.get(r, ""), 0) for r in ROLES]
        if not any(blue_ids) and not any(red_ids):
            return None

        patch_vocab = self.predictor["patch_vocab"]
        patch_id = patch_vocab.get(str(patch), patch_vocab.get("<UNK>", 0))
        blue_t = torch.tensor([blue_ids], dtype=torch.long)
        red_t = torch.tensor([red_ids], dtype=torch.long)
        patch_t = torch.tensor([patch_id], dtype=torch.long)

        with torch.inference_mode():
            _, win_logit, gold_pred, obj_logits = self.predictor["model"](blue_t, red_t, patch_t)
            win_prob = float(torch.sigmoid(win_logit).item())
            gold_curve = [round(float(x * self.predictor["gold_scale"]), 1) for x in gold_pred[0].cpu().numpy()]
            obj_probs = {}
            for name, logits in zip(OBJECTIVES, obj_logits):
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                obj_probs[name] = {
                    "none": round(float(probs[0]), 4),
                    "blue": round(float(probs[1]), 4),
                    "red": round(float(probs[2]), 4),
                }

        return {
            "blue_win_prob": round(win_prob, 4),
            "gold_curve": gold_curve,
            "obj_probs": obj_probs,
        }

    def analyze(
        self,
        blue: dict[str, str],
        red: dict[str, str],
        active_side: str = "blue",
        active_role: str = "sup",
        mode: str = "Auto (recommended)",
        min_evidence: int = 250,
        patch: str = "All patches",
    ) -> dict[str, Any]:
        """Execute full vectorized historical + neural analysis in ~12 ms."""
        if patch and patch != "All patches":
            scope_mask = self.patches == str(patch)
        else:
            scope_mask = np.ones(self.total_matches, dtype=np.bool_)

        scoped_indices = np.flatnonzero(scope_mask)
        n_scoped = len(scoped_indices)
        sub_mat = self.draft_mat[scoped_indices]

        selected: list[tuple[int, int, int]] = []  # (col_idx 0..9, role_idx 0..4, champ_idx)
        for r_idx, r in enumerate(ROLES):
            b_c = self.name_to_idx.get(blue.get(r, ""), 0)
            if b_c > 0:
                selected.append((r_idx, r_idx, b_c))
            r_c = self.name_to_idx.get(red.get(r, ""), 0)
            if r_c > 0:
                selected.append((r_idx + 5, r_idx, r_c))

        n_sel = len(selected)
        if n_sel == 0:
            sim_local_idx = np.arange(n_scoped)
            resolved_mode = "All matches"
            ladder = [
                {"mode": "Exact roles & sides", "matches": n_scoped},
                {"mode": "Role-aware (either side)", "matches": n_scoped},
                {"mode": "Champion presence", "matches": n_scoped},
            ]
        else:
            exact_hits = np.zeros(n_scoped, dtype=np.int8)
            role_hits = np.zeros(n_scoped, dtype=np.int8)
            pres_hits = np.zeros(n_scoped, dtype=np.int8)

            for col_idx, role_idx, cid in selected:
                exact_hits += sub_mat[:, col_idx] == cid
                role_hits += (sub_mat[:, role_idx] == cid) | (sub_mat[:, role_idx + 5] == cid)
                pres_hits += (sub_mat == cid).any(axis=1)

            ladder = [
                {"mode": "Exact roles & sides", "matches": int(np.sum(exact_hits == n_sel))},
                {"mode": "Role-aware (either side)", "matches": int(np.sum(role_hits == n_sel))},
                {"mode": "Champion presence", "matches": int(np.sum(pres_hits == n_sel))},
            ]

            sim_local_idx = None
            resolved_mode = mode
            if mode == "Exact roles & sides":
                sim_local_idx = np.flatnonzero(exact_hits == n_sel)
            elif mode == "Role-aware (either side)":
                sim_local_idx = np.flatnonzero(role_hits == n_sel)
            elif mode == "Champion presence":
                sim_local_idx = np.flatnonzero(pres_hits == n_sel)
            else:
                # Auto relaxation using precomputed hit vectors (0.1 ms!)
                for req in range(n_sel, 0, -1):
                    for m_name, vec in (
                        ("Exact roles & sides", exact_hits),
                        ("Role-aware (either side)", role_hits),
                        ("Champion presence", pres_hits),
                    ):
                        cnt = int(np.sum(vec >= req))
                        if cnt >= min_evidence:
                            sim_local_idx = np.flatnonzero(vec >= req)
                            resolved_mode = f"{m_name} · ≥{req}/{n_sel} picks"
                            break
                    if sim_local_idx is not None:
                        break
                if sim_local_idx is None:
                    sim_local_idx = np.flatnonzero(pres_hits >= 1)
                    resolved_mode = f"Champion presence · ≥1/{n_sel} picks"

        sim_global_idx = scoped_indices[sim_local_idx]
        n_sim = len(sim_global_idx)

        baseline_wr = float(self.blue_win[scoped_indices].mean()) if n_scoped > 0 else 0.5
        if n_sim > 0:
            sim_wins = self.blue_win[sim_global_idx]
            obs_wr = float(sim_wins.mean())
            ci_low, ci_high = wilson_ci(float(sim_wins.sum()), n_sim)

            ahead_mask = self.blue_ahead_10[sim_global_idx]
            behind_mask = ~ahead_mask
            conv_rate = float(sim_wins[ahead_mask].mean()) if np.any(ahead_mask) else 0.5
            comeback_rate = float(sim_wins[behind_mask].mean()) if np.any(behind_mask) else 0.5

            gold_curve_hist = [
                round(float(self.gold_Arrays[m][sim_global_idx].mean()), 1)
                for m in (5, 10, 15, 20, 25, 30)
            ]

            objectives_out = {}
            for obj in OBJECTIVES:
                t_arr = self.obj_team[obj][sim_global_idx]
                valid = (t_arr == 100) | (t_arr == 200)
                n_val = int(valid.sum())
                b_first = float((t_arr[valid] == 100).mean()) if n_val > 0 else 0.0
                r_first = float((t_arr[valid] == 200).mean()) if n_val > 0 else 0.0

                # Conditional win rates when securing objective
                b_sec = t_arr == 100
                win_if_first = float(sim_wins[b_sec].mean()) if np.any(b_sec) else 0.0
                ahead_sec = b_sec & ahead_mask
                behind_sec = b_sec & behind_mask
                objectives_out[obj] = {
                    "label": OBJECTIVE_LABELS[obj],
                    "observed": n_val,
                    "blue_first": round(b_first, 4),
                    "red_first": round(r_first, 4),
                    "blue_win_if_first": round(win_if_first, 4),
                    "blue_win_ahead_first": round(float(sim_wins[ahead_sec].mean()), 4) if np.any(ahead_sec) else None,
                    "blue_win_behind_first": round(float(sim_wins[behind_sec].mean()), 4) if np.any(behind_sec) else None,
                }

            # Fast vectorized archetype distribution on similar matches
            sample_idx = sim_global_idx[:5000]
            arch_dist = {}
            for s_idx, s_name in ((0, "blue"), (5, "red")):
                team_ids = self.draft_mat[sample_idx, s_idx : s_idx + 5]  # (M, 5)
                # Sum archetype tags across the 5 champions per match: (M, 4)
                tag_sums = self.champ_arch_mask[team_ids].sum(axis=1)
                top_a = np.argmax(tag_sums, axis=1)
                max_tags = np.max(tag_sums, axis=1)
                counts = {}
                for a_i, a_label in enumerate(self.arch_labels):
                    c = int(np.sum((top_a == a_i) & (max_tags >= 1)))
                    if c > 0:
                        counts[a_label] = round(c / len(sample_idx) * 100, 1)
                bal = int(np.sum(max_tags == 0))
                if bal > 0:
                    counts["Balanced"] = round(bal / len(sample_idx) * 100, 1)
                arch_dist[s_name] = counts
        else:
            obs_wr, ci_low, ci_high = baseline_wr, 0.5, 0.5
            conv_rate, comeback_rate = 0.5, 0.5
            gold_curve_hist = [0.0] * 6
            objectives_out = {}
            arch_dist = {"blue": {}, "red": {}}

        # Head-to-head lane matchups across scoped dataset
        lane_matchups = []
        for r_idx, r in enumerate(ROLES):
            b_name = blue.get(r, "")
            r_name = red.get(r, "")
            if b_name and r_name:
                b_cid = self.name_to_idx.get(b_name, -1)
                r_cid = self.name_to_idx.get(r_name, -1)
                m_mask = (sub_mat[:, r_idx] == b_cid) & (sub_mat[:, r_idx + 5] == r_cid)
                m_global = scoped_indices[m_mask]
                m_cnt = len(m_global)
                lane_matchups.append({
                    "role": r,
                    "role_label": ROLE_LABELS[r],
                    "blue_champ": b_name,
                    "red_champ": r_name,
                    "matches": m_cnt,
                    "blue_wr": round(float(self.blue_win[m_global].mean()), 4) if m_cnt > 0 else None,
                    "gold_10": round(float(self.gold_Arrays[10][m_global].mean()), 1) if m_cnt > 0 else None,
                })

        # Vectorized Pick & Ban Recommendations for active_side & active_role
        recs = self._recommend_for_slot(
            scoped_indices, sim_global_idx, blue, red, active_side, active_role, baseline_wr, top_k=10
        )

        # Priority Bans targeting remaining open enemy roles
        enemy_side = "red" if active_side == "blue" else "blue"
        enemy_picks = red if active_side == "blue" else blue
        open_enemy_roles = [r for r in ROLES if not enemy_picks.get(r)] or list(ROLES)
        ban_pool = []
        for er in open_enemy_roles:
            er_recs = self._recommend_for_slot(
                scoped_indices, sim_global_idx, blue, red, enemy_side, er, baseline_wr, top_k=4
            )
            ban_pool.extend(er_recs)
        ban_pool.sort(key=lambda x: (x["score"], x["role_matches"]), reverse=True)
        seen_bans = set()
        top_bans = []
        for item in ban_pool:
            if item["champion"] not in seen_bans:
                seen_bans.add(item["champion"])
                top_bans.append(item)
                if len(top_bans) >= 6:
                    break

        neural = self.run_neural(blue, red, patch)

        return {
            "scoped_matches": n_scoped,
            "similar_matches": n_sim,
            "resolved_mode": resolved_mode,
            "baseline_wr": round(baseline_wr, 4),
            "observed_blue_wr": round(obs_wr, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "lead_conversion_10": round(conv_rate, 4),
            "comeback_rate_10": round(comeback_rate, 4),
            "gold_curve_hist": gold_curve_hist,
            "objectives": objectives_out,
            "arch_dist": arch_dist,
            "lane_matchups": lane_matchups,
            "recommendations": recs,
            "bans": top_bans,
            "ladder": ladder,
            "neural": neural,
        }

    def _recommend_for_slot(
        self,
        scoped_indices: np.ndarray,
        sim_global_idx: np.ndarray,
        blue: dict[str, str],
        red: dict[str, str],
        target_side: str,
        target_role: str,
        baseline_blue_wr: float,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        side = target_side.lower()
        r_idx = ROLES.index(target_role)
        own_col = r_idx if side == "blue" else r_idx + 5
        opp_col = r_idx + 5 if side == "blue" else r_idx
        opp_picks = red if side == "blue" else blue
        opp_laner = opp_picks.get(target_role, "")
        opp_cid = self.name_to_idx.get(opp_laner, 0)

        excluded = {c for c in [*blue.values(), *red.values()] if c}
        prior = baseline_blue_wr if side == "blue" else (1.0 - baseline_blue_wr)

        # Comparable-draft performance per champion in own_col
        comp_n_arr = np.zeros(len(self.all_champions) + 1, dtype=np.int32)
        comp_w_arr = np.zeros(len(self.all_champions) + 1, dtype=np.float64)
        if len(sim_global_idx) > 0:
            sim_cids = self.draft_mat[sim_global_idx, own_col]
            sim_w = self.blue_win[sim_global_idx] if side == "blue" else ~self.blue_win[sim_global_idx]
            comp_n_arr = np.bincount(sim_cids, minlength=len(self.all_champions) + 1)
            comp_w_arr = np.bincount(sim_cids, weights=sim_w.astype(np.float64), minlength=len(self.all_champions) + 1)

        # Head-to-head lane performance against opp_cid if opponent laner is locked
        lane_n_arr = np.zeros(len(self.all_champions) + 1, dtype=np.int32)
        lane_w_arr = np.zeros(len(self.all_champions) + 1, dtype=np.float64)
        lane_g_arr = np.zeros(len(self.all_champions) + 1, dtype=np.float64)
        if opp_cid > 0 and len(scoped_indices) > 0:
            opp_mask = self.draft_mat[scoped_indices, opp_col] == opp_cid
            lane_idx = scoped_indices[opp_mask]
            if len(lane_idx) > 0:
                l_cids = self.draft_mat[lane_idx, own_col]
                l_w = self.blue_win[lane_idx] if side == "blue" else ~self.blue_win[lane_idx]
                l_g = self.gold_Arrays[10][lane_idx] if side == "blue" else -self.gold_Arrays[10][lane_idx]
                lane_n_arr = np.bincount(l_cids, minlength=len(self.all_champions) + 1)
                lane_w_arr = np.bincount(l_cids, weights=l_w.astype(np.float64), minlength=len(self.all_champions) + 1)
                lane_g_arr = np.bincount(l_cids, weights=l_g.astype(np.float64), minlength=len(self.all_champions) + 1)

        candidates = []
        for entry in self.role_stats_map.get(target_role, []):
            champ = entry["champion"]
            if champ in excluded or entry["matches"] < 15:
                continue
            cid = self.name_to_idx.get(champ, 0)
            n_role = entry["matches"]
            wr_role = entry["win_rate"]
            smoothed_role = (n_role * wr_role + 50.0 * prior) / (n_role + 50.0)

            cn = int(comp_n_arr[cid])
            c_wr = float(comp_w_arr[cid] / cn) if cn > 0 else smoothed_role
            smoothed_comp = (cn * c_wr + 20.0 * smoothed_role) / (cn + 20.0)

            ln = int(lane_n_arr[cid])
            l_wr = float(lane_w_arr[cid] / ln) if ln > 0 else smoothed_role
            l_g10 = float(lane_g_arr[cid] / ln) if ln > 0 else entry["mean_gold_diff_10"]
            smoothed_lane = (ln * l_wr + 15.0 * smoothed_role) / (ln + 15.0)

            if opp_cid > 0:
                score = 0.40 * smoothed_role + 0.35 * smoothed_lane + 0.25 * smoothed_comp
            else:
                score = 0.55 * smoothed_role + 0.45 * smoothed_comp

            candidates.append({
                "champion": champ,
                "role": target_role,
                "role_label": ROLE_LABELS[target_role],
                "archetypes": self.champ_meta.get(champ, {}).get("archetypes", "Flexible"),
                "score": round(score * 100, 1),
                "role_wr": round(wr_role * 100, 1),
                "role_matches": n_role,
                "comp_wr": round(c_wr * 100, 1) if cn > 0 else None,
                "comp_matches": cn,
                "lane_wr": round(l_wr * 100, 1) if (opp_cid > 0 and ln > 0) else None,
                "lane_matches": ln if opp_cid > 0 else 0,
                "opp_laner": opp_laner if (opp_cid > 0 and ln > 0) else None,
                "gold_10": round(l_g10, 1),
            })

        candidates.sort(key=lambda x: (x["score"], x["role_matches"]), reverse=True)
        return candidates[:top_k]


def build_spa_html(engine: FastDraftEngine, api_port: int) -> str:
    """Build the self-contained zero-refresh HTML/CSS/JS Champion Select Studio."""
    initial_draft = DEMO_PRESETS["Mid-Draft (7/10)"]
    initial_analysis = engine.analyze(
        initial_draft["blue"], initial_draft["red"], active_side="blue", active_role="sup"
    )

    boot_payload = {
        "apiPort": api_port,
        "totalMatches": engine.total_matches,
        "patches": engine.unique_patches,
        "roles": ROLES,
        "roleLabels": ROLE_LABELS,
        "roleIcons": ROLE_ICONS,
        "presets": DEMO_PRESETS,
        "champMeta": engine.champ_meta,
        "roleStats": engine.role_stats_map,
        "initialDraft": initial_draft,
        "initialAnalysis": initial_analysis,
    }
    boot_json = json.dumps(boot_payload)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoL Draft Studio</title>
<style>
  :root {{
    --bg-deep: #070b14;
    --bg-panel: #0e1628;
    --bg-card: #131e36;
    --bg-hover: #1b2a4a;
    --border: #24355a;
    --gold: #c8aa6e;
    --gold-bright: #f59e0b;
    --blue: #3b82f6;
    --blue-glow: rgba(59, 130, 246, 0.25);
    --red: #f43f5e;
    --red-glow: rgba(244, 63, 94, 0.25);
    --text: #f1f5f9;
    --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg-deep);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.4;
    overflow-x: hidden;
  }}

  /* Top Command Bar */
  .topbar {{
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
    padding: 10px 16px; background: linear-gradient(90deg, #0b1324 0%, #131f3a 50%, #1a1228 100%);
    border-bottom: 1px solid var(--border);
  }}
  .brand {{ display: flex; align-items: center; gap: 10px; }}
  .brand-title {{ font-size: 16px; font-weight: 800; letter-spacing: -0.02em; color: #fff; }}
  .brand-sub {{ font-size: 11px; color: var(--gold); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }}

  .toolbar {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
  .btn {{
    background: var(--bg-card); color: var(--text); border: 1px solid var(--border);
    padding: 5px 10px; border-radius: 7px; font-size: 11.5px; font-weight: 600;
    cursor: pointer; transition: all 0.12s ease; display: inline-flex; align-items: center; gap: 5px;
  }}
  .btn:hover {{ background: var(--bg-hover); border-color: var(--gold); }}
  .btn.active, .btn.gold {{ background: rgba(200, 170, 110, 0.18); border-color: var(--gold); color: #fde68a; }}
  .btn.danger:hover {{ border-color: var(--red); color: #fda4af; }}
  select.ctrl {{
    background: var(--bg-card); color: var(--text); border: 1px solid var(--border);
    padding: 5px 8px; border-radius: 7px; font-size: 11.5px; font-weight: 600; outline: none;
  }}

  /* Live Tug-of-War Win Bar Strip */
  .win-strip {{
    padding: 8px 16px; background: #0a1020; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px;
  }}
  .win-side {{ min-width: 195px; font-weight: 800; font-size: 13px; }}
  .win-side.blue {{ color: #60a5fa; }}
  .win-side.red {{ color: #fb7185; text-align: right; }}
  .win-side small {{ display: block; font-size: 10.5px; font-weight: 500; color: var(--muted); }}
  .tug-container {{ flex: 1; }}
  .tug-meta {{ display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 4px; }}
  .tug-track {{
    width: 100%; height: 10px; border-radius: 999px; background: var(--red);
    overflow: hidden; display: flex; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
  }}
  .tug-fill-blue {{
    height: 100%; background: linear-gradient(90deg, #2563eb, #60a5fa);
    transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }}

  /* Main 3-Column Champion Select Arena */
  .arena {{
    display: grid;
    grid-template-columns: 270px 1fr 270px;
    gap: 12px;
    padding: 12px 16px;
    align-items: start;
  }}
  @media (max-width: 1100px) {{
    .arena {{ grid-template-columns: 1fr; }}
  }}

  /* Side Team Columns */
  .team-col {{
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; padding: 10px;
  }}
  .team-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: 8px; margin-bottom: 8px; border-bottom: 1px solid var(--border);
    font-weight: 800; font-size: 12.5px; letter-spacing: 0.05em; text-transform: uppercase;
  }}
  .team-header.blue {{ color: #60a5fa; }}
  .team-header.red {{ color: #fb7185; }}

  .slot {{
    position: relative; overflow: hidden; border-radius: 10px; height: 68px;
    margin-bottom: 7px; border: 1.5px solid #1f2e4d; background: #0b1222;
    display: flex; align-items: center; padding: 0 10px; gap: 10px;
    cursor: pointer; transition: transform 0.1s ease, border-color 0.12s ease, box-shadow 0.12s ease;
    user-select: none;
  }}
  .slot:last-child {{ margin-bottom: 0; }}
  .slot:hover {{ border-color: #64748b; transform: translateY(-1px); }}
  .slot.blue {{ border-left: 4px solid var(--blue); }}
  .slot.red {{ border-right: 4px solid var(--red); flex-direction: row-reverse; text-align: right; }}
  .slot.active {{
    border: 2px solid var(--gold-bright) !important;
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.35);
  }}
  .slot-splash {{
    position: absolute; inset: 0; background-size: cover; background-position: center 20%;
    opacity: 0.26; z-index: 0; transition: opacity 0.2s;
  }}
  .slot:hover .slot-splash {{ opacity: 0.36; }}
  .slot-scrim {{
    position: absolute; inset: 0; z-index: 1;
    background: linear-gradient(90deg, rgba(11,18,34,0.92) 0%, rgba(11,18,34,0.55) 60%, rgba(11,18,34,0.88) 100%);
  }}
  .slot-icon {{
    position: relative; z-index: 2; width: 46px; height: 46px; border-radius: 8px;
    border: 1.5px solid var(--gold); object-fit: cover; flex-shrink: 0; background: #162038;
  }}
  .slot-icon.empty {{
    border: 1px dashed #475569; padding: 10px; opacity: 0.65;
  }}
  .slot-body {{ position: relative; z-index: 2; flex: 1; min-width: 0; }}
  .slot-role-row {{
    display: flex; align-items: center; gap: 4px; font-size: 10px; font-weight: 700;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }}
  .slot.red .slot-role-row {{ justify-content: flex-end; }}
  .slot-name {{
    font-size: 13.5px; font-weight: 800; color: #fff;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .slot-name.empty {{ color: #64748b; font-weight: 600; font-style: italic; }}
  .slot-sub {{
    font-size: 10.5px; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .arch-tag {{
    display: inline-block; padding: 1px 5px; border-radius: 4px; font-size: 9.5px; font-weight: 700;
    background: rgba(200, 170, 110, 0.2); border: 1px solid rgba(200, 170, 110, 0.45); color: #fde68a;
  }}
  .slot-clear {{
    position: relative; z-index: 3; width: 22px; height: 22px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.2); background: rgba(15,23,42,0.85); color: #cbd5e1;
    font-size: 12px; line-height: 20px; text-align: center; cursor: pointer; opacity: 0.75;
    flex-shrink: 0;
  }}
  .slot-clear:hover {{ background: var(--red); color: #fff; opacity: 1; border-color: var(--red); }}

  /* Center Workspace */
  .center-panel {{
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px;
  }}
  .active-banner {{
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
    padding: 8px 12px; border-radius: 9px; margin-bottom: 10px;
    background: linear-gradient(90deg, rgba(245,158,11,0.14) 0%, rgba(19,30,54,0.9) 100%);
    border: 1px solid rgba(245, 158, 11, 0.55);
  }}
  .active-title {{ font-weight: 800; font-size: 13px; color: #fef3c7; display: flex; align-items: center; gap: 6px; }}

  /* Live AI Advisor Strip inside Center */
  .advisor-strip {{
    display: grid; grid-template-columns: 1.5fr 1fr; gap: 10px; margin-bottom: 10px;
  }}
  @media (max-width: 900px) {{ .advisor-strip {{ grid-template-columns: 1fr; }} }}
  .adv-box {{
    background: #0b1222; border: 1px solid var(--border); border-radius: 9px; padding: 8px 10px;
  }}
  .adv-box-header {{
    font-size: 10.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--gold); margin-bottom: 6px; display: flex; justify-content: space-between;
  }}
  .adv-chips {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 6px; }}
  .adv-chip {{
    display: flex; align-items: center; gap: 7px; padding: 5px 7px; border-radius: 7px;
    background: var(--bg-card); border: 1px solid #26385e; cursor: pointer; transition: all 0.12s;
  }}
  .adv-chip:hover {{ border-color: var(--gold-bright); background: var(--bg-hover); transform: translateY(-1px); }}
  .adv-chip.ban:hover {{ border-color: var(--red); }}
  .adv-chip img {{ width: 32px; height: 32px; border-radius: 6px; border: 1px solid var(--gold); object-fit: cover; }}
  .adv-chip-info {{ min-width: 0; flex: 1; }}
  .adv-chip-name {{ font-weight: 800; font-size: 11.5px; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .adv-chip-sub {{ font-size: 10px; color: var(--muted); white-space: nowrap; }}

  /* Filter Bar */
  .filter-bar {{
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;
    padding: 6px 8px; background: #0b1222; border-radius: 8px; border: 1px solid var(--border);
  }}
  .role-pills {{ display: flex; gap: 4px; }}
  .role-btn {{
    background: var(--bg-card); border: 1px solid var(--border); color: var(--muted);
    padding: 4px 8px; border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 700;
    display: inline-flex; align-items: center; gap: 4px;
  }}
  .role-btn img {{ width: 14px; height: 14px; }}
  .role-btn.active {{ background: rgba(59,130,246,0.2); border-color: var(--blue); color: #fff; }}
  .search-input {{
    flex: 1; min-width: 140px; background: var(--bg-card); border: 1px solid var(--border);
    color: #fff; padding: 5px 10px; border-radius: 6px; font-size: 12px; outline: none;
  }}
  .search-input:focus {{ border-color: var(--gold); }}

  /* Champion Portrait Grid */
  .champ-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(74px, 1fr));
    gap: 6px;
    max-height: 270px;
    overflow-y: auto;
    padding: 4px 2px;
  }}
  .champ-cell {{
    background: #0b1222; border: 1px solid #1f2e4d; border-radius: 8px;
    padding: 5px 4px; text-align: center; cursor: pointer; transition: all 0.1s ease;
    position: relative; user-select: none;
  }}
  .champ-cell:hover {{ border-color: var(--gold-bright); background: var(--bg-hover); transform: translateY(-1px); }}
  .champ-cell.picked {{ opacity: 0.32; pointer-events: none; filter: grayscale(0.85); }}
  .champ-cell img {{
    width: 48px; height: 48px; border-radius: 7px; border: 1px solid rgba(200,170,110,0.6);
    object-fit: cover; display: block; margin: 0 auto 3px;
  }}
  .champ-cell-name {{
    font-size: 11px; font-weight: 700; color: #f8fafc;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .champ-cell-wr {{ font-size: 9.5px; color: #93c5fd; font-weight: 600; }}

  /* Bottom Analytics Tabs */
  .analytics-section {{ padding: 0 16px 20px; }}
  .tab-bar {{
    display: flex; gap: 6px; border-bottom: 1px solid var(--border); margin-bottom: 12px;
  }}
  .tab-btn {{
    background: transparent; border: none; border-bottom: 2.5px solid transparent;
    color: var(--muted); padding: 8px 14px; font-size: 12.5px; font-weight: 700; cursor: pointer;
  }}
  .tab-btn:hover {{ color: #fff; }}
  .tab-btn.active {{ color: var(--gold-bright); border-bottom-color: var(--gold-bright); }}

  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px; }}
  @media (max-width: 800px) {{ .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} }}
  .kpi-card {{
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
  }}
  .kpi-label {{ font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700; }}
  .kpi-val {{ font-size: 20px; font-weight: 800; color: #fff; margin-top: 2px; }}
  .kpi-sub {{ font-size: 11px; color: #93c5fd; margin-top: 2px; }}

  .grid-2 {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 12px; }}
  @media (max-width: 950px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px;
  }}
  .card-title {{ font-size: 13px; font-weight: 800; color: #fff; margin-bottom: 8px; display: flex; justify-content: space-between; }}

  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  table.data-table th {{
    text-align: left; padding: 6px 8px; color: var(--muted); border-bottom: 1px solid var(--border);
    font-size: 10.5px; text-transform: uppercase;
  }}
  table.data-table td {{ padding: 6px 8px; border-bottom: 1px solid rgba(36,53,90,0.45); vertical-align: middle; }}
  .td-champ {{ display: flex; align-items: center; gap: 7px; font-weight: 700; }}
  .td-champ img {{ width: 26px; height: 26px; border-radius: 5px; object-fit: cover; border: 1px solid var(--gold); }}
</style>
</head>
<body>

<!-- Top Command Bar -->
<div class="topbar">
  <div class="brand">
    <div>
      <div class="brand-sub">Riot Data Dragon · <span id="hdr-total">751,339</span> Matches · Wide & Deep Neural Model</div>
      <div class="brand-title">⚔️ LoL Champion Select Studio & Draft Pathways</div>
    </div>
  </div>
  <div class="toolbar">
    <span style="color:var(--muted); font-size:11px; font-weight:700;">PRESETS:</span>
    <button class="btn" onclick="loadPreset('Dive vs. Poke')">⚡ Dive vs. Poke</button>
    <button class="btn" onclick="loadPreset('Scaling vs. Split')">🛡️ Scaling vs. Split</button>
    <button class="btn" onclick="loadPreset('Teamfight vs. Pick')">🏆 Teamfight vs. Pick</button>
    <button class="btn" onclick="loadPreset('Mid-Draft (7/10)')">🧩 Mid-Draft (7/10)</button>
    <button class="btn gold" onclick="randomizeDraft()">🎲 Random</button>
    <button class="btn" onclick="swapSides()">⇄ Swap</button>
    <button class="btn danger" onclick="clearDraft()">🗑️ Clear</button>
    <select id="sel-mode" class="ctrl" onchange="triggerAnalyze()">
      <option value="Auto (recommended)">Mode: Auto (Recommended)</option>
      <option value="Role-aware (either side)">Mode: Role-Aware</option>
      <option value="Exact roles & sides">Mode: Exact Roles & Sides</option>
      <option value="Champion presence">Mode: Champion Presence</option>
    </select>
    <select id="sel-patch" class="ctrl" onchange="triggerAnalyze()">
      <option value="All patches">All Patches (49)</option>
    </select>
  </div>
</div>

<!-- Live Win Probability Tug-of-War Strip -->
<div class="win-strip">
  <div class="win-side blue">
    <span id="win-blue-pct">50.0%</span> 🟦 BLUE WIN
    <small id="win-blue-sub">Neural: — · Empirical: —</small>
  </div>
  <div class="tug-container">
    <div class="tug-meta">
      <span id="tug-mode-label">Loading evidence...</span>
      <span id="tug-sample-label">751,339 matches</span>
    </div>
    <div class="tug-track">
      <div id="tug-bar-blue" class="tug-fill-blue" style="width: 50%;"></div>
    </div>
  </div>
  <div class="win-side red">
    RED WIN 🟥 <span id="win-red-pct">50.0%</span>
    <small id="win-red-sub">95% CI: —</small>
  </div>
</div>

<!-- Main 3-Column Champion Select Arena -->
<div class="arena">
  <!-- Blue Team Slots -->
  <div class="team-col">
    <div class="team-header blue">
      <span>🟦 Blue Team</span>
      <span id="blue-count-badge">0/5 Locked</span>
    </div>
    <div id="blue-slots"></div>
  </div>

  <!-- Center Interactive Picker & AI Advisor -->
  <div class="center-panel">
    <div class="active-banner">
      <div class="active-title" id="active-slot-banner">🎯 Active Slot: BLUE · SUPPORT</div>
      <div style="font-size:11px; color:var(--muted);">Click any slot card on the left/right or click a champion below to lock in</div>
    </div>

    <!-- Top Recommendations & Priority Bans Strip -->
    <div class="advisor-strip">
      <div class="adv-box">
        <div class="adv-box-header">
          <span id="rec-strip-title">✨ Top AI Picks for Active Slot</span>
          <span style="color:var(--muted); font-weight:500;">1-Click Lock-In</span>
        </div>
        <div id="rec-chips" class="adv-chips"></div>
      </div>
      <div class="adv-box">
        <div class="adv-box-header">
          <span>🚫 Priority Enemy Bans</span>
          <span style="color:var(--muted); font-weight:500;">Highest Threat</span>
        </div>
        <div id="ban-chips" class="adv-chips"></div>
      </div>
    </div>

    <!-- Filter & Search Controls -->
    <div class="filter-bar">
      <div class="role-pills" id="role-filter-pills"></div>
      <input type="text" id="champ-search" class="search-input" placeholder="Search 172 champions by name or archetype (e.g. Ahri, Ornn, Engage)..." oninput="renderChampGrid()">
      <select id="arch-filter" class="ctrl" onchange="renderChampGrid()">
        <option value="all">All Archetypes</option>
        <option value="Engage">Engage</option>
        <option value="Poke">Poke</option>
        <option value="Scaling">Scaling</option>
        <option value="Split push">Split Push</option>
      </select>
      <select id="sort-filter" class="ctrl" onchange="renderChampGrid()">
        <option value="matches">Sort: Popularity</option>
        <option value="wr">Sort: Role Win Rate</option>
        <option value="az">Sort: A → Z</option>
      </select>
    </div>

    <!-- Scrollable Data Dragon Portrait Grid -->
    <div id="champ-grid" class="champ-grid"></div>
  </div>

  <!-- Red Team Slots -->
  <div class="team-col">
    <div class="team-header red">
      <span id="red-count-badge">0/5 Locked</span>
      <span>Red Team 🟥</span>
    </div>
    <div id="red-slots"></div>
  </div>
</div>

<!-- Bottom Analytics Section -->
<div class="analytics-section">
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('pathways', this)">📈 Gold Curve & Victory Pathways</button>
    <button class="tab-btn" onclick="switchTab('objectives', this)">🐉 Objective Control & Neural Forecast</button>
    <button class="tab-btn" onclick="switchTab('matchups', this)">⚔️ Head-to-Head Lane Matchups & Full Advisor Table</button>
  </div>

  <!-- Tab 1: Gold Curve & Victory Pathways -->
  <div id="tab-pathways" class="tab-pane">
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Observed Blue Win Rate</div>
        <div class="kpi-val" id="kpi-wr">—</div>
        <div class="kpi-sub" id="kpi-wr-ci">—</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">10m Lead Conversion</div>
        <div class="kpi-val" id="kpi-conv">—</div>
        <div class="kpi-sub">Blue win % when ahead @10m</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Comeback Rate (@10m Behind)</div>
        <div class="kpi-val" id="kpi-comeback">—</div>
        <div class="kpi-sub">Blue win % when behind @10m</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Neural Predicted 15m Gold</div>
        <div class="kpi-val" id="kpi-g15">—</div>
        <div class="kpi-sub" id="kpi-g15-hist">Historical mean: —</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">
          <span>Gold Differential Trajectory (5m → 30m · Blue Minus Red)</span>
          <span style="font-size:11px; color:var(--muted);">🔵 Historical Mean &nbsp; 🟡 Neural Model</span>
        </div>
        <div id="gold-chart-box" style="width:100%; height:210px;"></div>
      </div>
      <div class="card">
        <div class="card-title">
          <span>Composition Archetype Identity & Evidence Ladder</span>
        </div>
        <div id="arch-ladder-box"></div>
      </div>
    </div>
  </div>

  <!-- Tab 2: Objective Control & Neural Breakdown -->
  <div id="tab-objectives" class="tab-pane" style="display:none;">
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span>Historical First Objective Control & Conditional Win Rate</span></div>
        <div id="obj-hist-table"></div>
      </div>
      <div class="card">
        <div class="card-title"><span>Wide & Deep Neural Encoder Multi-Task Forecast</span></div>
        <div id="obj-neural-table"></div>
      </div>
    </div>
  </div>

  <!-- Tab 3: Lane Matchups & Full Candidate Table -->
  <div id="tab-matchups" class="tab-pane" style="display:none;">
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span>Direct Head-to-Head Lane Matchups</span></div>
        <div id="lane-matchup-table"></div>
      </div>
      <div class="card">
        <div class="card-title"><span id="full-rec-table-title">Top Role Candidates Breakdown</span></div>
        <div id="full-rec-table"></div>
      </div>
    </div>
  </div>
</div>

<script>
const BOOT = {boot_json};

let draftState = {{
  blue: Object.assign({{top:"", jng:"", mid:"", bot:"", sup:""}}, BOOT.initialDraft.blue),
  red: Object.assign({{top:"", jng:"", mid:"", bot:"", sup:""}}, BOOT.initialDraft.red),
}};
let activeSlot = {{ side: "blue", role: "sup" }};
let roleFilter = "sup"; // "all" or role key
let currentAnalysis = BOOT.initialAnalysis;

const FALLBACK_ICON = "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/-1.png";

function champImg(name) {{
  return (name && BOOT.champMeta[name]) ? BOOT.champMeta[name].image : FALLBACK_ICON;
}}

function champSplash(name) {{
  return (name && BOOT.champMeta[name]) ? BOOT.champMeta[name].splash : "";
}}

function initStudio() {{
  document.getElementById("hdr-total").textContent = BOOT.totalMatches.toLocaleString();
  const patchSel = document.getElementById("sel-patch");
  BOOT.patches.forEach(p => {{
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = "Patch " + p;
    patchSel.appendChild(opt);
  }});
  renderRoleFilterPills();
  renderAll();
}}

function selectSlot(side, role) {{
  activeSlot = {{ side, role }};
  roleFilter = role; // Auto-sync grid role filter to clicked slot!
  renderRoleFilterPills();
  renderSlots();
  renderBannerAndGrid();
  triggerAnalyze();
}}

function clearSlot(ev, side, role) {{
  ev.stopPropagation();
  draftState[side][role] = "";
  activeSlot = {{ side, role }};
  roleFilter = role;
  renderRoleFilterPills();
  renderSlots();
  renderBannerAndGrid();
  triggerAnalyze();
}}

function pickChampion(champName) {{
  const {{ side, role }} = activeSlot;
  draftState[side][role] = champName;

  // Auto-advance cursor to next open slot
  const order = [];
  BOOT.roles.forEach(r => {{ order.push({{side:"blue", role:r}}); order.push({{side:"red", role:r}}); }});
  const curIdx = order.findIndex(x => x.side === side && x.role === role);
  for (let step = 1; step <= order.length; step++) {{
    const cand = order[(curIdx + step) % order.length];
    if (!draftState[cand.side][cand.role]) {{
      activeSlot = cand;
      roleFilter = cand.role;
      break;
    }}
  }}
  renderRoleFilterPills();
  renderSlots();
  renderBannerAndGrid();
  triggerAnalyze();
}}

function loadPreset(name) {{
  const p = BOOT.presets[name];
  if (!p) return;
  BOOT.roles.forEach(r => {{
    draftState.blue[r] = p.blue[r] || "";
    draftState.red[r] = p.red[r] || "";
  }});
  // Set activeSlot to first open slot or blue top
  let found = null;
  BOOT.roles.forEach(r => {{
    if (!found && !draftState.blue[r]) found = {{side:"blue", role:r}};
    if (!found && !draftState.red[r]) found = {{side:"red", role:r}};
  }});
  activeSlot = found || {{side:"blue", role:"top"}};
  roleFilter = activeSlot.role;
  renderRoleFilterPills();
  renderSlots();
  renderBannerAndGrid();
  triggerAnalyze();
}}

function randomizeDraft() {{
  const used = new Set();
  BOOT.roles.forEach(r => {{
    const pool = (BOOT.roleStats[r] || []).slice(0, 22).map(x => x.champion).filter(c => !used.has(c));
    if (pool.length >= 2) {{
      const i1 = Math.floor(Math.random() * pool.length);
      const c1 = pool.splice(i1, 1)[0];
      const i2 = Math.floor(Math.random() * pool.length);
      const c2 = pool.splice(i2, 1)[0];
      draftState.blue[r] = c1;
      draftState.red[r] = c2;
      used.add(c1); used.add(c2);
    }}
  }});
  renderSlots();
  renderBannerAndGrid();
  triggerAnalyze();
}}

function swapSides() {{
  const bCopy = Object.assign({{}}, draftState.blue);
  draftState.blue = Object.assign({{}}, draftState.red);
  draftState.red = bCopy;
  renderSlots();
  renderBannerAndGrid();
  triggerAnalyze();
}}

function clearDraft() {{
  BOOT.roles.forEach(r => {{ draftState.blue[r] = ""; draftState.red[r] = ""; }});
  activeSlot = {{side:"blue", role:"top"}};
  roleFilter = "top";
  renderRoleFilterPills();
  renderSlots();
  renderBannerAndGrid();
  triggerAnalyze();
}}

function setRoleFilter(r) {{
  roleFilter = r;
  renderRoleFilterPills();
  renderChampGrid();
}}

function renderRoleFilterPills() {{
  const container = document.getElementById("role-filter-pills");
  let html = `<button class="role-btn ${{roleFilter === 'all' ? 'active' : ''}}" onclick="setRoleFilter('all')">All (172)</button>`;
  BOOT.roles.forEach(r => {{
    const act = roleFilter === r ? "active" : "";
    html += `<button class="role-btn ${{act}}" onclick="setRoleFilter('${{r}}')">
      <img src="${{BOOT.roleIcons[r]}}" alt=""> ${{BOOT.roleLabels[r]}}
    </button>`;
  }});
  container.innerHTML = html;
}}

function renderSlots() {{
  ["blue", "red"].forEach(side => {{
    const container = document.getElementById(side + "-slots");
    let count = 0;
    let html = "";
    BOOT.roles.forEach(role => {{
      const champ = draftState[side][role];
      if (champ) count++;
      const isAct = (activeSlot.side === side && activeSlot.role === role) ? "active" : "";
      const rIcon = BOOT.roleIcons[role];
      const rLabel = BOOT.roleLabels[role];

      if (champ && BOOT.champMeta[champ]) {{
        const meta = BOOT.champMeta[champ];
        const rStat = (BOOT.roleStats[role] || []).find(x => x.champion === champ);
        const wrTxt = rStat ? `${{(rStat.win_rate * 100).toFixed(1)}}% WR` : meta.title;
        html += `
          <div class="slot ${{side}} ${{isAct}}" onclick="selectSlot('${{side}}', '${{role}}')">
            <div class="slot-splash" style="background-image:url('${{meta.splash}}')"></div>
            <div class="slot-scrim"></div>
            <img class="slot-icon" src="${{meta.image}}" alt="${{champ}}">
            <div class="slot-body">
              <div class="slot-role-row"><img src="${{rIcon}}" width="12" height="12"> ${{rLabel}}</div>
              <div class="slot-name">${{champ}}</div>
              <div class="slot-sub">${{wrTxt}} · <span class="arch-tag">${{meta.archetypes}}</span></div>
            </div>
            <div class="slot-clear" title="Clear slot" onclick="clearSlot(event, '${{side}}', '${{role}}')">×</div>
          </div>`;
      }} else {{
        html += `
          <div class="slot ${{side}} ${{isAct}}" onclick="selectSlot('${{side}}', '${{role}}')">
            <div class="slot-scrim"></div>
            <img class="slot-icon empty" src="${{rIcon}}" alt="">
            <div class="slot-body">
              <div class="slot-role-row">${{rLabel}}</div>
              <div class="slot-name empty">Open Slot</div>
              <div class="slot-sub">Click to target ${{rLabel}}</div>
            </div>
          </div>`;
      }}
    }});
    container.innerHTML = html;
    document.getElementById(side + "-count-badge").textContent = `${{count}}/5 Locked`;
  }});
}}

function renderBannerAndGrid() {{
  const sBadge = activeSlot.side === "blue" ? "🟦 BLUE" : "🟥 RED";
  const cur = draftState[activeSlot.side][activeSlot.role] || "Open Slot";
  document.getElementById("active-slot-banner").innerHTML =
    `🎯 Active Slot: <span>${{sBadge}} · ${{BOOT.roleLabels[activeSlot.role].toUpperCase()}}</span> <span style="color:var(--gold);">(${{cur}})</span>`;
  document.getElementById("rec-strip-title").textContent =
    `✨ Top AI Picks · ${{activeSlot.side.toUpperCase()}} ${{BOOT.roleLabels[activeSlot.role]}}`;
  renderChampGrid();
}}

function renderChampGrid() {{
  const query = (document.getElementById("champ-search").value || "").trim().toLowerCase();
  const arch = document.getElementById("arch-filter").value;
  const sortMode = document.getElementById("sort-filter").value;

  const pickedSet = new Set([
    ...Object.values(draftState.blue).filter(Boolean),
    ...Object.values(draftState.red).filter(Boolean),
  ]);
  const curSlotChamp = draftState[activeSlot.side][activeSlot.role];

  const targetRole = roleFilter === "all" ? activeSlot.role : roleFilter;
  const roleMap = {{}};
  (BOOT.roleStats[targetRole] || []).forEach(entry => {{
    roleMap[entry.champion] = entry;
  }});

  let pool = Object.keys(BOOT.champMeta);
  if (roleFilter !== "all" && !query) {{
    pool = pool.filter(c => roleMap[c] && roleMap[c].matches >= 15);
  }}
  if (query) {{
    pool = pool.filter(c => {{
      const m = BOOT.champMeta[c];
      return c.toLowerCase().includes(query) || (m.archetypes || "").toLowerCase().includes(query);
    }});
  }}
  if (arch !== "all") {{
    pool = pool.filter(c => (BOOT.champMeta[c].arch_list || []).includes(arch));
  }}

  pool.sort((a, b) => {{
    if (sortMode === "az") return a.localeCompare(b);
    const ra = roleMap[a] || {{ matches: 0, win_rate: 0 }};
    const rb = roleMap[b] || {{ matches: 0, win_rate: 0 }};
    if (sortMode === "wr") return (rb.win_rate - ra.win_rate) || (rb.matches - ra.matches);
    return (rb.matches - ra.matches) || (rb.win_rate - ra.win_rate);
  }});

  const grid = document.getElementById("champ-grid");
  let html = "";
  pool.forEach(c => {{
    const meta = BOOT.champMeta[c];
    const rStat = roleMap[c];
    const subTxt = rStat ? `${{(rStat.win_rate * 100).toFixed(1)}}%` : (meta.arch_list[0] || "Flex");
    const isPicked = pickedSet.has(c) && c !== curSlotChamp;
    const safeName = c.replace(/'/g, "\\'");
    html += `
      <div class="champ-cell ${{isPicked ? 'picked' : ''}}" onclick="pickChampion('${{safeName}}')" title="${{c}} (${{meta.archetypes}})">
        <img src="${{meta.image}}" loading="lazy" alt="${{c}}">
        <div class="champ-cell-name">${{c}}</div>
        <div class="champ-cell-wr">${{subTxt}}</div>
      </div>`;
  }});
  grid.innerHTML = html || `<div style="grid-column:1/-1; color:var(--muted); padding:16px; text-align:center;">No champions match filter. Click 'All (172)' to view full roster.</div>`;
}}

function renderAnalysis(data) {{
  currentAnalysis = data;
  const hasNeural = Boolean(data.neural);
  const bWr = hasNeural ? data.neural.blue_win_prob : data.observed_blue_wr;
  const rWr = 1.0 - bWr;

  document.getElementById("win-blue-pct").textContent = (bWr * 100).toFixed(1) + "%";
  document.getElementById("win-red-pct").textContent = (rWr * 100).toFixed(1) + "%";
  document.getElementById("tug-bar-blue").style.width = (bWr * 100).toFixed(1) + "%";
  document.getElementById("win-blue-sub").textContent =
    `Neural: ${{hasNeural ? (data.neural.blue_win_prob*100).toFixed(1)+'%' : '—'}} · Empirical: ${{(data.observed_blue_wr*100).toFixed(1)}}%`;
  document.getElementById("win-red-sub").textContent =
    `Empirical 95% CI: ${{(data.ci_low*100).toFixed(1)}}%–${{(data.ci_high*100).toFixed(1)}}%`;
  document.getElementById("tug-mode-label").textContent = data.resolved_mode;
  document.getElementById("tug-sample-label").textContent = `${{data.similar_matches.toLocaleString()}} comparable matches`;

  // Top AI Pick Chips
  const recBox = document.getElementById("rec-chips");
  const topRecs = (data.recommendations || []).slice(0, 4);
  recBox.innerHTML = topRecs.map(r => {{
    const safe = r.champion.replace(/'/g, "\\'");
    const laneTxt = r.lane_wr !== null ? `Lane ${{r.lane_wr}}%` : `WR ${{r.role_wr}}%`;
    return `<div class="adv-chip" onclick="pickChampion('${{safe}}')" title="Click to lock in ${{r.champion}}">
      <img src="${{champImg(r.champion)}}" alt="">
      <div class="adv-chip-info">
        <div class="adv-chip-name">${{r.champion}}</div>
        <div class="adv-chip-sub">Score ${{r.score}} · ${{laneTxt}}</div>
      </div>
    </div>`;
  }}).join("") || `<div style="color:var(--muted); font-size:11px;">Select picks to compute recommendations.</div>`;

  // Top Ban Chips
  const banBox = document.getElementById("ban-chips");
  const topBans = (data.bans || []).slice(0, 3);
  banBox.innerHTML = topBans.map(b => {{
    return `<div class="adv-chip ban" title="High threat in enemy ${{b.role_label}}">
      <img src="${{champImg(b.champion)}}" alt="">
      <div class="adv-chip-info">
        <div class="adv-chip-name">${{b.champion}}</div>
        <div class="adv-chip-sub">${{b.role_label}} · Threat ${{b.score}}</div>
      </div>
    </div>`;
  }}).join("");

  // KPIs
  const delta = ((data.observed_blue_wr - data.baseline_wr) * 100).toFixed(1);
  document.getElementById("kpi-wr").textContent = (data.observed_blue_wr * 100).toFixed(1) + "%";
  document.getElementById("kpi-wr-ci").textContent = `${{delta >= 0 ? '+' : ''}}${{delta}} pts vs baseline (${{(data.ci_low*100).toFixed(1)}}%–${{(data.ci_high*100).toFixed(1)}}%)`;
  document.getElementById("kpi-conv").textContent = (data.lead_conversion_10 * 100).toFixed(1) + "%";
  document.getElementById("kpi-comeback").textContent = (data.comeback_rate_10 * 100).toFixed(1) + "%";

  const g15Hist = data.gold_curve_hist[2] || 0;
  const g15Neural = hasNeural ? data.neural.gold_curve[2] : g15Hist;
  document.getElementById("kpi-g15").textContent = `${{g15Neural >= 0 ? '+' : ''}}${{Math.round(g15Neural)}}g`;
  document.getElementById("kpi-g15-hist").textContent = `Historical mean @15m: ${{g15Hist >= 0 ? '+' : ''}}${{Math.round(g15Hist)}}g`;

  renderGoldSvg(data.gold_curve_hist, hasNeural ? data.neural.gold_curve : null);
  renderArchAndLadder(data);
  renderObjectivesTables(data);
  renderMatchupsAndAdvisorTables(data);
}}

function renderGoldSvg(histCurve, neuralCurve) {{
  const box = document.getElementById("gold-chart-box");
  const W = 560, H = 200, padL = 48, padR = 20, padT = 18, padB = 28;
  const allVals = [...histCurve, ...(neuralCurve || []), -500, 500];
  const minV = Math.min(...allVals) * 1.15;
  const maxV = Math.max(...allVals) * 1.15;
  const range = Math.max(100, maxV - minV);

  const xPt = i => padL + (i / 5) * (W - padL - padR);
  const yPt = v => padT + (1 - (v - minV) / range) * (H - padT - padB);
  const yZero = yPt(0);

  const makePath = arr => arr.map((v, i) => `${{i===0?'M':'L'}}${{xPt(i).toFixed(1)}},${{yPt(v).toFixed(1)}}`).join(" ");
  const mins = [5, 10, 15, 20, 25, 30];

  let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%; height:100%; overflow:visible;">
    <line x1="${{padL}}" y1="${{yZero}}" x2="${{W-padR}}" y2="${{yZero}}" stroke="#475569" stroke-dasharray="4,4" stroke-width="1"/>
    <text x="${{padL-6}}" y="${{yZero+4}}" fill="#94a3b8" font-size="10" text-anchor="end">0g</text>
    <text x="${{padL-6}}" y="${{padT+6}}" fill="#60a5fa" font-size="10" text-anchor="end">+${{Math.round(maxV)}}g</text>
    <text x="${{padL-6}}" y="${{H-padB}}" fill="#fb7185" font-size="10" text-anchor="end">${{Math.round(minV)}}g</text>`;

  mins.forEach((m, i) => {{
    const x = xPt(i);
    svg += `<text x="${{x}}" y="${{H-6}}" fill="#94a3b8" font-size="10.5" text-anchor="middle">${{m}}m</text>`;
  }});

  svg += `<path d="${{makePath(histCurve)}}" fill="none" stroke="#3b82f6" stroke-width="2.5"/>`;
  histCurve.forEach((v, i) => {{
    svg += `<circle cx="${{xPt(i)}}" cy="${{yPt(v)}}" r="3.5" fill="#60a5fa"><title>Historical @${{mins[i]}}m: ${{v}}g</title></circle>`;
  }});

  if (neuralCurve) {{
    svg += `<path d="${{makePath(neuralCurve)}}" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="5,3"/>`;
    neuralCurve.forEach((v, i) => {{
      svg += `<circle cx="${{xPt(i)}}" cy="${{yPt(v)}}" r="3.5" fill="#fbbf24"><title>Neural @${{mins[i]}}m: ${{v}}g</title></circle>`;
    }});
  }}
  svg += `</svg>`;
  box.innerHTML = svg;
}}

function renderArchAndLadder(data) {{
  const box = document.getElementById("arch-ladder-box");
  const fmtArch = obj => Object.entries(obj || {{}}).map(([k, v]) => `<span class="arch-tag" style="margin-right:5px;">${{k}}: ${{v}}%</span>`).join("") || "Balanced";
  let html = `
    <div style="margin-bottom:10px;">
      <div style="font-size:11px; color:#60a5fa; font-weight:700; margin-bottom:3px;">🟦 Blue Archetype Mix</div>
      <div>${{fmtArch(data.arch_dist.blue)}}</div>
    </div>
    <div style="margin-bottom:12px;">
      <div style="font-size:11px; color:#fb7185; font-weight:700; margin-bottom:3px;">🟥 Red Archetype Mix</div>
      <div>${{fmtArch(data.arch_dist.red)}}</div>
    </div>
    <table class="data-table">
      <thead><tr><th>Evidence Breadth Tier</th><th>Exact Full-Selection Matches</th></tr></thead>
      <tbody>
        ${{(data.ladder || []).map(r => `<tr><td>${{r.mode}}</td><td><b>${{r.matches.toLocaleString()}}</b></td></tr>`).join("")}}
      </tbody>
    </table>`;
  box.innerHTML = html;
}}

function renderObjectivesTables(data) {{
  const histBox = document.getElementById("obj-hist-table");
  const objs = Object.values(data.objectives || {{}});
  histBox.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Objective</th><th>Blue First</th><th>Red First</th><th>Blue Win If First</th><th>Behind @10m Win If First</th></tr></thead>
      <tbody>
        ${{objs.map(o => `<tr>
          <td><b>${{o.label}}</b></td>
          <td style="color:#60a5fa; font-weight:700;">${{(o.blue_first*100).toFixed(1)}}%</td>
          <td style="color:#fb7185; font-weight:700;">${{(o.red_first*100).toFixed(1)}}%</td>
          <td><b>${{(o.blue_win_if_first*100).toFixed(1)}}%</b></td>
          <td>${{o.blue_win_behind_first !== null ? (o.blue_win_behind_first*100).toFixed(1)+'%' : '—'}}</td>
        </tr>`).join("")}}
      </tbody>
    </table>`;

  const nBox = document.getElementById("obj-neural-table");
  if (!data.neural) {{
    nBox.innerHTML = `<div style="color:var(--muted);">Checkpoint not loaded.</div>`;
    return;
  }}
  const nObjs = data.neural.obj_probs;
  nBox.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Objective</th><th>Neural Blue First %</th><th>Neural Red First %</th><th>Unclaimed %</th></tr></thead>
      <tbody>
        ${{["dragon", "herald", "baron", "tower"].map(k => {{
          const p = nObjs[k];
          return `<tr>
            <td><b>${{k.charAt(0).toUpperCase() + k.slice(1)}}</b></td>
            <td style="color:#60a5fa; font-weight:700;">${{(p.blue*100).toFixed(1)}}%</td>
            <td style="color:#fb7185; font-weight:700;">${{(p.red*100).toFixed(1)}}%</td>
            <td style="color:var(--muted);">${{(p.none*100).toFixed(1)}}%</td>
          </tr>`;
        }}).join("")}}
      </tbody>
    </table>`;
}}

function renderMatchupsAndAdvisorTables(data) {{
  const mBox = document.getElementById("lane-matchup-table");
  const matchups = data.lane_matchups || [];
  if (!matchups.length) {{
    mBox.innerHTML = `<div style="color:var(--muted); padding:10px;">Lock both Blue and Red champions in the same role to compare direct lane matchups.</div>`;
  }} else {{
    mBox.innerHTML = `
      <table class="data-table">
        <thead><tr><th>Role</th><th>Blue</th><th>Red</th><th>Matches</th><th>Blue Lane WR</th><th>Gold @10m</th></tr></thead>
        <tbody>
          ${{matchups.map(m => `<tr>
            <td><b>${{m.role_label}}</b></td>
            <td><div class="td-champ"><img src="${{champImg(m.blue_champ)}}">${{m.blue_champ}}</div></td>
            <td><div class="td-champ"><img src="${{champImg(m.red_champ)}}">${{m.red_champ}}</div></td>
            <td>${{m.matches.toLocaleString()}}</td>
            <td style="color:${{(m.blue_wr||0.5)>=0.5?'#60a5fa':'#fb7185'}}; font-weight:700;">${{m.blue_wr !== null ? (m.blue_wr*100).toFixed(1)+'%' : '—'}}</td>
            <td>${{m.gold_10 !== null ? (m.gold_10>=0?'+':'')+m.gold_10+'g' : '—'}}</td>
          </tr>`).join("")}}
        </tbody>
      </table>`;
  }}

  document.getElementById("full-rec-table-title").textContent =
    `Top Candidates · ${{activeSlot.side.toUpperCase()}} ${{BOOT.roleLabels[activeSlot.role]}} (Click row to lock in)`;
  const rBox = document.getElementById("full-rec-table");
  const recs = data.recommendations || [];
  rBox.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Champion</th><th>Archetype</th><th>Score</th><th>Role WR</th><th>Lane vs Opp</th><th>Gold @10m</th></tr></thead>
      <tbody>
        ${{recs.map(r => {{
          const safe = r.champion.replace(/'/g, "\\'");
          return `<tr style="cursor:pointer;" onclick="pickChampion('${{safe}}')">
            <td><div class="td-champ"><img src="${{champImg(r.champion)}}">${{r.champion}}</div></td>
            <td><span class="arch-tag">${{r.archetypes}}</span></td>
            <td><b style="color:var(--gold-bright);">${{r.score}}</b></td>
            <td>${{r.role_wr}}% <small style="color:var(--muted);">(${{r.role_matches.toLocaleString()}})</small></td>
            <td>${{r.lane_wr !== null ? r.lane_wr+'% ('+r.lane_matches+')' : '—'}}</td>
            <td>${{r.gold_10 >= 0 ? '+' : ''}}${{r.gold_10}}g</td>
          </tr>`;
        }}).join("")}}
      </tbody>
    </table>`;
}}

function switchTab(tabId, btn) {{
  document.querySelectorAll(".tab-pane").forEach(el => el.style.display = "none");
  document.getElementById("tab-" + tabId).style.display = "block";
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
  btn.classList.add("active");
}}

let analyzeController = null;
async function triggerAnalyze() {{
  const payload = {{
    blue: draftState.blue,
    red: draftState.red,
    active_side: activeSlot.side,
    active_role: activeSlot.role,
    mode: document.getElementById("sel-mode").value,
    patch: document.getElementById("sel-patch").value,
  }};
  try {{
    if (analyzeController) analyzeController.abort();
    analyzeController = new AbortController();
    const res = await fetch(`http://127.0.0.1:${{BOOT.apiPort}}/api/analyze`, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(payload),
      signal: analyzeController.signal,
    }});
    if (res.ok) {{
      const data = await res.json();
      renderAnalysis(data);
    }}
  }} catch (err) {{
    if (err.name !== "AbortError") console.warn("Local API call error:", err);
  }}
}}

function renderAll() {{
  renderSlots();
  renderBannerAndGrid();
  renderAnalysis(currentAnalysis);
}}

initStudio();
</script>
</body>
</html>"""


class _StudioAPIHandler(BaseHTTPRequestHandler):
    engine: FastDraftEngine = None
    api_port: int = 8765

    def log_message(self, format: str, *args: Any) -> None:
        return  # Keep console clean

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html_bytes = build_spa_html(self.engine, self.api_port).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._send_cors()
            self.end_headers()
            self.wfile.write(html_bytes)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/api/analyze":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            result = self.engine.analyze(
                blue=payload.get("blue", {}),
                red=payload.get("red", {}),
                active_side=payload.get("active_side", "blue"),
                active_role=payload.get("active_role", "sup"),
                mode=payload.get("mode", "Auto (recommended)"),
                patch=payload.get("patch", "All patches"),
            )
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_cors()
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()


def _find_free_port(preferred: int = 8765) -> int:
    for port in (preferred, 8766, 8767, 8768, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def start_background_studio_server() -> tuple[FastDraftEngine, int]:
    engine = FastDraftEngine()
    port = _find_free_port(8765)
    _StudioAPIHandler.engine = engine
    _StudioAPIHandler.api_port = port
    server = ThreadingHTTPServer(("127.0.0.1", port), _StudioAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return engine, port


def main() -> None:
    """Launch inside Streamlit (or standalone if executed directly with python)."""
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(
        page_title="LoL Draft Studio",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        #MainMenu, header, footer {visibility: hidden; height: 0;}
        .block-container {padding: 0 !important; max-width: 100% !important;}
        iframe {border: none !important; display: block; width: 100% !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def _get_singleton_studio():
        return start_background_studio_server()

    engine, port = _get_singleton_studio()
    spa_html = build_spa_html(engine, port)
    components.html(spa_html, height=1060, scrolling=True)


if __name__ == "__main__":
    # Support both `streamlit run app.py` and direct `python app.py`
    if "streamlit" in sys.modules or any("streamlit" in arg.lower() for arg in sys.argv):
        main()
    else:
        try:
            from streamlit.runtime.scriptrunner import get_script_run_ctx
            if get_script_run_ctx() is not None:
                main()
                sys.exit(0)
        except Exception:
            pass
        print("Loading 751,339-match vectorized engine & neural model...")
        eng, p = start_background_studio_server()
        print(f"LoL Draft Studio running at http://127.0.0.1:{p} (Press Ctrl+C to stop)")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass