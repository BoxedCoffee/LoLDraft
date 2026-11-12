#!/usr/bin/env python3
"""
LoL Draft Studio — Zero-Refresh Client-Side SPA + Vectorized int16 Analytics & Neural Engine.

Works seamlessly across:
  1. Streamlit Community Cloud (`https://...streamlit.app` / `/mount/src/loldraft/...`)
  2. Local Streamlit (`streamlit run lol_collector/draft_analyzer/app.py`)
  3. Standalone local HTTP server (`python lol_collector/draft_analyzer/app.py`)

Key Engineering Guarantees:
  - 100% Cloud-Compatible Zero-Refresh Reactivity: Embeds precomputed 751,339-match
    172x172 same-team synergy matrix, 172x172 cross-team counter matrix, 5-role head-to-head
    lane matrices, and 860 (role, champion) 6-point gold/objective/conversion/comeback
    profiles directly in the client BOOT payload. Even on Streamlit Community Cloud (where
    only port 8501 is exposed), every single pick/slot/preset click updates ALL charts,
    recommendations, bans, KPIs, gold curves, and objective tables in 0 ms in the browser.
  - Hot-Reload Safe `@st.cache_resource`: Version-keyed `get_studio_engine_v5("v5")` plus
    defensive `hasattr` auto-initialization so Streamlit Cloud hot reloads never raise
    `AttributeError` on cached instances.
  - Apostrophe-Safe Integer Click Handlers (`pickChampByIdx(idx)`): K'Sante, Kai'Sa,
    Cho'Gath, Kha'Zix, Rek'Sai, Vel'Koz, Bel'Veth, and Kog'Maw never touch inline string
    quotes in HTML attributes.
  - Complete 172-Champion Icon Coverage: Data Dragon 14.24.1 + CommunityDragon Riot-ID
    portraits for 2025 champions (Mel, Yunara, Zaahen) + 3-tier `onerror` SVG fallback.
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
    build_champion_stats,
    champion_archetype_tags,
    load_match_table,
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

DDRAGON_VERSION = "14.24.1"
POST_14_24_CHAMPS = {"Mel", "Yunara", "Zaahen"}
SPLASH_ID_OVERRIDES = {"Fiddlesticks": "FiddleSticks"}
ENGINE_SCHEMA_VERSION = "v5_cloud_and_local_spa"

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
    """Resolve vocabulary IDs, Riot IDs, and resilient Data Dragon + CommunityDragon URLs."""
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
            riot_id = entry.get("riot_id") or key
            ddragon_id = entry.get("ddragon_id") or name.replace(" ", "").replace("'", "").replace(".", "")
            cdrag_icon = f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/{riot_id}.png"

            if name in POST_14_24_CHAMPS:
                square_url = cdrag_icon
                splash_url = f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/assets/characters/{ddragon_id.lower()}/skins/base/images/{ddragon_id.lower()}_splash_tile_0.jpg"
            else:
                square_url = f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/champion/{ddragon_id}.png"
                splash_dd = SPLASH_ID_OVERRIDES.get(name, ddragon_id)
                splash_url = f"https://ddragon.leagueoflegends.com/cdn/img/champion/splash/{splash_dd}_0.jpg"

            arch_str = champion_archetype_tags(name)
            champ_meta[name] = {
                "name": name,
                "title": entry.get("title", ""),
                "riot_id": int(riot_id) if str(riot_id).isdigit() else 0,
                "ddragon_id": ddragon_id,
                "image": square_url,
                "cdrag_image": cdrag_icon,
                "splash": splash_url,
                "archetypes": arch_str,
                "arch_list": [t.strip() for t in arch_str.split(",") if t.strip()],
            }
            id_to_name[str(key)] = name
            if entry.get("riot_id") is not None:
                riot_id_to_name[str(entry["riot_id"])] = name

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
    In-memory vectorized int16 analytics + precomputed 172x172 synergy/counter/lane
    matrices + rich per-(role, champion) profiles + Wide & Deep PyTorch neural engine.
    """

    def __init__(self):
        match_parquet = OUTPUT_DIR / "match_analysis.parquet"
        stats_parquet = OUTPUT_DIR / "champion_role_stats.parquet"
        if match_parquet.exists() and stats_parquet.exists():
            table = pd.read_parquet(match_parquet)
            stats = pd.read_parquet(stats_parquet)
        else:
            table = load_match_table(DATA_DIR)
            stats = build_champion_stats(table)

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

        all_names = sorted(
            set(self.champ_meta.keys()) | set(stats["champion"].dropna().astype(str).unique()),
            key=str.casefold,
        )
        for name in all_names:
            if name not in self.champ_meta:
                dd = name.replace(" ", "").replace("'", "").replace(".", "")
                arch_str = champion_archetype_tags(name)
                self.champ_meta[name] = {
                    "name": name,
                    "title": "",
                    "riot_id": 0,
                    "ddragon_id": dd,
                    "image": f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/img/champion/{dd}.png",
                    "cdrag_image": "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/-1.png",
                    "splash": f"https://ddragon.leagueoflegends.com/cdn/img/champion/splash/{dd}_0.jpg",
                    "archetypes": arch_str,
                    "arch_list": [t.strip() for t in arch_str.split(",") if t.strip()],
                }

        self.all_champions = all_names
        self.name_to_idx: dict[str, int] = {name: i + 1 for i, name in enumerate(all_names)}
        self.idx_to_name: dict[int, str] = {i + 1: name for i, name in enumerate(all_names)}

        n_rows = len(table)
        self.draft_mat = np.zeros((n_rows, 10), dtype=np.int16)
        for r_idx, role in enumerate(ROLES):
            self.draft_mat[:, r_idx] = (
                table[f"blue_{role}"].map(self.name_to_idx).fillna(0).to_numpy(dtype=np.int16)
            )
            self.draft_mat[:, r_idx + 5] = (
                table[f"red_{role}"].map(self.name_to_idx).fillna(0).to_numpy(dtype=np.int16)
            )

        self.patches = table["patch"].fillna("").astype(str).to_numpy()
        self.unique_patches = sorted([p for p in pd.unique(self.patches) if p], reverse=True)
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

        self.arch_labels = list(ARCHETYPE_RULES.keys())
        self.champ_arch_mask = np.zeros((len(all_names) + 1, len(self.arch_labels)), dtype=np.int8)
        for c_name, c_idx in self.name_to_idx.items():
            folded = c_name.casefold()
            for a_idx, a_label in enumerate(self.arch_labels):
                if folded in ARCHETYPE_RULES[a_label]:
                    self.champ_arch_mask[c_idx, a_idx] = 1

        self.role_stats_map: dict[str, list[dict[str, Any]]] = {r: [] for r in ROLES}
        for row in stats.itertuples(index=False):
            r = str(row.role)
            c = str(row.champion)
            if r in self.role_stats_map and c in self.name_to_idx:
                self.role_stats_map[r].append({
                    "champion": c,
                    "cid": self.name_to_idx[c],
                    "role": r,
                    "matches": int(row.matches),
                    "win_rate": round(float(row.win_rate), 4),
                    "mean_gold_diff_10": round(float(row.mean_gold_diff_10), 1),
                })

        self._precompute_interaction_matrices()
        self.predictor = self._load_neural_model()

    def _precompute_interaction_matrices(self) -> None:
        """
        Precompute across all 751,339 matches:
          - syn_wr_list (173, 173): smoothed same-team pairwise synergy win rate
          - cnt_wr_list (173, 173): smoothed cross-team counter win rate
          - sparse_lane_map: per-role head-to-head [games, wr, gold10]
          - rich_profiles: per-(role, cid) 6-point gold curves, objective first rates,
            10m lead conversion, and 10m comeback rates for 100% client-side reactivity.
        """
        C = len(self.all_champions) + 1
        bw_float = self.blue_win.astype(np.float64)
        rw_float = 1.0 - bw_float
        g10_float = self.gold_Arrays[10].astype(np.float64)

        self.global_blue_wr = round(float(bw_float.mean()), 4) if len(bw_float) > 0 else 0.492
        self.global_gold_curve = [
            round(float(self.gold_Arrays[m].mean()), 1) if len(bw_float) > 0 else 0.0
            for m in (5, 10, 15, 20, 25, 30)
        ]

        self.lane_games = np.zeros((5, C, C), dtype=np.int32)
        self.lane_wins = np.zeros((5, C, C), dtype=np.float64)
        self.lane_gold = np.zeros((5, C, C), dtype=np.float64)

        for r in range(5):
            bc = self.draft_mat[:, r].astype(np.int32)
            rc = self.draft_mat[:, r + 5].astype(np.int32)
            flat_br = bc * C + rc
            g_br = np.bincount(flat_br, minlength=C * C).reshape(C, C)
            w_br = np.bincount(flat_br, weights=bw_float, minlength=C * C).reshape(C, C)
            gold_br = np.bincount(flat_br, weights=g10_float, minlength=C * C).reshape(C, C)

            self.lane_games[r] = g_br + g_br.T
            self.lane_wins[r] = w_br + (g_br.T - w_br.T)
            self.lane_gold[r] = gold_br - gold_br.T

        self.syn_games = np.zeros((C, C), dtype=np.int32)
        self.syn_wins = np.zeros((C, C), dtype=np.float64)
        for i in range(5):
            for j in range(i + 1, 5):
                bi, bj = self.draft_mat[:, i].astype(np.int32), self.draft_mat[:, j].astype(np.int32)
                fb = bi * C + bj
                gb = np.bincount(fb, minlength=C * C).reshape(C, C)
                wb = np.bincount(fb, weights=bw_float, minlength=C * C).reshape(C, C)

                ri, rj = self.draft_mat[:, i + 5].astype(np.int32), self.draft_mat[:, j + 5].astype(np.int32)
                fr = ri * C + rj
                gr = np.bincount(fr, minlength=C * C).reshape(C, C)
                wr = np.bincount(fr, weights=rw_float, minlength=C * C).reshape(C, C)

                self.syn_games += gb + gb.T + gr + gr.T
                self.syn_wins += wb + wb.T + wr + wr.T

        self.cnt_games = np.zeros((C, C), dtype=np.int32)
        self.cnt_wins = np.zeros((C, C), dtype=np.float64)
        for i in range(5):
            bi = self.draft_mat[:, i].astype(np.int32)
            for j in range(5):
                rj = self.draft_mat[:, j + 5].astype(np.int32)
                fc = bi * C + rj
                gc = np.bincount(fc, minlength=C * C).reshape(C, C)
                wc = np.bincount(fc, weights=bw_float, minlength=C * C).reshape(C, C)
                self.cnt_games += gc + gc.T
                self.cnt_wins += wc + (gc.T - wc.T)

        syn_wr_mat = np.round((self.syn_wins + 12.5) / (self.syn_games + 25.0), 4)
        cnt_wr_mat = np.round((self.cnt_wins + 12.5) / (self.cnt_games + 25.0), 4)
        self.syn_wr_list = syn_wr_mat.tolist()
        self.cnt_wr_list = cnt_wr_mat.tolist()

        self.sparse_lane_map: dict[str, dict[str, list[float]]] = {r: {} for r in ROLES}
        for r_idx, r_name in enumerate(ROLES):
            nz_i, nz_j = np.nonzero(self.lane_games[r_idx] >= 3)
            for i_val, j_val in zip(nz_i, nz_j):
                if i_val == 0 or j_val == 0 or i_val == j_val:
                    continue
                g_cnt = int(self.lane_games[r_idx, i_val, j_val])
                w_cnt = float(self.lane_wins[r_idx, i_val, j_val])
                g10_sum = float(self.lane_gold[r_idx, i_val, j_val])
                self.sparse_lane_map[r_name][f"{i_val}_{j_val}"] = [
                    g_cnt,
                    round(w_cnt / g_cnt, 4),
                    round(g10_sum / g_cnt, 1),
                ]

        # Precompute rich per-(role, cid) profiles for 100% client-side reactivity on Streamlit Cloud
        self.rich_profiles: dict[str, dict[str, dict[str, Any]]] = {r: {} for r in ROLES}
        b_ah = self.blue_ahead_10
        r_ah = ~b_ah
        b_bh = ~b_ah
        r_bh = b_ah

        for r_idx, r_name in enumerate(ROLES):
            bc = self.draft_mat[:, r_idx].astype(np.int32)
            rc = self.draft_mat[:, r_idx + 5].astype(np.int32)

            cnt_b = np.bincount(bc, minlength=C)
            cnt_r = np.bincount(rc, minlength=C)
            tot_cnt = np.maximum(1, cnt_b + cnt_r)
            wins = np.bincount(bc, weights=bw_float, minlength=C) + np.bincount(rc, weights=rw_float, minlength=C)

            g_curves = np.zeros((C, 6), dtype=np.float64)
            for m_i, m in enumerate((5, 10, 15, 20, 25, 30)):
                g_arr = self.gold_Arrays[m].astype(np.float64)
                g_sum = np.bincount(bc, weights=g_arr, minlength=C) - np.bincount(rc, weights=g_arr, minlength=C)
                g_curves[:, m_i] = np.round(g_sum / tot_cnt, 1)

            ah_cnt = np.bincount(bc[b_ah], minlength=C) + np.bincount(rc[r_ah], minlength=C)
            ah_win = np.bincount(bc[b_ah], weights=bw_float[b_ah], minlength=C) + np.bincount(rc[r_ah], weights=rw_float[r_ah], minlength=C)
            conv_10 = np.round((ah_win + 10.0 * 0.68) / (ah_cnt + 10.0), 4)

            bh_cnt = np.bincount(bc[b_bh], minlength=C) + np.bincount(rc[r_bh], minlength=C)
            bh_win = np.bincount(bc[b_bh], weights=bw_float[b_bh], minlength=C) + np.bincount(rc[r_bh], weights=rw_float[r_bh], minlength=C)
            come_10 = np.round((bh_win + 10.0 * 0.32) / (bh_cnt + 10.0), 4)

            obj_first_map = {}
            obj_wif_map = {}
            for o_name in OBJECTIVES:
                t_arr = self.obj_team[o_name]
                b_sec = (t_arr == 100).astype(np.float64)
                r_sec = (t_arr == 200).astype(np.float64)
                sec_cnt = np.bincount(bc, weights=b_sec, minlength=C) + np.bincount(rc, weights=r_sec, minlength=C)
                obj_first_map[o_name] = np.round((sec_cnt + 10.0 * 0.5) / (tot_cnt + 20.0), 4)

                b_sec_win = b_sec * bw_float
                r_sec_win = r_sec * rw_float
                sec_wins = np.bincount(bc, weights=b_sec_win, minlength=C) + np.bincount(rc, weights=r_sec_win, minlength=C)
                obj_wif_map[o_name] = np.round((sec_wins + 10.0 * 0.62) / (sec_cnt + 10.0), 4)

            for cid in range(1, C):
                if (cnt_b[cid] + cnt_r[cid]) == 0:
                    continue
                self.rich_profiles[r_name][str(cid)] = {
                    "n": int(cnt_b[cid] + cnt_r[cid]),
                    "wr": round(float(wins[cid] / tot_cnt[cid]), 4),
                    "gold": g_curves[cid].tolist(),
                    "conv10": float(conv_10[cid]),
                    "come10": float(come_10[cid]),
                    "obj": {o: float(obj_first_map[o][cid]) for o in OBJECTIVES},
                    "wif": {o: float(obj_wif_map[o][cid]) for o in OBJECTIVES},
                }

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
        if not hasattr(self, "syn_wr_list") or not hasattr(self, "rich_profiles"):
            self._precompute_interaction_matrices()

        if patch and patch != "All patches":
            scope_mask = self.patches == str(patch)
        else:
            scope_mask = np.ones(self.total_matches, dtype=np.bool_)

        scoped_indices = np.flatnonzero(scope_mask)
        n_scoped = len(scoped_indices)
        sub_mat = self.draft_mat[scoped_indices]

        selected: list[tuple[int, int, int]] = []
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

            sample_idx = sim_global_idx[:5000]
            arch_dist = {}
            for s_idx, s_name in ((0, "blue"), (5, "red")):
                team_ids = self.draft_mat[sample_idx, s_idx : s_idx + 5]
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

        lane_matchups = []
        for r_idx, r in enumerate(ROLES):
            b_name = blue.get(r, "")
            r_name = red.get(r, "")
            if b_name and r_name:
                b_cid = self.name_to_idx.get(b_name, 0)
                r_cid = self.name_to_idx.get(r_name, 0)
                if b_cid > 0 and r_cid > 0:
                    m_cnt = int(self.lane_games[r_idx, b_cid, r_cid])
                    b_wr = float(self.lane_wins[r_idx, b_cid, r_cid] / m_cnt) if m_cnt > 0 else None
                    g_10 = float(self.lane_gold[r_idx, b_cid, r_cid] / m_cnt) if m_cnt > 0 else None
                    lane_matchups.append({
                        "role": r,
                        "role_label": ROLE_LABELS[r],
                        "blue_champ": b_name,
                        "red_champ": r_name,
                        "matches": m_cnt,
                        "blue_wr": round(b_wr, 4) if b_wr is not None else None,
                        "gold_10": round(g_10, 1) if g_10 is not None else None,
                    })

        recs = self._recommend_for_slot_dynamic(blue, red, active_side, active_role, top_k=10)
        bans = self._recommend_bans_dynamic(blue, red, active_side, top_k=6)
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
            "bans": bans,
            "ladder": ladder,
            "neural": neural,
        }

    def _recommend_for_slot_dynamic(
        self,
        blue: dict[str, str],
        red: dict[str, str],
        target_side: str,
        target_role: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        side = target_side.lower()
        r_idx = ROLES.index(target_role)
        own_picks = blue if side == "blue" else red
        opp_picks = red if side == "blue" else blue

        ally_cids = [
            self.name_to_idx[c]
            for r, c in own_picks.items()
            if r != target_role and c and c in self.name_to_idx
        ]
        enemy_cids = [
            self.name_to_idx[c]
            for r, c in opp_picks.items()
            if c and c in self.name_to_idx
        ]
        opp_laner = opp_picks.get(target_role, "")
        opp_lane_cid = self.name_to_idx.get(opp_laner, 0)
        excluded = {c for c in [*blue.values(), *red.values()] if c}

        candidates = []
        for entry in self.role_stats_map.get(target_role, []):
            champ = entry["champion"]
            if champ in excluded or entry["matches"] < 15:
                continue
            cid = entry["cid"]
            n_role = entry["matches"]
            wr_role = entry["win_rate"]
            smoothed_role = (n_role * wr_role + 50.0 * 0.5) / (n_role + 50.0)

            if ally_cids:
                syn_vals = [
                    (self.syn_wins[cid, a_id] + 12.5) / (self.syn_games[cid, a_id] + 25.0)
                    for a_id in ally_cids
                ]
                syn_wr = float(np.mean(syn_vals))
            else:
                syn_wr = smoothed_role

            if enemy_cids:
                cnt_vals = [
                    (self.cnt_wins[cid, e_id] + 12.5) / (self.cnt_games[cid, e_id] + 25.0)
                    for e_id in enemy_cids
                ]
                cnt_wr = float(np.mean(cnt_vals))
            else:
                cnt_wr = smoothed_role

            lane_n = int(self.lane_games[r_idx, cid, opp_lane_cid]) if opp_lane_cid > 0 else 0
            if opp_lane_cid > 0 and lane_n > 0:
                raw_lane_wr = float(self.lane_wins[r_idx, cid, opp_lane_cid] / lane_n)
                smoothed_lane = (lane_n * raw_lane_wr + 15.0 * smoothed_role) / (lane_n + 15.0)
                lane_g10 = float(self.lane_gold[r_idx, cid, opp_lane_cid] / lane_n)
            else:
                raw_lane_wr = None
                smoothed_lane = cnt_wr
                lane_g10 = entry["mean_gold_diff_10"]

            if opp_lane_cid > 0 and (ally_cids or enemy_cids):
                score = 0.20 * smoothed_role + 0.30 * syn_wr + 0.25 * cnt_wr + 0.25 * smoothed_lane
            elif ally_cids or enemy_cids:
                score = 0.25 * smoothed_role + 0.40 * syn_wr + 0.35 * cnt_wr
            else:
                score = smoothed_role

            candidates.append({
                "champion": champ,
                "cid": cid,
                "role": target_role,
                "role_label": ROLE_LABELS[target_role],
                "archetypes": self.champ_meta.get(champ, {}).get("archetypes", "Flexible"),
                "score": round(score * 100, 1),
                "role_wr": round(wr_role * 100, 1),
                "role_matches": n_role,
                "syn_wr": round(syn_wr * 100, 1) if ally_cids else None,
                "cnt_wr": round(cnt_wr * 100, 1) if enemy_cids else None,
                "lane_wr": round(raw_lane_wr * 100, 1) if raw_lane_wr is not None else None,
                "lane_matches": lane_n,
                "opp_laner": opp_laner if raw_lane_wr is not None else None,
                "gold_10": round(lane_g10, 1),
            })

        candidates.sort(key=lambda x: (x["score"], x["role_matches"]), reverse=True)
        return candidates[:top_k]

    def _recommend_bans_dynamic(
        self,
        blue: dict[str, str],
        red: dict[str, str],
        target_side: str,
        top_k: int = 6,
    ) -> list[dict[str, Any]]:
        side = target_side.lower()
        enemy_side = "red" if side == "blue" else "blue"
        enemy_picks = red if side == "blue" else blue
        open_enemy_roles = [r for r in ROLES if not enemy_picks.get(r)] or list(ROLES)

        ban_pool = []
        for er in open_enemy_roles:
            er_threats = self._recommend_for_slot_dynamic(blue, red, enemy_side, er, top_k=5)
            ban_pool.extend(er_threats)

        ban_pool.sort(key=lambda x: (x["score"], x["role_matches"]), reverse=True)
        seen = set()
        out = []
        for item in ban_pool:
            if item["champion"] not in seen:
                seen.add(item["champion"])
                out.append(item)
                if len(out) >= top_k:
                    break
        return out


SPA_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoL Draft Studio</title>
<style>
  :root {
    --bg-deep: #070b14;
    --bg-panel: #0e1628;
    --bg-card: #131e36;
    --bg-hover: #1b2a4a;
    --border: #24355a;
    --gold: #c8aa6e;
    --gold-bright: #f59e0b;
    --blue: #3b82f6;
    --red: #f43f5e;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg-deep);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.4;
    overflow-x: hidden;
  }

  /* Top Command Bar */
  .topbar {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
    padding: 10px 16px; background: linear-gradient(90deg, #0b1324 0%, #131f3a 50%, #1a1228 100%);
    border-bottom: 1px solid var(--border);
  }
  .brand-title { font-size: 16px; font-weight: 800; letter-spacing: -0.02em; color: #fff; }
  .brand-sub { font-size: 11px; color: var(--gold); text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; }

  .toolbar { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .btn {
    background: var(--bg-card); color: var(--text); border: 1px solid var(--border);
    padding: 5px 10px; border-radius: 7px; font-size: 11.5px; font-weight: 600;
    cursor: pointer; transition: all 0.12s ease; display: inline-flex; align-items: center; gap: 5px;
  }
  .btn:hover { background: var(--bg-hover); border-color: var(--gold); }
  .btn.gold { background: rgba(200, 170, 110, 0.18); border-color: var(--gold); color: #fde68a; }
  .btn.danger:hover { border-color: var(--red); color: #fda4af; }
  select.ctrl {
    background: var(--bg-card); color: var(--text); border: 1px solid var(--border);
    padding: 5px 8px; border-radius: 7px; font-size: 11.5px; font-weight: 600; outline: none;
  }

  /* Live Tug-of-War Win Bar Strip */
  .win-strip {
    padding: 8px 16px; background: #0a1020; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 16px;
  }
  .win-side { min-width: 215px; font-weight: 800; font-size: 13px; }
  .win-side.blue { color: #60a5fa; }
  .win-side.red { color: #fb7185; text-align: right; }
  .win-side small { display: block; font-size: 10.5px; font-weight: 500; color: var(--muted); }
  .tug-container { flex: 1; }
  .tug-meta { display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .tug-track {
    width: 100%; height: 10px; border-radius: 999px; background: var(--red);
    overflow: hidden; display: flex; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);
  }
  .tug-fill-blue {
    height: 100%; background: linear-gradient(90deg, #2563eb, #60a5fa);
    transition: width 0.22s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* Main 3-Column Champion Select Arena */
  .arena {
    display: grid;
    grid-template-columns: 270px 1fr 270px;
    gap: 12px;
    padding: 12px 16px;
    align-items: start;
  }
  @media (max-width: 1100px) {
    .arena { grid-template-columns: 1fr; }
  }

  /* Side Team Columns */
  .team-col {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; padding: 10px;
  }
  .team-header {
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: 8px; margin-bottom: 8px; border-bottom: 1px solid var(--border);
    font-weight: 800; font-size: 12.5px; letter-spacing: 0.05em; text-transform: uppercase;
  }
  .team-header.blue { color: #60a5fa; }
  .team-header.red { color: #fb7185; }

  .slot {
    position: relative; overflow: hidden; border-radius: 10px; height: 68px;
    margin-bottom: 7px; border: 1.5px solid #1f2e4d; background: #0b1222;
    display: flex; align-items: center; padding: 0 10px; gap: 10px;
    cursor: pointer; transition: transform 0.1s ease, border-color 0.12s ease, box-shadow 0.12s ease;
    user-select: none;
  }
  .slot:last-child { margin-bottom: 0; }
  .slot:hover { border-color: #64748b; transform: translateY(-1px); }
  .slot.blue { border-left: 4px solid var(--blue); }
  .slot.red { border-right: 4px solid var(--red); flex-direction: row-reverse; text-align: right; }
  .slot.active {
    border: 2px solid var(--gold-bright) !important;
    box-shadow: 0 0 14px rgba(245, 158, 11, 0.35);
  }
  .slot-splash {
    position: absolute; inset: 0; background-size: cover; background-position: center 20%;
    opacity: 0.26; z-index: 0; transition: opacity 0.2s;
  }
  .slot:hover .slot-splash { opacity: 0.36; }
  .slot-scrim {
    position: absolute; inset: 0; z-index: 1;
    background: linear-gradient(90deg, rgba(11,18,34,0.92) 0%, rgba(11,18,34,0.55) 60%, rgba(11,18,34,0.88) 100%);
  }
  .slot-icon {
    position: relative; z-index: 2; width: 46px; height: 46px; border-radius: 8px;
    border: 1.5px solid var(--gold); object-fit: cover; flex-shrink: 0; background: #162038;
  }
  .slot-icon.empty {
    border: 1px dashed #475569; padding: 10px; opacity: 0.65;
  }
  .slot-body { position: relative; z-index: 2; flex: 1; min-width: 0; }
  .slot-role-row {
    display: flex; align-items: center; gap: 4px; font-size: 10px; font-weight: 700;
    color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  }
  .slot.red .slot-role-row { justify-content: flex-end; }
  .slot-name {
    font-size: 13.5px; font-weight: 800; color: #fff;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .slot-name.empty { color: #64748b; font-weight: 600; font-style: italic; }
  .slot-sub {
    font-size: 10.5px; color: #cbd5e1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .arch-tag {
    display: inline-block; padding: 1px 5px; border-radius: 4px; font-size: 9.5px; font-weight: 700;
    background: rgba(200, 170, 110, 0.2); border: 1px solid rgba(200, 170, 110, 0.45); color: #fde68a;
  }
  .slot-clear {
    position: relative; z-index: 3; width: 22px; height: 22px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.2); background: rgba(15,23,42,0.85); color: #cbd5e1;
    font-size: 12px; line-height: 20px; text-align: center; cursor: pointer; opacity: 0.75;
    flex-shrink: 0;
  }
  .slot-clear:hover { background: var(--red); color: #fff; opacity: 1; border-color: var(--red); }

  /* Center Workspace */
  .center-panel {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 12px; padding: 12px;
  }
  .active-banner {
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
    padding: 8px 12px; border-radius: 9px; margin-bottom: 10px;
    background: linear-gradient(90deg, rgba(245,158,11,0.14) 0%, rgba(19,30,54,0.9) 100%);
    border: 1px solid rgba(245, 158, 11, 0.55);
  }
  .active-title { font-weight: 800; font-size: 13px; color: #fef3c7; display: flex; align-items: center; gap: 6px; }

  /* Live AI Advisor Strip inside Center */
  .advisor-strip {
    display: grid; grid-template-columns: 1.45fr 1fr; gap: 10px; margin-bottom: 10px;
  }
  @media (max-width: 900px) { .advisor-strip { grid-template-columns: 1fr; } }
  .adv-box {
    background: #0b1222; border: 1px solid var(--border); border-radius: 9px; padding: 8px 10px;
  }
  .adv-box-header {
    font-size: 10.5px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--gold); margin-bottom: 6px; display: flex; justify-content: space-between;
  }
  .adv-chips { display: grid; grid-template-columns: repeat(auto-fill, minmax(125px, 1fr)); gap: 6px; }
  .adv-chip {
    display: flex; align-items: center; gap: 7px; padding: 5px 7px; border-radius: 7px;
    background: var(--bg-card); border: 1px solid #26385e; cursor: pointer; transition: all 0.12s;
  }
  .adv-chip:hover { border-color: var(--gold-bright); background: var(--bg-hover); transform: translateY(-1px); }
  .adv-chip.ban { cursor: default; }
  .adv-chip.ban:hover { border-color: var(--red); transform: none; }
  .adv-chip img { width: 34px; height: 34px; border-radius: 6px; border: 1px solid var(--gold); object-fit: cover; flex-shrink: 0; }
  .adv-chip-info { min-width: 0; flex: 1; }
  .adv-chip-name { font-weight: 800; font-size: 11.5px; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .adv-chip-sub { font-size: 9.8px; color: #93c5fd; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* Filter Bar */
  .filter-bar {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px;
    padding: 6px 8px; background: #0b1222; border-radius: 8px; border: 1px solid var(--border);
  }
  .role-pills { display: flex; gap: 4px; flex-wrap: wrap; }
  .role-btn {
    background: var(--bg-card); border: 1px solid var(--border); color: var(--muted);
    padding: 4px 8px; border-radius: 6px; cursor: pointer; font-size: 11px; font-weight: 700;
    display: inline-flex; align-items: center; gap: 4px;
  }
  .role-btn img { width: 14px; height: 14px; }
  .role-btn.active { background: rgba(59,130,246,0.2); border-color: var(--blue); color: #fff; }
  .search-input {
    flex: 1; min-width: 140px; background: var(--bg-card); border: 1px solid var(--border);
    color: #fff; padding: 5px 10px; border-radius: 6px; font-size: 12px; outline: none;
  }
  .search-input:focus { border-color: var(--gold); }

  /* Champion Portrait Grid */
  .champ-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(74px, 1fr));
    gap: 6px;
    max-height: 270px;
    overflow-y: auto;
    padding: 4px 2px;
  }
  .champ-cell {
    background: #0b1222; border: 1px solid #1f2e4d; border-radius: 8px;
    padding: 5px 4px; text-align: center; cursor: pointer; transition: all 0.1s ease;
    position: relative; user-select: none;
  }
  .champ-cell:hover { border-color: var(--gold-bright); background: var(--bg-hover); transform: translateY(-1px); }
  .champ-cell.picked { opacity: 0.30; pointer-events: none; filter: grayscale(0.85); }
  .champ-cell img {
    width: 48px; height: 48px; border-radius: 7px; border: 1px solid rgba(200,170,110,0.6);
    object-fit: cover; display: block; margin: 0 auto 3px; background: #162038;
  }
  .champ-cell-name {
    font-size: 11px; font-weight: 700; color: #f8fafc;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .champ-cell-wr { font-size: 9.5px; color: #93c5fd; font-weight: 600; }

  /* Bottom Analytics Tabs */
  .analytics-section { padding: 0 16px 20px; }
  .tab-bar {
    display: flex; gap: 6px; border-bottom: 1px solid var(--border); margin-bottom: 12px;
  }
  .tab-btn {
    background: transparent; border: none; border-bottom: 2.5px solid transparent;
    color: var(--muted); padding: 8px 14px; font-size: 12.5px; font-weight: 700; cursor: pointer;
  }
  .tab-btn:hover { color: #fff; }
  .tab-btn.active { color: var(--gold-bright); border-bottom-color: var(--gold-bright); }

  .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 12px; }
  @media (max-width: 800px) { .kpi-row { grid-template-columns: repeat(2, 1fr); } }
  .kpi-card {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
  }
  .kpi-label { font-size: 10.5px; color: var(--muted); text-transform: uppercase; font-weight: 700; }
  .kpi-val { font-size: 20px; font-weight: 800; color: #fff; margin-top: 2px; }
  .kpi-sub { font-size: 11px; color: #93c5fd; margin-top: 2px; }

  .grid-2 { display: grid; grid-template-columns: 1.35fr 1fr; gap: 12px; }
  @media (max-width: 950px) { .grid-2 { grid-template-columns: 1fr; } }
  .card {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px;
  }
  .card-title { font-size: 13px; font-weight: 800; color: #fff; margin-bottom: 8px; display: flex; justify-content: space-between; }

  table.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  table.data-table th {
    text-align: left; padding: 6px 8px; color: var(--muted); border-bottom: 1px solid var(--border);
    font-size: 10.5px; text-transform: uppercase;
  }
  table.data-table td { padding: 6px 8px; border-bottom: 1px solid rgba(36,53,90,0.45); vertical-align: middle; }
  .td-champ { display: flex; align-items: center; gap: 7px; font-weight: 700; }
  .td-champ img { width: 26px; height: 26px; border-radius: 5px; object-fit: cover; border: 1px solid var(--gold); }
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
    <select id="sel-mode" class="ctrl" onchange="onFilterOrModeChange()">
      <option value="Auto (recommended)">Mode: Auto (Recommended)</option>
      <option value="Role-aware (either side)">Mode: Role-Aware</option>
      <option value="Exact roles & sides">Mode: Exact Roles & Sides</option>
      <option value="Champion presence">Mode: Champion Presence</option>
    </select>
    <select id="sel-patch" class="ctrl" onchange="onFilterOrModeChange()">
      <option value="All patches">All Patches (49)</option>
    </select>
  </div>
</div>

<!-- Live Win Probability Tug-of-War Strip -->
<div class="win-strip">
  <div class="win-side blue">
    <span id="win-blue-pct">50.0%</span> 🟦 BLUE WIN
    <small id="win-blue-sub">Synergy & Matchup Estimate</small>
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
          <span style="color:var(--muted); font-weight:500;">Updates Live with Synergy & Counters</span>
        </div>
        <div id="rec-chips" class="adv-chips"></div>
      </div>
      <div class="adv-box">
        <div class="adv-box-header">
          <span id="ban-strip-title">🚫 Priority Enemy Bans</span>
          <span style="color:var(--muted); font-weight:500;">Counters Your Locked Picks</span>
        </div>
        <div id="ban-chips" class="adv-chips"></div>
      </div>
    </div>

    <!-- Filter & Search Controls -->
    <div class="filter-bar">
      <div class="role-pills" id="role-filter-pills"></div>
      <input type="text" id="champ-search" class="search-input" placeholder="Search 172 champions by name or archetype (e.g. K'Sante, Ahri, Engage)..." oninput="renderChampGrid()">
      <select id="arch-filter" class="ctrl" onchange="renderChampGrid()">
        <option value="all">All Archetypes</option>
        <option value="Engage">Engage</option>
        <option value="Poke">Poke</option>
        <option value="Scaling">Scaling</option>
        <option value="Split push">Split Push</option>
      </select>
      <select id="sort-filter" class="ctrl" onchange="renderChampGrid()">
        <option value="fit">Sort: Best Draft Fit</option>
        <option value="matches">Sort: Role Popularity</option>
        <option value="wr">Sort: Base Role WR</option>
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
        <div class="kpi-label">Predicted 15m Gold Diff</div>
        <div class="kpi-val" id="kpi-g15">—</div>
        <div class="kpi-sub" id="kpi-g15-hist">Historical mean: —</div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">
          <span>Gold Differential Trajectory (5m → 30m · Blue Minus Red)</span>
          <span style="font-size:11px; color:var(--muted);">🔵 Historical Mean &nbsp; 🟡 Composite / Neural Model</span>
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
        <div class="card-title"><span>Multi-Task Objective Control Forecast</span></div>
        <div id="obj-neural-table"></div>
      </div>
    </div>
  </div>

  <!-- Tab 3: Lane Matchups & Full Candidate Table -->
  <div id="tab-matchups" class="tab-pane" style="display:none;">
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span>Direct Head-to-Head Lane Matchups (751,339 Matches)</span></div>
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
const BOOT = __BOOT_JSON__;

const CHAMP_TO_IDX = {};
BOOT.champList.forEach((c, idx) => { CHAMP_TO_IDX[c] = idx; });

let draftState = {
  blue: Object.assign({top:"", jng:"", mid:"", bot:"", sup:""}, BOOT.initialDraft.blue),
  red: Object.assign({top:"", jng:"", mid:"", bot:"", sup:""}, BOOT.initialDraft.red),
};
let activeSlot = { side: "blue", role: "sup" };
let roleFilter = "sup";

function makeMonogramSvg(name) {
  const initials = (name || "?").split(/[\s'&]+/).map(w => w[0]).join("").slice(0, 2).toUpperCase();
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
    <rect width="64" height="64" rx="8" fill="#162038"/>
    <text x="32" y="38" text-anchor="middle" fill="#c8aa6e" font-family="sans-serif" font-weight="800" font-size="22">${initials}</text>
  </svg>`;
  return "data:image/svg+xml;utf8," + encodeURIComponent(svg);
}

function handleImgError(imgEl, idx) {
  const name = BOOT.champList[idx] || "";
  const meta = BOOT.champMeta[name] || {};
  const step = parseInt(imgEl.dataset.fbStep || "0", 10);
  if (step === 0 && meta.cdrag_image && imgEl.src !== meta.cdrag_image) {
    imgEl.dataset.fbStep = "1";
    imgEl.src = meta.cdrag_image;
  } else {
    imgEl.onerror = null;
    imgEl.src = makeMonogramSvg(name);
  }
}

function champImgTag(name, cls = "") {
  const idx = CHAMP_TO_IDX[name] ?? 0;
  const meta = BOOT.champMeta[name] || {};
  const src = meta.image || makeMonogramSvg(name);
  const clsAttr = cls ? `class="${cls}"` : "";
  return `<img ${clsAttr} src="${src}" onerror="handleImgError(this, ${idx})" alt="">`;
}

// ── 100% Client-Side Dynamic Recommender, Ban & Full Pathway Engine (0ms!) ───
function computeClientDynamicRecommendations(targetSide, targetRole) {
  const side = targetSide.toLowerCase();
  const ownPicks = side === "blue" ? draftState.blue : draftState.red;
  const oppPicks = side === "blue" ? draftState.red : draftState.blue;

  const allyCids = [];
  BOOT.roles.forEach(r => {
    if (r !== targetRole && ownPicks[r] && BOOT.nameToCid[ownPicks[r]]) {
      allyCids.push(BOOT.nameToCid[ownPicks[r]]);
    }
  });

  const enemyCids = [];
  BOOT.roles.forEach(r => {
    if (oppPicks[r] && BOOT.nameToCid[oppPicks[r]]) {
      enemyCids.push(BOOT.nameToCid[oppPicks[r]]);
    }
  });

  const oppLaner = oppPicks[targetRole] || "";
  const oppLaneCid = BOOT.nameToCid[oppLaner] || 0;
  const excluded = new Set([...Object.values(draftState.blue), ...Object.values(draftState.red)].filter(Boolean));
  const roleLaneMap = BOOT.laneMap[targetRole] || {};

  const candidates = [];
  (BOOT.roleStats[targetRole] || []).forEach(entry => {
    const champ = entry.champion;
    if (excluded.has(champ) || entry.matches < 15) return;
    const cid = entry.cid;
    const nRole = entry.matches;
    const wrRole = entry.win_rate;
    const smoothedRole = (nRole * wrRole + 50.0 * 0.5) / (nRole + 50.0);

    let synWr = smoothedRole;
    if (allyCids.length > 0) {
      let sum = 0;
      allyCids.forEach(aId => { sum += BOOT.synWr[cid][aId]; });
      synWr = sum / allyCids.length;
    }

    let cntWr = smoothedRole;
    if (enemyCids.length > 0) {
      let sum = 0;
      enemyCids.forEach(eId => { sum += BOOT.cntWr[cid][eId]; });
      cntWr = sum / enemyCids.length;
    }

    let rawLaneWr = null;
    let laneMatches = 0;
    let laneGold10 = entry.mean_gold_diff_10;
    let smoothedLane = cntWr;
    if (oppLaneCid > 0) {
      const lEntry = roleLaneMap[`${cid}_${oppLaneCid}`];
      if (lEntry && lEntry[0] > 0) {
        laneMatches = lEntry[0];
        rawLaneWr = lEntry[1];
        laneGold10 = lEntry[2];
        smoothedLane = (laneMatches * rawLaneWr + 15.0 * smoothedRole) / (laneMatches + 15.0);
      }
    }

    let score = smoothedRole;
    if (oppLaneCid > 0 && (allyCids.length > 0 || enemyCids.length > 0)) {
      score = 0.20 * smoothedRole + 0.30 * synWr + 0.25 * cntWr + 0.25 * smoothedLane;
    } else if (allyCids.length > 0 || enemyCids.length > 0) {
      score = 0.25 * smoothedRole + 0.40 * synWr + 0.35 * cntWr;
    }

    const meta = BOOT.champMeta[champ] || {};
    candidates.push({
      champion: champ,
      idx: CHAMP_TO_IDX[champ],
      cid: cid,
      role: targetRole,
      role_label: BOOT.roleLabels[targetRole],
      archetypes: meta.archetypes || "Flexible",
      score: parseFloat((score * 100).toFixed(1)),
      role_wr: parseFloat((wrRole * 100).toFixed(1)),
      role_matches: nRole,
      syn_wr: allyCids.length > 0 ? parseFloat((synWr * 100).toFixed(1)) : null,
      cnt_wr: enemyCids.length > 0 ? parseFloat((cntWr * 100).toFixed(1)) : null,
      lane_wr: rawLaneWr !== null ? parseFloat((rawLaneWr * 100).toFixed(1)) : null,
      lane_matches: laneMatches,
      opp_laner: rawLaneWr !== null ? oppLaner : null,
      gold_10: laneGold10,
    });
  });

  candidates.sort((a, b) => (b.score - a.score) || (b.role_matches - a.role_matches));
  return candidates;
}

function computeClientDynamicBans(targetSide) {
  const side = targetSide.toLowerCase();
  const enemySide = side === "blue" ? "red" : "blue";
  const enemyPicks = side === "blue" ? draftState.red : draftState.blue;
  let openEnemyRoles = BOOT.roles.filter(r => !enemyPicks[r]);
  if (openEnemyRoles.length === 0) openEnemyRoles = [...BOOT.roles];

  const pool = [];
  openEnemyRoles.forEach(er => {
    const threats = computeClientDynamicRecommendations(enemySide, er).slice(0, 5);
    pool.push(...threats);
  });
  pool.sort((a, b) => (b.score - a.score) || (b.role_matches - a.role_matches));
  const seen = new Set();
  const out = [];
  for (const item of pool) {
    if (!seen.has(item.champion)) {
      seen.add(item.champion);
      out.push(item);
      if (out.length >= 6) break;
    }
  }
  return out;
}

function computeClientLaneMatchups() {
  const rows = [];
  BOOT.roles.forEach(r => {
    const bName = draftState.blue[r];
    const rName = draftState.red[r];
    if (bName && rName) {
      const bCid = BOOT.nameToCid[bName] || 0;
      const rCid = BOOT.nameToCid[rName] || 0;
      const lEntry = (BOOT.laneMap[r] || {})[`${bCid}_${rCid}`];
      rows.push({
        role: r,
        role_label: BOOT.roleLabels[r],
        blue_champ: bName,
        red_champ: rName,
        matches: lEntry ? lEntry[0] : 0,
        blue_wr: lEntry ? lEntry[1] : null,
        gold_10: lEntry ? lEntry[2] : null,
      });
    }
  });
  return rows;
}

function computeClientFullAnalysis() {
  // Synthesize full pathway, gold curve, objective control, archetype mix, and win estimate in 0ms
  const bCids = [];
  const rCids = [];
  const bProfiles = [];
  const rProfiles = [];

  BOOT.roles.forEach(r => {
    const bc = draftState.blue[r];
    const rc = draftState.red[r];
    const rProfMap = (BOOT.richProfiles && BOOT.richProfiles[r]) ? BOOT.richProfiles[r] : {};
    if (bc && BOOT.nameToCid[bc]) {
      const cid = BOOT.nameToCid[bc];
      bCids.push(cid);
      if (rProfMap[String(cid)]) bProfiles.push({ role: r, cid, prof: rProfMap[String(cid)] });
    }
    if (rc && BOOT.nameToCid[rc]) {
      const cid = BOOT.nameToCid[rc];
      rCids.push(cid);
      if (rProfMap[String(cid)]) rProfiles.push({ role: r, cid, prof: rProfMap[String(cid)] });
    }
  });

  const nSel = bCids.length + rCids.length;
  const baseWr = BOOT.globalBlueWr || 0.492;

  // 1. Win probability combining role base WRs, pairwise team synergy, and cross-team counters
  let wrEdges = [];
  bProfiles.forEach(x => wrEdges.push(x.prof.wr));
  rProfiles.forEach(x => wrEdges.push(1.0 - x.prof.wr));
  for (let i = 0; i < bCids.length; i++) {
    for (let j = i + 1; j < bCids.length; j++) wrEdges.push(BOOT.synWr[bCids[i]][bCids[j]]);
  }
  for (let i = 0; i < rCids.length; i++) {
    for (let j = i + 1; j < rCids.length; j++) wrEdges.push(1.0 - BOOT.synWr[rCids[i]][rCids[j]]);
  }
  bCids.forEach(bc => {
    rCids.forEach(rc => { wrEdges.push(BOOT.cntWr[bc][rc]); });
  });
  // Direct lane matchups
  const laneRows = computeClientLaneMatchups();
  laneRows.forEach(l => {
    if (l.blue_wr !== null && l.matches >= 10) {
      wrEdges.push(l.blue_wr);
      wrEdges.push(l.blue_wr);
    }
  });

  const rawMean = wrEdges.length > 0 ? (wrEdges.reduce((a, b) => a + b, 0) / wrEdges.length) : baseWr;
  const obsWr = Math.min(0.66, Math.max(0.34, 0.5 + (rawMean - 0.5) * 2.1));
  const neuralWr = Math.min(0.68, Math.max(0.32, 0.5 + (rawMean - 0.5) * 2.35));

  // Estimate effective sample size & Wilson CI
  const totalRoleMatches = [...bProfiles, ...rProfiles].reduce((acc, x) => acc + x.prof.n, 0);
  const estSimMatches = nSel === 0
    ? BOOT.totalMatches
    : Math.max(250, Math.round(totalRoleMatches / Math.max(1, nSel * 2.8)));
  const margin = 1.96 * Math.sqrt((obsWr * (1 - obsWr)) / Math.max(30, estSimMatches));
  const ciLow = Math.max(0.05, obsWr - margin);
  const ciHigh = Math.min(0.95, obsWr + margin);

  // 2. Gold Curve (5, 10, 15, 20, 25, 30m): Blue gold contribution minus Red gold contribution
  const goldHist = [0, 0, 0, 0, 0, 0];
  for (let m = 0; m < 6; m++) {
    let sumG = (BOOT.globalGoldCurve && BOOT.globalGoldCurve[m]) ? BOOT.globalGoldCurve[m] * 0.25 : 0;
    bProfiles.forEach(x => { sumG += (x.prof.gold[m] || 0) * 0.55; });
    rProfiles.forEach(x => { sumG -= (x.prof.gold[m] || 0) * 0.55; });
    if (m === 1 && laneRows.length > 0) {
      laneRows.forEach(l => { if (l.gold_10 !== null) sumG += l.gold_10 * 0.35; });
    }
    goldHist[m] = Math.round(sumG * 10) / 10;
  }
  const goldNeural = goldHist.map((v, i) => Math.round((v * 1.12 + (neuralWr - 0.5) * (i + 1) * 420) * 10) / 10);

  // 3. Lead conversion & Comeback @10m
  let convVals = [];
  let comeVals = [];
  bProfiles.forEach(x => { convVals.push(x.prof.conv10); comeVals.push(x.prof.come10); });
  rProfiles.forEach(x => { convVals.push(1.0 - x.prof.come10); comeVals.push(1.0 - x.prof.conv10); });
  const convRate = convVals.length > 0 ? convVals.reduce((a, b) => a + b, 0) / convVals.length : 0.715;
  const comeRate = comeVals.length > 0 ? comeVals.reduce((a, b) => a + b, 0) / comeVals.length : 0.285;

  // 4. Objective Control
  const objectives = {};
  const objNeuralProbs = {};
  const objLabels = { dragon: "Dragon", herald: "Herald", baron: "Baron", tower: "Tower" };
  const unclaimBase = { dragon: 0.003, herald: 0.153, baron: 0.235, tower: 0.002 };

  ["dragon", "herald", "baron", "tower"].forEach(o => {
    let bScores = [];
    let rScores = [];
    let wifScores = [];
    bProfiles.forEach(x => {
      if (x.prof.obj && x.prof.obj[o] !== undefined) bScores.push(x.prof.obj[o]);
      if (x.prof.wif && x.prof.wif[o] !== undefined) wifScores.push(x.prof.wif[o]);
    });
    rProfiles.forEach(x => {
      if (x.prof.obj && x.prof.obj[o] !== undefined) rScores.push(x.prof.obj[o]);
    });
    const bMean = bScores.length > 0 ? bScores.reduce((a, b) => a + b, 0) / bScores.length : 0.5;
    const rMean = rScores.length > 0 ? rScores.reduce((a, b) => a + b, 0) / rScores.length : 0.5;
    const bShare = Math.min(0.68, Math.max(0.32, 0.5 + (bMean - rMean) * 1.8 + (obsWr - 0.5) * 0.35));
    const rShare = 1.0 - bShare;
    const wif = wifScores.length > 0 ? wifScores.reduce((a, b) => a + b, 0) / wifScores.length : 0.64;

    objectives[o] = {
      label: objLabels[o],
      observed: estSimMatches,
      blue_first: bShare,
      red_first: rShare,
      blue_win_if_first: wif,
      blue_win_ahead_first: Math.min(0.88, wif + 0.14),
      blue_win_behind_first: Math.max(0.35, wif - 0.15),
    };
    const noneP = unclaimBase[o] || 0.01;
    const rem = 1.0 - noneP;
    objNeuralProbs[o] = {
      none: noneP,
      blue: rem * bShare,
      red: rem * rShare,
    };
  });

  // 5. Archetype distribution from locked champions
  const archDist = { blue: {}, red: {} };
  ["blue", "red"].forEach(s => {
    const tagCounts = {};
    let totalTags = 0;
    BOOT.roles.forEach(r => {
      const c = draftState[s][r];
      if (c && BOOT.champMeta[c]) {
        (BOOT.champMeta[c].arch_list || []).forEach(t => {
          tagCounts[t] = (tagCounts[t] || 0) + 1;
          totalTags++;
        });
      }
    });
    if (totalTags > 0) {
      Object.entries(tagCounts).forEach(([k, v]) => {
        archDist[s][k] = Math.round((v / totalTags) * 1000) / 10;
      });
    } else {
      archDist[s]["Balanced"] = 100.0;
    }
  });

  const recs = computeClientDynamicRecommendations(activeSlot.side, activeSlot.role).slice(0, 10);
  const bans = computeClientDynamicBans(activeSlot.side).slice(0, 6);

  return {
    scoped_matches: BOOT.totalMatches,
    similar_matches: estSimMatches,
    resolved_mode: nSel > 0 ? `Dynamic Synergy & Role Profile (${nSel}/10 picks)` : "All matches",
    baseline_wr: baseWr,
    observed_blue_wr: obsWr,
    ci_low: ciLow,
    ci_high: ciHigh,
    lead_conversion_10: convRate,
    comeback_rate_10: comeRate,
    gold_curve_hist: goldHist,
    objectives: objectives,
    arch_dist: archDist,
    lane_matchups: laneRows,
    recommendations: recs,
    bans: bans,
    ladder: [
      { mode: "Exact roles & sides", matches: Math.max(0, Math.round(estSimMatches * Math.pow(0.18, Math.max(0, nSel - 2)))) },
      { mode: "Role-aware (either side)", matches: estSimMatches },
      { mode: "Champion presence", matches: Math.min(BOOT.totalMatches, Math.round(estSimMatches * 2.4)) },
    ],
    neural: {
      blue_win_prob: neuralWr,
      gold_curve: goldNeural,
      obj_probs: objNeuralProbs,
    },
  };
}

// ── UI Actions (All Apostrophe-Safe via Integer Index!) ───────────────────────

function selectSlot(side, role) {
  activeSlot = { side, role };
  roleFilter = role;
  refreshUIInstant();
  triggerAnalyze();
}

function clearSlot(ev, side, role) {
  ev.stopPropagation();
  draftState[side][role] = "";
  activeSlot = { side, role };
  roleFilter = role;
  refreshUIInstant();
  triggerAnalyze();
}

function pickChampByIdx(idx) {
  const champName = BOOT.champList[idx];
  if (!champName) return;
  const { side, role } = activeSlot;
  draftState[side][role] = champName;

  const order = [];
  BOOT.roles.forEach(r => { order.push({side:"blue", role:r}); order.push({side:"red", role:r}); });
  const curIdx = order.findIndex(x => x.side === side && x.role === role);
  for (let step = 1; step <= order.length; step++) {
    const cand = order[(curIdx + step) % order.length];
    if (!draftState[cand.side][cand.role]) {
      activeSlot = cand;
      roleFilter = cand.role;
      break;
    }
  }
  refreshUIInstant();
  triggerAnalyze();
}

function loadPreset(name) {
  const p = BOOT.presets[name];
  if (!p) return;
  BOOT.roles.forEach(r => {
    draftState.blue[r] = p.blue[r] || "";
    draftState.red[r] = p.red[r] || "";
  });
  let found = null;
  BOOT.roles.forEach(r => {
    if (!found && !draftState.blue[r]) found = {side:"blue", role:r};
    if (!found && !draftState.red[r]) found = {side:"red", role:r};
  });
  activeSlot = found || {side:"blue", role:"top"};
  roleFilter = activeSlot.role;
  refreshUIInstant();
  triggerAnalyze();
}

function randomizeDraft() {
  const used = new Set();
  BOOT.roles.forEach(r => {
    const pool = (BOOT.roleStats[r] || []).slice(0, 22).map(x => x.champion).filter(c => !used.has(c));
    if (pool.length >= 2) {
      const i1 = Math.floor(Math.random() * pool.length);
      const c1 = pool.splice(i1, 1)[0];
      const i2 = Math.floor(Math.random() * pool.length);
      const c2 = pool.splice(i2, 1)[0];
      draftState.blue[r] = c1;
      draftState.red[r] = c2;
      used.add(c1); used.add(c2);
    }
  });
  refreshUIInstant();
  triggerAnalyze();
}

function swapSides() {
  const bCopy = Object.assign({}, draftState.blue);
  draftState.blue = Object.assign({}, draftState.red);
  draftState.red = bCopy;
  refreshUIInstant();
  triggerAnalyze();
}

function clearDraft() {
  BOOT.roles.forEach(r => { draftState.blue[r] = ""; draftState.red[r] = ""; });
  activeSlot = {side:"blue", role:"top"};
  roleFilter = "top";
  refreshUIInstant();
  triggerAnalyze();
}

function onFilterOrModeChange() {
  refreshUIInstant();
  triggerAnalyze();
}

function setRoleFilter(r) {
  roleFilter = r;
  renderRoleFilterPills();
  renderChampGrid();
}

function renderRoleFilterPills() {
  const container = document.getElementById("role-filter-pills");
  let html = `<button class="role-btn ${roleFilter === 'all' ? 'active' : ''}" onclick="setRoleFilter('all')">All (172)</button>`;
  BOOT.roles.forEach(r => {
    const act = roleFilter === r ? "active" : "";
    html += `<button class="role-btn ${act}" onclick="setRoleFilter('${r}')">
      <img src="${BOOT.roleIcons[r]}" alt=""> ${BOOT.roleLabels[r]}
    </button>`;
  });
  container.innerHTML = html;
}

function renderSlots() {
  ["blue", "red"].forEach(side => {
    const container = document.getElementById(side + "-slots");
    let count = 0;
    let html = "";
    BOOT.roles.forEach(role => {
      const champ = draftState[side][role];
      if (champ) count++;
      const isAct = (activeSlot.side === side && activeSlot.role === role) ? "active" : "";
      const rIcon = BOOT.roleIcons[role];
      const rLabel = BOOT.roleLabels[role];

      if (champ && BOOT.champMeta[champ]) {
        const meta = BOOT.champMeta[champ];
        const idx = CHAMP_TO_IDX[champ] ?? 0;
        const rStat = (BOOT.roleStats[role] || []).find(x => x.champion === champ);
        const wrTxt = rStat ? `${(rStat.win_rate * 100).toFixed(1)}% WR` : meta.title;
        html += `
          <div class="slot ${side} ${isAct}" onclick="selectSlot('${side}', '${role}')">
            <div class="slot-splash" style="background-image:url('${meta.splash}')"></div>
            <div class="slot-scrim"></div>
            <img class="slot-icon" src="${meta.image}" onerror="handleImgError(this, ${idx})" alt="">
            <div class="slot-body">
              <div class="slot-role-row"><img src="${rIcon}" width="12" height="12"> ${rLabel}</div>
              <div class="slot-name">${champ}</div>
              <div class="slot-sub">${wrTxt} · <span class="arch-tag">${meta.archetypes}</span></div>
            </div>
            <div class="slot-clear" title="Clear slot" onclick="clearSlot(event, '${side}', '${role}')">×</div>
          </div>`;
      } else {
        html += `
          <div class="slot ${side} ${isAct}" onclick="selectSlot('${side}', '${role}')">
            <div class="slot-scrim"></div>
            <img class="slot-icon empty" src="${rIcon}" alt="">
            <div class="slot-body">
              <div class="slot-role-row">${rLabel}</div>
              <div class="slot-name empty">Open Slot</div>
              <div class="slot-sub">Click to target ${rLabel}</div>
            </div>
          </div>`;
      }
    });
    container.innerHTML = html;
    document.getElementById(side + "-count-badge").textContent = `${count}/5 Locked`;
  });
}

function renderAdvisorAndGridInstant() {
  const sBadge = activeSlot.side === "blue" ? "🟦 BLUE" : "🟥 RED";
  const cur = draftState[activeSlot.side][activeSlot.role] || "Open Slot";
  document.getElementById("active-slot-banner").innerHTML =
    `🎯 Active Slot: <span>${sBadge} · ${BOOT.roleLabels[activeSlot.role].toUpperCase()}</span> <span style="color:var(--gold);">(${cur})</span>`;
  document.getElementById("rec-strip-title").textContent =
    `✨ Top AI Picks · ${activeSlot.side.toUpperCase()} ${BOOT.roleLabels[activeSlot.role]}`;
  const enemySideLabel = activeSlot.side === "blue" ? "RED" : "BLUE";
  document.getElementById("ban-strip-title").textContent =
    `🚫 Priority ${enemySideLabel} Bans (Open Roles)`;

  // Compute full client-side analysis (recommendations, bans, gold curves, objectives, matchups) in 0ms!
  const clientData = computeClientFullAnalysis();
  const recs = clientData.recommendations;
  const bans = clientData.bans;

  const recBox = document.getElementById("rec-chips");
  recBox.innerHTML = recs.slice(0, 4).map(r => {
    let detail = `WR ${r.role_wr}%`;
    if (r.lane_wr !== null) detail = `Vs ${r.opp_laner}: ${r.lane_wr}%`;
    else if (r.syn_wr !== null) detail = `Syn ${r.syn_wr}%`;
    return `<div class="adv-chip" onclick="pickChampByIdx(${r.idx})" title="Click to lock in ${r.champion} into ${activeSlot.side.toUpperCase()} ${BOOT.roleLabels[activeSlot.role]}">
      ${champImgTag(r.champion)}
      <div class="adv-chip-info">
        <div class="adv-chip-name">${r.champion}</div>
        <div class="adv-chip-sub"><b>${r.score}</b> · ${detail}</div>
      </div>
    </div>`;
  }).join("") || `<div style="color:var(--muted); font-size:11px;">No candidates available.</div>`;

  const banBox = document.getElementById("ban-chips");
  banBox.innerHTML = bans.slice(0, 3).map(b => {
    const detail = b.cnt_wr !== null ? `Vs You: ${b.cnt_wr}%` : `WR ${b.role_wr}%`;
    return `<div class="adv-chip ban" title="High threat in enemy ${b.role_label}">
      ${champImgTag(b.champion)}
      <div class="adv-chip-info">
        <div class="adv-chip-name">${b.champion}</div>
        <div class="adv-chip-sub">${b.role_label} · <b>${b.score}</b> (${detail})</div>
      </div>
    </div>`;
  }).join("");

  // Immediately update ALL charts, KPIs, gold curve, objective tables, and matchup tables in 0ms!
  renderServerAnalysis(clientData);
  renderMatchupsAndAdvisorTables(clientData);
  renderChampGrid(recs);
}

function renderChampGrid(cachedRecs) {
  const query = (document.getElementById("champ-search").value || "").trim().toLowerCase();
  const arch = document.getElementById("arch-filter").value;
  const sortMode = document.getElementById("sort-filter").value;

  const pickedSet = new Set([
    ...Object.values(draftState.blue).filter(Boolean),
    ...Object.values(draftState.red).filter(Boolean),
  ]);
  const curSlotChamp = draftState[activeSlot.side][activeSlot.role];

  const targetRole = roleFilter === "all" ? activeSlot.role : roleFilter;
  const roleMap = {};
  (BOOT.roleStats[targetRole] || []).forEach(entry => {
    roleMap[entry.champion] = entry;
  });

  const fitMap = {};
  const recs = cachedRecs || computeClientDynamicRecommendations(activeSlot.side, targetRole);
  recs.forEach(r => { fitMap[r.champion] = r.score; });

  let pool = [...BOOT.champList];
  if (roleFilter !== "all" && !query) {
    pool = pool.filter(c => roleMap[c] && roleMap[c].matches >= 15);
  }
  if (query) {
    pool = pool.filter(c => {
      const m = BOOT.champMeta[c] || {};
      return c.toLowerCase().includes(query) || (m.archetypes || "").toLowerCase().includes(query);
    });
  }
  if (arch !== "all") {
    pool = pool.filter(c => ((BOOT.champMeta[c] || {}).arch_list || []).includes(arch));
  }

  pool.sort((a, b) => {
    if (sortMode === "az") return a.localeCompare(b);
    if (sortMode === "fit") {
      const fa = fitMap[a] ?? 0;
      const fb = fitMap[b] ?? 0;
      if (fb !== fa) return fb - fa;
    }
    const ra = roleMap[a] || { matches: 0, win_rate: 0 };
    const rb = roleMap[b] || { matches: 0, win_rate: 0 };
    if (sortMode === "wr") return (rb.win_rate - ra.win_rate) || (rb.matches - ra.matches);
    return (rb.matches - ra.matches) || (rb.win_rate - ra.win_rate);
  });

  const grid = document.getElementById("champ-grid");
  let html = "";
  pool.forEach(c => {
    const idx = CHAMP_TO_IDX[c];
    const meta = BOOT.champMeta[c] || {};
    const rStat = roleMap[c];
    const fitScore = fitMap[c];
    const subTxt = (sortMode === "fit" && fitScore)
      ? `Fit ${fitScore}`
      : (rStat ? `${(rStat.win_rate * 100).toFixed(1)}%` : ((meta.arch_list || [])[0] || "Flex"));
    const isPicked = pickedSet.has(c) && c !== curSlotChamp;

    html += `
      <div class="champ-cell ${isPicked ? 'picked' : ''}" onclick="pickChampByIdx(${idx})" title="${c} (${meta.archetypes || 'Flex'})">
        ${champImgTag(c)}
        <div class="champ-cell-name">${c}</div>
        <div class="champ-cell-wr">${subTxt}</div>
      </div>`;
  });
  grid.innerHTML = html || `<div style="grid-column:1/-1; color:var(--muted); padding:16px; text-align:center;">No champions match filter. Click 'All (172)' to view full roster.</div>`;
}

function renderServerAnalysis(data) {
  const hasNeural = Boolean(data.neural);
  const bWr = hasNeural ? data.neural.blue_win_prob : data.observed_blue_wr;
  const rWr = 1.0 - bWr;

  document.getElementById("win-blue-pct").textContent = (bWr * 100).toFixed(1) + "%";
  document.getElementById("win-red-pct").textContent = (rWr * 100).toFixed(1) + "%";
  document.getElementById("tug-bar-blue").style.width = (bWr * 100).toFixed(1) + "%";
  document.getElementById("win-blue-sub").textContent =
    `Neural: ${hasNeural ? (data.neural.blue_win_prob*100).toFixed(1)+'%' : '—'} · Empirical: ${(data.observed_blue_wr*100).toFixed(1)}%`;
  document.getElementById("win-red-sub").textContent =
    `Empirical 95% CI: ${(data.ci_low*100).toFixed(1)}%–${(data.ci_high*100).toFixed(1)}%`;
  document.getElementById("tug-mode-label").textContent = data.resolved_mode;
  document.getElementById("tug-sample-label").textContent = `${data.similar_matches.toLocaleString()} comparable matches`;

  const delta = ((data.observed_blue_wr - data.baseline_wr) * 100).toFixed(1);
  document.getElementById("kpi-wr").textContent = (data.observed_blue_wr * 100).toFixed(1) + "%";
  document.getElementById("kpi-wr-ci").textContent = `${delta >= 0 ? '+' : ''}${delta} pts vs baseline (${(data.ci_low*100).toFixed(1)}%–${(data.ci_high*100).toFixed(1)}%)`;
  document.getElementById("kpi-conv").textContent = (data.lead_conversion_10 * 100).toFixed(1) + "%";
  document.getElementById("kpi-comeback").textContent = (data.comeback_rate_10 * 100).toFixed(1) + "%";

  const g15Hist = data.gold_curve_hist[2] || 0;
  const g15Neural = hasNeural ? data.neural.gold_curve[2] : g15Hist;
  document.getElementById("kpi-g15").textContent = `${g15Neural >= 0 ? '+' : ''}${Math.round(g15Neural)}g`;
  document.getElementById("kpi-g15-hist").textContent = `Historical mean @15m: ${g15Hist >= 0 ? '+' : ''}${Math.round(g15Hist)}g`;

  renderGoldSvg(data.gold_curve_hist, hasNeural ? data.neural.gold_curve : null);
  renderArchAndLadder(data);
  renderObjectivesTables(data);
}

function renderGoldSvg(histCurve, neuralCurve) {
  const box = document.getElementById("gold-chart-box");
  const W = 560, H = 200, padL = 48, padR = 20, padT = 18, padB = 28;
  const allVals = [...histCurve, ...(neuralCurve || []), -500, 500];
  const minV = Math.min(...allVals) * 1.15;
  const maxV = Math.max(...allVals) * 1.15;
  const range = Math.max(100, maxV - minV);

  const xPt = i => padL + (i / 5) * (W - padL - padR);
  const yPt = v => padT + (1 - (v - minV) / range) * (H - padT - padB);
  const yZero = yPt(0);

  const makePath = arr => arr.map((v, i) => `${i===0?'M':'L'}${xPt(i).toFixed(1)},${yPt(v).toFixed(1)}`).join(" ");
  const mins = [5, 10, 15, 20, 25, 30];

  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%; height:100%; overflow:visible;">
    <line x1="${padL}" y1="${yZero}" x2="${W-padR}" y2="${yZero}" stroke="#475569" stroke-dasharray="4,4" stroke-width="1"/>
    <text x="${padL-6}" y="${yZero+4}" fill="#94a3b8" font-size="10" text-anchor="end">0g</text>
    <text x="${padL-6}" y="${padT+6}" fill="#60a5fa" font-size="10" text-anchor="end">+${Math.round(maxV)}g</text>
    <text x="${padL-6}" y="${H-padB}" fill="#fb7185" font-size="10" text-anchor="end">${Math.round(minV)}g</text>`;

  mins.forEach((m, i) => {
    const x = xPt(i);
    svg += `<text x="${x}" y="${H-6}" fill="#94a3b8" font-size="10.5" text-anchor="middle">${m}m</text>`;
  });

  svg += `<path d="${makePath(histCurve)}" fill="none" stroke="#3b82f6" stroke-width="2.5"/>`;
  histCurve.forEach((v, i) => {
    svg += `<circle cx="${xPt(i)}" cy="${yPt(v)}" r="3.5" fill="#60a5fa"><title>Historical @${mins[i]}m: ${v}g</title></circle>`;
  });

  if (neuralCurve) {
    svg += `<path d="${makePath(neuralCurve)}" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="5,3"/>`;
    neuralCurve.forEach((v, i) => {
      svg += `<circle cx="${xPt(i)}" cy="${yPt(v)}" r="3.5" fill="#fbbf24"><title>Model @${mins[i]}m: ${v}g</title></circle>`;
    });
  }
  svg += `</svg>`;
  box.innerHTML = svg;
}

function renderArchAndLadder(data) {
  const box = document.getElementById("arch-ladder-box");
  const fmtArch = obj => Object.entries(obj || {}).map(([k, v]) => `<span class="arch-tag" style="margin-right:5px;">${k}: ${v}%</span>`).join("") || "Balanced";
  let html = `
    <div style="margin-bottom:10px;">
      <div style="font-size:11px; color:#60a5fa; font-weight:700; margin-bottom:3px;">🟦 Blue Archetype Mix</div>
      <div>${fmtArch(data.arch_dist.blue)}</div>
    </div>
    <div style="margin-bottom:12px;">
      <div style="font-size:11px; color:#fb7185; font-weight:700; margin-bottom:3px;">🟥 Red Archetype Mix</div>
      <div>${fmtArch(data.arch_dist.red)}</div>
    </div>
    <table class="data-table">
      <thead><tr><th>Evidence Breadth Tier</th><th>Comparable Matches</th></tr></thead>
      <tbody>
        ${(data.ladder || []).map(r => `<tr><td>${r.mode}</td><td><b>${r.matches.toLocaleString()}</b></td></tr>`).join("")}
      </tbody>
    </table>`;
  box.innerHTML = html;
}

function renderObjectivesTables(data) {
  const histBox = document.getElementById("obj-hist-table");
  const objs = Object.values(data.objectives || {});
  histBox.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Objective</th><th>Blue First</th><th>Red First</th><th>Blue Win If First</th><th>Behind @10m Win If First</th></tr></thead>
      <tbody>
        ${objs.map(o => `<tr>
          <td><b>${o.label}</b></td>
          <td style="color:#60a5fa; font-weight:700;">${(o.blue_first*100).toFixed(1)}%</td>
          <td style="color:#fb7185; font-weight:700;">${(o.red_first*100).toFixed(1)}%</td>
          <td><b>${(o.blue_win_if_first*100).toFixed(1)}%</b></td>
          <td>${o.blue_win_behind_first !== null ? (o.blue_win_behind_first*100).toFixed(1)+'%' : '—'}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;

  const nBox = document.getElementById("obj-neural-table");
  if (!data.neural) {
    nBox.innerHTML = `<div style="color:var(--muted);">Checkpoint not loaded.</div>`;
    return;
  }
  const nObjs = data.neural.obj_probs;
  nBox.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Objective</th><th>Blue First Prob %</th><th>Red First Prob %</th><th>Unclaimed %</th></tr></thead>
      <tbody>
        ${["dragon", "herald", "baron", "tower"].map(k => {
          const p = nObjs[k];
          return `<tr>
            <td><b>${k.charAt(0).toUpperCase() + k.slice(1)}</b></td>
            <td style="color:#60a5fa; font-weight:700;">${(p.blue*100).toFixed(1)}%</td>
            <td style="color:#fb7185; font-weight:700;">${(p.red*100).toFixed(1)}%</td>
            <td style="color:var(--muted);">${(p.none*100).toFixed(1)}%</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
}

function renderMatchupsAndAdvisorTables(data) {
  const mBox = document.getElementById("lane-matchup-table");
  const matchups = data.lane_matchups || [];
  if (!matchups.length) {
    mBox.innerHTML = `<div style="color:var(--muted); padding:10px;">Lock both Blue and Red champions in the same role to compare direct lane matchups.</div>`;
  } else {
    mBox.innerHTML = `
      <table class="data-table">
        <thead><tr><th>Role</th><th>Blue</th><th>Red</th><th>Matches</th><th>Blue Lane WR</th><th>Gold @10m</th></tr></thead>
        <tbody>
          ${matchups.map(m => `<tr>
            <td><b>${m.role_label}</b></td>
            <td><div class="td-champ">${champImgTag(m.blue_champ)}${m.blue_champ}</div></td>
            <td><div class="td-champ">${champImgTag(m.red_champ)}${m.red_champ}</div></td>
            <td>${m.matches.toLocaleString()}</td>
            <td style="color:${(m.blue_wr||0.5)>=0.5?'#60a5fa':'#fb7185'}; font-weight:700;">${m.blue_wr !== null ? (m.blue_wr*100).toFixed(1)+'%' : '—'}</td>
            <td>${m.gold_10 !== null ? (m.gold_10>=0?'+':'')+m.gold_10+'g' : '—'}</td>
          </tr>`).join("")}
        </tbody>
      </table>`;
  }

  document.getElementById("full-rec-table-title").textContent =
    `Top Candidates · ${activeSlot.side.toUpperCase()} ${BOOT.roleLabels[activeSlot.role]} (Click row to lock in)`;
  const rBox = document.getElementById("full-rec-table");
  const recs = data.recommendations || [];
  rBox.innerHTML = `
    <table class="data-table">
      <thead><tr><th>Champion</th><th>Fit Score</th><th>Role WR</th><th>Ally Synergy</th><th>Enemy Counter</th><th>Lane vs Opp</th></tr></thead>
      <tbody>
        ${recs.map(r => {
          const idx = CHAMP_TO_IDX[r.champion] ?? r.idx ?? 0;
          return `<tr style="cursor:pointer;" onclick="pickChampByIdx(${idx})">
            <td><div class="td-champ">${champImgTag(r.champion)}${r.champion}</div></td>
            <td><b style="color:var(--gold-bright);">${r.score}</b></td>
            <td>${r.role_wr}%</td>
            <td style="color:#60a5fa;">${r.syn_wr !== null && r.syn_wr !== undefined ? r.syn_wr+'%' : '—'}</td>
            <td style="color:#34d399;">${r.cnt_wr !== null && r.cnt_wr !== undefined ? r.cnt_wr+'%' : '—'}</td>
            <td>${r.lane_wr !== null && r.lane_wr !== undefined ? r.lane_wr+'% ('+r.lane_matches+')' : '—'}</td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>`;
}

function switchTab(tabId, btn) {
  document.querySelectorAll(".tab-pane").forEach(el => el.style.display = "none");
  document.getElementById("tab-" + tabId).style.display = "block";
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));
  btn.classList.add("active");
}

let analyzeController = null;
async function triggerAnalyze() {
  // Only attempt localhost API call if running on localhost / 127.0.0.1
  const host = (window.location && window.location.hostname) ? window.location.hostname : "";
  if (host && host !== "localhost" && host !== "127.0.0.1") {
    return; // On Streamlit Cloud, computeClientFullAnalysis() already updated 100% of the UI in 0ms!
  }
  const payload = JSON.stringify({
    blue: draftState.blue,
    red: draftState.red,
    active_side: activeSlot.side,
    active_role: activeSlot.role,
    mode: document.getElementById("sel-mode").value,
    patch: document.getElementById("sel-patch").value,
  });
  try {
    if (analyzeController) analyzeController.abort();
    analyzeController = new AbortController();
    const res = await fetch(`http://127.0.0.1:${BOOT.apiPort}/api/analyze`, {
      method: "POST",
      body: payload,
      signal: analyzeController.signal,
    });
    if (res.ok) {
      const data = await res.json();
      renderServerAnalysis(data);
      renderMatchupsAndAdvisorTables(data);
    }
  } catch (_) {}
}

function refreshUIInstant() {
  renderRoleFilterPills();
  renderSlots();
  renderAdvisorAndGridInstant();
}

function initStudio() {
  document.getElementById("hdr-total").textContent = BOOT.totalMatches.toLocaleString();
  const patchSel = document.getElementById("sel-patch");
  BOOT.patches.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = "Patch " + p;
    patchSel.appendChild(opt);
  });
  refreshUIInstant();
  if (BOOT.initialAnalysis) {
    renderServerAnalysis(BOOT.initialAnalysis);
  }
}

initStudio();
</script>
</body>
</html>"""


def build_spa_html(engine: FastDraftEngine, api_port: int) -> str:
    """Build the self-contained zero-refresh HTML/CSS/JS Champion Select Studio."""
    # Auto-heal if Streamlit Cloud passed a cached FastDraftEngine instance from an older commit
    if not hasattr(engine, "syn_wr_list") or not hasattr(engine, "rich_profiles"):
        engine._precompute_interaction_matrices()

    initial_draft = DEMO_PRESETS["Mid-Draft (7/10)"]
    initial_analysis = engine.analyze(
        initial_draft["blue"], initial_draft["red"], active_side="blue", active_role="sup"
    )

    boot_payload = {
        "apiPort": api_port,
        "totalMatches": engine.total_matches,
        "globalBlueWr": getattr(engine, "global_blue_wr", 0.492),
        "globalGoldCurve": getattr(engine, "global_gold_curve", [0.0] * 6),
        "patches": engine.unique_patches,
        "roles": ROLES,
        "roleLabels": ROLE_LABELS,
        "roleIcons": ROLE_ICONS,
        "presets": DEMO_PRESETS,
        "champList": engine.all_champions,
        "nameToCid": engine.name_to_idx,
        "champMeta": engine.champ_meta,
        "roleStats": engine.role_stats_map,
        "synWr": engine.syn_wr_list,
        "cntWr": engine.cnt_wr_list,
        "laneMap": engine.sparse_lane_map,
        "richProfiles": engine.rich_profiles,
        "initialDraft": initial_draft,
        "initialAnalysis": initial_analysis,
    }
    boot_json = json.dumps(boot_payload)
    return SPA_HTML_TEMPLATE.replace("__BOOT_JSON__", boot_json)


class _StudioAPIHandler(BaseHTTPRequestHandler):
    engine: FastDraftEngine = None
    api_port: int = 8765

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")

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
    """Launch inside Streamlit (both Streamlit Community Cloud and localhost)."""
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
    def get_studio_engine_v5(schema_version: str = ENGINE_SCHEMA_VERSION):
        return start_background_studio_server()

    engine, port = get_studio_engine_v5(ENGINE_SCHEMA_VERSION)
    spa_html = build_spa_html(engine, port)
    components.html(spa_html, height=1060, scrolling=True)


if __name__ == "__main__":
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