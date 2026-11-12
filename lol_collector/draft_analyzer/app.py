#!/usr/bin/env python3
"""
Interactive League of Legends Champion Select Studio, Historical Pathways & Neural Advisor.
Features:
  - Zero st.rerun() recursion: all buttons use native Streamlit `on_click` callbacks
    so every interaction updates in a single script pass.
  - Zero @st.cache_data hashing/pickling overhead: uses @st.cache_resource to keep the
    751,339-row match table in memory without recursive hashing or DataFrame copying.
  - 3-Column League Client Layout: Clickable Blue Slots (Left) | Interactive Data Dragon
    Champion Grid & AI Recommendations (Center) | Clickable Red Slots (Right).
"""

from __future__ import annotations

import html
import json
import math
import random
import sys
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ENCODER_DIR = APP_DIR.parent / "draft_encoder"
for _p in (APP_DIR, ENCODER_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis import (
    ARCHETYPE_RULES,
    OBJECTIVES,
    ROLE_LABELS,
    ROLES,
    champion_archetype_tags,
    recommend_bans_from_evidence,
    recommend_role_candidates,
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
    "⚡ Dive vs. Poke (5v5)": {
        "blue": {"top": "Ornn", "jng": "Jarvan IV", "mid": "Orianna", "bot": "Kai'Sa", "sup": "Leona"},
        "red": {"top": "Jayce", "jng": "Nidalee", "mid": "Viktor", "bot": "Ezreal", "sup": "Karma"},
    },
    "🛡️ Scaling vs. Split (5v5)": {
        "blue": {"top": "K'Sante", "jng": "Sejuani", "mid": "Azir", "bot": "Jinx", "sup": "Lulu"},
        "red": {"top": "Fiora", "jng": "Viego", "mid": "Twisted Fate", "bot": "Caitlyn", "sup": "Lux"},
    },
    "🏆 Teamfight vs. Pick (5v5)": {
        "blue": {"top": "Malphite", "jng": "Amumu", "mid": "Syndra", "bot": "Miss Fortune", "sup": "Rell"},
        "red": {"top": "Camille", "jng": "Lee Sin", "mid": "Ahri", "bot": "Jhin", "sup": "Thresh"},
    },
    "🧩 Mid-Draft Advisor (7/10)": {
        "blue": {"top": "Aatrox", "jng": "Lee Sin", "mid": "Ahri", "bot": "Jinx", "sup": ""},
        "red": {"top": "Renekton", "jng": "Vi", "mid": "", "bot": "Caitlyn", "sup": ""},
    },
}

st.set_page_config(
    page_title="LoL Draft Studio",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    .block-container {max-width: 1540px; padding-top: 1.1rem; padding-bottom: 2.5rem;}
    .hero {
        padding: 1rem 1.4rem; border-radius: 14px; margin-bottom: .75rem;
        background: radial-gradient(circle at 15% 20%, #1a2c54 0%, #0d1527 60%, #1a1429 100%);
        border: 1px solid #2e436e; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .8rem;
    }
    .hero h1 {color: #f8fafc; margin: 0; font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em;}
    .hero p {color: #b4c6e7; margin: .2rem 0 0; font-size: .88rem;}
    .kicker {text-transform: uppercase; letter-spacing: .14em; color: #c8aa6e; font-size: .68rem; font-weight: 700;}

    /* Interactive Role Slot Cards */
    .slot-card {
        position: relative; overflow: hidden; border-radius: 11px; height: 68px;
        margin-bottom: .2rem; border: 1.5px solid #283650; background: #0f172a;
        display: flex; align-items: center; padding: 0 .65rem; gap: .65rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25); transition: all 0.15s ease;
    }
    .slot-card.blue {border-left: 4px solid #3b82f6;}
    .slot-card.red {border-right: 4px solid #f43f5e; flex-direction: row-reverse; text-align: right;}
    .slot-card.active-slot {
        border: 2px solid #f59e0b !important;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.42);
    }
    .slot-splash {
        position: absolute; inset: 0; background-size: cover; background-position: center 20%;
        opacity: 0.25; filter: saturate(1.15); z-index: 0;
    }
    .slot-overlay {
        position: absolute; inset: 0; z-index: 1;
        background: linear-gradient(90deg, rgba(15,23,42,0.92) 0%, rgba(15,23,42,0.58) 60%, rgba(15,23,42,0.88) 100%);
    }
    .slot-avatar {
        position: relative; z-index: 2; width: 48px; height: 48px; border-radius: 9px;
        border: 2px solid #c8aa6e; object-fit: cover; flex-shrink: 0; background: #1e293b;
    }
    .slot-avatar.empty {
        border: 1px dashed #475569; display: flex; align-items: center; justify-content: center;
        padding: 10px; opacity: 0.75;
    }
    .slot-info {position: relative; z-index: 2; flex: 1; min-width: 0;}
    .slot-role {
        font-size: .65rem; text-transform: uppercase; letter-spacing: .08em;
        color: #94a3b8; font-weight: 700; display: flex; align-items: center; gap: .3rem;
    }
    .slot-card.red .slot-role {justify-content: flex-end;}
    .slot-champ {
        font-size: .95rem; font-weight: 800; color: #f8fafc; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis; line-height: 1.15;
    }
    .slot-champ.empty {color: #64748b; font-weight: 600; font-style: italic;}
    .slot-sub {
        font-size: .68rem; color: #cbd5e1; white-space: nowrap;
        overflow: hidden; text-overflow: ellipsis; margin-top: .08rem;
    }
    .arch-pill {
        display: inline-block; padding: .05rem .38rem; border-radius: 5px;
        font-size: .62rem; font-weight: 700; background: rgba(200, 170, 110, 0.18);
        border: 1px solid rgba(200, 170, 110, 0.45); color: #f3e5ab;
    }
    .active-tag {
        display: inline-block; padding: .05rem .4rem; border-radius: 5px;
        font-size: .62rem; font-weight: 800; background: #f59e0b; color: #0f172a; margin-left: .3rem;
    }

    /* Center Picker Header */
    .picker-banner {
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #f59e0b; border-radius: 11px; padding: .55rem .9rem;
        margin-bottom: .55rem; display: flex; justify-content: space-between; align-items: center;
    }
    .picker-target {font-weight: 800; font-size: .95rem; color: #fef3c7;}
    .picker-hint {font-size: .76rem; color: #94a3b8;}

    /* Champion Grid Portrait Tile */
    .champ-tile {
        background: #111a2e; border: 1px solid #293856; border-radius: 10px;
        padding: .35rem; text-align: center; margin-bottom: .15rem;
    }
        .champ-pick-link {
            display: block; color: inherit; text-decoration: none; border-radius: 9px;
            transition: transform .12s ease, filter .12s ease;
        }
        .champ-pick-link:hover {transform: translateY(-2px); filter: brightness(1.2);}
        .champ-pick-link:focus-visible {outline: 2px solid #f59e0b; outline-offset: 2px;}
    .champ-tile img {
        width: 54px; height: 54px; border-radius: 8px; border: 1.5px solid #c8aa6e;
        object-fit: cover; display: block; margin: 0 auto .2rem;
    }
    .champ-tile-name {
        font-size: .74rem; font-weight: 700; color: #f1f5f9;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .champ-tile-meta {font-size: .65rem; color: #94a3b8;}

    /* Win Probability Tug-of-War Bar */
    .tug-wrap {
        background: #0f172a; border: 1px solid #2b3954; border-radius: 12px;
        padding: .7rem 1rem; margin: .5rem 0 .9rem;
    }
    .tug-labels {display: flex; justify-content: space-between; font-weight: 800; font-size: .88rem; margin-bottom: .35rem;}
    .tug-bar {
        width: 100%; height: 13px; border-radius: 999px; overflow: hidden;
        background: #f43f5e; display: flex; border: 1px solid #334155;
    }
    .tug-blue {background: linear-gradient(90deg, #2563eb, #60a5fa); height: 100%; transition: width 0.3s ease;}
    </style>
    """,
    unsafe_allow_html=True,
)


def build_champion_metadata(encoder_data_dir: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """
    Resolve vocabulary IDs, Riot IDs, and Data Dragon metadata (square icon, splash art, title).
    """
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
            champ_meta[name] = {
                "name": name,
                "title": entry.get("title", ""),
                "riot_id": entry.get("riot_id"),
                "ddragon_id": ddragon_id,
                "image": square_url,
                "splash": splash_url,
                "archetypes": champion_archetype_tags(name),
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


@st.cache_resource
def load_studio_artifacts():
    """
    Keep the 751,339-row match table and metadata in memory as a shared resource
    so Streamlit never hashes or re-pickles the DataFrame on button clicks.
    """
    table = pd.read_parquet(OUTPUT_DIR / "match_analysis.parquet")
    stats = pd.read_parquet(OUTPUT_DIR / "champion_role_stats.parquet")
    encoder_data_dir = APP_DIR.parent / "draft_encoder" / "data"
    mapping, champ_meta = build_champion_metadata(encoder_data_dir)
    if mapping:
        for role in ROLES:
            for side in ("blue", "red"):
                column = f"{side}_{role}"
                table[column] = table[column].astype(str).map(mapping).fillna(table[column].astype(str))
        stats["champion"] = stats["champion"].astype(str).map(mapping).fillna(stats["champion"].astype(str))
    return table, stats, mapping, champ_meta


@st.cache_resource
def load_neural_predictor():
    """Load trained Wide & Deep DraftModel checkpoint if present."""
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


def run_neural_inference(
    predictor: dict | None, blue_picks: dict[str, str], red_picks: dict[str, str], selected_patch: str
) -> dict | None:
    if predictor is None:
        return None
    name_to_id = predictor["name_to_vocab_id"]
    blue_ids = [name_to_id.get(blue_picks.get(r, ""), 0) for r in ROLES]
    red_ids = [name_to_id.get(red_picks.get(r, ""), 0) for r in ROLES]
    if not any(blue_ids) and not any(red_ids):
        return None

    patch_vocab = predictor["patch_vocab"]
    patch_id = patch_vocab.get(str(selected_patch), patch_vocab.get("<UNK>", 0))
    blue_t = torch.tensor([blue_ids], dtype=torch.long)
    red_t = torch.tensor([red_ids], dtype=torch.long)
    patch_t = torch.tensor([patch_id], dtype=torch.long)

    with torch.inference_mode():
        _, win_logit, gold_pred, obj_logits = predictor["model"](blue_t, red_t, patch_t)
        win_prob = float(torch.sigmoid(win_logit).item())
        gold_curve = (gold_pred[0].cpu().numpy() * predictor["gold_scale"]).tolist()
        obj_probs = {}
        for name, logits in zip(OBJECTIVES, obj_logits):
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            obj_probs[name] = {"none": float(probs[0]), "blue": float(probs[1]), "red": float(probs[2])}

    return {
        "blue_win_prob": win_prob,
        "gold_curve": gold_curve,
        "obj_probs": obj_probs,
    }


def get_champion_image(champ_name: str, champ_meta: dict[str, dict]) -> str:
    if champ_name and champ_name in champ_meta:
        return champ_meta[champ_name]["image"]
    return "https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/-1.png"


def pct(value) -> str:
    return "Unavailable" if pd.isna(value) else f"{float(value) * 100:.1f}%"


def filtered_matches(
    table: pd.DataFrame,
    blue: dict[str, str],
    red: dict[str, str],
    mode: str,
) -> pd.DataFrame:
    mask = pd.Series(True, index=table.index)
    if mode == "Champion presence":
        for champion in [*blue.values(), *red.values()]:
            if champion:
                champion_columns = [f"{side}_{role}" for side in ("blue", "red") for role in ROLES]
                mask &= table[champion_columns].eq(champion).any(axis=1)
        return table.loc[mask]
    for team, picks in (("blue", blue), ("red", red)):
        for role, champion in picks.items():
            if champion:
                if mode == "Role-aware (either side)":
                    mask &= table[[f"blue_{role}", f"red_{role}"]].eq(champion).any(axis=1)
                else:
                    mask &= table[f"{team}_{role}"].eq(champion)
    return table.loc[mask]


def confidence_label(matches: int) -> tuple[str, str]:
    if matches < 30:
        return "Very limited evidence", "Use this as a prompt for exploration; the sample is too small for a stable estimate."
    if matches < 250:
        return "Small evidence base", "Patterns may be noisy. Broaden the draft selection for a more reliable comparison."
    if matches < 1000:
        return "Moderate evidence base", "Useful directional evidence, but not a guaranteed outcome."
    return "Strong evidence base", "This comparison has a substantial historical sample."


def wilson_interval(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float)
    n = len(values)
    if not n:
        return float("nan"), float("nan")
    successes = values.sum()
    z = 1.96
    center = (successes + z * z / 2) / (n + z * z)
    margin = z * math.sqrt(successes * (n - successes) / n + z * z / 4) / (n + z * z)
    return center - margin, center + margin


def resolve_mode(
    table, blue, red, requested: str, minimum: int
) -> tuple[str, pd.DataFrame, int | None]:
    if requested != "Auto (recommended)":
        return requested, filtered_matches(table, blue, red, requested), None
    selected = [
        (team, role, champion)
        for team, picks in (("blue", blue), ("red", red))
        for role, champion in picks.items()
        if champion
    ]
    if not selected:
        return "All matches", table, 0
    for mode in ("Exact roles & sides", "Role-aware (either side)", "Champion presence"):
        conditions = match_conditions(table, blue, red, mode)
        if not conditions:
            continue
        matched = pd.concat(conditions, axis=1).sum(axis=1)
        for required in range(len(conditions), 0, -1):
            if int((matched >= required).sum()) >= minimum:
                return (
                    f"{mode} · at least {required}/{len(selected)} picks",
                    table.loc[matched >= required],
                    required,
                )
    return "Champion presence · at least 1 pick", relaxed_matches(table, blue, red, "Champion presence", 1), 1


def match_conditions(
    table: pd.DataFrame,
    blue: dict[str, str],
    red: dict[str, str],
    mode: str,
) -> list[pd.Series]:
    """Build matching masks once so auto evidence does not rescan the table."""
    conditions = []
    champion_columns = [f"{side}_{role}" for side in ("blue", "red") for role in ROLES]
    if mode == "Champion presence":
        for champion in [*blue.values(), *red.values()]:
            if champion:
                conditions.append(table[champion_columns].eq(champion).any(axis=1))
        return conditions
    for team, picks in (("blue", blue), ("red", red)):
        for role, champion in picks.items():
            if not champion:
                continue
            if mode == "Role-aware (either side)":
                conditions.append(table[[f"blue_{role}", f"red_{role}"]].eq(champion).any(axis=1))
            else:
                conditions.append(table[f"{team}_{role}"].eq(champion))
    return conditions


def relaxed_matches(
    table: pd.DataFrame,
    blue: dict[str, str],
    red: dict[str, str],
    mode: str,
    required: int,
) -> pd.DataFrame:
    conditions = match_conditions(table, blue, red, mode)
    if not conditions:
        return table
    matched = pd.concat(conditions, axis=1).sum(axis=1)
    return table.loc[matched >= required]


def classify_team_fast(table: pd.DataFrame, side: str) -> pd.Series:
    """Fast vectorized archetype classification over up to 5,000 sampled matches."""
    sample_df = table if len(table) <= 5000 else table.sample(n=5000, random_state=42)
    role_cols = [sample_df[f"{side}_{r}"].astype(str).str.casefold() for r in ROLES]

    arch_names = list(ARCHETYPE_RULES.keys())
    counts_by_arch = {}
    for label, pool in ARCHETYPE_RULES.items():
        c = sum(col.isin(pool).astype(int) for col in role_cols)
        counts_by_arch[label] = c

    count_df = pd.DataFrame(counts_by_arch, index=sample_df.index)
    top_arch = count_df.idxmax(axis=1)
    max_val = count_df.max(axis=1)
    return top_arch.where(max_val >= 1, "Balanced / mixed")


def matchup_evidence(
    table: pd.DataFrame, blue: dict[str, str], red: dict[str, str], champ_meta: dict[str, dict]
) -> pd.DataFrame:
    rows = []
    for role in ROLES:
        blue_champion, red_champion = blue.get(role), red.get(role)
        if not blue_champion or not red_champion:
            continue
        mask = table[f"blue_{role}"].eq(blue_champion) & table[f"red_{role}"].eq(red_champion)
        frame = table.loc[mask]
        rows.append({
            "Role": ROLE_LABELS[role],
            "Blue icon": get_champion_image(blue_champion, champ_meta),
            "Blue champion": blue_champion,
            "Red icon": get_champion_image(red_champion, champ_meta),
            "Red champion": red_champion,
            "Matches": len(frame),
            "Blue win rate": pct(frame.blue_win.mean()),
            "Avg gold diff 10m": round(frame.gold_diff_10.mean(), 1) if len(frame) else None,
        })
    return pd.DataFrame(rows)


# ── Pure Session-State Callbacks (Zero st.rerun() Calls!) ─────────────────────

def init_draft_state() -> None:
    """Initialize dedicated draft state dict and active slot cursor."""
    if "draft_state" not in st.session_state:
        default_preset = DEMO_PRESETS["🧩 Mid-Draft Advisor (7/10)"]
        st.session_state["draft_state"] = {
            "blue": {r: default_preset["blue"].get(r, "") for r in ROLES},
            "red": {r: default_preset["red"].get(r, "") for r in ROLES},
        }
    if "active_slot" not in st.session_state:
        st.session_state["active_slot"] = ("blue", "sup")


def cb_target_slot(side: str, role: str) -> None:
    st.session_state["active_slot"] = (side, role)


def cb_clear_slot(side: str, role: str) -> None:
    st.session_state["draft_state"][side][role] = ""
    st.session_state["active_slot"] = (side, role)


def cb_pick_champion(side: str, role: str, champion: str) -> None:
    st.session_state["draft_state"][side][role] = champion
    order = [(s, r) for r in ROLES for s in ("blue", "red")]
    try:
        cur_idx = order.index((side, role))
    except ValueError:
        cur_idx = 0
    for step in range(1, len(order) + 1):
        next_side, next_role = order[(cur_idx + step) % len(order)]
        if not st.session_state["draft_state"][next_side].get(next_role):
            st.session_state["active_slot"] = (next_side, next_role)
            return


def cb_load_preset(preset_name: str) -> None:
    preset = DEMO_PRESETS[preset_name]
    st.session_state["draft_state"] = {
        "blue": {r: preset["blue"].get(r, "") for r in ROLES},
        "red": {r: preset["red"].get(r, "") for r in ROLES},
    }
    for r in ROLES:
        for s in ("blue", "red"):
            if not st.session_state["draft_state"][s][r]:
                st.session_state["active_slot"] = (s, r)
                return
    st.session_state["active_slot"] = ("blue", "top")


def cb_randomize_draft(role_pools: dict[str, list[str]]) -> None:
    used: set[str] = set()
    new_state = {"blue": {}, "red": {}}
    for r in ROLES:
        avail = [c for c in role_pools.get(r, []) if c not in used]
        picks = random.sample(avail, k=min(2, len(avail)))
        if len(picks) == 2:
            new_state["blue"][r] = picks[0]
            new_state["red"][r] = picks[1]
            used.update(picks)
        else:
            new_state["blue"][r] = ""
            new_state["red"][r] = ""
    st.session_state["draft_state"] = new_state
    st.session_state["active_slot"] = ("blue", "top")


def cb_swap_sides() -> None:
    b_copy = dict(st.session_state["draft_state"]["blue"])
    r_copy = dict(st.session_state["draft_state"]["red"])
    st.session_state["draft_state"]["blue"] = r_copy
    st.session_state["draft_state"]["red"] = b_copy


def cb_clear_all() -> None:
    st.session_state["draft_state"] = {
        "blue": {r: "" for r in ROLES},
        "red": {r: "" for r in ROLES},
    }
    st.session_state["active_slot"] = ("blue", "top")


def pick_query_href(side: str, role: str, champion: str) -> str:
    return "?" + urlencode(
        {"pick_side": side, "pick_role": role, "pick_champion": champion}
    )


def target_query_href(side: str, role: str) -> str:
    return "?" + urlencode({"target_side": side, "target_role": role})


def apply_query_action(champ_meta: dict[str, dict]) -> None:
    """Apply a portrait/link action once, then remove it from the URL."""
    if hasattr(st, "query_params"):
        params = st.query_params
    else:
        params = st.experimental_get_query_params()

    def value(key: str) -> str | None:
        item = params.get(key)
        if isinstance(item, list):
            return item[0] if item else None
        return item

    pick_side = value("pick_side")
    pick_role = value("pick_role")
    pick_champion = value("pick_champion")
    target_side = value("target_side")
    target_role = value("target_role")

    if pick_side in ("blue", "red") and pick_role in ROLES and pick_champion in champ_meta:
        selected = {
            value
            for side in ("blue", "red")
            for role, value in st.session_state["draft_state"][side].items()
            if value and (side, role) != (pick_side, pick_role)
        }
        if pick_champion not in selected:
            cb_pick_champion(pick_side, pick_role, pick_champion)
    elif target_side in ("blue", "red") and target_role in ROLES:
        cb_target_slot(target_side, target_role)

    if pick_side or target_side:
        if hasattr(st, "query_params"):
            st.query_params.clear()
        else:
            st.experimental_set_query_params()


def render_slot_card_html(
    side: str, role: str, champion: str, is_active: bool, champ_meta: dict[str, dict]
) -> str:
    role_label = html.escape(ROLE_LABELS[role])
    role_icon = ROLE_ICONS[role]
    active_cls = " active-slot" if is_active else ""
    active_badge = '<span class="active-tag">ACTIVE SLOT</span>' if is_active else ""

    if champion and champion in champ_meta:
        meta = champ_meta[champion]
        esc_name = html.escape(champion)
        esc_title = html.escape(meta.get("title", ""))
        esc_arch = html.escape(meta.get("archetypes", "Flexible"))
        img_url = meta["image"]
        splash_url = meta["splash"]
        return (
            f'<div class="slot-card {side}{active_cls}">'
            f'<div class="slot-splash" style="background-image: url(\'{splash_url}\');"></div>'
            f'<div class="slot-overlay"></div>'
            f'<img class="slot-avatar" src="{img_url}" alt="{esc_name}">'
            f'<div class="slot-info">'
            f'<div class="slot-role"><img src="{role_icon}" width="13" height="13"> {role_label} {active_badge}</div>'
            f'<div class="slot-champ">{esc_name}</div>'
            f'<div class="slot-sub">{esc_title} · <span class="arch-pill">{esc_arch}</span></div>'
            f'</div></div>'
        )
    return (
        f'<div class="slot-card {side}{active_cls}">'
        f'<div class="slot-overlay"></div>'
        f'<div class="slot-avatar empty"><img src="{role_icon}" width="24" height="24"></div>'
        f'<div class="slot-info">'
        f'<div class="slot-role">{role_label} {active_badge}</div>'
        f'<div class="slot-champ empty">Open Slot</div>'
        f'<div class="slot-sub">Click button below to target this slot</div>'
        f'</div></div>'
    )


def render_interactive_team_column(
    side: str,
    picks: dict[str, str],
    active_slot: tuple[str, str],
    champ_meta: dict[str, dict],
) -> None:
    side_title = "🟦 BLUE TEAM" if side == "blue" else "🟥 RED TEAM"
    filled = sum(bool(v) for v in picks.values())
    st.markdown(f"**{side_title}** ({filled}/5 locked)")

    for role in ROLES:
        champ = picks.get(role, "")
        is_active = active_slot == (side, role)
        target_href = html.escape(target_query_href(side, role), quote=True)
        card_html = render_slot_card_html(side, role, champ, is_active, champ_meta)
        st.markdown(
            f'<a class="champ-pick-link" href="{target_href}" '
            f'aria-label="Edit {html.escape(side.title())} {html.escape(ROLE_LABELS[role])}">'
            f"{card_html}</a>",
            unsafe_allow_html=True,
        )
        if champ:
            b1, b2 = st.columns([4, 1])
            with b1:
                btn_label = f"🎯 Editing {ROLE_LABELS[role]}" if is_active else f"✏️ Change {ROLE_LABELS[role]} ({champ})"
                st.button(
                    btn_label,
                    key=f"target_{side}_{role}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                    on_click=cb_target_slot,
                    args=(side, role),
                )
            with b2:
                st.button(
                    "✖",
                    key=f"clear_{side}_{role}",
                    help=f"Clear {side.title()} {ROLE_LABELS[role]}",
                    use_container_width=True,
                    on_click=cb_clear_slot,
                    args=(side, role),
                )
        else:
            btn_label = f"🎯 Picking {side.title()} {ROLE_LABELS[role]}..." if is_active else f"➕ Pick {side.title()} {ROLE_LABELS[role]}"
            st.button(
                btn_label,
                key=f"target_{side}_{role}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
                on_click=cb_target_slot,
                args=(side, role),
            )


def main() -> None:
    table, stats, mapping, champ_meta = load_studio_artifacts()
    predictor = load_neural_predictor()
    init_draft_state()
    apply_query_action(champ_meta)

    all_champions = sorted(
        set(champ_meta.keys()) if champ_meta else (
            set(mapping.values()) if mapping else {
                str(value) for role in ROLES for side in ("blue", "red")
                for value in table[f"{side}_{role}"].unique()
            }
        ),
        key=str.casefold,
    )

    role_pools = {
        r: stats[stats["role"] == r].sort_values("matches", ascending=False).head(25)["champion"].tolist()
        for r in ROLES
    }

    st.markdown(
        '<div class="hero">'
        '<div><div class="kicker">Riot Data Dragon · 751,339 Matches · Wide & Deep Neural Encoder</div>'
        '<h1>⚔️ LoL Champion Select Studio & Draft Pathways</h1>'
        '<p>Click any Blue or Red role slot below, then click a champion portrait in the center grid (or load a 1-click preset).</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Top 1-Click Preset & Utility Bar (Using Native Callbacks) ──
    preset_names = list(DEMO_PRESETS.keys())
    top_cols = st.columns([1.3, 1.3, 1.3, 1.3, 1.0, 0.9, 0.9])
    for idx, p_name in enumerate(preset_names):
        with top_cols[idx]:
            st.button(
                p_name,
                key=f"preset_btn_{idx}",
                use_container_width=True,
                on_click=cb_load_preset,
                args=(p_name,),
            )
    with top_cols[4]:
        st.button(
            "🎲 Random Draft",
            key="btn_random_draft",
            use_container_width=True,
            on_click=cb_randomize_draft,
            args=(role_pools,),
        )
    with top_cols[5]:
        st.button(
            "⇄ Swap Sides",
            key="btn_swap_sides",
            use_container_width=True,
            on_click=cb_swap_sides,
        )
    with top_cols[6]:
        st.button(
            "🗑️ Clear All",
            key="btn_clear_all",
            use_container_width=True,
            on_click=cb_clear_all,
        )

    blue_picks: dict[str, str] = st.session_state["draft_state"]["blue"]
    red_picks: dict[str, str] = st.session_state["draft_state"]["red"]
    active_side, active_role = st.session_state["active_slot"]

    with st.sidebar:
        st.header("🔍 Evidence & Patch Settings")
        comparison_mode = st.selectbox(
            "Comparison breadth",
            ["Auto (recommended)", "Role-aware (either side)", "Exact roles & sides", "Champion presence"],
        )
        minimum_evidence = st.slider(
            "Minimum evidence (matches)",
            min_value=30,
            max_value=2000,
            value=250,
            step=10,
        )
        patch_options = ["All patches"] + sorted(table["patch"].dropna().astype(str).unique(), reverse=True)
        selected_patch = st.selectbox("Patch scope", patch_options)

    draft_key = (
        selected_patch,
        comparison_mode,
        minimum_evidence,
        active_side,
        active_role,
        tuple((side, role, blue_picks[role] if side == "blue" else red_picks[role])
              for side in ("blue", "red") for role in ROLES),
    )
    if st.session_state.get("_analysis_key") != draft_key:
        scoped_table = (
            table
            if selected_patch == "All patches"
            else table[table["patch"].astype(str).eq(selected_patch)]
        )
        resolved_mode, similar, matched_picks = resolve_mode(
            scoped_table, blue_picks, red_picks, comparison_mode, minimum_evidence
        )
        neural_out = run_neural_inference(predictor, blue_picks, red_picks, selected_patch)
        slot_recs = recommend_role_candidates(
            scoped_table=scoped_table,
            similar_table=similar if not similar.empty else scoped_table,
            stats=stats,
            blue_picks=blue_picks,
            red_picks=red_picks,
            target_side=active_side,
            target_role=active_role,
            min_role_matches=15,
            top_k=4,
        )
        st.session_state["_analysis_key"] = draft_key
        st.session_state["_analysis_cache"] = (
            scoped_table,
            resolved_mode,
            similar,
            matched_picks,
            neural_out,
            slot_recs,
        )
    else:
        (
            scoped_table,
            resolved_mode,
            similar,
            matched_picks,
            neural_out,
            slot_recs,
        ) = st.session_state["_analysis_cache"]

    # ── 3-Column League Client Champion Select Board ──
    col_blue, col_center, col_red = st.columns([1.05, 1.9, 1.05], gap="medium")

    with col_blue:
        render_interactive_team_column("blue", blue_picks, (active_side, active_role), champ_meta)

    with col_red:
        render_interactive_team_column("red", red_picks, (active_side, active_role), champ_meta)

    with col_center:
        side_badge = "🟦 BLUE" if active_side == "blue" else "🟥 RED"
        current_in_slot = st.session_state["draft_state"][active_side].get(active_role) or "Open Slot"
        st.markdown(
            f'<div class="picker-banner">'
            f'<div class="picker-target">🎯 Active Slot: {side_badge} · {ROLE_LABELS[active_role].upper()} ({html.escape(current_in_slot)})</div>'
            f'<div class="picker-hint">Click any champion below to lock in</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if not slot_recs.empty:
            st.caption(f"✨ **Top AI Recommendations for {active_side.title()} {ROLE_LABELS[active_role]}** (1-click lock-in):")
            rec_cols = st.columns(len(slot_recs), gap="small")
            for r_idx, row in enumerate(slot_recs.itertuples(index=False)):
                c_name = str(row.Champion)
                c_img = get_champion_image(c_name, champ_meta)
                c_score = getattr(row, "_3", 50.0)
                c_wr = getattr(row, "_4", "—")
                with rec_cols[r_idx]:
                    href = html.escape(
                        pick_query_href(active_side, active_role, c_name), quote=True
                    )
                    st.markdown(
                        f'<a class="champ-pick-link" href="{href}" '
                        f'aria-label="Pick {html.escape(c_name)}">'
                        f'<div class="champ-tile">'
                        f'<img src="{c_img}" alt="{html.escape(c_name)}">'
                        f'<div class="champ-tile-name">{html.escape(c_name)}</div>'
                        f'<div class="champ-tile-meta">Score {c_score} · {html.escape(str(c_wr))}</div>'
                        f'</div></a>',
                        unsafe_allow_html=True,
                    )

        f_col1, f_col2, f_col3 = st.columns([1.2, 1.1, 1.1], gap="small")
        with f_col1:
            search_query = st.text_input(
                "🔍 Search champion",
                placeholder="Type champion name (e.g. Ahri, Ornn, Jinx)...",
                label_visibility="collapsed",
            )
        with f_col2:
            role_filter = st.selectbox(
                "Role filter",
                ["Active Slot Role", "All 172 Champions", "Top", "Jungle", "Mid", "Bot", "Support"],
                label_visibility="collapsed",
            )
        with f_col3:
            arch_filter = st.selectbox(
                "Archetype filter",
                ["All Archetypes", "Engage", "Poke", "Scaling", "Split push"],
                label_visibility="collapsed",
            )

        already_picked = {
            c for s in ("blue", "red") for c in st.session_state["draft_state"][s].values() if c
        }

        role_map_reverse = {"Top": "top", "Jungle": "jng", "Mid": "mid", "Bot": "bot", "Support": "sup"}
        effective_role = (
            active_role
            if role_filter == "Active Slot Role"
            else role_map_reverse.get(role_filter)
        )

        if effective_role and not search_query.strip():
            role_sub = stats[(stats["role"] == effective_role) & (stats["matches"] >= 10)].sort_values(
                ["matches", "win_rate"], ascending=[False, False]
            )
            ordered_champs = [c for c in role_sub["champion"].tolist() if c in champ_meta]
            if not ordered_champs:
                ordered_champs = list(all_champions)
        else:
            ordered_champs = list(all_champions)

        if search_query.strip():
            q = search_query.strip().casefold()
            ordered_champs = [c for c in all_champions if q in c.casefold()]

        if arch_filter != "All Archetypes":
            pool_set = ARCHETYPE_RULES.get(arch_filter, set())
            ordered_champs = [c for c in ordered_champs if c.casefold() in pool_set]

        role_wr_lookup: dict[str, str] = {}
        target_stat_role = effective_role or active_role
        for r_row in stats[stats["role"] == target_stat_role].itertuples(index=False):
            role_wr_lookup[str(r_row.champion)] = f"{float(r_row.win_rate) * 100:.1f}%"

        display_champs = ordered_champs[:18]
        st.caption(
            f"Showing **{len(display_champs)}** of **{len(ordered_champs)}** matching champions "
            f"({'filtered to ' + ROLE_LABELS[effective_role] if effective_role and not search_query.strip() else 'All Roles'}):"
        )
        if not display_champs:
            st.info("No champions match that search/filter. Try clearing the search box or switching to 'All 172 Champions'.")
        else:
            grid_cols = st.columns(6, gap="small")
            for idx, c_name in enumerate(display_champs):
                c_img = get_champion_image(c_name, champ_meta)
                c_wr = role_wr_lookup.get(c_name, champ_meta.get(c_name, {}).get("archetypes", "Flex")[:10])
                is_taken = c_name in already_picked and c_name != current_in_slot
                with grid_cols[idx % 6]:
                    tile = (
                        f'<div class="champ-tile" style="opacity: '
                        f'{"0.4" if is_taken else "1.0"};">'
                        f'<img src="{c_img}" alt="{html.escape(c_name)}">'
                        f'<div class="champ-tile-name">{html.escape(c_name)}</div>'
                        f'<div class="champ-tile-meta">{html.escape(str(c_wr))}</div>'
                        f'</div>'
                    )
                    if is_taken:
                        st.markdown(tile, unsafe_allow_html=True)
                    else:
                        href = html.escape(
                            pick_query_href(active_side, active_role, c_name), quote=True
                        )
                        st.markdown(
                            f'<a class="champ-pick-link" href="{href}" '
                            f'aria-label="Pick {html.escape(c_name)}">{tile}</a>',
                            unsafe_allow_html=True,
                        )

    # ── Live Summary Metrics & Tug-of-War Win Bar ──
    selected_count = sum(bool(v) for v in [*blue_picks.values(), *red_picks.values()])
    overview = st.columns(5)
    overview[0].metric("Dataset", f"{len(table):,} matches")
    overview[1].metric("Patches", f"{table.patch.nunique():,}")
    overview[2].metric("Comparable Matches", f"{len(similar):,}")
    label, explanation = confidence_label(len(similar))
    overview[3].metric("Evidence Tier", label)
    if neural_out is not None:
        overview[4].metric(
            "Neural Model Blue Win",
            f"{neural_out['blue_win_prob'] * 100:.1f}%",
            f"{(neural_out['blue_win_prob'] - 0.5) * 100:+.1f} pts vs 50%",
        )
    else:
        overview[4].metric("Selected Picks", f"{selected_count}/10")

    if not selected_count:
        st.info("Click any preset button at the top or click a champion portrait in the center grid to begin analyzing.")
        st.stop()

    hist_win = float(similar.blue_win.mean()) if not similar.empty else 0.5
    display_win = neural_out["blue_win_prob"] if (neural_out is not None and selected_count >= 4) else hist_win
    source_tag = "Wide & Deep Neural Model + Historical" if (neural_out is not None and selected_count >= 4) else "Historical Empirical"
    st.markdown(
        f'<div class="tug-wrap">'
        f'<div class="tug-labels">'
        f'<span style="color:#60a5fa;">🟦 Blue Win Estimate: {display_win*100:.1f}% ({source_tag})</span>'
        f'<span style="color:#94a3b8; font-size:.8rem;">{resolved_mode} · {selected_patch}</span>'
        f'<span style="color:#fb7185;">Red Win Estimate: {(1-display_win)*100:.1f}% 🟥</span>'
        f'</div>'
        f'<div class="tug-bar"><div class="tug-blue" style="width: {display_win*100:.1f}%;"></div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if similar.empty:
        st.warning("No historical matches support this comparison. Clear one or two picks, or broaden the comparison mode in the sidebar.")
        st.stop()

    tabs = st.tabs([
        "🎯 Full Pick & Ban Advisor Table",
        "📈 Pathway to Victory & Gold Curve",
        "🐉 Objective Control",
        "⚔️ Lane Matchups & Archetypes",
        "🧠 Neural Encoder Multi-Task Breakdown",
        "🌍 Patch Trends & Pick Evidence",
    ])

    with tabs[0]:
        st.subheader(f"Detailed Pick & Ban Tables · Currently Targeting {active_side.title()} {ROLE_LABELS[active_role]}")
        pick_candidates = recommend_role_candidates(
            scoped_table=scoped_table,
            similar_table=similar,
            stats=stats,
            blue_picks=blue_picks,
            red_picks=red_picks,
            target_side=active_side,
            target_role=active_role,
            min_role_matches=15,
            top_k=10,
        )
        if not pick_candidates.empty:
            pick_display = pick_candidates.copy()
            pick_display.insert(0, "Portrait", pick_display["Champion"].map(lambda c: get_champion_image(c, champ_meta)))
            st.dataframe(
                pick_display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Portrait": st.column_config.ImageColumn("Icon", width="small"),
                    "Composite score": st.column_config.ProgressColumn(
                        "Composite score", min_value=40.0, max_value=65.0, format="%.1f"
                    ),
                },
            )

        ban_candidates = recommend_bans_from_evidence(
            scoped_table=scoped_table,
            similar_table=similar,
            stats=stats,
            blue_picks=blue_picks,
            red_picks=red_picks,
            target_side=active_side,
            min_role_matches=25,
            top_k=6,
        )
        st.markdown("#### Priority Ban Targets · Remaining Enemy Open Roles")
        if not ban_candidates.empty:
            ban_display = ban_candidates.copy()
            ban_display.insert(0, "Portrait", ban_display["Champion"].map(lambda c: get_champion_image(c, champ_meta)))
            st.dataframe(
                ban_display,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Portrait": st.column_config.ImageColumn("Icon", width="small"),
                    "Threat score": st.column_config.ProgressColumn(
                        "Threat score", min_value=40.0, max_value=65.0, format="%.1f"
                    ),
                },
            )

    with tabs[1]:
        st.subheader("What happened in comparable historical matches?")
        cards = st.columns(4)
        win_rate = similar.blue_win.mean()
        win_low, win_high = wilson_interval(similar.blue_win)
        baseline = scoped_table.blue_win.mean()
        cards[0].metric(
            "Observed Blue win rate",
            pct(win_rate),
            f"{(win_rate - baseline) * 100:+.1f} pts vs dataset",
        )
        cards[1].metric("First Dragon", pct(similar.blue_first_dragon.mean()))
        cards[2].metric("First Tower", pct(similar.blue_first_tower.mean()))
        cards[3].metric(
            "Comeback when behind @10m",
            pct(similar.loc[~similar.blue_ahead_10, "blue_win"].mean()),
        )
        st.caption(
            f"Blue win-rate 95% confidence interval: {pct(win_low)}–{pct(win_high)} "
            f"(dataset baseline: {pct(baseline)})."
        )

        st.subheader("Gold Differential Trajectory (5m → 30m)")
        gold_data = {
            "Minute": [5, 10, 15, 20, 25, 30],
            "Historical Mean Gold Diff": [
                float(similar[f"gold_diff_{minute}"].mean()) for minute in (5, 10, 15, 20, 25, 30)
            ],
        }
        if neural_out is not None:
            gold_data["Neural Predicted Gold Diff"] = neural_out["gold_curve"]
        gold_df = pd.DataFrame(gold_data).set_index("Minute")
        st.line_chart(gold_df)
        ahead = similar.loc[similar.blue_ahead_10, "blue_win"]
        conversion = ahead.mean()
        conversion_low, conversion_high = wilson_interval(ahead)
        st.info(
            f"When Blue was ahead at 10 minutes in this comparison, it converted {pct(conversion)} "
            f"of those leads into wins (95% CI: {pct(conversion_low)}–{pct(conversion_high)})."
        )

    with tabs[2]:
        st.subheader("First Objective Control & Conditional Win Rates")
        objective_frame = pd.DataFrame(
            {
                "Objective": [OBJECTIVE_LABELS[name] for name in OBJECTIVES],
                "Blue": [similar[f"blue_first_{name}"].mean() for name in OBJECTIVES],
                "Red": [similar[f"red_first_{name}"].mean() for name in OBJECTIVES],
            }
        ).set_index("Objective")
        st.bar_chart(objective_frame, color=["#4b9dff", "#ff667c"])

        state_rows = []
        for objective in OBJECTIVES:
            available = similar[similar[f"first_{objective}_team"].isin([100, 200])].copy()
            for state, state_mask in (
                ("Blue ahead at 10m", available.blue_ahead_10),
                ("Blue behind at 10m", ~available.blue_ahead_10),
            ):
                state_frame = available.loc[state_mask]
                state_rows.append({
                    "Objective": OBJECTIVE_LABELS[objective],
                    "Game State": state,
                    "Matches": len(state_frame),
                    "Blue First Rate": pct(state_frame[f"blue_first_{objective}"].mean()),
                    "Blue Win If Secured": pct(
                        state_frame.loc[state_frame[f"blue_first_{objective}"], "blue_win"].mean()
                    ),
                })
        st.dataframe(pd.DataFrame(state_rows), hide_index=True, use_container_width=True)

    with tabs[3]:
        st.subheader("Head-to-Head Lane Matchups")
        matchup = matchup_evidence(scoped_table, blue_picks, red_picks, champ_meta)
        if matchup.empty:
            st.info("Select both Blue and Red champions in at least one shared lane to compare head-to-head stats.")
        else:
            st.dataframe(
                matchup,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Blue icon": st.column_config.ImageColumn("Blue", width="small"),
                    "Red icon": st.column_config.ImageColumn("Red", width="small"),
                },
            )

        st.subheader("Strategic Composition Identity (172-Champion Taxonomy)")
        id_cols = st.columns(2)
        for idx, side in enumerate(("blue", "red")):
            with id_cols[idx]:
                classified = classify_team_fast(similar, side)
                counts = classified.value_counts().rename_axis("Archetype Mix").reset_index(name="Matches")
                counts["Share (%)"] = (counts["Matches"] / len(classified) * 100).round(1)
                st.markdown(f"**{side.title()} Side Archetype Distribution**")
                st.dataframe(
                    counts,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Share (%)": st.column_config.ProgressColumn("Share (%)", min_value=0, max_value=100, format="%.1f%%")
                    },
                )

    with tabs[4]:
        st.subheader("Wide & Deep Multi-Task Neural Encoder (`checkpoints/best.pt`)")
        if neural_out is None:
            st.warning("Trained checkpoint not found at `lol_collector/draft_encoder/checkpoints/best.pt`.")
        else:
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric("Neural Blue Win Probability", f"{neural_out['blue_win_prob']*100:.1f}%")
            nc2.metric("Predicted 10m Gold Diff", f"{neural_out['gold_curve'][1]:+.0f}g")
            nc3.metric("Predicted 20m Gold Diff", f"{neural_out['gold_curve'][3]:+.0f}g")

            obj_rows = []
            for name in OBJECTIVES:
                p = neural_out["obj_probs"][name]
                obj_rows.append({
                    "Objective": OBJECTIVE_LABELS[name],
                    "Blue First Prob (%)": round(p["blue"] * 100, 1),
                    "Red First Prob (%)": round(p["red"] * 100, 1),
                    "Neither / Unclaimed (%)": round(p["none"] * 100, 1),
                })
            st.dataframe(
                pd.DataFrame(obj_rows),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Blue First Prob (%)": st.column_config.ProgressColumn("Blue First (%)", min_value=0, max_value=100, format="%.1f%%"),
                    "Red First Prob (%)": st.column_config.ProgressColumn("Red First (%)", min_value=0, max_value=100, format="%.1f%%"),
                },
            )

    with tabs[5]:
        st.subheader("Individual Pick Evidence & Meta Trends")
        selected_rows = []
        for team, picks in (("Blue", blue_picks), ("Red", red_picks)):
            for role, champion in picks.items():
                if champion:
                    row = stats[(stats.role == role) & (stats.champion == champion)]
                    selected_rows.append({
                        "Icon": get_champion_image(champion, champ_meta),
                        "Side": team,
                        "Role": ROLE_LABELS[role],
                        "Champion": champion,
                        "Archetypes": champion_archetype_tags(champion),
                        "Historical Picks": int(row.matches.iloc[0]) if not row.empty else 0,
                        "Role Win Rate": pct(row.win_rate.iloc[0]) if not row.empty else "Unavailable",
                        "Avg Gold @10m": round(float(row.mean_gold_diff_10.iloc[0]), 1) if not row.empty else None,
                    })
        st.dataframe(
            pd.DataFrame(selected_rows),
            hide_index=True,
            use_container_width=True,
            column_config={"Icon": st.column_config.ImageColumn("Icon", width="small")},
        )

        patch_summary = (
            scoped_table.groupby("patch", as_index=False)
            .agg(
                matches=("match_id", "size"),
                blue_win_rate=("blue_win", "mean"),
                gold_diff_10=("gold_diff_10", "mean"),
                first_dragon=("blue_first_dragon", "mean"),
                first_tower=("blue_first_tower", "mean"),
            )
            .sort_values("patch")
        )
        st.line_chart(patch_summary.set_index("patch")[["blue_win_rate", "first_dragon", "first_tower"]])


if __name__ == "__main__":
    main()