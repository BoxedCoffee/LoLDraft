#!/usr/bin/env python3
"""Streamlit interface for League of Legends draft prediction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from model import DraftModel


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_PATH = APP_DIR / "checkpoints" / "best.pt"
ROLES = ["Top", "Jungle", "Mid", "Bot", "Support"]
OBJECTIVES = ["Dragon", "Herald", "Baron", "Tower"]
ROLE_ICONS = {
    "Top": (
        "https://wiki.leagueoflegends.com/en-us/images/thumb/Top_icon.png/120px-Top_icon.png",
        "Top lane",
    ),
    "Jungle": (
        "https://wiki.leagueoflegends.com/en-us/images/thumb/Jungle_icon.png/120px-Jungle_icon.png",
        "Jungle",
    ),
    "Mid": (
        "https://wiki.leagueoflegends.com/en-us/images/thumb/Middle_icon.png/120px-Middle_icon.png",
        "Mid lane",
    ),
    "Bot": (
        "https://wiki.leagueoflegends.com/en-us/images/thumb/Bottom_icon.png/120px-Bottom_icon.png",
        "Bot lane",
    ),
    "Support": (
        "https://wiki.leagueoflegends.com/en-us/images/thumb/Support_icon.png/120px-Support_icon.png",
        "Support",
    ),
}

st.set_page_config(
    page_title="LoL Draft Predictor",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_resources():
    """Load the checkpoint, model vocabulary, and complete local champion roster."""
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    champion_vocab = checkpoint["champion_vocab"]
    patch_vocab = checkpoint.get("patch_vocab") or {}
    model_cfg = checkpoint.get("config", {}).get("model", {})

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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(DATA_DIR / "champion_data.json", encoding="utf-8") as file:
        champion_data = json.load(file)
    with open(DATA_DIR / "full_champion_mapping.json", encoding="utf-8") as file:
        champion_id_mapping = json.load(file)
    artifact_dirs = [PROCESSED_DIR, DATA_DIR / "processed_riot"]
    artifact_dir = PROCESSED_DIR
    artifact_score = -1
    for candidate in artifact_dirs:
        vocab_path = candidate / "champion_vocab.json"
        if not vocab_path.exists():
            continue
        with open(vocab_path, encoding="utf-8") as file:
            candidate_vocab = json.load(file)
        if len(candidate_vocab) != len(champion_vocab):
            continue
        candidate_objectives = candidate / "objectives.parquet"
        score = 0
        if candidate_objectives.exists():
            objective_frame = pd.read_parquet(candidate_objectives)
            score = sum(
                (
                    f"first_{objective.lower()}_team" in objective_frame
                    and (objective_frame[f"first_{objective.lower()}_team"] > 0).any()
                )
                for objective in OBJECTIVES
            )
        if score > artifact_score:
            artifact_dir = candidate
            artifact_score = score
    objectives_path = artifact_dir / "objectives.parquet"
    objective_support = {}
    if objectives_path.exists():
        objectives = pd.read_parquet(objectives_path)
        for objective in OBJECTIVES:
            column = f"first_{objective.lower()}_team"
            objective_support[objective] = bool(
                column in objectives
                and (objectives[column].fillna(0).astype(int) > 0).any()
            )

    roster = {}
    for entry in champion_data.values():
        name = entry.get("name")
        if name:
            roster[name] = entry

    # The trained vocabulary is authoritative for model inputs. The full
    # roster is still shown so users can see every available champion.
    name_to_model_id = {
        name: champion_vocab[str(riot_id)]
        for name, riot_id in champion_id_mapping.items()
        if str(riot_id) in champion_vocab and champion_vocab[str(riot_id)] > 0
    }
    champion_names = sorted(roster, key=str.casefold)
    return (
        model,
        patch_vocab,
        roster,
        name_to_model_id,
        champion_names,
        objective_support,
        checkpoint.get("checkpoint_metric"),
    )


def reset_draft() -> None:
    for team in ("blue", "red"):
        for index in range(5):
            st.session_state[f"{team}_{index}"] = ""
    st.session_state.pop("prediction", None)


def champion_label(name: str, supported: dict[str, int]) -> str:
    if not name:
        return "— Select a champion —"
    if name in supported:
        return name
    return f"{name} · unavailable in current model"


def render_team(team: str, title: str, color: str, champion_names, supported, roster):
    """Render one team with searchable image-grid champion pickers."""
    st.markdown(
        f'<div class="team-heading {color}"><span>{title}</span>'
        f"<small>Choose one champion per role</small></div>",
        unsafe_allow_html=True,
    )
    selections = []
    for index, role in enumerate(ROLES):
        key = f"{team}_{index}"
        current = st.session_state.get(key, "")
        role_image, role_controls = st.columns([1, 2.8], gap="medium")
        with role_image:
            if current and current in roster:
                st.markdown(
                    f'<div class="selected-pick {team}-pick">'
                    f'<img src="{roster[current].get("image")}" alt="{current}">'
                    f'<span><strong>{role}</strong> · {current}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="empty-pick {team}-pick"><img class="role-icon" '
                    f'src="{ROLE_ICONS[role][0]}" alt="{ROLE_ICONS[role][1]}">'
                    f'<span>{role}</span>'
                    f'<strong>Open slot</strong></div>',
                    unsafe_allow_html=True,
                )
        with role_controls:
            st.markdown(f"**{role}** · {current or 'Choose a champion'}")
            search_key = f"{team}_{index}_search"
            search = st.text_input(
                f"Search {role} champions",
                key=search_key,
                placeholder="Search by champion name...",
                label_visibility="collapsed",
            )
        query = search.casefold().strip()
        filtered_names = [
            name for name in champion_names if not query or query in name.casefold()
        ]
        if not filtered_names:
            st.caption("No champions match that search.")
        else:
            with st.expander(
                f"Choose {role} champion · {len(filtered_names)} matches",
                expanded=bool(search),
            ):
                grid = st.columns(6, gap="small")
                picked = {
                    value
                    for other_team in ("blue", "red")
                    for other_index in range(5)
                    if (value := st.session_state.get(f"{other_team}_{other_index}", ""))
                }
                for champion_index, name in enumerate(filtered_names):
                    entry = roster[name]
                    with grid[champion_index % 6]:
                        st.image(entry.get("image"), use_column_width=True)
                        already_picked = name in picked and name != current
                        button_label = "Selected" if name == current else name
                        if st.button(
                            button_label,
                            key=f"{team}_{index}_champion_{name}",
                            disabled=already_picked,
                            use_container_width=True,
                            help=(
                                "Already selected in this draft."
                                if already_picked
                                else f"Pick {name} for {role}."
                            ),
                        ):
                            st.session_state[key] = name
                            st.session_state.pop("prediction", None)
                            st.rerun()
                if current and st.button(
                    f"Clear {role} pick", key=f"{team}_{index}_clear"
                ):
                    st.session_state[key] = ""
                    st.session_state.pop("prediction", None)
                    st.rerun()
        selected = st.session_state.get(key, "")
        selections.append(selected)
    return selections


def validate_draft(blue_team, red_team, supported):
    errors = []
    all_picks = blue_team + red_team
    if not all(blue_team) or not all(red_team):
        errors.append("Select all five champions for both teams.")
    duplicates = sorted(
        {name for name in all_picks if name and all_picks.count(name) > 1},
        key=str.casefold,
    )
    if duplicates:
        errors.append(f"Champions cannot be picked twice: {', '.join(duplicates)}.")
    unsupported = sorted(
        {name for name in all_picks if name and name not in supported},
        key=str.casefold,
    )
    if unsupported:
        errors.append(
            "The current checkpoint does not contain: "
            + ", ".join(unsupported)
            + ". Retrain the model with the complete roster to use them."
        )
    return errors


def predict(model, patch_vocab, name_to_model_id, blue_team, red_team):
    blue_ids = [name_to_model_id[name] for name in blue_team]
    red_ids = [name_to_model_id[name] for name in red_team]
    patch_id = patch_vocab.get("<UNK>", 0)
    blue_tensor = torch.tensor([blue_ids], dtype=torch.long)
    red_tensor = torch.tensor([red_ids], dtype=torch.long)
    patch_tensor = torch.tensor([patch_id], dtype=torch.long)

    with torch.inference_mode():
        embedding, win_logit, gold_pred, objective_logits = model(
            blue_tensor, red_tensor, patch_tensor
        )
        win_probability = torch.sigmoid(win_logit).item()
        objective_predictions = [
            int(torch.argmax(logits, dim=-1).item()) for logits in objective_logits
        ]
        objective_probabilities = [
            torch.softmax(logits, dim=-1)[0].cpu().numpy().tolist()
            for logits in objective_logits
        ]

    return {
        "embedding": embedding[0].cpu(),
        "win_probability": win_probability,
        "gold_curve": np.nan_to_num(
            gold_pred[0].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0
        ),
        "objective_predictions": objective_predictions,
        "objective_probabilities": objective_probabilities,
    }


def render_results(result, objective_support):
    st.markdown("## Prediction results")
    probability = result["win_probability"]
    blue_probability = probability * 100
    red_probability = (1 - probability) * 100
    winner = "Blue Team" if probability >= 0.5 else "Red Team"
    margin = abs(probability - 0.5)
    if margin < 0.05:
        matchup_assessment = "Very close matchup"
        assessment_type = "warning"
    elif margin < 0.12:
        matchup_assessment = "Slight model lean"
        assessment_type = "info"
    else:
        matchup_assessment = "Clearer model lean"
        assessment_type = "success"

    left, right = st.columns([1, 1])
    with left:
        st.metric("Blue team win probability", f"{blue_probability:.1f}%")
        st.progress(probability)
    with right:
        st.metric("Red team win probability", f"{red_probability:.1f}%")
        st.progress(1 - probability)
    getattr(st, assessment_type)(
        f"**{matchup_assessment}** · Model lean: **{winner}** "
        f"(experimental draft-only estimate)"
    )

    curve = pd.DataFrame(
        {"Predicted gold difference": result["gold_curve"]},
        index=["5 min", "10 min", "15 min", "20 min", "25 min", "30 min"],
    )
    curve.index.name = "Game time"
    st.markdown("### Predicted gold curve")
    st.line_chart(curve)

    st.markdown("### First objective predictions")
    columns = st.columns(4)
    for column, objective, prediction, probabilities in zip(
        columns,
        OBJECTIVES,
        result["objective_predictions"],
        result["objective_probabilities"],
    ):
        if not objective_support.get(objective, False):
            owner = "Unavailable"
        else:
            owner = {0: "No objective", 1: "Blue Team", 2: "Red Team"}[prediction]
        column.metric(objective, owner)
        if objective_support.get(objective, False):
            column.caption(
                f"Blue {probabilities[1]:.0%} · Red {probabilities[2]:.0%}"
            )
    unavailable = [
        objective for objective in OBJECTIVES if not objective_support.get(objective, False)
    ]
    if unavailable:
        st.warning(
            "Objective training labels are missing for "
            + ", ".join(unavailable)
            + ". Regenerate objectives.parquet from event data and retrain the "
            "checkpoint before using those predictions."
        )
    st.caption(
        "Predictions use champion draft composition only. Player skill, matchup "
        "execution, and in-game events are not included."
    )

    with st.expander("Technical details"):
        st.caption("The encoder output is a 256-dimensional draft embedding.")
        st.write(result["embedding"].numpy())


try:
    (
        model,
        patch_vocab,
        champion_data,
        name_to_model_id,
        champion_names,
        objective_support,
        checkpoint_metric,
    ) = load_resources()
except (FileNotFoundError, KeyError, RuntimeError, OSError) as error:
    st.error(f"Unable to load the predictor: {error}")
    st.stop()

st.title("🎮 League of Legends Draft Predictor")
st.caption(
    "Explore a five-role draft and estimate win probability, gold trajectory, "
    "and first-objective control."
)

filled_slots = sum(bool(st.session_state.get(f"{team}_{index}", "")) for team in ("blue", "red") for index in range(5))
st.progress(filled_slots / 10, text=f"Draft progress · {filled_slots}/10 champions selected")

with st.sidebar:
    st.header("How to use")
    st.write(
        "Choose one champion for each role. Select boxes are searchable: click a "
        "field and type a champion name."
    )
    st.divider()
    st.metric("Complete roster", len(champion_names))
    st.metric("Supported by checkpoint", len(name_to_model_id))
    if checkpoint_metric:
        st.caption(f"Checkpoint selected by `{checkpoint_metric}`")
    if len(name_to_model_id) < len(champion_names):
        st.warning(
            "Some champions are shown for roster completeness but are not present "
            "in this checkpoint. They cannot be used for prediction until retraining."
        )
    if st.button("Reset draft", use_container_width=True):
        reset_draft()
        st.rerun()

st.markdown(
    """
    <style>
    .team-heading {
        border-radius: 0.6rem;
        padding: 0.8rem 1rem;
        margin: 0 0 0.8rem;
        color: white;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
    }
    .team-heading small {
        display: block;
        opacity: 0.85;
        font-weight: 400;
        margin-top: 0.2rem;
    }
    .blue { background: linear-gradient(90deg, #1769aa, #2196f3); }
    .red { background: linear-gradient(90deg, #a32929, #e74c3c); }
    .team-blue-stack {
        border-left: 0.35rem solid #2196f3;
        padding-left: 0.65rem;
    }
    .team-red-stack {
        border-left: 0.35rem solid #e74c3c;
        padding-left: 0.65rem;
    }
    .selected-pick {
        border-radius: 0.55rem;
        overflow: hidden;
        background: rgba(248, 250, 252, 0.8);
    }
    .selected-pick img {
        display: block;
        width: 100%;
        aspect-ratio: 1 / 1.18;
        object-fit: cover;
    }
    .selected-pick span {
        display: block;
        padding: 0.35rem 0.45rem;
        color: #334155;
        font-size: 0.78rem;
    }
    .blue-pick { border-left: 0.35rem solid #2196f3; }
    .red-pick { border-left: 0.35rem solid #e74c3c; }
    div[data-testid="stMetric"] {
        background: rgba(247, 249, 252, 0.9);
        padding: 0.7rem;
        border-radius: 0.6rem;
        border: 1px solid rgba(100, 116, 139, 0.15);
    }
    .empty-pick {
        aspect-ratio: 1 / 1.18;
        min-height: 116px;
        border: 1px solid rgba(148, 163, 184, 0.55);
        border-radius: 0.55rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #64748b;
        font-size: 0.78rem;
        background: linear-gradient(145deg, rgba(226, 232, 240, 0.72), rgba(248, 250, 252, 0.88));
    }
    .role-icon {
        width: 2.4rem;
        height: 2.4rem;
        display: grid;
        place-items: center;
        margin-bottom: 0.35rem;
        border-radius: 50%;
        color: #475569;
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(100, 116, 139, 0.28);
        font-size: 1.45rem;
        line-height: 1;
    }
    .empty-pick strong { color: #94a3b8; margin-top: 0.25rem; }
    [data-testid="stImage"] img { border-radius: 0.55rem; }
    [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
        padding: 0.45rem;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(100, 116, 139, 0.16);
        border-radius: 0.7rem;
        background: rgba(248, 250, 252, 0.55);
    }
    [data-testid="stExpander"] {
        border-radius: 0.55rem;
        border-color: rgba(100, 116, 139, 0.25);
        margin-bottom: 0.65rem;
    }
    [data-testid="stExpander"] [data-testid="stImage"] img {
        aspect-ratio: 1;
        object-fit: cover;
    }
    [data-testid="stExpander"] button:disabled {
        opacity: 0.38;
        filter: grayscale(1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Build your draft")
st.caption("Work from Top to Support. Each role has its own image preview and champion picker.")
blue_team = render_team(
    "blue", "🔵 Blue Team", "blue team-blue-stack", champion_names, name_to_model_id, champion_data
)
red_team = render_team(
    "red", "🔴 Red Team", "red team-red-stack", champion_names, name_to_model_id, champion_data
)

blue_preview, red_preview = st.columns(2)
with blue_preview:
    st.caption("Blue picks")
    st.write(" · ".join(name or "Open slot" for name in blue_team))
with red_preview:
    st.caption("Red picks")
    st.write(" · ".join(name or "Open slot" for name in red_team))

errors = validate_draft(blue_team, red_team, name_to_model_id)
if errors:
    st.session_state.pop("prediction", None)
    for error in errors:
        st.warning(error)
else:
    st.success("Draft is ready. Review the image-backed picks, then run the prediction.")

if st.button(
    "🔮 Predict draft",
    type="primary",
    use_container_width=True,
    disabled=bool(errors),
):
    st.session_state.prediction = predict(
        model, patch_vocab, name_to_model_id, blue_team, red_team
    )

if "prediction" in st.session_state:
    render_results(st.session_state.prediction, objective_support)
