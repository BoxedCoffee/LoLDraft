#!/usr/bin/env python3
"""Interactive Draft Analyzer focused on pathways to victory."""

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "draft_encoder" / "data" / "processed_riot"
OUTPUT_DIR = APP_DIR / "analysis_output"
ROLES = ["top", "jng", "mid", "bot", "sup"]
ROLE_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
OBJECTIVES = ["dragon", "herald", "baron", "tower"]
OBJECTIVE_LABELS = {"dragon": "Dragon", "herald": "Herald", "baron": "Baron", "tower": "Tower"}
ARCHETYPE_RULES = {
    "Engage": {"malphite", "amumu", "leona", "nautilus", "rell", "sejuani", "ornn", "wukong", "jarvan iv", "zac"},
    "Poke": {"xerath", "ezreal", "jayce", "nidalee", "varus", "karma", "lux", "vel'koz", "zoe", "jayce"},
    "Scaling": {"vayne", "nasus", "kayle", "kassadin", "senna", "jinx", "azir", "sivir", "twitch", "gangplank"},
    "Split push": {"fiora", "tryndamere", "camille", "yorick", "trundle", "jax", "shen"},
}

st.set_page_config(page_title="LoL Draft Analyzer", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem;}
    .hero {padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1.2rem;
           background: linear-gradient(120deg, #101b35 0%, #182b4f 55%, #24203d 100%);
           border: 1px solid #344b76;}
    .hero h1 {color: #f5f7ff; margin: 0; font-size: 2.2rem;}
    .hero p {color: #becbeb; margin: .45rem 0 0; font-size: 1rem;}
    .team-card {border-radius: 14px; padding: .8rem 1rem; margin-bottom: .6rem;
                background: #121a2c; border: 1px solid #2b3a5a;}
    .team-card.blue {border-left: 5px solid #4b9dff;}
    .team-card.red {border-left: 5px solid #ff667c;}
    .team-title {font-weight: 700; font-size: 1.05rem; margin-bottom: .5rem;}
    .team-title.blue {color: #75b7ff;} .team-title.red {color: #ff8d9d;}
    .pick {display: inline-block; min-width: 90px; padding: .38rem .55rem; margin: .15rem;
           border-radius: 8px; background: #1b2740; color: #e7edff; font-size: .82rem;}
    .empty-pick {color: #8290ad; background: #172034;}
    .kicker {text-transform: uppercase; letter-spacing: .1em; color: #91a3c6; font-size: .7rem;}
    .evidence {padding: .75rem 1rem; border-radius: 12px; background: #111a2b; border: 1px solid #293958;}
    .muted {color: #8e9bb7; font-size: .86rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_artifacts():
    table = pd.read_parquet(OUTPUT_DIR / "match_analysis.parquet")
    stats = pd.read_parquet(OUTPUT_DIR / "champion_role_stats.parquet")
    champion_data_path = APP_DIR.parent / "draft_encoder" / "data" / "champion_data.json"
    mapping = {}
    if champion_data_path.exists():
        champion_data = json.loads(champion_data_path.read_text(encoding="utf-8"))
        mapping = {
            str(champion_id): entry["name"]
            for champion_id, entry in champion_data.items()
            if entry.get("name")
        }
        for role in ROLES:
            for side in ("blue", "red"):
                column = f"{side}_{role}"
                table[column] = table[column].map(
                    lambda value: mapping.get(str(value), str(value))
                )
        stats["champion"] = stats["champion"].map(
            lambda value: mapping.get(str(value), str(value))
        )
    return table, stats, mapping


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


def render_team_card(team: str, picks: dict[str, str]) -> None:
    labels = []
    for role in ROLES:
        champion = picks.get(role) or "Open slot"
        empty = " empty-pick" if champion == "Open slot" else ""
        labels.append(
            f'<span class="pick{empty}"><b>{ROLE_LABELS[role]}</b><br>{champion}</span>'
        )
    color = "blue" if team == "Blue" else "red"
    st.markdown(
        f'<div class="team-card {color}"><div class="team-title {color}">{team} side</div>'
        + "".join(labels)
        + "</div>",
        unsafe_allow_html=True,
    )


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
    margin = z * math.sqrt(
        successes * (n - successes) / n + z * z / 4
    ) / (n + z * z)
    return center - margin, center + margin


def evidence_ladder(table, blue, red) -> list[dict[str, str]]:
    rows = []
    for mode in ("Exact roles & sides", "Role-aware (either side)", "Champion presence"):
        count = len(filtered_matches(table, blue, red, mode))
        label, _ = confidence_label(count)
        rows.append({"Comparison": mode, "Matches": f"{count:,}", "Evidence": label})
    return rows


def resolve_mode(
    table, blue, red, requested: str, minimum: int
) -> tuple[str, pd.DataFrame, int | None]:
    if requested != "Auto (recommended)":
        return requested, filtered_matches(table, blue, red, requested), None
    selected = [(team, role, champion) for team, picks in (("blue", blue), ("red", red))
                for role, champion in picks.items() if champion]
    for required in range(len(selected), 0, -1):
        for mode in ("Exact roles & sides", "Role-aware (either side)", "Champion presence"):
            result = relaxed_matches(table, blue, red, mode, required)
            if len(result) >= minimum:
                return f"{mode} · at least {required}/{len(selected)} picks", result, required
    return "Champion presence · at least 1 pick", relaxed_matches(table, blue, red, "Champion presence", 1), 1


def relaxed_matches(
    table: pd.DataFrame,
    blue: dict[str, str],
    red: dict[str, str],
    mode: str,
    required: int,
) -> pd.DataFrame:
    conditions = []
    champion_columns = [f"{side}_{role}" for side in ("blue", "red") for role in ROLES]
    if mode == "Champion presence":
        for champion in [*blue.values(), *red.values()]:
            if champion:
                conditions.append(table[champion_columns].eq(champion).any(axis=1))
    else:
        for team, picks in (("blue", blue), ("red", red)):
            for role, champion in picks.items():
                if not champion:
                    continue
                if mode == "Role-aware (either side)":
                    conditions.append(table[[f"blue_{role}", f"red_{role}"]].eq(champion).any(axis=1))
                else:
                    conditions.append(table[f"{team}_{role}"].eq(champion))
    if not conditions:
        return table
    matched = pd.concat(conditions, axis=1).sum(axis=1)
    return table.loc[matched >= required]


def classify_team(table: pd.DataFrame, side: str, mapping: dict[str, str]) -> pd.DataFrame:
    def classify(row):
        names = [
            mapping.get(str(row[f"{side}_{role}"]), str(row[f"{side}_{role}"])).casefold()
            for role in ROLES
        ]
        labels = [label for label, champions in ARCHETYPE_RULES.items()
                  if any(name in champions for name in names)]
        return ", ".join(labels) if labels else "Balanced / mixed"

    return table.apply(classify, axis=1)


def matchup_evidence(table: pd.DataFrame, blue: dict[str, str], red: dict[str, str]) -> pd.DataFrame:
    rows = []
    for role in ROLES:
        blue_champion, red_champion = blue.get(role), red.get(role)
        if not blue_champion or not red_champion:
            continue
        mask = table[f"blue_{role}"].eq(blue_champion) & table[f"red_{role}"].eq(red_champion)
        frame = table.loc[mask]
        rows.append({
            "Role": ROLE_LABELS[role],
            "Blue champion": blue_champion,
            "Red champion": red_champion,
            "Matches": len(frame),
            "Blue win rate": pct(frame.blue_win.mean()),
            "Avg gold diff 10m": round(frame.gold_diff_10.mean(), 1) if len(frame) else None,
        })
    return pd.DataFrame(rows)


def smoothed_rate(series: pd.Series, prior: float, strength: int = 100) -> float:
    values = series.dropna().astype(float)
    return float((values.sum() + prior * strength) / (len(values) + strength))


table, stats, mapping = load_artifacts()
all_champions = sorted(
    set(mapping.values()) if mapping else {
        str(value) for role in ROLES for side in ("blue", "red")
        for value in table[f"{side}_{role}"].unique()
    },
    key=str.casefold,
)

st.markdown(
    '<div class="hero"><div class="kicker">League of Legends · historical draft research</div>'
    '<h1>Draft to Victory</h1>'
    '<p>Explore how compositions relate to early gold, objective control, comebacks, and lead conversion.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Build a draft")
    st.caption("Pick as many roles as you know. More selections make the historical comparison narrower.")
    comparison_mode = st.selectbox(
        "Comparison breadth",
        ["Auto (recommended)", "Role-aware (either side)", "Exact roles & sides", "Champion presence"],
        help=(
            "Role-aware comparisons keep the selected lane but allow the champion "
            "to appear on either team. Champion presence is the broadest mode."
        ),
    )
    minimum_evidence = st.slider(
        "Minimum evidence",
        min_value=30,
        max_value=2000,
        value=250,
        step=10,
        help="Auto mode broadens the comparison until at least this many matches are available.",
    )
    patch_options = ["All patches"] + sorted(table["patch"].dropna().astype(str).unique(), reverse=True)
    selected_patch = st.selectbox("Patch scope", patch_options)
    if st.button("Clear draft", use_container_width=True):
        for key in list(st.session_state):
            if key.startswith(("blue_", "red_")):
                del st.session_state[key]
        st.rerun()
    blue_picks = {}
    red_picks = {}
    for role in ROLES:
        blue_picks[role] = st.selectbox(
            f"Blue · {ROLE_LABELS[role]}", [""] + all_champions, key=f"blue_{role}"
        )
        red_picks[role] = st.selectbox(
            f"Red · {ROLE_LABELS[role]}", [""] + all_champions, key=f"red_{role}"
        )

render_team_card("Blue", blue_picks)
render_team_card("Red", red_picks)
selected_count = sum(bool(value) for value in [*blue_picks.values(), *red_picks.values()])
scoped_table = table if selected_patch == "All patches" else table[table["patch"].astype(str).eq(selected_patch)]
resolved_mode, similar, matched_picks = resolve_mode(
    scoped_table, blue_picks, red_picks, comparison_mode, minimum_evidence
)
st.caption(f"{selected_count}/10 roles selected · {resolved_mode} · {selected_patch}")

st.subheader("Evidence at a glance")
overview = st.columns(4)
overview[0].metric("Dataset", f"{len(table):,} matches")
overview[1].metric("Patches", f"{table.patch.nunique():,}")
overview[2].metric("Draft matches", f"{len(similar):,}")
label, explanation = confidence_label(len(similar))
overview[3].metric("Evidence", label)

if not selected_count:
    st.info("Start with one or two champions in the sidebar. The analyzer will compare historical matches using the selected breadth.")
    st.stop()

with st.expander("Evidence ladder", expanded=False):
    st.caption("See how much evidence is available before choosing how specific the comparison should be.")
    st.dataframe(pd.DataFrame(evidence_ladder(scoped_table, blue_picks, red_picks)), hide_index=True, use_container_width=True)

if similar.empty:
    st.warning(
        "No historical matches support this comparison. Clear one or two picks, "
        "or choose a broader comparison mode in the sidebar."
    )
    st.stop()

st.caption(explanation)
if comparison_mode == "Auto (recommended)":
    st.success(f"Auto mode selected **{resolved_mode}** to meet the {minimum_evidence:,}-match evidence target.")
elif resolved_mode != "Exact roles & sides":
    st.info(
        f"This view uses **{comparison_mode.lower()}** to keep the evidence useful. "
        "It should be read as a composition/role association, not an exact replay of the selected draft."
    )
if matched_picks is not None and matched_picks < selected_count:
    st.warning(
        f"Only {matched_picks} of {selected_count} selected picks are required in this "
        "comparison. Add fewer picks or lower the evidence threshold for a more specific view."
    )
tabs = st.tabs(["Pathway to victory", "Objective control", "Draft evidence", "Matchups & identity", "Patch context"])

with tabs[0]:
    st.subheader("What happened in comparable matches?")
    cards = st.columns(4)
    win_rate = similar.blue_win.mean()
    win_low, win_high = wilson_interval(similar.blue_win)
    baseline = scoped_table.blue_win.mean()
    cards[0].metric("Blue win rate", pct(win_rate), f"{(win_rate - baseline) * 100:+.1f} pts vs dataset")
    cards[1].metric("First Dragon", pct(similar.blue_first_dragon.mean()))
    cards[2].metric("First Tower", pct(similar.blue_first_tower.mean()))
    cards[3].metric(
        "Comeback when behind",
        pct(similar.loc[~similar.blue_ahead_10, "blue_win"].mean()),
        help="Among comparable matches where Blue was behind at 10 minutes.",
    )
    st.caption(
        f"Blue win-rate 95% interval: {pct(win_low)}–{pct(win_high)} "
        f"(dataset baseline: {pct(baseline)})."
    )
    st.subheader("Average Blue gold difference")
    gold = pd.DataFrame(
        {
            "Minute": [5, 10, 15, 20, 25, 30],
            "Gold difference": [
                similar[f"gold_diff_{minute}"].mean() for minute in (5, 10, 15, 20, 25, 30)
            ],
        }
    ).set_index("Minute")
    st.line_chart(gold, color="#4b9dff")
    st.caption("Positive values indicate a Blue-side gold lead. This describes the typical game shape; it does not imply the draft caused it.")
    ahead = similar.loc[similar.blue_ahead_10, "blue_win"]
    conversion = ahead.mean()
    conversion_low, conversion_high = wilson_interval(ahead)
    st.info(
        f"When Blue was ahead at 10 minutes in this comparison, it won {pct(conversion)} "
        f"of those matches (95% interval: {pct(conversion_low)}–{pct(conversion_high)})."
    )

with tabs[1]:
    st.subheader("Who secured the first major objective?")
    objective_frame = pd.DataFrame(
        {
            "Objective": [OBJECTIVE_LABELS[name] for name in OBJECTIVES],
            "Blue": [similar[f"blue_first_{name}"].mean() for name in OBJECTIVES],
            "Red": [similar[f"red_first_{name}"].mean() for name in OBJECTIVES],
        }
    ).set_index("Objective")
    st.bar_chart(objective_frame, color=["#4b9dff", "#ff667c"])
    details = []
    for objective in OBJECTIVES:
        available = similar[f"first_{objective}_team"].isin([100, 200]).sum()
        details.append(
            {
                "Objective": OBJECTIVE_LABELS[objective],
                "Blue first": pct(similar[f"blue_first_{objective}"].mean()),
                "Red first": pct(similar[f"red_first_{objective}"].mean()),
                "Observed matches": f"{available:,}",
            }
        )
    st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
    st.caption("Unavailable objective events are excluded from the Blue/Red rates rather than counted as losses.")
    st.caption(
        "Rates are shown raw; small samples should be interpreted with the observed-match count above."
    )
    st.subheader("Objective value by game state")
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
                "State": state,
                "Matches": len(state_frame),
                "Blue first": pct(state_frame[f"blue_first_{objective}"].mean()),
                "Blue win if first": pct(
                    state_frame.loc[state_frame[f"blue_first_{objective}"], "blue_win"].mean()
                ),
            })
    st.dataframe(pd.DataFrame(state_rows), hide_index=True, use_container_width=True)
    st.caption("This is conditional association: it does not prove that securing an objective caused the win.")

with tabs[2]:
    st.subheader("How much evidence supports this view?")
    st.write(
        "The table below shows the selected champions in their selected roles. "
        "The comparison is intentionally exact by side and role, so adding picks can reduce the sample quickly."
    )
    selected_rows = []
    for team, picks in (("Blue", blue_picks), ("Red", red_picks)):
        for role, champion in picks.items():
            if champion:
                row = stats[(stats.role == role) & (stats.champion == champion)]
                selected_rows.append(
                    {
                        "Side": team,
                        "Role": ROLE_LABELS[role],
                        "Champion": champion,
                        "Historical picks": int(row.matches.iloc[0]) if not row.empty else 0,
                        "Role win rate": pct(row.win_rate.iloc[0]) if not row.empty else "Unavailable",
                        "Avg gold diff at 10m": round(float(row.mean_gold_diff_10.iloc[0]), 1) if not row.empty else None,
                    }
                )
    st.dataframe(pd.DataFrame(selected_rows), hide_index=True, use_container_width=True)
    conversion = similar.loc[similar.blue_ahead_10, "blue_win"]
    st.metric(
        "Blue lead conversion at 10 minutes",
        pct(smoothed_rate(conversion, scoped_table.blue_win.mean())),
        help="A lightly smoothed estimate shrinks very small samples toward the full-dataset baseline.",
    )

with tabs[3]:
    st.subheader("Lane matchup evidence")
    matchup = matchup_evidence(scoped_table, blue_picks, red_picks)
    if matchup.empty:
        st.info("Select both Blue and Red champions in at least one shared role to compare a lane matchup.")
    else:
        st.dataframe(matchup, hide_index=True, use_container_width=True)
        st.caption("Matchup rows use exact Blue-versus-Red role assignments and may be sparse independently of the broader comparison.")

    st.subheader("Composition identity")
    for side in ("blue", "red"):
        classified = classify_team(similar, side, mapping)
        counts = classified.value_counts().rename_axis("Observed archetype mix").reset_index(name="Matches")
        counts["Share"] = counts["Matches"] / len(similar)
        st.write(f"{side.title()} archetype indicators")
        st.dataframe(counts, hide_index=True, use_container_width=True)
    st.caption("Archetypes are transparent champion-pool indicators, not expert labels or causal explanations.")

with tabs[4]:
    st.subheader("How does the broader meta move?")
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
    chart = patch_summary.set_index("patch")[["blue_win_rate", "first_dragon", "first_tower"]]
    st.line_chart(chart)
    st.dataframe(patch_summary, hide_index=True, use_container_width=True)
    st.caption("Patch context follows the selected patch scope. Select All patches to see the full meta trend.")

st.divider()
st.caption("Research note: these are historical associations among comparable matches. They are not causal claims, betting advice, or guaranteed predictions.")
