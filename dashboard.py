#!/usr/bin/env python3
"""
Streamlit dashboard for draft synergy and archetype analysis.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from recommendation_engine import DraftRecommendationEngine

st.set_page_config(
    page_title="LoL Draft Synergy Engine",
    page_icon="🎮",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #2196F3;
    }
    .metric-card {
        background-color: #f0f8ff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<h1 class="main-header">League of Legends Draft Synergy Engine</h1>', unsafe_allow_html=True)
st.markdown("---")


@st.cache_data
def load_clustering_data():
    """Load clustering results."""
    try:
        with open("./clustering_results/cluster_assignments.json", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading clustering data: {e}")
        return None


@st.cache_data
def load_embeddings():
    """Load draft embeddings."""
    try:
        embeddings = np.load("./embeddings/draft_embeddings.npy")
        with open("./embeddings/match_ids.json", encoding="utf-8") as f:
            match_ids = json.load(f)
        return embeddings, match_ids
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None, None


def get_cluster_stat(cluster_stats: dict, cluster_id) -> dict:
    """Safely look up cluster stats regardless of str/int key serialization."""
    if str(cluster_id) in cluster_stats:
        return cluster_stats[str(cluster_id)]
    if int(cluster_id) in cluster_stats:
        return cluster_stats[int(cluster_id)]
    return {"size": 0, "win_rate": 0.5, "mean_embedding": []}


def main():
    cluster_data = load_clustering_data()
    embeddings, match_ids = load_embeddings()

    if cluster_data is None or embeddings is None:
        st.warning("Please run the clustering analysis first.")
        return

    st.subheader("📊 Archetype Analysis")

    archetype_mapping = cluster_data["archetype_mapping"]
    cluster_stats = cluster_data["cluster_stats"]

    cols = st.columns(max(1, len(archetype_mapping)))
    for idx, (cluster_id, archetype) in enumerate(archetype_mapping.items()):
        stats = get_cluster_stat(cluster_stats, cluster_id)
        with cols[idx % len(cols)]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label=f"Cluster {cluster_id} ({archetype})",
                value=f"{stats['size']} samples",
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Embedding Visualization")

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ordered_keys = sorted(cluster_stats.keys(), key=lambda k: int(k))
    cluster_counts = [get_cluster_stat(cluster_stats, k)["size"] for k in ordered_keys]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    ax[0].bar(
        [f"{k} ({archetype_mapping.get(str(k), '')})" for k in ordered_keys],
        cluster_counts,
        color=colors[: len(ordered_keys)],
    )
    ax[0].set_xlabel("Cluster ID")
    ax[0].set_ylabel("Number of Samples")
    ax[0].set_title("Distribution of Drafts Across Archetypes")

    sample_embeddings = embeddings[:100]
    if len(sample_embeddings) > 0:
        ax[1].scatter(
            sample_embeddings[:, 0],
            sample_embeddings[:, 1],
            c=cluster_data["cluster_labels"][:100],
            cmap="viridis",
            alpha=0.6,
        )
        ax[1].set_xlabel("Embedding Dimension 1")
        ax[1].set_ylabel("Embedding Dimension 2")
        ax[1].set_title("Sample Embeddings (First 2 Dimensions)")

    st.pyplot(fig)
    st.markdown("---")
    st.subheader("🎯 Archetype Characteristics")

    for cluster_id, archetype in archetype_mapping.items():
        st.markdown(f"### {archetype} (Cluster {cluster_id})")
        stats = get_cluster_stat(cluster_stats, cluster_id)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sample Count", stats["size"])
        with col2:
            win_rate = stats.get("win_rate")
            st.metric("Avg. Win Rate", f"{win_rate * 100:.1f}%" if win_rate is not None else "N/A")
        with col3:
            st.metric("Embedding Dimension", str(embeddings.shape[1]))

        st.markdown("---")

    st.subheader("🎮 Recommendation Engine")
    st.markdown("### Make a Recommendation")

    col1, col2 = st.columns(2)
    with col1:
        blue_team = st.text_input("Blue Team Champions (comma-separated IDs)", "1,2,3,4,5")
        blue_champs = [int(x.strip()) for x in blue_team.split(",") if x.strip().isdigit()]
    with col2:
        red_team = st.text_input("Red Team Champions (comma-separated IDs)", "6,7,8,9,10")
        red_champs = [int(x.strip()) for x in red_team.split(",") if x.strip().isdigit()]

    if st.button("Generate Recommendations"):
        engine = DraftRecommendationEngine(
            model_path="./checkpoints/best.pt",
            embedding_dir="./embeddings",
            clustering_dir="./clustering_results",
            data_dir="./data/processed",
        )
        pick_rec = engine.recommend_pick(blue_champs, red_champs)
        ban_rec = engine.recommend_ban(blue_champs, red_champs)
        st.success("Recommendation generated!")
        st.markdown("#### Pick Recommendation:")
        st.info(
            f"Recommended: {pick_rec['recommended_champ']} "
            f"(Confidence: {pick_rec['confidence']:.0%}, Archetype: {pick_rec['archetype']})"
        )
        st.markdown("#### Ban Recommendation:")
        st.info(
            f"Recommended: {ban_rec['recommended_ban']} "
            f"(Confidence: {ban_rec['confidence']:.0%} — {ban_rec['reason']})"
        )


if __name__ == "__main__":
    main()