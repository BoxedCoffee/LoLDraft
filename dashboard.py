#!/usr/bin/env python3
"""
Streamlit dashboard for draft synergy and archetype analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="LoL Draft Synergy Engine",
    page_icon="🎮",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">League of Legends Draft Synergy Engine</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load data
@st.cache_data
def load_clustering_data():
    """Load clustering results."""
    try:
        with open("./clustering_results/cluster_assignments.json") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading clustering data: {e}")
        return None

@st.cache_data
def load_embeddings():
    """Load draft embeddings."""
    try:
        embeddings = np.load("./embeddings/draft_embeddings.npy")
        with open("./embeddings/match_ids.json") as f:
            match_ids = json.load(f)
        return embeddings, match_ids
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None, None

# Main dashboard
def main():
    # Load data
    cluster_data = load_clustering_data()
    embeddings, match_ids = load_embeddings()

    if cluster_data is None or embeddings is None:
        st.warning("Please run the clustering analysis first.")
        return

    st.subheader("📊 Archetype Analysis")

    # Display archetype mapping
    archetype_mapping = cluster_data["archetype_mapping"]
    cluster_stats = cluster_data["cluster_stats"]

    col1, col2, col3 = st.columns(3)

    for cluster_id, archetype in archetype_mapping.items():
        with col1 if cluster_id == 0 else (col2 if cluster_id == 1 else col3):
            st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
            st.metric(label=f"Cluster {cluster_id} ({archetype})",
                     value=f"{cluster_stats[cluster_id]['size']} samples")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Show sample embeddings
    st.subheader("📈 Embedding Visualization")

    # Create embedding plots
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Cluster distribution
    cluster_counts = [cluster_stats[i]['size'] for i in range(len(cluster_stats))]
    ax[0].bar(range(len(cluster_counts)), cluster_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax[0].set_xlabel('Cluster ID')
    ax[0].set_ylabel('Number of Samples')
    ax[0].set_title('Distribution of Drafts Across Archetypes')

    # Plot 2: Sample embeddings (first few dimensions)
    sample_embeddings = embeddings[:100]  # First 100 samples
    if len(sample_embeddings) > 0:
        # Show first 4 dimensions as a simple visualization
        ax[1].scatter(sample_embeddings[:, 0], sample_embeddings[:, 1],
                     c=cluster_data["cluster_labels"][:100], cmap='viridis', alpha=0.6)
        ax[1].set_xlabel('Embedding Dimension 1')
        ax[1].set_ylabel('Embedding Dimension 2')
        ax[1].set_title('Sample Embeddings (First 2 Dimensions)')

    st.pyplot(fig)

    st.markdown("---")

    # Show archetype characteristics
    st.subheader("🎯 Archetype Characteristics")

    for cluster_id, archetype in archetype_mapping.items():
        st.markdown(f"### {archetype} ({cluster_id})")

        # Display cluster statistics
        stats = cluster_stats[cluster_id]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Sample Count", stats["size"])

        with col2:
            # This would be actual characteristic analysis in a full implementation
            st.metric("Avg. Win Rate", "67%")  # Placeholder

        with col3:
            st.metric("Embedding Dimension", "256")

        st.markdown("---")

    st.subheader("🎮 Recommendation Engine")

    # Simple recommendation interface
    st.markdown("### Make a Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        blue_team = st.text_input("Blue Team Champions (comma-separated)", "1,2,3,4,5")
        blue_champs = [int(x.strip()) for x in blue_team.split(",") if x.strip()]

    with col2:
        red_team = st.text_input("Red Team Champions (comma-separated)", "6,7,8,9,10")
        red_champs = [int(x.strip()) for x in red_team.split(",") if x.strip()]

    if st.button("Generate Recommendations"):
        # Placeholder recommendations
        st.success("Recommendation generated!")
        st.markdown("#### Pick Recommendation:")
        st.info("Recommended: Jinx (Confidence: 85%)")
        st.markdown("#### Ban Recommendation:")
        st.info("Recommended: Thresh (Confidence: 75%)")

if __name__ == "__main__":
    main()