#!/usr/bin/env python3
"""
Perform archetype clustering on draft embeddings.
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from lol_collector.draft_encoder.dataset import DraftDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering")


def load_embeddings(embedding_dir: str) -> dict:
    """Load draft embeddings and metadata."""
    embeddings = np.load(Path(embedding_dir) / "draft_embeddings.npy")

    with open(Path(embedding_dir) / "match_ids.json") as f:
        match_ids = json.load(f)

    with open(Path(embedding_dir) / "win_labels.json") as f:
        win_labels = json.load(f)

    return {
        "embeddings": embeddings,
        "match_ids": match_ids,
        "win_labels": win_labels
    }


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 3) -> tuple:
    """Perform KMeans clustering on embeddings."""
    logger.info("Performing KMeans clustering...")

    # Standardize features
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)

    logger.info(f"Clustering completed. Cluster counts: {np.bincount(cluster_labels)}")

    return cluster_labels, kmeans, scaler


def assign_archetypes(cluster_labels: np.ndarray, win_labels: np.ndarray) -> dict:
    """Assign archetypes based on cluster labels and win rates."""
    # Create mapping from cluster to archetype
    archetype_mapping = {}

    # For each cluster, calculate win rate and determine archetype
    for cluster_id in range(len(np.bincount(cluster_labels))):
        cluster_mask = cluster_labels == cluster_id
        cluster_win_rate = np.mean(win_labels[cluster_mask])

        if cluster_win_rate > 0.65:
            archetype = "Engage"
        elif cluster_win_rate < 0.35:
            archetype = "Scaling"
        else:
            archetype = "Poke"

        archetype_mapping[cluster_id] = archetype
        logger.info(f"Cluster {cluster_id}: Win rate = {cluster_win_rate:.3f} -> {archetype}")

    return archetype_mapping


def analyze_cluster_properties(embeddings: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """Analyze properties of each cluster."""
    cluster_stats = {}

    for cluster_id in range(len(np.bincount(cluster_labels))):
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]

        # Calculate mean embedding for this cluster
        mean_embedding = np.mean(cluster_embeddings, axis=0)

        cluster_stats[cluster_id] = {
            "size": len(cluster_embeddings),
            "mean_embedding": mean_embedding.tolist()
        }

    return cluster_stats


def main():
    # Load embeddings
    embedding_dir = "./embeddings"
    if not Path(embedding_dir).exists():
        logger.error(f"Embedding directory {embedding_dir} does not exist")
        return

    data = load_embeddings(embedding_dir)
    embeddings = data["embeddings"]
    win_labels = np.array(data["win_labels"])

    logger.info(f"Loaded {len(embeddings)} embeddings")

    # Perform clustering
    cluster_labels, kmeans_model, scaler = perform_clustering(embeddings, n_clusters=3)

    # Assign archetypes
    archetype_mapping = assign_archetypes(cluster_labels, win_labels)

    # Analyze cluster properties
    cluster_stats = analyze_cluster_properties(embeddings, cluster_labels)

    # Save results
    output_dir = Path("./clustering_results")
    output_dir.mkdir(exist_ok=True)

    # Save cluster assignments
    cluster_assignments = {
        "match_ids": data["match_ids"],
        "cluster_labels": cluster_labels.tolist(),
        "archetype_mapping": archetype_mapping,
        "cluster_stats": cluster_stats
    }

    with open(output_dir / "cluster_assignments.json", "w") as f:
        json.dump(cluster_assignments, f, indent=2)

    # Save model artifacts
    np.save(output_dir / "kmeans_model.npy", kmeans_model.cluster_centers_)
    np.save(output_dir / "scaler.npy", scaler.scale_)

    logger.info("Clustering results saved to clustering_results/")

    # Print cluster statistics
    for cluster_id, stats in cluster_stats.items():
        archetype = archetype_mapping[cluster_id]
        print(f"Cluster {cluster_id} ({archetype}): {stats['size']} samples")


if __name__ == "__main__":
    main()