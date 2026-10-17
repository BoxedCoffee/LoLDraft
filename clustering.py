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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering")

ARCHETYPE_LABELS = ["Engage", "Poke", "Scaling"]


def load_embeddings(embedding_dir: str) -> dict:
    """Load draft embeddings and metadata."""
    emb_path = Path(embedding_dir)
    embeddings = np.load(emb_path / "draft_embeddings.npy")

    with open(emb_path / "match_ids.json", encoding="utf-8") as f:
        match_ids = json.load(f)

    with open(emb_path / "win_labels.json", encoding="utf-8") as f:
        win_labels = json.load(f)

    return {
        "embeddings": embeddings,
        "match_ids": match_ids,
        "win_labels": win_labels,
    }


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 3) -> tuple:
    """Perform KMeans clustering on embeddings."""
    logger.info("Performing KMeans clustering...")

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_scaled)

    logger.info(f"Clustering completed. Cluster counts: {np.bincount(cluster_labels)}")
    return cluster_labels, kmeans, scaler


def assign_archetypes(cluster_labels: np.ndarray, win_labels: np.ndarray) -> dict[str, str]:
    """Assign distinct strategic archetypes ordered by cluster win rate."""
    n_clusters = int(len(np.bincount(cluster_labels)))
    cluster_win_rates = {}
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_win_rates[cluster_id] = (
            float(np.mean(win_labels[cluster_mask])) if np.any(cluster_mask) else 0.5
        )

    # Sort clusters by win rate descending to map cleanly onto Engage / Poke / Scaling
    ranked_clusters = sorted(
        cluster_win_rates.keys(), key=lambda cid: cluster_win_rates[cid], reverse=True
    )
    archetype_mapping: dict[str, str] = {}
    for rank, cluster_id in enumerate(ranked_clusters):
        archetype = ARCHETYPE_LABELS[rank] if rank < len(ARCHETYPE_LABELS) else f"Archetype_{rank + 1}"
        archetype_mapping[str(cluster_id)] = archetype
        logger.info(
            f"Cluster {cluster_id}: Win rate = {cluster_win_rates[cluster_id]:.3f} -> {archetype}"
        )

    return archetype_mapping


def analyze_cluster_properties(
    embeddings: np.ndarray, cluster_labels: np.ndarray, win_labels: np.ndarray
) -> dict[str, dict]:
    """Analyze size, win rate, and centroid properties of each cluster."""
    cluster_stats: dict[str, dict] = {}

    for cluster_id in range(len(np.bincount(cluster_labels))):
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_wins = win_labels[cluster_mask]

        mean_embedding = (
            np.mean(cluster_embeddings, axis=0).tolist()
            if len(cluster_embeddings) > 0
            else [0.0] * embeddings.shape[1]
        )
        win_rate = float(np.mean(cluster_wins)) if len(cluster_wins) > 0 else 0.5

        cluster_stats[str(cluster_id)] = {
            "size": int(len(cluster_embeddings)),
            "win_rate": win_rate,
            "mean_embedding": mean_embedding,
        }

    return cluster_stats


def main():
    embedding_dir = Path("./embeddings")
    if not embedding_dir.exists():
        logger.error(f"Embedding directory {embedding_dir} does not exist")
        return

    data = load_embeddings(str(embedding_dir))
    embeddings = data["embeddings"]
    win_labels = np.array(data["win_labels"], dtype=np.float32)

    logger.info(f"Loaded {len(embeddings)} embeddings")

    cluster_labels, kmeans_model, scaler = perform_clustering(embeddings, n_clusters=3)
    archetype_mapping = assign_archetypes(cluster_labels, win_labels)
    cluster_stats = analyze_cluster_properties(embeddings, cluster_labels, win_labels)

    output_dir = Path("./clustering_results")
    output_dir.mkdir(exist_ok=True)

    cluster_assignments = {
        "match_ids": data["match_ids"],
        "cluster_labels": cluster_labels.tolist(),
        "archetype_mapping": archetype_mapping,
        "cluster_stats": cluster_stats,
    }

    for target_path in (
        output_dir / "cluster_assignments.json",
        embedding_dir / "cluster_assignments.json",
    ):
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(cluster_assignments, f, indent=2)

    np.save(output_dir / "kmeans_model.npy", kmeans_model.cluster_centers_)
    np.save(output_dir / "scaler.npy", scaler.scale_)

    logger.info("Clustering results saved to clustering_results/ and embeddings/")

    for cluster_id_str, stats in cluster_stats.items():
        archetype = archetype_mapping[cluster_id_str]
        print(
            f"Cluster {cluster_id_str} ({archetype}): {stats['size']} samples "
            f"(win rate: {stats['win_rate']:.1%})"
        )


if __name__ == "__main__":
    main()