#!/usr/bin/env python3
"""
Perform archetype clustering on draft embeddings with strategic profile alignment
and quantitative cluster quality metrics (Silhouette & Davies-Bouldin).
Uses scikit-learn when available, with a pure NumPy/SciPy fallback for lightweight environments.
"""

from itertools import permutations
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clustering")

ARCHETYPE_LABELS = ["Engage", "Poke", "Scaling", "Split Push"]
ROLES = ["top", "jng", "mid", "bot", "sup"]

# Full-roster champion strategic taxonomy (casefolded names)
ARCHETYPE_CHAMPIONS = {
    "Engage": {
        "aatrox", "alistar", "ambessa", "amumu", "annie", "bard", "blitzcrank", "braum",
        "briar", "camille", "cho'gath", "diana", "elise", "fiddlesticks", "galio",
        "gnar", "gragas", "hecarim", "irelia", "jarvan iv", "k'sante", "kennen", "kled",
        "lee sin", "leona", "lissandra", "malphite", "maokai", "mordekaiser", "nautilus",
        "neeko", "nocturne", "nunu & willump", "olaf", "orianna", "ornn", "pantheon",
        "poppy", "pyke", "qiyana", "rakan", "rammus", "rek'sai", "rell", "renata glasc",
        "renekton", "rengar", "rumble", "samira", "sejuani", "sett", "shen", "shyvana",
        "singed", "sion", "skarner", "swain", "sylas", "tahm kench", "taric", "thresh",
        "vi", "viego", "volibear", "warwick", "wukong", "xin zhao", "yasuo", "yone",
        "zaahen", "zac",
    },
    "Poke": {
        "ahri", "akshan", "anivia", "ashe", "aurora", "brand", "caitlyn", "corki",
        "draven", "ezreal", "gangplank", "graves", "heimerdinger", "hwei", "ivern",
        "janna", "jayce", "jhin", "kalista", "karma", "karthus", "kha'zix", "leblanc",
        "lucian", "lulu", "lux", "malzahar", "mel", "milio", "miss fortune", "morgana",
        "naafiri", "nami", "nidalee", "quinn", "seraphine", "shaco", "sivir", "soraka",
        "syndra", "taliyah", "talon", "teemo", "twisted fate", "varus", "vel'koz",
        "vex", "viktor", "xerath", "yunara", "yuumi", "zed", "ziggs", "zilean", "zoe",
        "zyra",
    },
    "Scaling": {
        "aphelios", "aurelion sol", "azir", "bel'veth", "cassiopeia", "gwen",
        "jinx", "kai'sa", "karthus", "kassadin", "katarina", "kayle", "kayn", "kindred",
        "kog'maw", "lillia", "master yi", "nasus", "nilah", "ornn", "ryze", "senna",
        "sivir", "smolder", "sona", "swain", "tristana", "twitch", "vayne", "veigar",
        "viktor", "vladimir", "xayah", "yasuo", "yone", "zeri",
    },
    "Split Push": {
        "akali", "camille", "darius", "dr. mundo", "ekko", "evelynn", "fiora", "fizz",
        "gangplank", "garen", "gwen", "illaoi", "irelia", "jax", "jayce", "k'sante",
        "kayle", "master yi", "mordekaiser", "nasus", "olaf", "quinn", "renekton",
        "riven", "ryze", "shen", "singed", "sion", "tristana", "trundle", "tryndamere",
        "twisted fate", "udyr", "urgot", "vayne", "volibear", "yasuo", "yone", "yorick",
        "zed",
    },
}


class _NumpyScaler:
    """Lightweight StandardScaler equivalent when scikit-learn is unavailable."""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.mean_ = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        self.scale_ = np.where(std > 1e-8, std, 1.0)
        return (x - self.mean_) / self.scale_


class _NumpyKMeans:
    """Deterministic Lloyd's K-Means with k-means++ initialization."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42, n_init: int = 10, max_iter: int = 100):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.cluster_centers_ = None

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        best_inertia = float("inf")
        best_labels = np.zeros(len(x), dtype=np.int64)
        best_centers = x[: self.n_clusters].copy()

        for _ in range(self.n_init):
            # k-means++ seeding
            centers = [x[rng.integers(len(x))]]
            for _ in range(1, self.n_clusters):
                dists_sq = np.min(
                    np.sum((x[:, None, :] - np.array(centers)[None, :, :]) ** 2, axis=-1),
                    axis=1,
                )
                probs = dists_sq / max(float(np.sum(dists_sq)), 1e-12)
                idx = rng.choice(len(x), p=probs)
                centers.append(x[idx])
            centers_arr = np.array(centers, dtype=np.float64)

            labels = np.zeros(len(x), dtype=np.int64)
            for _ in range(self.max_iter):
                dists = np.sum((x[:, None, :] - centers_arr[None, :, :]) ** 2, axis=-1)
                new_labels = np.argmin(dists, axis=1)
                if np.array_equal(new_labels, labels):
                    break
                labels = new_labels
                for k in range(self.n_clusters):
                    mask = labels == k
                    if np.any(mask):
                        centers_arr[k] = np.mean(x[mask], axis=0)

            inertia = float(np.sum((x - centers_arr[labels]) ** 2))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers_arr.copy()

        self.cluster_centers_ = best_centers
        return best_labels


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

    effective_k = min(n_clusters, max(1, len(embeddings)))
    if HAS_SKLEARN:
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        kmeans = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
    else:
        scaler = _NumpyScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        kmeans = _NumpyKMeans(n_clusters=effective_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)

    logger.info(f"Clustering completed. Cluster counts: {np.bincount(cluster_labels)}")
    return cluster_labels, kmeans, scaler


def compute_clustering_metrics(embeddings: np.ndarray, cluster_labels: np.ndarray) -> dict[str, Optional[float]]:
    """Compute Silhouette Score and Davies-Bouldin Index when valid."""
    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) < 2 or len(embeddings) <= len(unique_labels):
        return {"silhouette_score": None, "davies_bouldin_index": None}

    scaler = StandardScaler() if HAS_SKLEARN else _NumpyScaler()
    scaled = scaler.fit_transform(embeddings)

    if HAS_SKLEARN:
        try:
            sil = float(silhouette_score(scaled, cluster_labels))
            dbi = float(davies_bouldin_score(scaled, cluster_labels))
            return {"silhouette_score": round(sil, 4), "davies_bouldin_index": round(dbi, 4)}
        except Exception as exc:
            logger.warning("Could not compute cluster quality metrics: %s", exc)
            return {"silhouette_score": None, "davies_bouldin_index": None}

    # Pure NumPy Silhouette & Davies-Bouldin computation
    diffs = scaled[:, None, :] - scaled[None, :, :]
    dist_mat = np.sqrt(np.sum(diffs * diffs, axis=-1))
    n = len(scaled)
    sil_vals = np.zeros(n, dtype=np.float64)
    centroids = []
    dispersions = []

    for k in unique_labels:
        mask = cluster_labels == k
        c_pts = scaled[mask]
        c_center = np.mean(c_pts, axis=0)
        centroids.append(c_center)
        dispersions.append(float(np.mean(np.linalg.norm(c_pts - c_center, axis=1))))

    for i in range(n):
        own_k = cluster_labels[i]
        own_mask = (cluster_labels == own_k) & (np.arange(n) != i)
        a_i = float(np.mean(dist_mat[i, own_mask])) if np.any(own_mask) else 0.0
        b_i = min(
            float(np.mean(dist_mat[i, cluster_labels == other_k]))
            for other_k in unique_labels
            if other_k != own_k
        )
        denom = max(a_i, b_i, 1e-12)
        sil_vals[i] = (b_i - a_i) / denom

    k_count = len(unique_labels)
    db_vals = []
    for i in range(k_count):
        ratios = []
        for j in range(k_count):
            if i == j:
                continue
            c_dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            ratios.append((dispersions[i] + dispersions[j]) / max(c_dist, 1e-12))
        db_vals.append(max(ratios) if ratios else 0.0)

    return {
        "silhouette_score": round(float(np.mean(sil_vals)), 4),
        "davies_bouldin_index": round(float(np.mean(db_vals)), 4),
    }


def _load_id_to_name_map(data_dir: Optional[Path] = None) -> dict[int, str]:
    """Resolve vocabulary integer IDs to lowercase champion names."""
    mapping: dict[int, str] = {}
    champ_data_candidates = [
        Path("lol_collector/draft_encoder/data/champion_data.json"),
        Path("../draft_encoder/data/champion_data.json"),
    ]
    riot_to_name: dict[str, str] = {}
    for path in champ_data_candidates:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            for key, entry in raw.items():
                name = entry.get("name", "")
                if name:
                    if str(key).isdigit():
                        mapping[int(key)] = name.casefold()
                    if entry.get("riot_id") is not None:
                        riot_to_name[str(entry["riot_id"])] = name.casefold()
            break

    vocab_candidates = []
    if data_dir is not None:
        vocab_candidates.append(Path(data_dir) / "champion_vocab.json")
    vocab_candidates.extend([
        Path("data/processed/champion_vocab.json"),
        Path("lol_collector/draft_encoder/data/processed_riot/champion_vocab.json"),
    ])
    for vocab_path in vocab_candidates:
        if vocab_path.exists():
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            for k, vid in vocab.items():
                if k == "<UNK>":
                    continue
                if str(k) in riot_to_name:
                    mapping[int(vid)] = riot_to_name[str(k)]
                elif not str(k).isdigit():
                    mapping[int(vid)] = str(k).casefold()
            break
    return mapping


def assign_archetypes(
    cluster_labels: np.ndarray,
    win_labels: np.ndarray,
    embeddings: Optional[np.ndarray] = None,
    match_ids: Optional[list[str]] = None,
    data_dir: Optional[str | Path] = None,
) -> dict[str, str]:
    """
    Assign distinct strategic archetypes (Engage, Poke, Scaling, Split Push)
    using champion composition tags + gold scaling profiles when available, or
    embedding geometry + win-rate tie-breaking as a deterministic fallback.
    """
    n_clusters = int(len(np.bincount(cluster_labels)))
    target_labels = ARCHETYPE_LABELS[:n_clusters]

    resolved_dir = Path(data_dir) if data_dir else Path("./data/processed")
    if not (resolved_dir / "drafts.parquet").exists():
        alt = Path("lol_collector/draft_encoder/data/processed_riot")
        if (alt / "drafts.parquet").exists():
            resolved_dir = alt

    score_matrix = np.zeros((n_clusters, len(target_labels)), dtype=np.float64)
    used_domain_profiles = False

    if match_ids is not None and (resolved_dir / "drafts.parquet").exists():
        try:
            drafts_df = pd.read_parquet(resolved_dir / "drafts.parquet").set_index("match_id")
            id_to_name = _load_id_to_name_map(resolved_dir)
            gold_df = (
                pd.read_parquet(resolved_dir / "gold_curves.parquet").set_index("match_id")
                if (resolved_dir / "gold_curves.parquet").exists()
                else None
            )

            for cid in range(n_clusters):
                mask = cluster_labels == cid
                c_mids = [str(m) for m, flag in zip(match_ids, mask) if flag]
                matched_rows = drafts_df.loc[drafts_df.index.intersection(c_mids)]
                if matched_rows.empty:
                    continue

                champ_cols = [f"blue_{r}" for r in ROLES] + [f"red_{r}" for r in ROLES]
                champ_vals = matched_rows[champ_cols].to_numpy().ravel()
                champ_names = [id_to_name.get(int(v), "") for v in champ_vals if int(v) > 0]

                if champ_names:
                    total_c = max(1, len(champ_names))
                    for a_idx, arch in enumerate(target_labels):
                        arch_set = ARCHETYPE_CHAMPIONS.get(arch, set())
                        score_matrix[cid, a_idx] += sum(1 for n in champ_names if n in arch_set) / total_c
                    used_domain_profiles = True

                if gold_df is not None:
                    base_ids = [m.replace("_swap", "") for m in c_mids]
                    g_rows = gold_df.loc[gold_df.index.intersection(base_ids)]
                    if not g_rows.empty and {"gold_diff_10", "gold_diff_15", "gold_diff_30"}.issubset(g_rows.columns):
                        g10 = float(g_rows["gold_diff_10"].abs().mean())
                        g15 = float(g_rows["gold_diff_15"].abs().mean())
                        g_slope = float((g_rows["gold_diff_30"].abs() - g_rows["gold_diff_10"].abs()).mean())
                        if len(target_labels) >= 3:
                            score_matrix[cid, 0] += 1e-4 * g15       # Engage: mid-game 15m spike
                            score_matrix[cid, 1] += 1e-4 * g10       # Poke: early 10m lane pressure
                            score_matrix[cid, 2] += 1e-4 * g_slope   # Scaling: 10m->30m growth
        except Exception as exc:
            logger.warning("Domain profile scoring fallback triggered: %s", exc)

    if not used_domain_profiles:
        for cid in range(n_clusters):
            mask = cluster_labels == cid
            wr = float(np.mean(win_labels[mask])) if np.any(mask) else 0.5
            if embeddings is not None and np.any(mask) and embeddings.shape[1] >= 3:
                c_emb = embeddings[mask]
                third = max(1, embeddings.shape[1] // max(1, len(target_labels)))
                for a_idx in range(len(target_labels)):
                    sl = slice(a_idx * third, (a_idx + 1) * third)
                    score_matrix[cid, a_idx] = float(np.mean(c_emb[:, sl])) + 0.1 * wr * (len(target_labels) - a_idx)
            else:
                for a_idx in range(len(target_labels)):
                    score_matrix[cid, a_idx] = wr * (len(target_labels) - a_idx)

    best_perm = tuple(range(len(target_labels)))
    best_score = -float("inf")
    for perm in permutations(range(len(target_labels))):
        total = sum(score_matrix[cid, perm[cid]] for cid in range(min(n_clusters, len(target_labels))))
        if total > best_score:
            best_score = total
            best_perm = perm

    archetype_mapping: dict[str, str] = {}
    for cluster_id in range(n_clusters):
        if cluster_id < len(target_labels):
            archetype = target_labels[best_perm[cluster_id]]
        else:
            archetype = f"Archetype_{cluster_id + 1}"
        archetype_mapping[str(cluster_id)] = archetype
        cluster_mask = cluster_labels == cluster_id
        wr = float(np.mean(win_labels[cluster_mask])) if np.any(cluster_mask) else 0.5
        logger.info(
            f"Cluster {cluster_id}: Win rate = {wr:.3f} -> {archetype}"
        )

    return archetype_mapping


def analyze_cluster_properties(
    embeddings: np.ndarray, cluster_labels: np.ndarray, win_labels: np.ndarray
) -> dict[str, dict]:
    """Analyze size, win rate, dispersion, and centroid properties of each cluster."""
    cluster_stats: dict[str, dict] = {}

    for cluster_id in range(len(np.bincount(cluster_labels))):
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_wins = win_labels[cluster_mask]

        if len(cluster_embeddings) > 0:
            centroid = np.mean(cluster_embeddings, axis=0)
            mean_embedding = centroid.tolist()
            dispersion = float(np.mean(np.linalg.norm(cluster_embeddings - centroid, axis=1)))
        else:
            mean_embedding = [0.0] * embeddings.shape[1]
            dispersion = 0.0

        win_rate = float(np.mean(cluster_wins)) if len(cluster_wins) > 0 else 0.5

        cluster_stats[str(cluster_id)] = {
            "size": int(len(cluster_embeddings)),
            "win_rate": win_rate,
            "dispersion": round(dispersion, 4),
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
    archetype_mapping = assign_archetypes(
        cluster_labels,
        win_labels,
        embeddings=embeddings,
        match_ids=data["match_ids"],
    )
    cluster_stats = analyze_cluster_properties(embeddings, cluster_labels, win_labels)
    quality_metrics = compute_clustering_metrics(embeddings, cluster_labels)

    output_dir = Path("./clustering_results")
    output_dir.mkdir(exist_ok=True)

    cluster_assignments = {
        "match_ids": data["match_ids"],
        "cluster_labels": cluster_labels.tolist(),
        "archetype_mapping": archetype_mapping,
        "cluster_stats": cluster_stats,
        "quality_metrics": quality_metrics,
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
    if quality_metrics["silhouette_score"] is not None:
        logger.info(
            "Cluster quality — Silhouette: %.4f | Davies-Bouldin: %.4f",
            quality_metrics["silhouette_score"],
            quality_metrics["davies_bouldin_index"],
        )

    for cluster_id_str, stats in cluster_stats.items():
        archetype = archetype_mapping[cluster_id_str]
        print(
            f"Cluster {cluster_id_str} ({archetype}): {stats['size']} samples "
            f"(win rate: {stats['win_rate']:.1%})"
        )


if __name__ == "__main__":
    main()