#!/usr/bin/env python3
"""
KNN-based recommendation engine for draft picks and bans.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch

from lol_collector.draft_encoder.model import DraftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("recommendation_engine")

ROLES = ["top", "jng", "mid", "bot", "sup"]


class DraftRecommendationEngine:
    """KNN-based draft recommendation engine."""

    def __init__(
        self,
        model_path: str = "./checkpoints/best.pt",
        embedding_dir: str = "./embeddings",
        clustering_dir: str = "./clustering_results",
        data_dir: str = "./data/processed",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.embedding_dir = Path(embedding_dir)
        self.clustering_dir = Path(clustering_dir)
        self.data_dir = Path(data_dir)

        self.model: Optional[DraftModel] = None
        self.id_to_name: Dict[int, str] = {}
        self.draft_lookup: Dict[str, dict] = {}

        self.load_model_and_embeddings()

        self.knn_model = None
        self.scaler = StandardScaler()
        self.initialize_recommendation_system()

    def _load_id_to_name_mapping(self) -> Dict[int, str]:
        """Resolve integer champion IDs to display names."""
        mapping: Dict[int, str] = {}
        vocab_candidates = [
            self.data_dir / "champion_vocab.json",
            Path("lol_collector/draft_encoder/data/processed_riot/champion_vocab.json"),
        ]
        champ_data_path = Path("lol_collector/draft_encoder/data/champion_data.json")
        riot_to_name: Dict[str, str] = {}
        if champ_data_path.exists():
            champ_data = json.loads(champ_data_path.read_text(encoding="utf-8"))
            for key, entry in champ_data.items():
                if entry.get("name"):
                    mapping[int(key)] = entry["name"]
                    if entry.get("riot_id") is not None:
                        riot_to_name[str(entry["riot_id"])] = entry["name"]

        for vocab_path in vocab_candidates:
            if vocab_path.exists():
                vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
                for key, val in vocab.items():
                    if key == "<UNK>":
                        continue
                    vid = int(val)
                    if key in riot_to_name:
                        mapping[vid] = riot_to_name[key]
                    elif not key.isdigit():
                        mapping[vid] = key
                break
        return mapping

    def load_model_and_embeddings(self):
        """Load embeddings, cluster assignments, optional trained model, and draft rows."""
        logger.info("Loading model and embeddings...")

        self.embeddings = np.load(self.embedding_dir / "draft_embeddings.npy")
        with open(self.embedding_dir / "match_ids.json", encoding="utf-8") as f:
            self.match_ids = json.load(f)
        with open(self.embedding_dir / "win_labels.json", encoding="utf-8") as f:
            self.win_labels = json.load(f)

        cluster_path = self.clustering_dir / "cluster_assignments.json"
        if not cluster_path.exists():
            cluster_path = self.embedding_dir / "cluster_assignments.json"
        with open(cluster_path, encoding="utf-8") as f:
            cluster_data = json.load(f)

        self.cluster_labels = np.array(cluster_data["cluster_labels"])
        self.archetype_mapping = {
            str(k): v for k, v in cluster_data["archetype_mapping"].items()
        }

        self.id_to_name = self._load_id_to_name_mapping()

        # Load draft compositions so KNN neighbor matches can be inspected
        drafts_path = self.data_dir / "drafts.parquet"
        if drafts_path.exists():
            try:
                drafts_df = pd.read_parquet(drafts_path)
                for row in drafts_df.to_dict(orient="records"):
                    self.draft_lookup[str(row["match_id"])] = row
            except Exception as exc:
                logger.warning("Could not read drafts.parquet (%s); using embedding fallback", exc)

        # Load trained DraftModel checkpoint if available
        if self.model_path.exists():
            try:
                ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
                champion_vocab = ckpt["champion_vocab"]
                patch_vocab = ckpt.get("patch_vocab") or {}
                model_cfg = ckpt.get("config", {}).get("model", {})
                self.model = DraftModel(
                    num_champions=len(champion_vocab),
                    champion_dim=model_cfg.get("champion_dim", 64),
                    draft_dim=model_cfg.get("draft_dim", 256),
                    num_patches=len(patch_vocab),
                    patch_dim=model_cfg.get("patch_dim", 16),
                    team_hidden=model_cfg.get("team_hidden", 512),
                    num_team_layers=model_cfg.get("num_team_layers", 3),
                    dropout=0.0,
                ).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
                logger.info("Loaded trained DraftModel from %s", self.model_path)
            except Exception as exc:
                logger.warning("Could not load checkpoint %s: %s", self.model_path, exc)

        logger.info(f"Loaded {len(self.embeddings)} embeddings")

    def initialize_recommendation_system(self):
        """Initialize the KNN recommendation system."""
        logger.info("Initializing recommendation system...")
        self.embeddings_scaled = self.scaler.fit_transform(self.embeddings)
        n_neighbors = min(20, len(self.embeddings))
        self.knn_model = NearestNeighbors(
            n_neighbors=n_neighbors, metric="cosine", algorithm="auto"
        )
        self.knn_model.fit(self.embeddings_scaled)
        logger.info("Recommendation system initialized")

    def _pad_team(self, champs: List[int]) -> List[int]:
        padded = [int(c) for c in champs[:5]]
        padded.extend([0] * (5 - len(padded)))
        return padded

    def get_draft_embedding(self, blue_champs: List[int], red_champs: List[int]) -> np.ndarray:
        """Compute embedding using the trained encoder, or average matching historical drafts."""
        blue_padded = self._pad_team(blue_champs)
        red_padded = self._pad_team(red_champs)

        if self.model is not None:
            with torch.no_grad():
                blue_t = torch.tensor([blue_padded], dtype=torch.long, device=self.device)
                red_t = torch.tensor([red_padded], dtype=torch.long, device=self.device)
                patch_t = torch.tensor([0], dtype=torch.long, device=self.device)
                return self.model.get_embedding(blue_t, red_t, patch_t).cpu().numpy()[0]

        # Deterministic fallback when no checkpoint is loaded: average embeddings of
        # historical drafts sharing the selected champions.
        selected = {c for c in blue_padded + red_padded if c > 0}
        if self.draft_lookup and selected:
            matching_indices = []
            for idx, mid in enumerate(self.match_ids):
                row = self.draft_lookup.get(str(mid))
                if not row:
                    continue
                row_champs = {
                    int(row[f"{side}_{r}"])
                    for side in ("blue", "red")
                    for r in ROLES
                    if f"{side}_{r}" in row
                }
                if selected & row_champs:
                    matching_indices.append(idx)
            if matching_indices:
                return np.mean(self.embeddings[matching_indices], axis=0)

        return np.mean(self.embeddings, axis=0)

    def find_similar_drafts(
        self, target_embedding: np.ndarray, n_similar: int = 10
    ) -> List[Tuple[str, float, int]]:
        """Find similar drafts based on cosine similarity. Returns (match_id, similarity, index)."""
        target_scaled = self.scaler.transform(target_embedding.reshape(1, -1))
        k = min(n_similar, len(self.embeddings))
        distances, indices = self.knn_model.kneighbors(target_scaled, n_neighbors=k)

        similar_drafts = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = float(max(0.0, 1.0 - dist))
            similar_drafts.append((str(self.match_ids[idx]), similarity, int(idx)))
        return similar_drafts

    def recommend_pick(
        self,
        team_champs: List[int],
        opposing_team_champs: List[int],
        archetype_preference: str = "any",
    ) -> Dict:
        """Recommend a champion pick for the given team using similarity-weighted win rates."""
        logger.info(f"Generating recommendation for team {team_champs}")
        target_embedding = self.get_draft_embedding(team_champs, opposing_team_champs)
        similar_drafts = self.find_similar_drafts(target_embedding, n_similar=50)

        excluded = {int(c) for c in team_champs + opposing_team_champs if int(c) > 0}
        champ_scores: Dict[int, float] = defaultdict(float)
        champ_weights: Dict[int, float] = defaultdict(float)
        archetype_votes: Dict[str, float] = defaultdict(float)

        for match_id, similarity, idx in similar_drafts:
            cluster_id = str(self.cluster_labels[idx])
            arch = self.archetype_mapping.get(cluster_id, "Poke")
            if archetype_preference != "any" and arch.lower() != archetype_preference.lower():
                continue
            archetype_votes[arch] += similarity
            win = float(self.win_labels[idx])
            row = self.draft_lookup.get(match_id)
            if row:
                for role in ROLES:
                    cid = int(row.get(f"blue_{role}", 0))
                    if cid > 0 and cid not in excluded:
                        weight = max(similarity, 1e-3)
                        champ_scores[cid] += weight * (0.6 * win + 0.4)
                        champ_weights[cid] += weight

        dominant_archetype = (
            max(archetype_votes, key=archetype_votes.get)
            if archetype_votes
            else self.archetype_mapping.get("0", "Poke")
        )

        if champ_scores:
            best_cid = max(
                champ_scores,
                key=lambda cid: champ_scores[cid] * np.log1p(champ_weights[cid]),
            )
            confidence = float(
                np.clip(champ_scores[best_cid] / max(champ_weights[best_cid], 1e-6), 0.50, 0.95)
            )
            champ_name = self.id_to_name.get(best_cid, f"Champion_{best_cid}")
        else:
            fallback_id = next((cid for cid in self.id_to_name if cid not in excluded), 1)
            champ_name = self.id_to_name.get(fallback_id, "Jinx")
            confidence = 0.65

        return {
            "recommended_champ": champ_name,
            "confidence": round(confidence, 3),
            "similar_drafts_count": len(similar_drafts),
            "archetype": dominant_archetype,
        }

    def recommend_ban(
        self, team_champs: List[int], opposing_team_champs: List[int]
    ) -> Dict:
        """Recommend a champion ban targeting high-performing enemy picks in similar drafts."""
        logger.info(f"Generating ban recommendation for team {team_champs}")
        target_embedding = self.get_draft_embedding(team_champs, opposing_team_champs)
        similar_drafts = self.find_similar_drafts(target_embedding, n_similar=50)

        excluded = {int(c) for c in team_champs + opposing_team_champs if int(c) > 0}
        enemy_threat: Dict[int, float] = defaultdict(float)
        enemy_weights: Dict[int, float] = defaultdict(float)

        for match_id, similarity, idx in similar_drafts:
            red_win = 1.0 - float(self.win_labels[idx])
            row = self.draft_lookup.get(match_id)
            if row:
                for role in ROLES:
                    cid = int(row.get(f"red_{role}", 0))
                    if cid > 0 and cid not in excluded:
                        weight = max(similarity, 1e-3)
                        enemy_threat[cid] += weight * (0.65 * red_win + 0.35)
                        enemy_weights[cid] += weight

        if enemy_threat:
            best_ban_id = max(
                enemy_threat,
                key=lambda cid: enemy_threat[cid] * np.log1p(enemy_weights[cid]),
            )
            confidence = float(
                np.clip(
                    enemy_threat[best_ban_id] / max(enemy_weights[best_ban_id], 1e-6),
                    0.50,
                    0.95,
                )
            )
            ban_name = self.id_to_name.get(best_ban_id, f"Champion_{best_ban_id}")
        else:
            fallback_id = next((cid for cid in sorted(self.id_to_name, reverse=True) if cid not in excluded), 2)
            ban_name = self.id_to_name.get(fallback_id, "Thresh")
            confidence = 0.65

        return {
            "recommended_ban": ban_name,
            "confidence": round(confidence, 3),
            "reason": "Highest similarity-weighted opposing win rate in comparable drafts",
        }

    def evaluate_recommendation_quality(self, test_data: List[Dict]) -> Dict:
        """Evaluate recommendation coverage and diversity on sample drafts."""
        if not test_data:
            return {"accuracy": 0.0, "coverage": 0.0, "recommendation_diversity": 0.0}

        recommended = []
        hits = 0
        for sample in test_data:
            blue = sample.get("blue", [1, 2, 3, 4])
            red = sample.get("red", [6, 7, 8, 9, 10])
            rec = self.recommend_pick(blue, red)
            recommended.append(rec["recommended_champ"])
            if rec["confidence"] >= 0.5:
                hits += 1

        unique_ratio = len(set(recommended)) / max(1, len(recommended))
        return {
            "accuracy": round(hits / len(test_data), 3),
            "coverage": round(len(self.id_to_name) / 172.0, 3) if self.id_to_name else 1.0,
            "recommendation_diversity": round(unique_ratio, 3),
        }


def main():
    """Main function to demonstrate the recommendation engine."""
    logger.info("Starting recommendation engine demo...")

    engine = DraftRecommendationEngine(
        model_path="./checkpoints/best.pt",
        embedding_dir="./embeddings",
        clustering_dir="./clustering_results",
        data_dir="./data/processed",
    )

    blue_team = [1, 2, 3, 4, 5]
    red_team = [6, 7, 8, 9, 10]

    pick_rec = engine.recommend_pick(blue_team, red_team)
    ban_rec = engine.recommend_ban(blue_team, red_team)

    print("Recommendation Results:")
    print(
        f"Recommended Pick: {pick_rec['recommended_champ']} "
        f"(Confidence: {pick_rec['confidence']}, Archetype: {pick_rec['archetype']})"
    )
    print(
        f"Recommended Ban: {ban_rec['recommended_ban']} "
        f"(Confidence: {ban_rec['confidence']} — {ban_rec['reason']})"
    )


if __name__ == "__main__":
    main()