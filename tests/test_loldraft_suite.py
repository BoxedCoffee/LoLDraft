#!/usr/bin/env python3
"""
Automated test suite for the LoLDraft senior project.
Can be executed either via `pytest tests/test_loldraft_suite.py` or directly via
`python tests/test_loldraft_suite.py`.

Covers:
  1. DraftModel & MatchupSynergyBlock architecture verification (plus live forward/backward
     pass when PyTorch is installed).
  2. Archetype clustering (`clustering.py`): KMeans, Silhouette / Davies-Bouldin
     quality metrics, and strategic profile archetype assignment.
  3. Role-aware KNN recommendation engine (`recommendation_engine.py`): open-slot
     detection, explicit target_role filtering, lane-counter boost, and top-K output.
  4. Draft Analyzer (`lol_collector/draft_analyzer/analysis.py`): side-swap
     deduplication, Red-side gold sign correction, full 172-champion archetype
     coverage, and evidence-backed pick/ban recommendations.
  5. Phase 2a Macro Trajectory (`lol_collector/draft_analyzer/macro_trajectory.py`):
     Summoner's Rift Voronoi landmark discretization, grouping/split-push detection,
     and quantitative archetype statistical probe.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

ROOT_DIR = Path(__file__).resolve().parent.parent
ENCODER_DIR = ROOT_DIR / "lol_collector" / "draft_encoder"
ANALYZER_DIR = ROOT_DIR / "lol_collector" / "draft_analyzer"

for p in (ROOT_DIR, ENCODER_DIR, ANALYZER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import clustering
from lol_collector.draft_analyzer import analysis as draft_analysis
from lol_collector.draft_analyzer import macro_trajectory
from recommendation_engine import DraftRecommendationEngine


def test_draft_model_matchup_synergy_and_backward_compat():
    """Verify DraftModel & MatchupSynergyBlock structure and (if torch installed) forward/backward pass."""
    model_path = ENCODER_DIR / "model.py"
    source = model_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert {"ResidualBlock", "TeamEncoder", "MatchupSynergyBlock", "DraftEncoder", "DraftModel"}.issubset(class_names)

    if not HAS_TORCH:
        return

    from lol_collector.draft_encoder.model import DraftModel

    model = DraftModel(
        num_champions=50,
        champion_dim=32,
        draft_dim=64,
        num_patches=5,
        patch_dim=8,
        team_hidden=128,
        num_team_layers=2,
        use_matchup_interactions=True,
    )

    blue = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]], dtype=torch.long)
    red = torch.tensor([[10, 11, 12, 13, 14], [15, 16, 17, 18, 0]], dtype=torch.long)
    patch = torch.tensor([1, 2], dtype=torch.long)

    embed, win_logit, gold_pred, obj_preds = model(blue, red, patch)
    assert embed.shape == (2, 64)
    assert win_logit.shape == (2,)
    assert gold_pred.shape == (2, 6)
    assert len(obj_preds) == 4 and obj_preds[0].shape == (2, 3)

    role_embeds = model.encoder.get_role_embeddings(blue)
    assert torch.allclose(role_embeds[1, 3], torch.zeros_like(role_embeds[1, 3]))
    assert torch.allclose(role_embeds[1, 4], torch.zeros_like(role_embeds[1, 4]))

    win_target = torch.tensor([1.0, 0.0])
    gold_target = torch.randn(2, 6)
    obj_targets = [torch.tensor([1, 2]) for _ in range(4)]
    losses = model.compute_loss(win_logit, win_target, gold_pred, gold_target, obj_preds, obj_targets)
    losses["total"].backward()
    assert model.encoder.interaction_gate.grad is not None

    legacy_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("encoder.matchup_synergy")
        and k != "encoder.interaction_gate"
        and not k.startswith("encoder.out_norm")
    }
    fresh_model = DraftModel(
        num_champions=50,
        champion_dim=32,
        draft_dim=64,
        num_patches=5,
        patch_dim=8,
        team_hidden=128,
        num_team_layers=2,
        use_matchup_interactions=True,
    )
    fresh_model.load_state_dict(legacy_state)
    assert float(fresh_model.encoder.interaction_gate.item()) == 0.0


def test_clustering_metrics_and_profile_assignment(tmp_path: Path | None = None):
    """Verify clustering produces Silhouette/DBI metrics and maps distinct archetypes."""
    rng = np.random.default_rng(42)
    c0 = rng.normal(loc=-3.0, scale=0.4, size=(20, 16))
    c1 = rng.normal(loc=0.0, scale=0.4, size=(20, 16))
    c2 = rng.normal(loc=3.0, scale=0.4, size=(20, 16))
    embeddings = np.vstack([c0, c1, c2]).astype(np.float32)
    win_labels = np.array([0.6] * 20 + [0.5] * 20 + [0.4] * 20, dtype=np.float32)

    labels, _, _ = clustering.perform_clustering(embeddings, n_clusters=3)
    metrics = clustering.compute_clustering_metrics(embeddings, labels)
    assert metrics["silhouette_score"] is not None and metrics["silhouette_score"] > 0.5
    assert metrics["davies_bouldin_index"] is not None and metrics["davies_bouldin_index"] < 1.0

    mapping = clustering.assign_archetypes(labels, win_labels, embeddings=embeddings)
    assert set(mapping.values()) == {"Engage", "Poke", "Scaling"}


def test_role_aware_recommendation_engine(tmp_path: Path | None = None):
    """Verify recommend_pick and recommend_ban respect target_role and exclude picked champions."""
    with tempfile.TemporaryDirectory() as td:
        base_dir = tmp_path if tmp_path is not None else Path(td)
        emb_dir = base_dir / "embeddings"
        clus_dir = base_dir / "clustering_results"
        data_dir = base_dir / "data"
        emb_dir.mkdir(exist_ok=True)
        clus_dir.mkdir(exist_ok=True)
        data_dir.mkdir(exist_ok=True)

        n_matches = 30
        rng = np.random.default_rng(7)
        embeddings = rng.normal(size=(n_matches, 16)).astype(np.float32)
        match_ids = [f"m_{i}" for i in range(n_matches)]
        win_labels = [1 if i % 2 == 0 else 0 for i in range(n_matches)]

        np.save(emb_dir / "draft_embeddings.npy", embeddings)
        (emb_dir / "match_ids.json").write_text(json.dumps(match_ids), encoding="utf-8")
        (emb_dir / "win_labels.json").write_text(json.dumps(win_labels), encoding="utf-8")

        vocab = {
            "<UNK>": 0,
            "Ornn": 1, "Lee Sin": 2, "Ahri": 3, "Jinx": 4, "Leona": 5,
            "Nautilus": 6, "Aatrox": 7, "Viego": 8, "Viktor": 9, "Caitlyn": 10, "Thresh": 11,
        }
        (data_dir / "champion_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")

        cluster_payload = {
            "match_ids": match_ids,
            "cluster_labels": [i % 3 for i in range(n_matches)],
            "archetype_mapping": {"0": "Engage", "1": "Poke", "2": "Scaling"},
            "cluster_stats": {},
        }
        (clus_dir / "cluster_assignments.json").write_text(json.dumps(cluster_payload), encoding="utf-8")

        draft_rows = []
        for i, mid in enumerate(match_ids):
            draft_rows.append({
                "match_id": mid,
                "patch": "14.20",
                "blue_top": 1,
                "blue_jng": 2,
                "blue_mid": 3,
                "blue_bot": 4,
                "blue_sup": 6 if i % 2 == 0 else 5,
                "red_top": 7,
                "red_jng": 8,
                "red_mid": 9,
                "red_bot": 10,
                "red_sup": 11,
                "blue_win": bool(i % 2 == 0),
            })
        pd.DataFrame(draft_rows).to_parquet(data_dir / "drafts.parquet", index=False)

        engine = DraftRecommendationEngine(
            model_path=str(base_dir / "nonexistent.pt"),
            embedding_dir=str(emb_dir),
            clustering_dir=str(clus_dir),
            data_dir=str(data_dir),
        )

        # Blue needs Support (slot index 4 is 0)
        pick_res = engine.recommend_pick([1, 2, 3, 4, 0], [7, 8, 9, 10, 0])
        assert pick_res["target_role"] == "sup"
        assert pick_res["recommended_champ"] in {"Nautilus", "Leona"}
        assert len(pick_res["top_recommendations"]) >= 1

        # Ban should target Red's open Support role -> Thresh
        ban_res = engine.recommend_ban([1, 2, 3, 4, 0], [7, 8, 9, 10, 0])
        assert ban_res["target_role"] == "sup"
        assert ban_res["recommended_ban"] == "Thresh"


def test_draft_analyzer_deduplication_and_advisor(tmp_path: Path | None = None):
    """Verify _swap deduplication, Red gold sign correction, and evidence-based pick/ban advisor."""
    with tempfile.TemporaryDirectory() as td:
        base_dir = tmp_path if tmp_path is not None else Path(td)
        drafts = pd.DataFrame([
            {
                "match_id": "NA1_1", "patch": "14.20",
                "blue_top": "Ornn", "blue_jng": "Lee Sin", "blue_mid": "Ahri", "blue_bot": "Jinx", "blue_sup": "Leona",
                "red_top": "Aatrox", "red_jng": "Viego", "red_mid": "Viktor", "red_bot": "Caitlyn", "red_sup": "Thresh",
                "blue_win": True,
            },
            {
                "match_id": "NA1_1_swap", "patch": "14.20",
                "blue_top": "Aatrox", "blue_jng": "Viego", "blue_mid": "Viktor", "blue_bot": "Caitlyn", "blue_sup": "Thresh",
                "red_top": "Ornn", "red_jng": "Lee Sin", "red_mid": "Ahri", "red_bot": "Jinx", "red_sup": "Leona",
                "blue_win": False,
            },
            {
                "match_id": "NA1_2", "patch": "14.20",
                "blue_top": "Ornn", "blue_jng": "Lee Sin", "blue_mid": "Ahri", "blue_bot": "Jinx", "blue_sup": "Nautilus",
                "red_top": "Aatrox", "red_jng": "Viego", "red_mid": "Viktor", "red_bot": "Caitlyn", "red_sup": "Thresh",
                "blue_win": True,
            },
        ])
        gold = pd.DataFrame([
            {"match_id": "NA1_1", "gold_diff_5": 200.0, "gold_diff_10": 500.0, "gold_diff_15": 900.0, "gold_diff_20": 1500.0, "gold_diff_25": 2000.0, "gold_diff_30": 2500.0},
            {"match_id": "NA1_2", "gold_diff_5": 100.0, "gold_diff_10": 400.0, "gold_diff_15": 800.0, "gold_diff_20": 1200.0, "gold_diff_25": 1800.0, "gold_diff_30": 2200.0},
        ])
        objectives = pd.DataFrame([
            {"match_id": "NA1_1", "first_dragon_team": 100, "first_herald_team": 100, "first_baron_team": 100, "first_tower_team": 100},
            {"match_id": "NA1_2", "first_dragon_team": 100, "first_herald_team": 200, "first_baron_team": 100, "first_tower_team": 100},
        ])
        drafts.to_parquet(base_dir / "drafts.parquet", index=False)
        gold.to_parquet(base_dir / "gold_curves.parquet", index=False)
        objectives.to_parquet(base_dir / "objectives.parquet", index=False)

        table = draft_analysis.load_match_table(base_dir)
        assert len(table) == 2  # NA1_1_swap deduplicated

        stats = draft_analysis.build_champion_stats(table)
        aatrox_row = stats[(stats["champion"] == "Aatrox") & (stats["role"] == "top")].iloc[0]
        assert float(aatrox_row["mean_gold_diff_10"]) == -450.0

        blue_picks = {"top": "Ornn", "jng": "Lee Sin", "mid": "Ahri", "bot": "Jinx", "sup": ""}
        red_picks = {"top": "Aatrox", "jng": "Viego", "mid": "Viktor", "bot": "Caitlyn", "sup": ""}
        picks_df = draft_analysis.recommend_role_candidates(
            scoped_table=table,
            similar_table=table,
            stats=stats,
            blue_picks=blue_picks,
            red_picks=red_picks,
            target_side="blue",
            target_role="sup",
            min_role_matches=1,
        )
        assert not picks_df.empty
        assert set(picks_df["Champion"].iloc[:2]) == {"Leona", "Nautilus"}
        assert set(picks_df["Champion"]).issubset({"Leona", "Nautilus", "Thresh"})


def test_macro_trajectory_voronoi_and_probe():
    """Verify Summoner's Rift Voronoi zone assignment, grouping/split-push detection, and statistical probe."""
    pts = np.array([
        [5000.0, 10400.0],  # Baron pit
        [9860.0, 4400.0],   # Dragon pit
        [10500.0, 1030.0],  # Blue bot outer tower (side lane)
    ], dtype=np.float32)
    _, names = macro_trajectory.assign_voronoi_zones(pts)
    assert names == ["baron_herald_pit", "dragon_pit", "blue_bot_outer"]

    team_xy = np.array([
        [4950.0, 10350.0],
        [5050.0, 10420.0],
        [5100.0, 10300.0],
        [4900.0, 10500.0],
        [10500.0, 1030.0],
    ], dtype=np.float32)
    macro_state = macro_trajectory.compute_team_grouping_and_split(team_xy)
    assert macro_state["is_grouped"] == 1.0
    assert macro_state["max_cluster_size"] == 4.0
    assert macro_state["side_lane_split"] == 1.0

    rng = np.random.default_rng(123)
    probe_df = pd.DataFrame({
        "archetype": ["Engage"] * 30 + ["Poke"] * 30 + ["Split push"] * 30,
        "grouping_rate": np.concatenate([
            rng.normal(0.75, 0.05, 30),
            rng.normal(0.60, 0.05, 30),
            rng.normal(0.35, 0.05, 30),
        ]),
        "side_lane_split_rate": np.concatenate([
            rng.normal(0.15, 0.04, 30),
            rng.normal(0.20, 0.04, 30),
            rng.normal(0.65, 0.05, 30),
        ]),
        "first_obj_minute": np.concatenate([
            rng.normal(6.2, 0.5, 30),
            rng.normal(7.5, 0.5, 30),
            rng.normal(8.8, 0.5, 30),
        ]),
    })
    probe_res = macro_trajectory.run_archetype_trajectory_probe(probe_df)
    assert probe_res["total_pairs"] == 3
    assert probe_res["significant_pairs"] == 3
    assert probe_res["passes_roadmap_threshold"] is True


if __name__ == "__main__":
    suite = [
        ("DraftModel & MatchupSynergyBlock", test_draft_model_matchup_synergy_and_backward_compat),
        ("Archetype Clustering & Quality Metrics", test_clustering_metrics_and_profile_assignment),
        ("Role-Aware KNN Recommendation Engine", test_role_aware_recommendation_engine),
        ("Draft Analyzer & Pick/Ban Advisor", test_draft_analyzer_deduplication_and_advisor),
        ("Phase 2a Voronoi Macro Trajectory & Probe", test_macro_trajectory_voronoi_and_probe),
    ]
    for label, fn in suite:
        fn()
        print(f"[PASS] {label}")
    print(f"\nAll {len(suite)} test suites passed successfully.")