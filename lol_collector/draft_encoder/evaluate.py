#!/usr/bin/env python3
"""
Evaluate a trained draft encoder.

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --data-dir data/processed

Outputs:
  - Win prediction accuracy vs baselines
  - Gold curve MAE per timestep
  - Objective prediction accuracy
  - t-SNE / UMAP visualization of draft embeddings
  - Nearest neighbour queries
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import DraftModel
from dataset import DraftDataset


def normalize_champion_name(name: str) -> str:
    """Normalize common display-name differences used by evaluation queries."""
    return "".join(character.casefold() for character in name if character.isalnum())


def build_normalized_vocab(
    champion_vocab: dict, champion_mapping: dict | None = None
) -> dict[str, int]:
    if champion_mapping:
        return {
            normalize_champion_name(name): int(champion_vocab[str(riot_id)])
            for name, riot_id in champion_mapping.items()
            if str(riot_id) in champion_vocab and int(champion_vocab[str(riot_id)]) > 0
        }
    return {
        normalize_champion_name(name): int(champion_id)
        for name, champion_id in champion_vocab.items()
        if name != "<UNK>"
    }


def bootstrap_accuracy_interval(
    probabilities: np.ndarray, labels: np.ndarray, seed: int = 42, samples: int = 500
) -> list[float]:
    """Estimate a percentile confidence interval for threshold accuracy."""
    rng = np.random.default_rng(seed)
    predictions = probabilities > 0.5
    scores = np.empty(samples, dtype=np.float32)
    indices = np.arange(len(labels))
    for index in range(samples):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        scores[index] = np.mean(predictions[sampled] == labels[sampled])
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("evaluate")


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    champion_vocab = ckpt["champion_vocab"]
    patch_vocab = ckpt["patch_vocab"]

    model_cfg = cfg.get("model", {})
    model = DraftModel(
        num_champions=len(champion_vocab),
        champion_dim=model_cfg.get("champion_dim", 64),
        draft_dim=model_cfg.get("draft_dim", 256),
        num_patches=len(patch_vocab),
        patch_dim=model_cfg.get("patch_dim", 16),
        team_hidden=model_cfg.get("team_hidden", 512),
        num_team_layers=model_cfg.get("num_team_layers", 3),
        dropout=0.0,  # no dropout at eval
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, champion_vocab, patch_vocab, cfg


@torch.no_grad()
def compute_metrics(model, loader, device) -> dict:
    """Compute all evaluation metrics."""
    all_win_preds = []
    all_win_labels = []
    all_gold_preds = []
    all_gold_targets = []
    all_gold_masks = []
    all_obj_preds = [[] for _ in range(4)]
    all_obj_labels = [[] for _ in range(4)]
    all_obj_masks = [[] for _ in range(4)]
    all_embeddings = []

    for batch in loader:
        blue = batch["blue_champs"].to(device)
        red = batch["red_champs"].to(device)
        patch = batch["patch_id"].to(device)

        embed, win_logit, gold_pred, obj_preds = model(blue, red, patch)

        all_embeddings.append(embed.cpu().numpy())
        all_win_preds.append(torch.sigmoid(win_logit).cpu().numpy())
        all_win_labels.append(batch["win_label"].numpy())
        all_gold_preds.append(gold_pred.cpu().numpy())
        all_gold_targets.append(batch["gold_targets"].numpy())
        all_gold_masks.append(batch["gold_mask"].numpy())

        obj_t = batch["obj_targets"]
        obj_m = batch["obj_mask"]
        for i in range(4):
            mask = obj_m[:, i] > 0
            all_obj_preds[i].append(obj_preds[i].argmax(-1).cpu().numpy()[mask])
            all_obj_labels[i].append(obj_t[:, i].numpy()[mask])

    # Concatenate
    win_preds = np.concatenate(all_win_preds)
    win_labels = np.concatenate(all_win_labels)
    gold_preds = np.concatenate(all_gold_preds)
    gold_targets = np.concatenate(all_gold_targets)
    gold_masks = np.concatenate(all_gold_masks)
    embeddings = np.concatenate(all_embeddings)

    # Win accuracy
    win_acc = ((win_preds > 0.5) == win_labels).mean()
    majority_baseline = max(win_labels.mean(), 1 - win_labels.mean())
    win_logloss = float(
        -(win_labels * np.log(np.clip(win_preds, 1e-7, 1 - 1e-7))
          + (1 - win_labels) * np.log(np.clip(1 - win_preds, 1e-7, 1 - 1e-7))).mean()
    )
    baseline_probability = float(win_labels.mean())
    baseline_logloss = float(
        -(win_labels * np.log(np.clip(baseline_probability, 1e-7, 1 - 1e-7))
          + (1 - win_labels) * np.log(
              np.clip(1 - baseline_probability, 1e-7, 1 - 1e-7)
          )).mean()
    )
    win_brier = float(np.mean((win_preds - win_labels) ** 2))
    baseline_brier = float(np.mean((baseline_probability - win_labels) ** 2))
    win_accuracy_ci = bootstrap_accuracy_interval(win_preds, win_labels)
    try:
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        win_balanced_accuracy = float(
            balanced_accuracy_score(win_labels, win_preds > 0.5)
        )
        win_auc = float(roc_auc_score(win_labels, win_preds))
    except (ImportError, ValueError):
        win_balanced_accuracy = None
        win_auc = None

    # Gold MAE per timestep
    gold_mae = {}
    for i, minute in enumerate([5, 10, 15, 20, 25, 30]):
        mask = gold_masks[:, i] > 0
        if mask.sum() > 0:
            mae = np.abs(gold_preds[mask, i] - gold_targets[mask, i]).mean()
            mean_baseline = np.abs(gold_targets[mask, i] - gold_targets[mask, i].mean()).mean()
            gold_mae[f"min_{minute}"] = {"mae": float(mae), "mean_baseline": float(mean_baseline)}

    # Objective accuracy
    obj_acc = {}
    obj_names = ["dragon", "herald", "baron", "tower"]
    for i, name in enumerate(obj_names):
        preds = np.concatenate(all_obj_preds[i]) if all_obj_preds[i] else np.array([])
        labels = np.concatenate(all_obj_labels[i]) if all_obj_labels[i] else np.array([])
        if len(labels) == 0:
            obj_acc[name] = {"accuracy": None, "majority_baseline": None, "samples": 0}
            continue
        acc = (preds == labels).mean()
        majority = max(np.bincount(labels, minlength=3) / len(labels))
        obj_acc[name] = {
            "accuracy": float(acc),
            "majority_baseline": float(majority),
            "samples": int(len(labels)),
            "class_counts": {
                str(class_id): int((labels == class_id).sum())
                for class_id in range(3)
            },
        }

    return {
        "win_accuracy": float(win_acc),
        "win_majority_baseline": float(majority_baseline),
        "win_logloss": win_logloss,
        "win_baseline_logloss": baseline_logloss,
        "win_brier": win_brier,
        "win_baseline_brier": baseline_brier,
        "win_balanced_accuracy": win_balanced_accuracy,
        "win_auc": win_auc,
        "win_accuracy_ci_95": win_accuracy_ci,
        "gold_mae": gold_mae,
        "objective_accuracy": obj_acc,
        "embeddings": embeddings,
        "win_labels": win_labels,
    }


def visualize_embeddings(embeddings: np.ndarray, labels: np.ndarray, output_path: str):
    """t-SNE visualization of draft embeddings colored by win/loss."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping visualization")
        return

    logger.info("Running t-SNE on %d embeddings...", len(embeddings))

    # Subsample for speed
    max_points = 10000
    if len(embeddings) > max_points:
        idx = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        coords = tsne.fit_transform(embeddings)
    except ImportError:
        logger.warning("sklearn not available — skipping t-SNE")
        return

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="coolwarm",
                          alpha=0.3, s=5)
    plt.colorbar(scatter, label="Blue Win")
    plt.title("Draft Embeddings (t-SNE) — Colored by Blue Win")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info("  Saved t-SNE to %s", output_path)
    plt.close()


def nearest_neighbours(
    model, champion_vocab, patch_vocab, device, champion_mapping=None, n=5
):
    """
    Query nearest neighbours for some known compositions.
    """
    normalized_vocab = build_normalized_vocab(champion_vocab, champion_mapping)

    # Example compositions (update champion names to match your vocab)
    test_comps = {
        "Teamfight (Malphite/Amumu/Orianna/MissFortune/Leona)": {
            "blue": ["Malphite", "Amumu", "Orianna", "MissFortune", "Leona"],
            "red": ["Gnar", "LeeSin", "Ahri", "Jinx", "Thresh"],
        },
        "Poke (Jayce/Nidalee/Xerath/Ezreal/Karma)": {
            "blue": ["Jayce", "Nidalee", "Xerath", "Ezreal", "Karma"],
            "red": ["Gnar", "LeeSin", "Ahri", "Jinx", "Thresh"],
        },
    }

    logger.info("\nNearest Neighbour Queries:")
    for name, comp in test_comps.items():
        blue_ids = [
            normalized_vocab.get(normalize_champion_name(c), 0)
            for c in comp["blue"]
        ]
        red_ids = [
            normalized_vocab.get(normalize_champion_name(c), 0)
            for c in comp["red"]
        ]

        if any(id == 0 for id in blue_ids + red_ids):
            missing = [
                c for c in comp["blue"] + comp["red"]
                if normalized_vocab.get(normalize_champion_name(c), 0) == 0
            ]
            logger.info("  Skipping '%s' — missing champions: %s", name, missing)
            continue

        blue_t = torch.tensor([blue_ids], dtype=torch.long, device=device)
        red_t = torch.tensor([red_ids], dtype=torch.long, device=device)
        patch_t = torch.tensor([0], dtype=torch.long, device=device)

        embed = model.get_embedding(blue_t, red_t, patch_t)
        logger.info("  %s → embed norm: %.3f", name, embed.norm().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data/processed")
    parser.add_argument("--output-dir", type=str, default="./eval_output")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, champion_vocab, patch_vocab, cfg = load_model(args.checkpoint, device)

    # Test set
    test_ds = DraftDataset(args.data_dir, split="test", patch_vocab=patch_vocab)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    logger.info("Test set: %d samples", len(test_ds))

    # Compute metrics
    metrics = compute_metrics(model, test_loader, device)

    # Print results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info("Win Accuracy:     %.3f (majority baseline: %.3f)",
                metrics["win_accuracy"], metrics["win_majority_baseline"])
    logger.info(
        "Win accuracy 95%% CI: [%.3f, %.3f]",
        metrics["win_accuracy_ci_95"][0],
        metrics["win_accuracy_ci_95"][1],
    )
    logger.info(
        "Win log loss:      %.4f (baseline %.4f) | Brier: %.4f (baseline %.4f) | "
        "Balanced accuracy: %s | ROC-AUC: %s",
        metrics["win_logloss"],
        metrics["win_baseline_logloss"],
        metrics["win_brier"],
        metrics["win_baseline_brier"],
        f"{metrics['win_balanced_accuracy']:.3f}"
        if metrics["win_balanced_accuracy"] is not None else "n/a",
        f"{metrics['win_auc']:.3f}" if metrics["win_auc"] is not None else "n/a",
    )
    logger.info("")

    logger.info("Gold Curve MAE:")
    for minute, vals in metrics["gold_mae"].items():
        logger.info("  %s: MAE=%.1f (mean baseline=%.1f, improvement=%.1f%%)",
                     minute, vals["mae"], vals["mean_baseline"],
                     100 * (1 - vals["mae"] / vals["mean_baseline"]) if vals["mean_baseline"] > 0 else 0)
    logger.info("")

    logger.info("Objective Prediction:")
    for obj, vals in metrics["objective_accuracy"].items():
        logger.info("  %s: %.3f (majority baseline: %.3f)",
                     obj, vals["accuracy"], vals["majority_baseline"])
        logger.info("    class counts: %s", vals.get("class_counts", {}))

    # Save metrics
    save_metrics = {k: v for k, v in metrics.items() if k not in ("embeddings", "win_labels")}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(save_metrics, f, indent=2)

    # Visualize
    visualize_embeddings(
        metrics["embeddings"], metrics["win_labels"],
        str(output_dir / "tsne_embeddings.png")
    )

    # Nearest neighbours
    mapping_path = Path(__file__).resolve().parent / "data" / "full_champion_mapping.json"
    champion_mapping = None
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as mapping_file:
            champion_mapping = json.load(mapping_file)
    nearest_neighbours(model, champion_vocab, patch_vocab, device, champion_mapping)

    logger.info("=" * 50)
    logger.info("Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
