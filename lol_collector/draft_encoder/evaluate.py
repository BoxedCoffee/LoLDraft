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
        for i in range(4):
            all_obj_preds[i].append(obj_preds[i].argmax(-1).cpu().numpy())
            all_obj_labels[i].append(obj_t[:, i].numpy())

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
        preds = np.concatenate(all_obj_preds[i])
        labels = np.concatenate(all_obj_labels[i])
        acc = (preds == labels).mean()
        majority = max(np.bincount(labels, minlength=3) / len(labels))
        obj_acc[name] = {"accuracy": float(acc), "majority_baseline": float(majority)}

    return {
        "win_accuracy": float(win_acc),
        "win_majority_baseline": float(majority_baseline),
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


def nearest_neighbours(model, champion_vocab, patch_vocab, device, n=5):
    """
    Query nearest neighbours for some known compositions.
    """
    inv_vocab = {v: k for k, v in champion_vocab.items()}

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
        blue_ids = [champion_vocab.get(c, 0) for c in comp["blue"]]
        red_ids = [champion_vocab.get(c, 0) for c in comp["red"]]

        if any(id == 0 for id in blue_ids + red_ids):
            missing = [c for c in comp["blue"] + comp["red"] if champion_vocab.get(c, 0) == 0]
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
    nearest_neighbours(model, champion_vocab, patch_vocab, device)

    logger.info("=" * 50)
    logger.info("Output saved to %s", output_dir)


if __name__ == "__main__":
    main()
