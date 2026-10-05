#!/usr/bin/env python3
"""
Extract draft embeddings from trained model for archetype clustering.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import torch
import numpy as np
from torch.utils.data import DataLoader

from lol_collector.draft_encoder.dataset import DraftDataset
from lol_collector.draft_encoder.model import DraftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_embeddings")


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DraftModel, dict]:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    champion_vocab = checkpoint["champion_vocab"]
    patch_vocab = checkpoint.get("patch_vocab") or {}
    model_cfg = checkpoint.get("config", {}).get("model", {})

    # Reconstruct model with same parameters
    model = DraftModel(
        num_champions=len(champion_vocab),
        champion_dim=model_cfg.get("champion_dim", 64),
        draft_dim=model_cfg.get("draft_dim", 256),
        num_patches=len(patch_vocab),
        patch_dim=model_cfg.get("patch_dim", 16),
        team_hidden=model_cfg.get("team_hidden", 512),
        num_team_layers=model_cfg.get("num_team_layers", 3),
        num_gold_points=6,
        num_objectives=4,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, patch_vocab


def extract_embeddings(
    model: DraftModel, dataset: DraftDataset, device: torch.device
) -> Dict:
    """Extract all draft embeddings from dataset."""
    logger.info("Extracting embeddings...")

    if len(dataset) == 0:
        raise ValueError("The selected dataset split contains no drafts")

    dataloader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)

    embeddings = []
    match_ids = []
    win_labels = []

    with torch.no_grad():
        for batch in dataloader:
            blue_champs = batch["blue_champs"].to(device)
            red_champs = batch["red_champs"].to(device)
            patch_ids = batch["patch_id"].to(device)

            # Get just the draft embedding
            draft_embed = model.get_embedding(blue_champs, red_champs, patch_ids)
            embeddings.append(draft_embed.cpu().numpy())

            match_ids.extend(batch["match_id"])
            win_labels.extend(batch["win_label"].cpu().numpy())

    return {
        "embeddings": np.vstack(embeddings),
        "match_ids": match_ids,
        "win_labels": win_labels
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the checkpoint before constructing the dataset so its patch
    # vocabulary is reused exactly during inference.
    data_dir = Path("./data/processed")
    checkpoint_path = "./checkpoints/best.pt"
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    model, patch_vocab = load_model(checkpoint_path, device)

    # Extract embeddings
    test_ds = DraftDataset(data_dir, split="test", patch_vocab=patch_vocab)
    result = extract_embeddings(model, test_ds, device)

    # Save embeddings
    output_dir = Path("./embeddings")
    output_dir.mkdir(exist_ok=True)

    np.save(output_dir / "draft_embeddings.npy", result["embeddings"])
    with open(output_dir / "match_ids.json", "w") as f:
        json.dump(result["match_ids"], f)
    with open(output_dir / "win_labels.json", "w") as f:
        json.dump(result["win_labels"], f)

    logger.info(f"Extracted {len(result['embeddings'])} embeddings")
    logger.info(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()