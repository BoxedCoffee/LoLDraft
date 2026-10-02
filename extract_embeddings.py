#!/usr/bin/env python3
"""
Extract draft embeddings from trained model for archetype clustering.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from torch.utils.data import DataLoader

from lol_collector.draft_encoder.dataset import DraftDataset
from lol_collector.draft_encoder.model import DraftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("extract_embeddings")


def load_model(checkpoint_path: str, device: str) -> DraftModel:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model with same parameters
    model = DraftModel(
        num_champions=checkpoint["champion_vocab"]["<PAD>"] + 1,
        champion_dim=64,
        draft_dim=256,
        num_patches=len(checkpoint["patch_vocab"]) if checkpoint["patch_vocab"] else 0,
        patch_dim=16,
        team_hidden=512,
        num_team_layers=3,
        num_gold_points=6,
        num_objectives=4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def extract_embeddings(model: DraftModel, dataset: DraftDataset, device: str) -> Dict:
    """Extract all draft embeddings from dataset."""
    logger.info("Extracting embeddings...")

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

    # Load dataset (assuming it's already processed)
    data_dir = Path("./data/processed")
    test_ds = DraftDataset(data_dir, split="test")

    # Load model
    checkpoint_path = "./checkpoints/best.pt"
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    model = load_model(checkpoint_path, device)

    # Extract embeddings
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