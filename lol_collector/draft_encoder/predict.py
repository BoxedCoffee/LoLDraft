#!/usr/bin/env python3
"""
Simple inference script for the draft encoder model.
This demonstrates how to use the trained model for making predictions on new draft compositions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from model import DraftModel
OBJECTIVE_CLASSES = {0: "None", 1: "Blue Team", 2: "Red Team"}


def normalize_champion_name(name: str) -> str:
    return "".join(character.casefold() for character in name if character.isalnum())


def build_normalized_vocab(champion_vocab: dict, champion_mapping: dict | None = None) -> dict[str, int]:
    resolved = {
        normalize_champion_name(name): int(champion_id)
        for name, champion_id in champion_vocab.items()
        if name != "<UNK>" and int(champion_id) > 0
    }
    if champion_mapping:
        for name, riot_id in champion_mapping.items():
            if str(riot_id) in champion_vocab and int(champion_vocab[str(riot_id)]) > 0:
                resolved[normalize_champion_name(name)] = int(champion_vocab[str(riot_id)])
    return resolved

def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
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
    return model, champion_vocab, patch_vocab

def predict_draft(model, champion_vocab, patch_vocab, blue_team, red_team, device, champion_mapping=None):
    """
    Make a prediction for a draft composition.

    Args:
        model: trained DraftModel
        champion_vocab: vocabulary mapping champion names to IDs
        patch_vocab: vocabulary mapping patch names to IDs
        blue_team: list of 5 champion names for blue team
        red_team: list of 5 champion names for red team
        device: torch device

    Returns:
        dict with prediction results
    """

    # Convert champion names to IDs
    normalized_vocab = build_normalized_vocab(champion_vocab, champion_mapping)
    blue_ids = [normalized_vocab.get(normalize_champion_name(c), 0) for c in blue_team]
    red_ids = [normalized_vocab.get(normalize_champion_name(c), 0) for c in red_team]

    # Create tensors
    blue_tensor = torch.tensor([blue_ids], dtype=torch.long, device=device)
    red_tensor = torch.tensor([red_ids], dtype=torch.long, device=device)
    patch_tensor = torch.tensor([0], dtype=torch.long, device=device)  # Default patch

    with torch.no_grad():
        # Get the full model output
        draft_embed, win_logit, gold_pred, obj_preds = model(blue_tensor, red_tensor, patch_tensor)

        # Convert logits to probabilities
        win_prob = torch.sigmoid(win_logit).item()

        # Get objective predictions (argmax for each objective type)
        obj_predictions = []
        for pred in obj_preds:
            obj_pred = torch.argmax(pred, dim=-1).item()
            obj_predictions.append(obj_pred)  # 0=none, 1=blue, 2=red

        return {
            'draft_embedding': draft_embed.cpu().numpy()[0],
            'win_probability': win_prob,
            'gold_curve_prediction': gold_pred.cpu().numpy()[0],
            'objective_predictions': obj_predictions,
            'objective_labels': [OBJECTIVE_CLASSES.get(c, "Unknown") for c in obj_predictions],
            'blue_ids': blue_ids,
            'red_ids': red_ids,
            'blue_team': blue_team,
            'red_team': red_team
        }

def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained draft encoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run inference on (cuda or cpu)")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    model, champion_vocab, patch_vocab = load_model(args.checkpoint, device)
    mapping_path = Path(__file__).resolve().parent / "data" / "full_champion_mapping.json"
    champion_mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else None

    # Example prediction - you can modify this with your own teams
    print("\nMaking example predictions...")

    # Example 1: Teamfight composition
    blue_team_1 = ["Malphite", "Amumu", "Orianna", "MissFortune", "Leona"]
    red_team_1 = ["Gnar", "LeeSin", "Ahri", "Jinx", "Thresh"]

    result_1 = predict_draft(model, champion_vocab, patch_vocab, blue_team_1, red_team_1, device, champion_mapping)

    print(f"\nExample 1:")
    print(f"Blue Team: {blue_team_1}")
    print(f"Red Team: {red_team_1}")
    print(f"Win Probability: {result_1['win_probability']:.3f}")
    print(f"Gold Curve Prediction (6 time points): {result_1['gold_curve_prediction']}")
    print(f"Objective Predictions: {result_1['objective_predictions']}")

    # Example 2: Poke composition
    blue_team_2 = ["Jayce", "Nidalee", "Xerath", "Ezreal", "Karma"]
    red_team_2 = ["Gnar", "LeeSin", "Ahri", "Jinx", "Thresh"]

    result_2 = predict_draft(model, champion_vocab, patch_vocab, blue_team_2, red_team_2, device, champion_mapping)

    print(f"\nExample 2:")
    print(f"Blue Team: {blue_team_2}")
    print(f"Red Team: {red_team_2}")
    print(f"Win Probability: {result_2['win_probability']:.3f}")
    print(f"Gold Curve Prediction (6 time points): {result_2['gold_curve_prediction']}")
    print(f"Objective Predictions: {result_2['objective_predictions']}")

if __name__ == "__main__":
    main()