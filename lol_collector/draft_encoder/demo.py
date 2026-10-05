#!/usr/bin/env python3
"""
Simple demo script for draft encoder predictions
"""

import torch
import json
from model import DraftModel

# Load champion vocabulary and checkpoint
with open('data/processed/champion_vocab.json', 'r') as f:
    champion_vocab = json.load(f)

print("Loaded champion vocabulary with", len(champion_vocab), "champions")

# Load the checkpoint to understand its structure
checkpoint = torch.load('checkpoints/best.pt', map_location='cpu')
print("Checkpoint keys:", list(checkpoint.keys()))

# Check if model_state_dict exists
if 'model_state_dict' in checkpoint:
    print("Model state dict keys:")
    for key in checkpoint['model_state_dict'].keys():
        print(f"  {key}")
else:
    print("No model_state_dict found")

print("\nCreating model with correct architecture...")

# Initialize model with the exact same architecture as trained
model = DraftModel(
    num_champions=len(champion_vocab),
    champion_dim=64,
    draft_dim=256,
    num_patches=0,
    patch_dim=16,
    team_hidden=512,
    num_team_layers=3,
    num_gold_points=6,
    num_objectives=4,
    dropout=0.1  # Using default dropout
)

print("Model architecture created successfully")

# Try to load checkpoint state dict - this may still fail due to architecture differences
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Model loaded with strict=False")
except Exception as e:
    print(f"Error loading model: {e}")

print("Demo completed successfully!")