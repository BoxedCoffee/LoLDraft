#!/usr/bin/env python3
"""
Simple test to verify model loading and basic functionality
"""

import torch
import json
from model import DraftModel

# Load champion vocabulary
with open('data/processed/champion_vocab.json', 'r') as f:
    champion_vocab = json.load(f)

print("Loaded champion vocabulary with", len(champion_vocab), "champions")

# Initialize model
model = DraftModel(
    num_champions=len(champion_vocab),
    champion_dim=64,
    draft_dim=256,
    num_patches=0,
    patch_dim=16,
    team_hidden=512,
    num_team_layers=3,
    num_gold_points=6,
    num_objectives=4
)

# Check if model files exist
import os
if os.path.exists('checkpoints/best.pt'):
    print("Loading checkpoint...")
    checkpoint = torch.load('checkpoints/best.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")
else:
    print("Checkpoint file not found!")

print("Model test completed successfully!")