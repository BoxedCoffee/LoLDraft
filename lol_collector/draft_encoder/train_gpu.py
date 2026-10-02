#!/usr/bin/env python3
"""
GPU-Optimized Training Script for Draft Encoder with RTX 4090
Features: CUDA acceleration, mixed precision training, checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb
import os
import json
import argparse
from datetime import datetime
from model import DraftEncoder  # Import your model
from dataset import DraftDataset  # Import your dataset

def setup_device():
    """Setup CUDA device for RTX 4090"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        return device
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")

def setup_wandb(project_name="draft_encoder_gpu"):
    """Initialize Weights & Biases"""
    wandb.init(
        project=project_name,
        config={
            "epochs": 100,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "model_type": "DraftEncoder",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        }
    )
    return wandb

def train_model(model, train_loader, val_loader, device, config):
    """Main training loop with checkpointing"""

    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler()  # For mixed precision training

    # Checkpoint directory
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load existing checkpoint if available
    start_epoch = 0
    best_val_loss = float('inf')

    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        print("Loading from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    model.train()
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # Training phase
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move data to GPU
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(batch)
                loss = model.compute_loss(outputs, batch)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Validation phase
        val_loss = validate_model(model, val_loader, device)

        # Update learning rate
        scheduler.step()

        # Log metrics to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss/num_batches,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }

        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            print(f"New best model saved with validation loss: {val_loss:.4f}")

        print(f"Epoch {epoch+1} - Train Loss: {train_loss/num_batches:.4f}, Val Loss: {val_loss:.4f}")

def validate_model(model, val_loader, device):
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch)
            loss = model.compute_loss(outputs, batch)

            val_loss += loss.item()
            num_batches += 1

    model.train()
    return val_loss / num_batches

def main():
    """Main training function"""

    # Setup device
    device = setup_device()

    # Configuration
    config = {
        'epochs': 100,
        'batch_size': 256,  # Optimized for RTX 4090 (24GB VRAM)
        'learning_rate': 1e-3,
        'num_workers': 4,
        'pin_memory': True
    }

    print("Starting GPU-Optimized Training with Checkpointing...")
    print(f"Configuration: {config}")

    # Setup W&B
    try:
        wandb = setup_wandb("draft_encoder_gpu")
    except Exception as e:
        print(f"W&B setup failed: {e}")
        wandb = None

    # Initialize model and data
    # Initialize the model with required parameters
    num_champions = 174  # From our processed vocabulary
    model = DraftEncoder(num_champions=num_champions).to(device)

    # Create dataset (adjust path as needed)
    train_dataset = DraftDataset(data_dir="data/processed", split="train")
    val_dataset = DraftDataset(data_dir="data/processed", split="val")

    # Create data loaders with GPU optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    # Start training
    try:
        train_model(model, train_loader, val_loader, device, config)
        print("Training completed successfully!")

        # Save final model
        final_path = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        torch.save(model.state_dict(), final_path)
        print(f"Final model saved to {final_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save checkpoint before exiting
        checkpoint_path = os.path.join("checkpoints", "interrupted_checkpoint.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.AdamW(model.parameters()).state_dict(),
            'epoch': 0,
        }, checkpoint_path)
        print(f"Interrupted checkpoint saved to {checkpoint_path}")

    except Exception as e:
        print(f"Training error: {e}")
        # Save error checkpoint
        checkpoint_path = os.path.join("checkpoints", "error_checkpoint.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.AdamW(model.parameters()).state_dict(),
            'epoch': 0,
        }, checkpoint_path)
        print(f"Error checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()