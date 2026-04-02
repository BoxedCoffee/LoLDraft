#!/usr/bin/env python3
"""
Draft encoder training loop.

Usage:
    python train.py --config config/default.yaml
    python train.py --config config/default.yaml --wandb

Reads processed data from prepare_dataset.py output.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from model import DraftModel
from dataset import DraftDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    totals = {"total": 0, "win": 0, "gold": 0, "obj": 0, "n": 0}

    for batch in loader:
        blue = batch["blue_champs"].to(device)
        red = batch["red_champs"].to(device)
        patch = batch["patch_id"].to(device)
        win = batch["win_label"].to(device)
        gold_t = batch["gold_targets"].to(device)
        gold_m = batch["gold_mask"].to(device)
        obj_t = batch["obj_targets"].to(device)

        _, win_logit, gold_pred, obj_preds = model(blue, red, patch)

        obj_targets = [obj_t[:, i] for i in range(obj_t.shape[1])]
        losses = model.compute_loss(win_logit, win, gold_pred, gold_t, obj_preds, obj_targets, gold_m)

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = blue.shape[0]
        totals["total"] += losses["total"].item() * bs
        totals["win"] += losses["win"].item() * bs
        totals["gold"] += losses["gold"].item() * bs
        totals["obj"] += losses["obj"].item() * bs
        totals["n"] += bs

    n = totals["n"]
    return {k: totals[k] / n for k in ["total", "win", "gold", "obj"]}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {"total": 0, "win": 0, "gold": 0, "obj": 0, "n": 0}
    correct = 0
    total_samples = 0

    for batch in loader:
        blue = batch["blue_champs"].to(device)
        red = batch["red_champs"].to(device)
        patch = batch["patch_id"].to(device)
        win = batch["win_label"].to(device)
        gold_t = batch["gold_targets"].to(device)
        gold_m = batch["gold_mask"].to(device)
        obj_t = batch["obj_targets"].to(device)

        _, win_logit, gold_pred, obj_preds = model(blue, red, patch)

        obj_targets = [obj_t[:, i] for i in range(obj_t.shape[1])]
        losses = model.compute_loss(win_logit, win, gold_pred, gold_t, obj_preds, obj_targets, gold_m)

        bs = blue.shape[0]
        totals["total"] += losses["total"].item() * bs
        totals["win"] += losses["win"].item() * bs
        totals["gold"] += losses["gold"].item() * bs
        totals["obj"] += losses["obj"].item() * bs
        totals["n"] += bs

        # Win accuracy
        preds = (torch.sigmoid(win_logit) > 0.5).float()
        correct += (preds == win).sum().item()
        total_samples += bs

    n = totals["n"]
    metrics = {k: totals[k] / n for k in ["total", "win", "gold", "obj"]}
    metrics["win_acc"] = correct / total_samples if total_samples > 0 else 0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg.get("data_dir", "./data/processed")
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    logger.info("Config: %s", args.config)
    logger.info("Device: %s", device)
    logger.info("Data: %s", data_dir)

    # W&B
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="lol-draft-encoder", config=cfg)
            logger.info("W&B run: %s", wandb_run.name)
        except ImportError:
            logger.warning("wandb not installed — skipping")

    # Datasets
    train_ds = DraftDataset(data_dir, split="train")
    val_ds = DraftDataset(data_dir, split="val", patch_vocab=train_ds.patch_vocab)

    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # Load champion vocab for model sizing
    with open(Path(data_dir) / "champion_vocab.json") as f:
        champion_vocab = json.load(f)
    num_champions = len(champion_vocab)

    # Model
    model_cfg = cfg.get("model", {})
    model = DraftModel(
        num_champions=num_champions,
        champion_dim=model_cfg.get("champion_dim", 64),
        draft_dim=model_cfg.get("draft_dim", 256),
        num_patches=train_ds.num_patches,
        patch_dim=model_cfg.get("patch_dim", 16),
        team_hidden=model_cfg.get("team_hidden", 512),
        num_team_layers=model_cfg.get("num_team_layers", 3),
        num_gold_points=6,
        num_objectives=4,
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.1fM)", param_count, param_count / 1e6)

    # Training setup
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 512)
    lr = train_cfg.get("learning_rate", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 0.01)
    max_epochs = train_cfg.get("max_epochs", 100)
    patience = train_cfg.get("patience", 10)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine schedule with warmup
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    total_steps = max_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpointing
    ckpt_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(max_epochs):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, epoch)
        global_step += len(train_loader)
        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0

        # Log
        logger.info(
            "Epoch %3d | train_loss=%.4f | val_loss=%.4f | val_win_acc=%.3f | "
            "val_gold=%.4f | val_obj=%.4f | %.1fs",
            epoch + 1, train_metrics["total"], val_metrics["total"],
            val_metrics["win_acc"], val_metrics["gold"], val_metrics["obj"], elapsed,
        )

        # Log task weights
        with torch.no_grad():
            w_win = (0.5 * torch.exp(-model.log_var_win)).item()
            w_gold = (0.5 * torch.exp(-model.log_var_gold)).item()
            w_obj = (0.5 * torch.exp(-model.log_var_obj)).item()
        logger.info("  Task weights: win=%.3f gold=%.3f obj=%.3f", w_win, w_gold, w_obj)

        if wandb_run:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": train_metrics["total"],
                "train/win_loss": train_metrics["win"],
                "train/gold_loss": train_metrics["gold"],
                "train/obj_loss": train_metrics["obj"],
                "val/loss": val_metrics["total"],
                "val/win_acc": val_metrics["win_acc"],
                "val/gold_loss": val_metrics["gold"],
                "val/obj_loss": val_metrics["obj"],
                "weights/win": w_win,
                "weights/gold": w_gold,
                "weights/obj": w_obj,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)

        # Checkpoint
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            epochs_without_improvement = 0
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_win_acc": val_metrics["win_acc"],
                "config": cfg,
                "champion_vocab": champion_vocab,
                "patch_vocab": train_ds.patch_vocab,
            }, ckpt_path)
            logger.info("  ✓ New best model saved (val_loss=%.4f)", best_val_loss)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

    # Save final model
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "champion_vocab": champion_vocab,
        "patch_vocab": train_ds.patch_vocab,
    }, ckpt_dir / "final.pt")

    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best val loss: %.4f", best_val_loss)
    logger.info("  Checkpoints: %s", ckpt_dir)
    logger.info("=" * 50)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
