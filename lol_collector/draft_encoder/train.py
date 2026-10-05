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
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from model import DraftModel
from dataset import DraftDataset


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write a checkpoint atomically so interruption cannot corrupt it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def backup_checkpoint(path: Path, backup_dir: Path, keep: int) -> None:
    if not path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = backup_dir / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, destination)
    backups = sorted(backup_dir.glob(f"{path.stem}_*.pt"), key=lambda item: item.stat().st_mtime)
    for old_backup in backups[:-keep]:
        old_backup.unlink()


def checkpoint_payload(
    epoch, model, optimizer, scheduler, best_metric, epochs_without_improvement,
    global_step, checkpoint_metric, val_metrics, cfg, champion_vocab, patch_vocab,
    train_size, val_size,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": best_metric,
        "epochs_without_improvement": epochs_without_improvement,
        "global_step": global_step,
        "val_loss": val_metrics["total"],
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_metric_value": val_metrics.get(checkpoint_metric),
        "val_win_acc": val_metrics["win_acc"],
        "config": cfg,
        "model_signature": model_config_signature(cfg),
        "champion_vocab": champion_vocab,
        "patch_vocab": patch_vocab,
        "train_size": train_size,
        "val_size": val_size,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def model_config_signature(cfg: dict) -> dict:
    model_cfg = cfg.get("model", {})
    return {
        key: model_cfg.get(key, default)
        for key, default in (
            ("champion_dim", 64),
            ("draft_dim", 256),
            ("patch_dim", 16),
            ("team_hidden", 512),
            ("num_team_layers", 3),
            ("dropout", 0.1),
        )
    }


def resolve_resume_path(value: str | None, ckpt_dir: Path) -> Path | None:
    if not value:
        return None
    if value.lower() == "auto":
        candidates = [ckpt_dir / "latest.pt", ckpt_dir / "best.pt"]
        candidates.extend(sorted((ckpt_dir / "backups").glob("latest_*.pt"), reverse=True))
        return next((candidate for candidate in candidates if candidate.exists()), None)
    return Path(value)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


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
        obj_m = batch["obj_mask"].to(device)

        _, win_logit, gold_pred, obj_preds = model(blue, red, patch)

        obj_targets = [obj_t[:, i] for i in range(obj_t.shape[1])]
        losses = model.compute_loss(
            win_logit, win, gold_pred, gold_t, obj_preds, obj_targets, gold_m, obj_m
        )

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
        obj_m = batch["obj_mask"].to(device)

        _, win_logit, gold_pred, obj_preds = model(blue, red, patch)

        obj_targets = [obj_t[:, i] for i in range(obj_t.shape[1])]
        losses = model.compute_loss(
            win_logit, win, gold_pred, gold_t, obj_preds, obj_targets, gold_m, obj_m
        )

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
        totals.setdefault("win_logloss", 0)
        totals["win_logloss"] += nn.functional.binary_cross_entropy_with_logits(
            win_logit, win.float()
        ).item() * bs

    n = totals["n"]
    metrics = {k: totals[k] / n for k in ["total", "win", "gold", "obj"]}
    metrics["win_logloss"] = totals["win_logloss"] / n
    metrics["win_acc"] = correct / total_samples if total_samples > 0 else 0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None,
        help="Resume from a checkpoint path, or use 'auto' for latest.pt/best.pt",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", False)))
    data_dir = args.data_dir or cfg.get("data_dir", "./data/processed")
    requested_device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is unavailable; falling back to CPU")
        requested_device = "cpu"
    device = torch.device(requested_device)

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
    if not len(train_ds) or not len(val_ds):
        raise ValueError("Training and validation splits must both contain samples")
    objective_counts = train_ds._obj[
        [f"first_{name}_team" for name in ("dragon", "herald", "baron", "tower")]
    ].apply(lambda column: column.isin([0, 100, 200]).sum()) if train_ds._obj is not None else None
    if objective_counts is not None:
        logger.info("Objective rows with resolvable team labels: %s", objective_counts.to_dict())

    # Load champion vocab for model sizing
    with open(Path(data_dir) / "champion_vocab.json") as f:
        champion_vocab = json.load(f)
    num_champions = len(champion_vocab)
    champion_ids = [int(value) for value in champion_vocab.values()]
    if min(champion_ids) != 0 or max(champion_ids) >= num_champions:
        raise ValueError(
            "champion_vocab IDs must be contiguous and fit the embedding table"
        )

    # Preflight resume validation before constructing or mutating model state.
    preflight_ckpt_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
    resume_path = resolve_resume_path(args.resume, preflight_ckpt_dir)
    if args.resume and (resume_path is None or not resume_path.exists()):
        raise FileNotFoundError(
            f"Resume requested but no checkpoint was found for {args.resume!r}"
        )
    resume_checkpoint = None
    if resume_path is not None:
        logger.info("Preflighting resume checkpoint: %s", resume_path)
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_checkpoint.get("champion_vocab") != champion_vocab:
            raise ValueError("Resume checkpoint champion vocabulary does not match the dataset")
        if resume_checkpoint.get("patch_vocab") != train_ds.patch_vocab:
            raise ValueError("Resume checkpoint patch vocabulary does not match the dataset")
        if resume_checkpoint.get("train_size") not in (None, len(train_ds)):
            raise ValueError("Resume checkpoint training split size does not match the dataset")
        if resume_checkpoint.get("val_size") not in (None, len(val_ds)):
            raise ValueError("Resume checkpoint validation split size does not match the dataset")
        saved_signature = resume_checkpoint.get("model_signature")
        if saved_signature is not None and saved_signature != model_config_signature(cfg):
            raise ValueError(
                "Resume checkpoint model configuration does not match the current config: "
                f"saved={saved_signature}, current={model_config_signature(cfg)}"
            )

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
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine schedule with warmup
    total_steps = max_epochs * max(1, len(train_loader))
    warmup_steps = min(int(train_cfg.get("warmup_steps", 1000)), total_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpointing
    ckpt_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = ckpt_dir / "backups"
    backup_keep = max(1, int(train_cfg.get("checkpoint_backup_keep", 5)))
    checkpoint_interval = max(1, int(train_cfg.get("checkpoint_interval", 1)))

    checkpoint_metric = str(
        cfg.get("training", {}).get("checkpoint_metric", "win_logloss")
    )
    best_metric = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    start_epoch = 0

    resume_path = resolve_resume_path(args.resume, ckpt_dir)
    if args.resume and (resume_path is None or not resume_path.exists()):
        raise FileNotFoundError(
            f"Resume requested but no checkpoint was found for {args.resume!r}"
        )
    if resume_path is None:
        existing = [ckpt_dir / name for name in ("best.pt", "latest.pt", "final.pt")]
        existing = [path for path in existing if path.exists()]
        if existing:
            logger.warning(
                "Starting a fresh run; existing checkpoints will be backed up in %s",
                backup_dir,
            )
            for existing_path in existing:
                backup_checkpoint(existing_path, backup_dir, backup_keep)
    if resume_path is not None:
        logger.info("Loading resume checkpoint: %s", resume_path)
        checkpoint = resume_checkpoint or torch.load(
            resume_path, map_location=device, weights_only=False
        )
        if checkpoint.get("champion_vocab") != champion_vocab:
            raise ValueError("Resume checkpoint champion vocabulary does not match the dataset")
        if checkpoint.get("patch_vocab") != train_ds.patch_vocab:
            raise ValueError("Resume checkpoint patch vocabulary does not match the dataset")
        if checkpoint.get("train_size") not in (None, len(train_ds)):
            raise ValueError("Resume checkpoint training split size does not match the dataset")
        if checkpoint.get("val_size") not in (None, len(val_ds)):
            raise ValueError("Resume checkpoint validation split size does not match the dataset")
        saved_metric = checkpoint.get("checkpoint_metric", checkpoint_metric)
        if saved_metric != checkpoint_metric:
            raise ValueError(
                f"Resume checkpoint uses {saved_metric!r}, current config uses {checkpoint_metric!r}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        best_metric = float(checkpoint.get("best_metric", checkpoint.get("checkpoint_metric_value", float("inf"))))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        global_step = int(checkpoint.get("global_step", start_epoch * len(train_loader)))
        if checkpoint.get("python_random_state") is not None:
            random.setstate(checkpoint["python_random_state"])
        if checkpoint.get("numpy_random_state") is not None:
            np.random.set_state(checkpoint["numpy_random_state"])
        if checkpoint.get("torch_random_state") is not None:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_random_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        logger.info(
            "Resumed at epoch %d | best %s=%.6f | global_step=%d",
            start_epoch, checkpoint_metric, best_metric, global_step,
        )

    for epoch in range(start_epoch, max_epochs):
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
        monitored_value = val_metrics.get(checkpoint_metric)
        if monitored_value is None:
            raise ValueError(
                f"Unsupported checkpoint metric {checkpoint_metric!r}; "
                "choose win_logloss, win, or total."
            )
        should_stop = False
        if monitored_value < best_metric:
            best_metric = monitored_value
            epochs_without_improvement = 0
            ckpt_path = ckpt_dir / "best.pt"
            backup_checkpoint(ckpt_path, backup_dir, backup_keep)
            atomic_torch_save(
                checkpoint_payload(
                    epoch + 1, model, optimizer, scheduler, best_metric,
                    epochs_without_improvement, global_step, checkpoint_metric,
                    val_metrics, cfg, champion_vocab, train_ds.patch_vocab,
                    len(train_ds), len(val_ds),
                ),
                ckpt_path,
            )
            logger.info(
                "  ✓ New best model saved (%s=%.6f)",
                checkpoint_metric,
                monitored_value,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                should_stop = True

        latest_path = ckpt_dir / "latest.pt"
        if (epoch + 1) % checkpoint_interval == 0:
            backup_checkpoint(latest_path, backup_dir, backup_keep)
        atomic_torch_save(
            checkpoint_payload(
                epoch + 1, model, optimizer, scheduler, best_metric,
                epochs_without_improvement, global_step, checkpoint_metric,
                val_metrics, cfg, champion_vocab, train_ds.patch_vocab,
                len(train_ds), len(val_ds),
            ),
            latest_path,
        )
        logger.info("  ✓ Recovery checkpoint saved: %s", latest_path)
        if should_stop:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
            break

    # Save final model
    if start_epoch >= max_epochs:
        logger.info("Resume checkpoint is already at or beyond max_epochs; nothing to train")
        final_epoch = start_epoch
    else:
        final_epoch = epoch + 1
    final_path = ckpt_dir / "final.pt"
    backup_checkpoint(final_path, backup_dir, backup_keep)
    atomic_torch_save(
        checkpoint_payload(
            final_epoch, model, optimizer, scheduler, best_metric,
            epochs_without_improvement, global_step, checkpoint_metric,
            val_metrics, cfg, champion_vocab, train_ds.patch_vocab,
            len(train_ds), len(val_ds),
        ) if start_epoch < max_epochs else {
            "epoch": final_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "epochs_without_improvement": epochs_without_improvement,
            "global_step": global_step,
            "checkpoint_metric": checkpoint_metric,
            "checkpoint_metric_value": best_metric,
            "config": cfg,
            "model_signature": model_config_signature(cfg),
            "champion_vocab": champion_vocab,
            "patch_vocab": train_ds.patch_vocab,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
        },
        final_path,
    )

    logger.info("=" * 50)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best %s: %.6f", checkpoint_metric, best_metric)
    logger.info("  Checkpoints: %s", ckpt_dir)
    logger.info("=" * 50)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
