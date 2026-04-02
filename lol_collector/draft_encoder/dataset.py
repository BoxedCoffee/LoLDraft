"""
PyTorch Dataset for draft encoder training.

Reads processed parquet files from prepare_dataset.py output.
Handles joins across drafts, gold curves, and objectives.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROLES = ["top", "jng", "mid", "bot", "sup"]
GOLD_COLUMNS = ["gold_diff_5", "gold_diff_10", "gold_diff_15",
                "gold_diff_20", "gold_diff_25", "gold_diff_30"]
OBJ_TYPES = ["dragon", "herald", "baron", "tower"]

# Map objective team IDs to class indices: 0=none, 1=blue(100), 2=red(200)
TEAM_TO_CLASS = {0: 0, 100: 1, 200: 2}


class DraftDataset(Dataset):
    """
    Dataset for draft encoder training.

    Each item returns:
        blue_champs:  (5,) int tensor — champion IDs for blue team
        red_champs:   (5,) int tensor — champion IDs for red team
        patch_id:     int — patch index
        win_label:    float — 1.0 if blue wins
        gold_targets: (6,) float tensor — gold diffs at 5-min intervals
        gold_mask:    (6,) float tensor — 1.0 where data exists
        obj_targets:  (4,) int tensor — first obj team class per type
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        patch_vocab: Optional[dict] = None,
    ):
        data_dir = Path(data_dir)

        # Load splits
        with open(data_dir / "splits.json") as f:
            splits = json.load(f)
        split_ids = set(splits[split])

        # Load drafts
        drafts = pd.read_parquet(data_dir / "drafts.parquet")

        # For augmented data: include swapped versions if original is in split
        def in_split(match_id):
            base = match_id.replace("_swap", "")
            return base in split_ids

        drafts = drafts[drafts["match_id"].apply(in_split)].reset_index(drop=True)

        # Load gold curves
        gold_path = data_dir / "gold_curves.parquet"
        if gold_path.exists():
            gold = pd.read_parquet(gold_path)
            self._gold = gold.set_index("match_id")
        else:
            self._gold = None

        # Load objectives
        obj_path = data_dir / "objectives.parquet"
        if obj_path.exists():
            obj = pd.read_parquet(obj_path)
            self._obj = obj.set_index("match_id")
        else:
            self._obj = None

        # Build patch vocab if not provided
        if patch_vocab is None:
            patches = sorted(drafts["patch"].unique()) if "patch" in drafts.columns else []
            self.patch_vocab = {p: i + 1 for i, p in enumerate(patches)}
            self.patch_vocab["<UNK>"] = 0
        else:
            self.patch_vocab = patch_vocab

        # Store draft data as numpy for fast indexing
        self._match_ids = drafts["match_id"].values
        self._blue = np.stack([drafts[f"blue_{r}"].values for r in ROLES], axis=1)  # (N, 5)
        self._red = np.stack([drafts[f"red_{r}"].values for r in ROLES], axis=1)    # (N, 5)
        self._win = drafts["blue_win"].values.astype(np.float32)
        self._patches = np.array([
            self.patch_vocab.get(p, 0)
            for p in (drafts["patch"].values if "patch" in drafts.columns else ["unknown"] * len(drafts))
        ])

    def __len__(self) -> int:
        return len(self._match_ids)

    @property
    def num_patches(self) -> int:
        return len(self.patch_vocab)

    def __getitem__(self, idx: int) -> dict:
        match_id = self._match_ids[idx]
        # For swapped matches, gold/obj data uses the original match_id
        base_match_id = match_id.replace("_swap", "")
        is_swapped = match_id.endswith("_swap")

        blue_champs = torch.tensor(self._blue[idx], dtype=torch.long)
        red_champs = torch.tensor(self._red[idx], dtype=torch.long)
        patch_id = torch.tensor(self._patches[idx], dtype=torch.long)
        win_label = torch.tensor(self._win[idx], dtype=torch.float32)

        # Gold curves
        gold_targets = torch.zeros(6, dtype=torch.float32)
        gold_mask = torch.zeros(6, dtype=torch.float32)
        if self._gold is not None and base_match_id in self._gold.index:
            row = self._gold.loc[base_match_id]
            for i, col in enumerate(GOLD_COLUMNS):
                if col in row.index and pd.notna(row[col]):
                    val = float(row[col])
                    # If swapped, negate gold diff (blue perspective → red perspective)
                    gold_targets[i] = -val if is_swapped else val
                    gold_mask[i] = 1.0

        # Objectives
        obj_targets = torch.zeros(4, dtype=torch.long)  # default: 0 = none
        if self._obj is not None and base_match_id in self._obj.index:
            row = self._obj.loc[base_match_id]
            for i, obj_type in enumerate(OBJ_TYPES):
                col = f"first_{obj_type}_team"
                if col in row.index and pd.notna(row[col]):
                    team_id = int(row[col])
                    cls = TEAM_TO_CLASS.get(team_id, 0)
                    # If swapped, flip blue↔red
                    if is_swapped and cls > 0:
                        cls = 3 - cls  # 1→2, 2→1
                    obj_targets[i] = cls

        return {
            "blue_champs": blue_champs,
            "red_champs": red_champs,
            "patch_id": patch_id,
            "win_label": win_label,
            "gold_targets": gold_targets,
            "gold_mask": gold_mask,
            "obj_targets": obj_targets,
        }
