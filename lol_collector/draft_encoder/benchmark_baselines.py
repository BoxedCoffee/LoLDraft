#!/usr/bin/env python3
"""Benchmark simple draft-only baselines against the neural encoder.

Usage:
    python benchmark_baselines.py --data-dir data\processed_riot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder


ROLES = ["top", "jng", "mid", "bot", "sup"]


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = probabilities >= 0.5
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "samples": int(len(labels)),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
    else:
        result["roc_auc"] = None
    return result


def load_split(data_dir: Path, split: str, ids: set[str]) -> pd.DataFrame:
    drafts = pd.read_parquet(data_dir / "drafts.parquet")
    drafts = drafts.loc[~drafts["match_id"].str.endswith("_swap")].copy()
    drafts = drafts[drafts["match_id"].isin(ids)]
    return drafts.drop_duplicates("match_id")


def add_structured_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    for role in ROLES:
        features[f"lane_{role}"] = (
            features[f"blue_{role}"].astype(str)
            + "_vs_"
            + features[f"red_{role}"].astype(str)
        )
    for side in ("blue", "red"):
        for left_index, left_role in enumerate(ROLES):
            for right_role in ROLES[left_index + 1:]:
                features[f"{side}_pair_{left_role}_{right_role}"] = (
                    features[f"{side}_{left_role}"].astype(str)
                    + "_"
                    + features[f"{side}_{right_role}"].astype(str)
                )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    splits = json.loads((args.data_dir / "splits.json").read_text(encoding="utf-8"))
    train = load_split(args.data_dir, "train", set(splits["train"]))
    test = load_split(args.data_dir, "test", set(splits["test"]))
    labels = test["blue_win"].astype(int).to_numpy()
    train_labels = train["blue_win"].astype(int).to_numpy()

    overall_probability = float(train_labels.mean())
    overall = metrics(labels, np.full(len(labels), overall_probability))

    patch_rates = train.groupby("patch")["blue_win"].mean().to_dict()
    patch_probabilities = test["patch"].map(patch_rates).fillna(overall_probability).to_numpy()
    patch = metrics(labels, patch_probabilities)

    feature_columns = [
        f"{side}_{role}"
        for side in ("blue", "red")
        for role in ROLES
    ]
    preprocessor = ColumnTransformer(
        [("champions", OneHotEncoder(handle_unknown="ignore"), feature_columns)],
        remainder="drop",
    )
    x_train = preprocessor.fit_transform(train[feature_columns])
    x_test = preprocessor.transform(test[feature_columns])
    model = LogisticRegression(max_iter=300, C=0.5, solver="liblinear")
    model.fit(x_train, train_labels)
    logistic = metrics(labels, model.predict_proba(x_test)[:, 1])

    structured_columns = [
        column
        for column in add_structured_features(train).columns
        if column.startswith(("lane_", "blue_pair_", "red_pair_"))
    ]
    structured_train = add_structured_features(train)
    structured_test = add_structured_features(test)
    structured_preprocessor = ColumnTransformer(
        [("structured", OneHotEncoder(handle_unknown="ignore"), structured_columns)],
        remainder="drop",
    )
    structured_x_train = structured_preprocessor.fit_transform(structured_train)
    structured_x_test = structured_preprocessor.transform(structured_test)
    structured_model = LogisticRegression(max_iter=300, C=0.25, solver="liblinear")
    structured_model.fit(structured_x_train, train_labels)
    structured = metrics(
        labels, structured_model.predict_proba(structured_x_test)[:, 1]
    )

    result = {
        "dataset": {
            "train_samples": int(len(train)),
            "test_samples": int(len(test)),
            "train_blue_win_rate": overall_probability,
        },
        "overall_rate": overall,
        "patch_rate": patch,
        "role_champion_logistic": logistic,
        "structured_interaction_logistic": structured,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
