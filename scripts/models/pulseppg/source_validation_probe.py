#!/usr/bin/env python3
"""Small source-only HR-signal check for Pulse-PPG 8-second embeddings.

PTT-PPG is the declared held-out target. Its signals and labels are not read.
This is a diagnostic gate, not the Phase-12 benchmark or hyperparameter tuning.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from input_compatibility_smoke import CHECKPOINT, MANIFEST, OUTPUT, ROOT, Net, embed, input_array


ROLES = ROOT / "reports/phase7_frozen_dataset/subject_splits.csv"
RESULT = OUTPUT.with_name("source_validation_probe.json")
TARGET = "PTT-PPG"
SAMPLES_PER_SUBJECT = 12


def sampled_source_rows() -> dict[str, list[dict[str, str]]]:
    role_by_subject = {}
    with ROLES.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            role_by_subject[(row["dataset"], row["subject_id"])] = row[f"lodo_{TARGET}_role"]
    by_subject = defaultdict(list)
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] == TARGET:
                continue
            key = (row["dataset"], row["subject_id"])
            role = role_by_subject[key]
            if role not in ("source_train", "source_validation"):
                raise AssertionError(f"Unexpected source role: {role}")
            by_subject[(role, *key)].append(row)
    chosen = defaultdict(list)
    for (role, _, _), rows in sorted(by_subject.items()):
        # Evenly cover each subject's available windows, without HR-based selection.
        indices = np.linspace(0, len(rows) - 1, min(SAMPLES_PER_SUBJECT, len(rows)), dtype=int)
        chosen[role].extend(rows[index] for index in indices)
    return dict(chosen)


def make_matrix(model: Net, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features, labels, datasets = [], [], []
    for row in rows:
        values = input_array(row, 8)
        if values is None:
            continue
        features.append(embed(model, values))
        labels.append(float(row["hr_bpm"]))
        datasets.append(row["dataset"])
    return np.stack(features), np.asarray(labels), datasets


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = Net(in_channels=1, base_filters=128, kernel_size=11, stride=2,
                groups=1, n_block=12, finalpool="max")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state["net"], strict=True)
    model.eval()

    rows = sampled_source_rows()
    x_train, y_train, train_datasets = make_matrix(model, rows["source_train"])
    x_val, y_val, val_datasets = make_matrix(model, rows["source_validation"])
    scaler = StandardScaler().fit(x_train)
    predictor = Ridge(alpha=10.0).fit(scaler.transform(x_train), y_train)
    prediction = predictor.predict(scaler.transform(x_val))
    baseline = np.full_like(y_val, np.mean(y_train))
    result = {
        "purpose": "Small fixed-alpha source-validation HR-signal gate; not benchmark",
        "held_out_target_not_read": TARGET,
        "source_datasets": sorted(set(train_datasets)),
        "samples_per_subject_max": SAMPLES_PER_SUBJECT,
        "train_windows": len(y_train),
        "validation_windows": len(y_val),
        "source_validation_mae_bpm": float(mean_absolute_error(y_val, prediction)),
        "source_train_mean_baseline_mae_bpm": float(mean_absolute_error(y_val, baseline)),
        "source_validation_pearson_r": float(pearsonr(y_val, prediction).statistic),
        "validation_by_dataset": {},
    }
    for name in sorted(set(val_datasets)):
        indices = np.array([item == name for item in val_datasets])
        result["validation_by_dataset"][name] = {
            "n": int(np.sum(indices)),
            "mae_bpm": float(mean_absolute_error(y_val[indices], prediction[indices])),
            "mean_baseline_mae_bpm": float(mean_absolute_error(y_val[indices], baseline[indices])),
        }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
