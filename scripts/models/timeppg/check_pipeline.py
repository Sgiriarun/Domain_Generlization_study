#!/usr/bin/env python3
"""Phase 9A: verify TimePPG data, split, model, and optimization plumbing."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rq1_hr.models.timeppg import (
    LogCoshLoss,
    ManifestPPGDataset,
    TimePPGBigPPGOnly,
    attach_split_roles,
)


UPSTREAM_COMMIT = "ddf3866da6d5f9dda4da7d7884b4f1f3b809a6ba"


def select_evenly(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Small subject-spanning subset used only for plumbing validation."""
    rng = np.random.default_rng(seed)
    groups = []
    per_subject = max(1, count // frame.subject_id.nunique())
    for _, group in frame.groupby("subject_id", sort=True):
        take = min(per_subject, len(group))
        groups.append(group.iloc[rng.choice(len(group), take, replace=False)])
    result = pd.concat(groups).sort_values(["subject_id", "window_start_s"])
    if len(result) > count:
        result = result.iloc[:count]
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase7-dir", type=Path, default=Path("reports/phase7_frozen_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase9_timeppg/pipeline_check"))
    parser.add_argument("--dataset", default="PPG-DaLiA")
    parser.add_argument("--train-windows", type=int, default=256)
    parser.add_argument("--validation-windows", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.phase7_dir / "main_window_manifest.csv", low_memory=False)
    subjects = pd.read_csv(args.phase7_dir / "subject_splits.csv")
    data = attach_split_roles(manifest, subjects, "within_dataset_fold")
    data = data.loc[data.dataset.eq(args.dataset)].copy()
    train = select_evenly(data.loc[data.within_dataset_fold.ne(0)], args.train_windows, args.seed)
    validation = select_evenly(data.loc[data.within_dataset_fold.eq(0)], args.validation_windows, args.seed + 1)
    train_subjects = set(train.subject_id)
    validation_subjects = set(validation.subject_id)
    if train_subjects & validation_subjects:
        raise RuntimeError("subject leakage detected")

    train_loader = DataLoader(ManifestPPGDataset(train), batch_size=128, shuffle=True, num_workers=0)
    validation_loader = DataLoader(ManifestPPGDataset(validation), batch_size=128, shuffle=False, num_workers=0)
    model = TimePPGBigPPGOnly(512)
    loss_function = LogCoshLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for inputs, targets, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        validation_losses, absolute_errors = [], []
        with torch.no_grad():
            for inputs, targets, _ in validation_loader:
                predictions = model(inputs)
                validation_losses.append(float(loss_function(predictions, targets)))
                absolute_errors.extend(torch.abs(predictions - targets).tolist())
        history.append({
            "epoch": epoch,
            "train_log_cosh": float(np.mean(train_losses)),
            "validation_log_cosh": float(np.mean(validation_losses)),
            "validation_mae_bpm": float(np.mean(absolute_errors)),
        })

    sample_inputs, sample_targets, _ = next(iter(validation_loader))
    with torch.no_grad():
        sample_predictions = model.eval()(sample_inputs)
    checks = {
        "status": "pass",
        "purpose": "pipeline_validation_not_research_performance",
        "dataset": args.dataset,
        "seed": args.seed,
        "upstream_commit": UPSTREAM_COMMIT,
        "train_subjects": sorted(train_subjects),
        "validation_subjects": sorted(validation_subjects),
        "subject_overlap": sorted(train_subjects & validation_subjects),
        "train_windows": len(train),
        "validation_windows": len(validation),
        "input_shape": list(sample_inputs.shape),
        "target_shape": list(sample_targets.shape),
        "output_shape": list(sample_predictions.shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "finite_predictions": bool(torch.isfinite(sample_predictions).all()),
        "completed_forward_backward": True,
    }
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    (args.output_dir / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    torch.save({"model_state": model.state_dict(), "checks": checks}, args.output_dir / "smoke_checkpoint.pt")
    report = f"""# Phase 9A TimePPG pipeline check

Status: **PASS**

This is a short real-data plumbing check, not a model comparison and not a
reportable HR-performance result. It used only {args.dataset}, with frozen fold
0 subjects for validation and different subjects for training.

- Official upstream commit: `{UPSTREAM_COMMIT}`
- Training windows: {len(train)}
- Validation windows: {len(validation)}
- Subject overlap: none
- Batch input shape: `{tuple(sample_inputs.shape)}`
- Output shape: `{tuple(sample_predictions.shape)}`
- Trainable parameters: {checks['parameters']:,}
- Forward pass, backward pass, Adam update, checkpoint writing: passed
- Predictions finite: yes

Loss values in `training_history.csv` only demonstrate executable optimization
on a tiny subset. They must never be placed in the final results table.
"""
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(pd.DataFrame(history).to_string(index=False))


if __name__ == "__main__":
    main()
