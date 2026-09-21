#!/usr/bin/env python3
"""Run a time-bounded, balanced-source TimePPG LODO pilot."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rq1_hr.evaluation import hr_metrics
from rq1_hr.models.timeppg import (
    LogCoshLoss,
    ManifestPPGDataset,
    TimePPGBigPPGOnly,
    attach_split_roles,
    predict,
    run_epoch,
)


def loader(frame: pd.DataFrame, *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ManifestPPGDataset(frame), batch_size=batch_size, shuffle=shuffle,
        num_workers=0, generator=generator, pin_memory=False,
    )


def balanced_sample(frame: pd.DataFrame, role: str, per_dataset: int, seed: int) -> pd.DataFrame:
    selected = frame.loc[frame["lodo_role"].eq(role)]
    if per_dataset <= 0:
        return selected.sample(frac=1, random_state=seed).reset_index(drop=True)
    parts = []
    for offset, (_, group) in enumerate(selected.groupby("dataset", sort=True)):
        parts.append(group.sample(n=min(per_dataset, len(group)), random_state=seed + offset))
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="PPG-DaLiA")
    parser.add_argument("--phase7-dir", type=Path, default=Path("reports/phase7_frozen_dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/phase9_timeppg/lodo_pilot"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-per-source", type=int, default=5000)
    parser.add_argument("--validation-per-source", type=int, default=1500)
    parser.add_argument("--full-data", action="store_true", help="Use every frozen source window")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    if args.device == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but this PyTorch environment cannot access it")
    device = torch.device(device_name)

    manifest = pd.read_csv(args.phase7_dir / "main_window_manifest.csv", low_memory=False)
    subjects = pd.read_csv(args.phase7_dir / "subject_splits.csv")
    role_column = f"lodo_{args.target}_role"
    frame = attach_split_roles(manifest, subjects, role_column).rename(columns={role_column: "lodo_role"})
    train_limit = 0 if args.full_data else args.train_per_source
    validation_limit = 0 if args.full_data else args.validation_per_source
    train = balanced_sample(frame, "source_train", train_limit, args.seed)
    validation = balanced_sample(frame, "source_validation", validation_limit, args.seed + 100)
    test = frame.loc[frame["lodo_role"].eq("target_test")].reset_index(drop=True)

    if set(train.dataset) & {args.target} or set(validation.dataset) & {args.target}:
        raise RuntimeError("held-out target leaked into source training or validation")
    if set(test.dataset) != {args.target}:
        raise RuntimeError("target test set contains an unexpected dataset")
    source_subjects = set(zip(train.dataset, train.subject_id)) | set(zip(validation.dataset, validation.subject_id))
    target_subjects = set(zip(test.dataset, test.subject_id))
    if source_subjects & target_subjects:
        raise RuntimeError("subject leakage detected")

    slug = args.target.lower().replace("-", "_")
    if args.full_data and args.output_root == Path("reports/phase9_timeppg/lodo_pilot"):
        args.output_root = Path("reports/phase9_timeppg/lodo_full")
    output = args.output_root / f"target_{slug}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    train.to_csv(output / "sampled_source_train_manifest.csv", index=False)
    validation.to_csv(output / "sampled_source_validation_manifest.csv", index=False)

    train_loader = loader(train, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = loader(validation, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_loader = loader(test, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    model = TimePPGBigPPGOnly(512).to(device)
    loss_function = LogCoshLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)

    history, best_loss, best_epoch, waited = [], float("inf"), 0, 0
    checkpoint_path = output / "best_checkpoint.pt"
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_function, optimizer=optimizer, device=device)
        validation_metrics = run_epoch(model, validation_loader, loss_function, device=device)
        row = {
            "epoch": epoch, "elapsed_seconds": time.time() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        if validation_metrics["loss"] < best_loss:
            best_loss, best_epoch, waited = validation_metrics["loss"], epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "arguments": vars(args)}, checkpoint_path)
        else:
            waited += 1
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(
            f"epoch={epoch:02d} train_mae={train_metrics['mae_bpm']:.3f} "
            f"val_mae={validation_metrics['mae_bpm']:.3f} best={best_epoch:02d} wait={waited}",
            flush=True,
        )
        if waited >= args.patience:
            break

    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    references, predictions, row_indices = predict(model, test_loader, device=device)
    test_rows = test.iloc[row_indices].copy()
    test_rows["reference_hr_bpm"] = references
    test_rows["predicted_hr_bpm"] = predictions
    test_rows["error_bpm"] = predictions - references
    test_rows["absolute_error_bpm"] = np.abs(predictions - references)
    test_rows.to_csv(output / "target_test_predictions.csv", index=False)
    metrics = hr_metrics(references, predictions)
    result = {
        "status": "complete",
        "experiment": "full_data_lodo" if args.full_data else "pilot_lodo_balanced_source_subset",
        "claim_boundary": "full frozen-data LODO" if args.full_data else "preliminary pilot; not the final full-data LODO result",
        "target_dataset": args.target, "source_datasets": sorted(set(train.dataset)),
        "seed": args.seed, "device": device_name,
        "epochs_planned": args.epochs, "epochs_completed": len(history),
        "best_epoch": best_epoch, "train_windows": len(train),
        "validation_windows": len(validation), "target_test_windows": len(test),
        "train_windows_by_dataset": train.dataset.value_counts().sort_index().to_dict(),
        "validation_windows_by_dataset": validation.dataset.value_counts().sort_index().to_dict(),
        "target_used_for_training_or_validation": False, "target_test_evaluations": 1,
        "elapsed_seconds": time.time() - started, **metrics,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result | {"source_datasets": ", ".join(result["source_datasets"])}]).to_csv(output / "metrics.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
