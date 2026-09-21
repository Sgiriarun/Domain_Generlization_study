#!/usr/bin/env python3
"""Train one frozen subject-wise within-dataset TimePPG fold."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase7-dir", type=Path, default=Path("reports/phase7_frozen_dataset"))
    parser.add_argument("--manifest-file", type=Path,
                        help="Override the default main_window_manifest.csv (for controlled channel experiments)")
    parser.add_argument("--channel-name", help="Optionally retain one named PPG channel")
    parser.add_argument("--splits-file", type=Path)
    parser.add_argument("--role-column")
    parser.add_argument("--output-root", type=Path, default=Path("reports/phase9_timeppg/within_dataset"))
    parser.add_argument("--dataset", default="PPG-DaLiA")
    parser.add_argument("--test-fold", type=int, default=0)
    parser.add_argument("--validation-fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if args.test_fold == args.validation_fold:
        raise ValueError("test and validation folds must differ")

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
    slug = args.dataset.lower().replace("-", "_")
    output = args.output_root / slug / f"test_fold_{args.test_fold}" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest_file or args.phase7_dir / "main_window_manifest.csv"
    manifest = pd.read_csv(manifest_path, low_memory=False)
    splits_file = args.splits_file or args.phase7_dir / "subject_splits.csv"
    subjects = pd.read_csv(splits_file)
    split_column = args.role_column or "within_dataset_fold"
    frame = attach_split_roles(manifest, subjects, split_column)
    frame = frame.loc[frame.dataset.eq(args.dataset)].reset_index(drop=True)
    if args.channel_name:
        frame = frame.loc[frame.channel_name.eq(args.channel_name)].reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"no windows found for channel {args.channel_name}")
    if args.role_column:
        test = frame.loc[frame[split_column].eq("test")].reset_index(drop=True)
        validation = frame.loc[frame[split_column].eq("validation")].reset_index(drop=True)
        train = frame.loc[frame[split_column].eq("train")].reset_index(drop=True)
    else:
        test = frame.loc[frame[split_column].eq(args.test_fold)].reset_index(drop=True)
        validation = frame.loc[frame[split_column].eq(args.validation_fold)].reset_index(drop=True)
        train = frame.loc[~frame[split_column].isin([args.test_fold, args.validation_fold])].reset_index(drop=True)
    sets = {"train": set(train.subject_id), "validation": set(validation.subject_id), "test": set(test.subject_id)}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise RuntimeError("subject leakage detected")

    train_loader = loader(train, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = loader(validation, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    test_loader = loader(test, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    model = TimePPGBigPPGOnly(512).to(device)
    loss_function = LogCoshLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8
    )

    history, best_loss, best_epoch, waited = [], float("inf"), 0, 0
    checkpoint_path = output / "best_checkpoint.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_function, optimizer=optimizer, device=device)
        validation_metrics = run_epoch(model, validation_loader, loss_function, device=device)
        elapsed = time.time() - start
        row = {
            "epoch": epoch, "elapsed_seconds": elapsed,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        improved = validation_metrics["loss"] < best_loss
        if improved:
            best_loss, best_epoch, waited = validation_metrics["loss"], epoch, 0
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "validation_loss": best_loss, "arguments": vars(args),
                "subjects": {key: sorted(value) for key, value in sets.items()},
                "splits_file": str(splits_file), "split_column": split_column,
            }, checkpoint_path)
        else:
            waited += 1
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(
            f"epoch={epoch:03d} train_mae={train_metrics['mae_bpm']:.3f} "
            f"val_mae={validation_metrics['mae_bpm']:.3f} val_loss={validation_metrics['loss']:.3f} "
            f"best={best_epoch:03d} wait={waited:02d}", flush=True,
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
    test_rows.to_csv(output / "test_predictions.csv", index=False)
    metrics = hr_metrics(references, predictions)
    result = {
        "status": "complete", "dataset": args.dataset, "seed": args.seed,
        "device": device_name, "manifest_file": str(manifest_path),
        "channel_name": args.channel_name,
        "test_fold": args.test_fold, "validation_fold": args.validation_fold,
        "best_epoch": best_epoch, "stopped_epoch": len(history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_windows": len(train), "validation_windows": len(validation), "test_windows": len(test),
        "subjects": {key: sorted(value) for key, value in sets.items()},
        "subject_overlap": False, "test_evaluations": 1,
        "splits_file": str(splits_file), "split_column": split_column,
        **metrics,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([result | {"subjects": "see metrics.json"}]).to_csv(output / "metrics.csv", index=False)
    readme = f"""# TimePPG within-dataset result: {args.dataset}

- Test fold: {args.test_fold}; validation fold: {args.validation_fold}
- Training subjects: {len(sets['train'])}; validation subjects: {len(sets['validation'])}; test subjects: {len(sets['test'])}
- Subject overlap: none
- Best epoch selected by validation log-cosh loss: {best_epoch}
- Test set evaluated once after checkpoint selection
- Test MAE: {metrics['mae_bpm']:.3f} bpm
- Test RMSE: {metrics['rmse_bpm']:.3f} bpm
- Pearson correlation: {metrics['pearson_r']:.4f}
- Within +/-5 bpm: {metrics['within_5_bpm_percent']:.2f}%

This is one fold and one seed, not the final five-fold/three-seed aggregate.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
