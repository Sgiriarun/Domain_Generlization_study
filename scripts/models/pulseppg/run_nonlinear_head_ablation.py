#!/usr/bin/env python3
"""Frozen Pulse-PPG nonlinear-head ablation with source-only selection.

The pretrained encoder and cached 512-dimensional embeddings remain frozen.
Candidate MLP heads are trained on source-training subjects, early-stopped and
selected using source-validation subjects, and evaluated on the held-out test
subjects/dataset only after selection. No target data enter model selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/models/pulseppg"))
from rq1_hr.evaluation import hr_metrics  # noqa: E402
from run_full_linear_probe import (  # noqa: E402
    CACHE,
    CHECKPOINT,
    DATASETS,
    LODO_ROLES,
    MANIFEST,
    WITHIN_ROLES,
    cached_features,
    joined_roles,
    manifest_frame,
    predictions,
    sha256,
    subject_summary,
)


OUTPUT = ROOT / "reports/phase12_pulseppg/03_head_and_tuning_ablation/nonlinear_head"


@dataclass(frozen=True)
class HeadConfig:
    name: str
    hidden: tuple[int, ...]
    dropout: float


# Two predeclared capacities answer whether nonlinear access to the frozen
# representation helps without an uncontrolled architecture search.
CONFIGS = (
    HeadConfig("mlp_128", (128,), 0.10),
    HeadConfig("mlp_256_64", (256, 64), 0.10),
)


class MLPHead(nn.Module):
    def __init__(self, input_features: int, config: HeadConfig):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_features
        for width in config.hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(config.dropout)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def scaled_split(features: np.ndarray, labels: np.ndarray, train: np.ndarray,
                 validation: np.ndarray) -> tuple[StandardScaler, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not train.any() or not validation.any() or np.any(train & validation):
        raise RuntimeError("Invalid train/validation masks")
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train]).astype(np.float32, copy=False)
    x_validation = scaler.transform(features[validation]).astype(np.float32, copy=False)
    y_mean = float(np.mean(labels[train])); y_scale = float(np.std(labels[train]))
    if y_scale <= 0:
        raise RuntimeError("Training labels have zero variance")
    y_train = ((labels[train] - y_mean) / y_scale).astype(np.float32)
    y_validation = labels[validation].astype(np.float32)
    return scaler, y_mean, y_scale, x_train, y_train, x_validation, y_validation


def predict(model: nn.Module, values: np.ndarray, y_mean: float, y_scale: float,
            device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(values)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    parts = []
    model.eval()
    with torch.inference_mode():
        for (batch,) in loader:
            result = model(batch.to(device)).cpu().numpy() * y_scale + y_mean
            parts.append(result)
    return np.concatenate(parts)


def validation_score(reference: np.ndarray, predicted: np.ndarray,
                     domains: np.ndarray) -> tuple[float, dict[str, float]]:
    by_dataset = {}
    for dataset in sorted(set(domains)):
        mask = domains == dataset
        by_dataset[dataset] = float(np.mean(np.abs(predicted[mask] - reference[mask])))
    return float(np.mean(list(by_dataset.values()))), by_dataset


def fit_candidate(config: HeadConfig, seed: int, x_train: np.ndarray, y_train: np.ndarray,
                  x_validation: np.ndarray, y_validation: np.ndarray,
                  validation_domains: np.ndarray, y_mean: float, y_scale: float,
                  device: torch.device, batch_size: int, max_epochs: int,
                  patience: int) -> tuple[dict[str, torch.Tensor], dict]:
    seed_everything(seed)
    model = MLPHead(x_train.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Huber is robust to large HR errors while remaining smooth near the optimum.
    loss_function = nn.HuberLoss(delta=1.0)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
                        batch_size=batch_size, shuffle=True, generator=generator,
                        num_workers=0, drop_last=False)
    best_score = float("inf"); best_state = None; best_epoch = 0; stale = 0; history = []
    for epoch in range(1, max_epochs + 1):
        model.train(); total_loss = 0.0; seen = 0
        for values, targets in loader:
            values = values.to(device); targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(values), targets)
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(values); seen += len(values)
        val_prediction = predict(model, x_validation, y_mean, y_scale, device, batch_size * 2)
        score, by_dataset = validation_score(y_validation, val_prediction, validation_domains)
        history.append({"epoch": epoch, "train_huber": total_loss / seen,
                        "validation_dataset_macro_mae_bpm": score,
                        "validation_mae_by_dataset": by_dataset})
        if score < best_score - 1e-6:
            best_score = score; best_epoch = epoch; stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    return best_state, {"config": config.name, "hidden": config.hidden, "dropout": config.dropout,
                        "seed": seed, "best_epoch": best_epoch,
                        "best_validation_dataset_macro_mae_bpm": best_score,
                        "epochs_run": len(history), "history": history}


def fit_select_evaluate(frame: pd.DataFrame, features: np.ndarray, labels: np.ndarray,
                        domains: np.ndarray, train: np.ndarray, validation: np.ndarray,
                        test: np.ndarray, output: Path, seeds: list[int], device: torch.device,
                        batch_size: int, max_epochs: int, patience: int,
                        fold: int | None = None) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    scaler, y_mean, y_scale, x_train, y_train, x_val, y_val = scaled_split(
        features, labels, train, validation
    )
    validation_domains = domains[validation]
    states: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    trials = []
    for config in CONFIGS:
        for seed in seeds:
            state, trial = fit_candidate(config, seed, x_train, y_train, x_val, y_val,
                                         validation_domains, y_mean, y_scale, device,
                                         batch_size, max_epochs, patience)
            states[(config.name, seed)] = state; trials.append(trial)
            print(f"  {config.name} seed {seed}: source-val macro MAE "
                  f"{trial['best_validation_dataset_macro_mae_bpm']:.3f} at epoch {trial['best_epoch']}", flush=True)
    scores = {
        config.name: float(np.mean([trial["best_validation_dataset_macro_mae_bpm"]
                                    for trial in trials if trial["config"] == config.name]))
        for config in CONFIGS
    }
    selected_name = min(scores, key=scores.get)
    selected_config = next(config for config in CONFIGS if config.name == selected_name)
    indices = np.flatnonzero(test)
    x_test = scaler.transform(features[test]).astype(np.float32, copy=False)
    results = []
    for seed in seeds:
        model = MLPHead(features.shape[1], selected_config).to(device)
        model.load_state_dict(states[(selected_name, seed)], strict=True)
        test_prediction = predict(model, x_test, y_mean, y_scale, device, batch_size * 2)
        result = predictions(frame, indices, test_prediction, fold=fold)
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        result.to_csv(seed_output / "test_predictions.csv", index=False)
        torch.save({"state_dict": states[(selected_name, seed)], "config": selected_name,
                    "seed": seed, "y_mean": y_mean, "y_scale": y_scale,
                    "feature_mean": scaler.mean_, "feature_scale": scaler.scale_},
                   seed_output / "head_checkpoint.pt")
        macro, interval, subject = subject_summary(result)
        subject.to_csv(seed_output / "subject_metrics.csv", index=False)
        row = {"seed": seed, "selected_config": selected_name,
               "source_validation_macro_mae_bpm": scores[selected_name],
               "train_windows": int(train.sum()), "validation_windows": int(validation.sum()),
               "test_windows": int(test.sum()), "subject_macro_mae_bpm": macro,
               "subject_bootstrap_95_ci_low_bpm": interval[0],
               "subject_bootstrap_95_ci_high_bpm": interval[1],
               **hr_metrics(labels[test], test_prediction)}
        results.append(row)
    selection = {"selection_uses_target": False, "selected_config": selected_name,
                 "configuration_mean_source_validation_mae_bpm": scores,
                 "seeds": seeds, "trials": trials}
    (output / "source_validation_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(output / "test_metrics_by_seed.csv", index=False)
    return results


def evaluate(protocol: str, targets: list[str], frame: pd.DataFrame, features: np.ndarray,
             labels: np.ndarray, seeds: list[int], device: torch.device, batch_size: int,
             max_epochs: int, patience: int, output_root: Path) -> pd.DataFrame:
    domains = frame.dataset.to_numpy()
    summaries = []
    if protocol in ("within", "all"):
        columns = [f"within3_test_fold_{fold}_role" for fold in range(3)]
        roles = joined_roles(frame, WITHIN_ROLES, columns)
        for dataset in targets:
            base = domains == dataset
            seed_parts: dict[int, list[pd.DataFrame]] = {seed: [] for seed in seeds}
            for fold in range(3):
                role = roles[f"within3_test_fold_{fold}_role"].to_numpy()
                train = base & (role == "train"); validation = base & (role == "validation")
                test = base & (role == "test")
                print(f"Within {dataset}, fold {fold}", flush=True)
                rows = fit_select_evaluate(frame, features, labels, domains, train, validation, test,
                                           output_root / "within_dataset" / dataset.lower().replace("-", "_") / f"fold_{fold}",
                                           seeds, device, batch_size, max_epochs, patience, fold=fold)
                for row in rows:
                    path = output_root / "within_dataset" / dataset.lower().replace("-", "_") / f"fold_{fold}" / f"seed_{row['seed']}" / "test_predictions.csv"
                    seed_parts[row["seed"]].append(pd.read_csv(path))
            for seed, parts in seed_parts.items():
                combined = pd.concat(parts, ignore_index=True)
                if len(combined) != int(base.sum()) or combined.duplicated(
                    ["dataset", "record_id", "subject_id", "window_index"]
                ).any():
                    raise RuntimeError("Within-dataset predictions do not test every window exactly once")
                path = output_root / "within_dataset" / dataset.lower().replace("-", "_") / f"seed_{seed}"
                path.mkdir(parents=True, exist_ok=True)
                combined.to_csv(path / "all_test_predictions.csv", index=False)
                macro, interval, subject = subject_summary(combined)
                subject.to_csv(path / "subject_metrics.csv", index=False)
                summaries.append({"protocol": "within", "target": dataset, "seed": seed,
                                  "subject_macro_mae_bpm": macro,
                                  "subject_bootstrap_95_ci_low_bpm": interval[0],
                                  "subject_bootstrap_95_ci_high_bpm": interval[1],
                                  **hr_metrics(combined.reference_hr_bpm, combined.predicted_hr_bpm)})
    if protocol in ("lodo", "all"):
        columns = [f"lodo_{dataset}_role" for dataset in DATASETS]
        roles = joined_roles(frame, LODO_ROLES, columns)
        for target in targets:
            role = roles[f"lodo_{target}_role"].to_numpy()
            train = role == "source_train"; validation = role == "source_validation"
            test = role == "target_test"
            if np.any((train | validation) & (domains == target)) or not np.array_equal(test, domains == target):
                raise RuntimeError("Target leakage in LODO masks")
            print(f"LODO target {target}", flush=True)
            rows = fit_select_evaluate(frame, features, labels, domains, train, validation, test,
                                       output_root / "lodo" / f"target_{target.lower().replace('-', '_')}",
                                       seeds, device, batch_size, max_epochs, patience)
            summaries.extend({"protocol": "lodo", "target": target, **row} for row in rows)
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("within", "lodo", "all"), default="all")
    parser.add_argument("--targets", default=",".join(DATASETS),
                        help="Comma-separated subset of BIDMC,WESAD,PTT-PPG,PPG-DaLiA")
    parser.add_argument("--seeds", default="17", help="Comma-separated seeds; use 17,29,43 for replication")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    if not targets or not set(targets).issubset(DATASETS):
        raise ValueError(f"Targets must be selected from {DATASETS}")
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or args.batch_size < 1 or args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Invalid seeds or training arguments")
    device = device_from_name(args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print(f"Nonlinear-head device: {device}; seeds={seeds}", flush=True)
    frame = manifest_frame(); features = cached_features(len(frame)); labels = frame.hr_bpm.to_numpy(float)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = evaluate(args.protocol, targets, frame, features, labels, seeds, device,
                       args.batch_size, args.max_epochs, args.patience, args.output)
    summary.to_csv(args.output / "run_summary.csv", index=False)
    provenance = {
        "status": "complete", "protocol": args.protocol, "targets": targets, "seeds": seeds,
        "manifest_sha256": sha256(MANIFEST), "within_roles_sha256": sha256(WITHIN_ROLES),
        "lodo_roles_sha256": sha256(LODO_ROLES), "encoder_checkpoint_sha256": sha256(CHECKPOINT),
        "embedding_metadata": json.loads((CACHE / "metadata.json").read_text(encoding="utf-8")),
        "encoder_frozen": True, "target_used_for_selection": False,
        "candidate_heads": [config.__dict__ for config in CONFIGS],
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)", "loss": "Huber(delta=1.0) on train-standardized HR",
        "early_stopping_metric": "source-validation dataset-macro MAE",
        "batch_size": args.batch_size, "max_epochs": args.max_epochs, "patience": args.patience,
        "device": str(device),
    }
    (args.output / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
