#!/usr/bin/env python3
"""Source-only Pulse-PPG final-block fine-tuning, warm-started from Phase 12D.

The stem and residual blocks 0–10 are frozen in eval mode. Only residual
block 11 and the previously source-trained nonlinear HR head are updated.
The held-out target is never used for training, early stopping, or LR choice.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/models/pulseppg"))
from rq1_hr.evaluation import hr_metrics  # noqa: E402
from run_full_linear_probe import (  # noqa: E402
    CACHE, CHECKPOINT, DATASETS, LODO_ROLES, MANIFEST, WITHIN_ROLES,
    joined_roles, load_encoder, manifest_frame, predictions, sha256, subject_summary,
)
from run_nonlinear_head_ablation import CONFIGS, MLPHead, validation_score  # noqa: E402


INPUT_CACHE = ROOT / "artifacts/pulseppg/phase12_inputs_8s_50hz"
HEAD_ROOT = ROOT / "reports/phase12_pulseppg/03_head_and_tuning_ablation/nonlinear_head"
OUT = ROOT / "reports/phase12_pulseppg/03_head_and_tuning_ablation/partial_final_block"
LEARNING_RATES = (3e-5, 1e-4)
KEY = ["dataset", "record_id", "subject_id", "window_index"]


def cache_inputs(frame: pd.DataFrame, batch_size: int) -> None:
    INPUT_CACHE.mkdir(parents=True, exist_ok=True)
    path = INPUT_CACHE / "windows_8s_50hz.npy"
    metadata = INPUT_CACHE / "metadata.json"
    if metadata.exists():
        raise FileExistsError("Complete input cache already exists; use --stage train")
    if path.exists():
        raise FileExistsError("Incomplete input cache exists; inspect it before recreating")
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(len(frame), 400))
    from run_full_linear_probe import batch_windows
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        matrix[start:stop] = batch_windows(frame.iloc[start:stop])
        if stop % 8192 < batch_size or stop == len(frame):
            print(f"Prepared {stop:,}/{len(frame):,} windows", flush=True)
    matrix.flush()
    metadata.write_text(json.dumps({
        "status": "complete", "rows": len(frame), "shape": [len(frame), 400],
        "manifest_sha256": sha256(MANIFEST), "checkpoint_sha256": sha256(CHECKPOINT),
        "preprocessing": "frozen 64-Hz 0.5–4-Hz 512-sample window -> resample_poly(25,32) -> independent window z-score",
        "matrix": path.name,
    }, indent=2) + "\n", encoding="utf-8")


def load_inputs(frame: pd.DataFrame) -> np.ndarray:
    metadata = json.loads((INPUT_CACHE / "metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or metadata["manifest_sha256"] != sha256(MANIFEST):
        raise RuntimeError("Input cache is not complete or does not match the manifest")
    if metadata["checkpoint_sha256"] != sha256(CHECKPOINT):
        raise RuntimeError("Input cache checkpoint provenance differs")
    array = np.load(INPUT_CACHE / metadata["matrix"], mmap_mode="r", allow_pickle=False)
    if array.shape != (len(frame), 400):
        raise RuntimeError("Incorrect cached input shape")
    return array


class PartialModel(nn.Module):
    def __init__(self, encoder: nn.Module, head_checkpoint: dict):
        super().__init__()
        configs = {config.name: config for config in CONFIGS}
        config_name = head_checkpoint["config"]
        if config_name not in configs:
            raise RuntimeError(f"Unknown source-selected head: {config_name}")
        self.encoder = encoder
        self.head = MLPHead(512, configs[config_name])
        self.head.load_state_dict(head_checkpoint["state_dict"], strict=True)
        self.register_buffer("feature_mean", torch.as_tensor(head_checkpoint["feature_mean"], dtype=torch.float32))
        self.register_buffer("feature_scale", torch.as_tensor(head_checkpoint["feature_scale"], dtype=torch.float32))
        self.register_buffer("label_mean", torch.tensor(float(head_checkpoint["y_mean"])))
        self.register_buffer("label_scale", torch.tensor(float(head_checkpoint["y_scale"])))
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.encoder.basicblock_list[11].parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen prefix BN and dropout in the pretrained eval state.
        self.encoder.eval()
        self.encoder.basicblock_list[11].train(mode)
        self.head.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        with torch.no_grad():
            z = encoder.instnorm(x)
            z = encoder.first_block_conv(z)
            if encoder.use_bn:
                z = encoder.first_block_bn(z)
            z = encoder.first_block_relu(z)
            for block in encoder.basicblock_list[:11]:
                z = block(z)
        z = encoder.basicblock_list[11](z)
        z = torch.max(z, dim=-1).values
        z = (z - self.feature_mean) / self.feature_scale
        return self.head(z) * self.label_scale + self.label_mean


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(indices: np.ndarray, batch_size: int, *, seed: int | None = None):
    order = indices.copy()
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def predict(model: PartialModel, inputs: np.ndarray, indices: np.ndarray,
            device: torch.device, batch_size: int) -> np.ndarray:
    model.eval(); parts = []
    with torch.inference_mode():
        for index in batches(indices, batch_size):
            x = torch.from_numpy(np.asarray(inputs[index], dtype=np.float32)[:, None, :]).to(device)
            parts.append(model(x).cpu().numpy())
    return np.concatenate(parts)


def head_checkpoint_path(protocol: str, target: str, fold: int | None, seed: int) -> Path:
    slug = target.lower().replace("-", "_")
    if protocol == "within":
        return HEAD_ROOT / "within_dataset" / slug / f"fold_{fold}" / f"seed_{seed}" / "head_checkpoint.pt"
    return HEAD_ROOT / "lodo" / f"target_{slug}" / f"seed_{seed}" / "head_checkpoint.pt"


def fit_trial(inputs: np.ndarray, labels: np.ndarray, domains: np.ndarray,
              train: np.ndarray, validation: np.ndarray, head_checkpoint: dict,
              learning_rate: float, seed: int, device: torch.device,
              batch_size: int, max_epochs: int, patience: int) -> tuple[dict, dict]:
    seed_everything(seed)
    model = PartialModel(load_encoder(torch.device("cpu")), head_checkpoint).to(device)
    train_indices = np.flatnonzero(train); validation_indices = np.flatnonzero(validation)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.basicblock_list[11].parameters(), "lr": learning_rate},
        {"params": model.head.parameters(), "lr": learning_rate * 3},
    ], weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=10.0)
    best_score = float("inf"); best_state = None; best_epoch = 0; stale = 0; history = []
    for epoch in range(1, max_epochs + 1):
        model.train(); total = 0.0; seen = 0
        for index in batches(train_indices, batch_size, seed=seed * 1000 + epoch):
            x = torch.from_numpy(np.asarray(inputs[index], dtype=np.float32)[:, None, :]).to(device)
            y = torch.as_tensor(labels[index], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step(); total += float(loss.detach().cpu()) * len(index); seen += len(index)
        val_pred = predict(model, inputs, validation_indices, device, batch_size)
        score, by_domain = validation_score(labels[validation], val_pred, domains[validation])
        history.append({"epoch": epoch, "train_huber_bpm": total / seen,
                        "validation_dataset_macro_mae_bpm": score,
                        "validation_mae_by_dataset": by_domain})
        print(f"    lr={learning_rate:g} seed={seed} epoch={epoch} "
              f"source-val={score:.3f} bpm", flush=True)
        if score < best_score - 1e-6:
            best_score = score; best_epoch = epoch; stale = 0
            best_state = {
                "final_block": {key: val.detach().cpu().clone()
                                for key, val in model.encoder.basicblock_list[11].state_dict().items()},
                "head": {key: val.detach().cpu().clone() for key, val in model.head.state_dict().items()},
            }
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    return best_state, {"learning_rate": learning_rate, "seed": seed,
                        "best_epoch": best_epoch, "best_source_validation_macro_mae_bpm": best_score,
                        "epochs_run": len(history), "history": history}


def run_split(frame: pd.DataFrame, inputs: np.ndarray, labels: np.ndarray,
              domains: np.ndarray, train: np.ndarray, validation: np.ndarray,
              test: np.ndarray, *, protocol: str, target: str, fold: int | None,
              seeds: list[int], device: torch.device, batch_size: int,
              max_epochs: int, patience: int, output_root: Path) -> list[dict]:
    if np.any(train & validation) or np.any(train & test) or np.any(validation & test):
        raise RuntimeError("Split masks overlap")
    if protocol == "lodo" and (np.any((train | validation) & (domains == target)) or
                                not np.array_equal(test, domains == target)):
        raise RuntimeError("Held-out target leaks into LODO training or validation")
    slug = target.lower().replace("-", "_")
    split_out = output_root / protocol / f"target_{slug}"
    if fold is not None:
        split_out = split_out / f"fold_{fold}"
    split_out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in seeds:
        head_path = head_checkpoint_path(protocol, target, fold, seed)
        # Phase-12D locally generated checkpoints contain NumPy scaler arrays,
        # which are not accepted by torch's restricted weights-only loader.
        # Never point this at an untrusted or downloaded head checkpoint.
        head_state = torch.load(head_path, map_location="cpu", weights_only=False)
        expected_selection = head_path.parents[1] / "source_validation_selection.json"
        selection = json.loads(expected_selection.read_text(encoding="utf-8"))
        if head_state["config"] != selection["selected_config"]:
            raise RuntimeError("Warm-start head is not the source-selected configuration")
        initial_model = PartialModel(load_encoder(torch.device("cpu")), head_state).to(device)
        validation_indices = np.flatnonzero(validation)
        initial_pred = predict(initial_model, inputs, validation_indices, device, batch_size)
        initial_score, _ = validation_score(labels[validation], initial_pred, domains[validation])
        prior_trials = [trial for trial in selection["trials"]
                        if trial["config"] == head_state["config"] and trial["seed"] == seed]
        if len(prior_trials) != 1 or abs(
            initial_score - prior_trials[0]["best_validation_dataset_macro_mae_bpm"]
        ) > 0.01:
            raise RuntimeError("Raw-window warm start does not reproduce the frozen-head validation result")
        del initial_model
        trials = []; states = {}
        for learning_rate in LEARNING_RATES:
            state, trial = fit_trial(inputs, labels, domains, train, validation, head_state,
                                     learning_rate, seed, device, batch_size, max_epochs, patience)
            trials.append(trial); states[learning_rate] = state
        chosen = min(trials, key=lambda trial: trial["best_source_validation_macro_mae_bpm"])
        # The source-only warm-start is an explicit no-adaptation control.
        use_initial = initial_score <= chosen["best_source_validation_macro_mae_bpm"]
        model = PartialModel(load_encoder(torch.device("cpu")), head_state).to(device)
        if not use_initial:
            state = states[chosen["learning_rate"]]
            model.encoder.basicblock_list[11].load_state_dict(state["final_block"], strict=True)
            model.head.load_state_dict(state["head"], strict=True)
        test_indices = np.flatnonzero(test)
        prediction = predict(model, inputs, test_indices, device, batch_size)
        result = predictions(frame, test_indices, prediction, fold=fold)
        seed_out = split_out / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        result.to_csv(seed_out / "test_predictions.csv", index=False)
        macro, interval, subject = subject_summary(result)
        subject.to_csv(seed_out / "subject_metrics.csv", index=False)
        torch.save({
            "final_block": {key: val.detach().cpu() for key, val in model.encoder.basicblock_list[11].state_dict().items()},
            "head": {key: val.detach().cpu() for key, val in model.head.state_dict().items()},
            "head_config": head_state["config"], "head_checkpoint": str(head_path.relative_to(ROOT)),
            "source_selected_no_adaptation": use_initial,
        }, seed_out / "partial_checkpoint.pt")
        selected_score = initial_score if use_initial else chosen["best_source_validation_macro_mae_bpm"]
        record = {"target": target, "protocol": protocol, "fold": fold, "seed": seed,
                  "source_selected_no_adaptation": use_initial,
                  "initial_source_validation_macro_mae_bpm": initial_score,
                  "selected_source_validation_macro_mae_bpm": selected_score,
                  "selected_lr": None if use_initial else chosen["learning_rate"],
                  "selected_epoch": 0 if use_initial else chosen["best_epoch"],
                  "train_windows": int(train.sum()), "validation_windows": int(validation.sum()),
                  "subject_macro_mae_bpm": macro,
                  "subject_bootstrap_95_ci_low_bpm": interval[0],
                  "subject_bootstrap_95_ci_high_bpm": interval[1],
                  **hr_metrics(labels[test], prediction)}
        (seed_out / "selection.json").write_text(json.dumps({
            "target_data_used_for_selection": False, "initial_source_validation_macro_mae_bpm": initial_score,
            "selected": record, "trials": trials,
        }, indent=2) + "\n", encoding="utf-8")
        summaries.append(record)
        print(f"  {target} {protocol} fold={fold} seed={seed}: "
              f"selected source-val={selected_score:.3f}, test MAE={record['mae_bpm']:.3f}", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="train")
    parser.add_argument("--protocol", choices=("within", "lodo", "all"), default="all")
    parser.add_argument("--targets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="17,29,43")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--prepare-batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    targets = [value.strip() for value in args.targets.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not targets or not set(targets).issubset(DATASETS) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid targets or seeds")
    if min(args.batch_size, args.max_epochs, args.patience, args.prepare_batch_size) < 1:
        raise ValueError("Training and cache arguments must be positive")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    frame = manifest_frame()
    if args.stage in ("prepare", "all"):
        cache_inputs(frame, args.prepare_batch_size)
    if args.stage == "prepare":
        return
    inputs = load_inputs(frame); labels = frame.hr_bpm.to_numpy(float)
    domains = frame.dataset.to_numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.protocol in ("within", "all"):
        roles = joined_roles(frame, WITHIN_ROLES, [f"within3_test_fold_{fold}_role" for fold in range(3)])
        for target in targets:
            base = domains == target
            for fold in range(3):
                role = roles[f"within3_test_fold_{fold}_role"].to_numpy()
                rows.extend(run_split(frame, inputs, labels, domains,
                                      base & (role == "train"), base & (role == "validation"),
                                      base & (role == "test"), protocol="within", target=target,
                                      fold=fold, seeds=seeds, device=device, batch_size=args.batch_size,
                                      max_epochs=args.max_epochs, patience=args.patience,
                                      output_root=args.output))
    if args.protocol in ("lodo", "all"):
        roles = joined_roles(frame, LODO_ROLES, [f"lodo_{target}_role" for target in DATASETS])
        for target in targets:
            role = roles[f"lodo_{target}_role"].to_numpy()
            rows.extend(run_split(frame, inputs, labels, domains,
                                  role == "source_train", role == "source_validation",
                                  role == "target_test", protocol="lodo", target=target,
                                  fold=None, seeds=seeds, device=device, batch_size=args.batch_size,
                                  max_epochs=args.max_epochs, patience=args.patience,
                                  output_root=args.output))
    pd.DataFrame(rows).to_csv(args.output / "run_summary.csv", index=False)
    primary_complete = (args.protocol == "all" and set(targets) == set(DATASETS)
                        and set(seeds) == {17, 29, 43}
                        and args.max_epochs == 30 and args.patience == 5)
    (args.output / "run_provenance.json").write_text(json.dumps({
        "status": "primary_complete" if primary_complete else "subset_or_pilot_complete",
        "encoder_checkpoint_sha256": sha256(CHECKPOINT),
        "manifest_sha256": sha256(MANIFEST), "within_roles_sha256": sha256(WITHIN_ROLES),
        "lodo_roles_sha256": sha256(LODO_ROLES), "encoder_final_block": 11,
        "frozen_prefix_blocks": list(range(11)), "head_warm_start": "source-selected Phase 12D nonlinear head",
        "learning_rate_grid": LEARNING_RATES, "head_lr_multiplier": 3,
        "max_epochs": args.max_epochs, "patience": args.patience, "batch_size": args.batch_size,
        "seeds": seeds, "targets": targets, "protocol": args.protocol, "device": device_name,
        "source_validation_only_selection": True, "target_used_for_selection": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
