#!/usr/bin/env python3
"""Full-data, target-free Pulse-PPG Ridge benchmark on frozen RQ1 windows.

Stage 1 caches one embedding for every accepted Phase-7 window. Stage 2 runs
the frozen three-fold within-dataset and four LODO protocols. No target data
are used to fit scaling, Ridge weights, or its alpha hyperparameter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from scipy.signal import resample_poly
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from input_compatibility_smoke import CHECKPOINT, MANIFEST, ROOT, Net

sys.path.insert(0, str(ROOT / "src"))
from rq1_hr.evaluation import hr_metrics  # noqa: E402


OUT = ROOT / "reports/phase12_pulseppg/02_frozen_linear_probe"
CACHE = ROOT / "artifacts/pulseppg/rq1_8s_50hz_embeddings.npy"
CACHE_META = OUT / "embedding_cache_meta.json"
CACHE_PROGRESS = OUT / "embedding_cache_progress.json"
WITHIN_ROLES = ROOT / "reports/phase9_timeppg/three_fold_subject_roles.csv"
LODO_ROLES = ROOT / "reports/phase7_frozen_dataset/subject_splits.csv"
DATASETS = ("BIDMC", "WESAD", "PTT-PPG", "PPG-DaLiA")
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST, low_memory=False)
    if len(frame) != 136_625 or not frame.primary_analysis.fillna(False).all():
        raise RuntimeError("Primary frozen manifest changed; audit before proceeding")
    if frame.duplicated(["dataset", "record_id", "subject_id", "window_index"]).any():
        raise RuntimeError("Duplicate primary window key")
    frame.insert(0, "embedding_index", np.arange(len(frame)))
    return frame


def load_encoder() -> Net:
    model = Net(in_channels=1, base_filters=128, kernel_size=11, stride=2,
                groups=1, n_block=12, finalpool="max")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state["net"], strict=True)
    model.eval()
    return model


def build_embeddings(frame: pd.DataFrame, *, batch_size: int) -> np.ndarray:
    expected = {
        "manifest_sha256": sha256(MANIFEST),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "shape": [len(frame), 512],
        "preprocessing": "frozen 64Hz 0.5-4Hz PPG; resample_poly 25/32 to 50Hz; per-8s z-score",
        "input_seconds": 8,
        "output_dtype": "float32",
    }
    if CACHE_META.exists() and CACHE.exists():
        saved = json.loads(CACHE_META.read_text())
        if saved != expected:
            raise RuntimeError("Embedding cache provenance differs; do not silently reuse it")
        cached = np.load(CACHE, mmap_mode="r", allow_pickle=False)
        if cached.shape != (len(frame), 512) or cached.dtype != np.float32:
            raise RuntimeError("Embedding cache shape or dtype mismatch")
        print(f"Reusing {CACHE.relative_to(ROOT)}", flush=True)
        return cached
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PROGRESS.exists() and CACHE.exists():
        if json.loads(CACHE_PROGRESS.read_text()) != expected:
            raise RuntimeError("Partial embedding cache provenance differs")
        cache = open_memmap(CACHE, mode="r+")
        if cache.shape != (len(frame), 512) or cache.dtype != np.float32:
            raise RuntimeError("Partial embedding cache shape or dtype mismatch")
        print("Resuming partial embedding cache", flush=True)
    else:
        # An interrupted cache without this progress marker is untrusted.
        cache = open_memmap(CACHE, mode="w+", dtype=np.float32, shape=(len(frame), 512))
        cache[:] = np.nan
        cache.flush()
        CACHE_PROGRESS.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    model = load_encoder()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    count = 0
    for signal_path, group in frame.groupby("signal_path", sort=False):
        signal = np.load(ROOT / signal_path, mmap_mode="r", allow_pickle=False)
        for start in range(0, len(group), batch_size):
            rows = group.iloc[start:start + batch_size]
            batch_count = len(rows)
            row_indices = rows.embedding_index.to_numpy(dtype=int)
            completed = np.isfinite(cache[row_indices, 0])
            if completed.all():
                count += batch_count
                continue
            if completed.any():
                rows = rows.iloc[np.flatnonzero(~completed)]
                row_indices = rows.embedding_index.to_numpy(dtype=int)
            raw = np.stack([
                signal[int(row.start_sample_64hz):int(row.end_sample_64hz), int(row.channel_index)]
                for row in rows.itertuples(index=False)
            ]).astype(np.float64)
            if raw.shape[1] != 512 or not np.isfinite(raw).all():
                raise RuntimeError(f"Invalid frozen input from {signal_path}")
            sampled = resample_poly(raw, 25, 32, axis=1)
            std = sampled.std(axis=1, keepdims=True)
            if sampled.shape[1] != 400 or (std <= np.finfo(np.float32).eps).any():
                raise RuntimeError(f"Invalid resampled input from {signal_path}")
            normalized = ((sampled - sampled.mean(axis=1, keepdims=True)) / std).astype(np.float32)
            with torch.inference_mode():
                embedding = model(torch.from_numpy(normalized[:, None, :])).cpu().numpy()
            if embedding.shape != (len(rows), 512) or not np.isfinite(embedding).all():
                raise RuntimeError(f"Invalid encoder output for {signal_path}")
            cache[row_indices] = embedding
            count += batch_count
            if count % 1024 < batch_count:
                cache.flush()
                print(f"Embedded {count}/{len(frame)} windows", flush=True)
        print(f"Embedded {count}/{len(frame)} windows", flush=True)
    cache.flush()
    if not np.isfinite(cache[:, 0]).all():
        raise RuntimeError("Embedding cache contains incomplete rows")
    CACHE_META.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return np.load(CACHE, mmap_mode="r", allow_pickle=False)


def fit_source_only(embeddings: np.ndarray, train_ids: np.ndarray, val_ids: np.ndarray,
                    labels: np.ndarray) -> tuple[StandardScaler, Ridge, float, list[dict]]:
    if len(train_ids) == 0 or len(val_ids) == 0:
        raise RuntimeError("Empty source-training or source-validation split")
    x_train = np.asarray(embeddings[train_ids], dtype=np.float64)
    x_val = np.asarray(embeddings[val_ids], dtype=np.float64)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    trials = []
    best = None
    with threadpool_limits(limits=4):
        for alpha in ALPHAS:
            probe = Ridge(alpha=alpha, solver="cholesky").fit(x_train, labels[train_ids])
            score = hr_metrics(labels[val_ids], probe.predict(x_val))["mae_bpm"]
            trials.append({"alpha": alpha, "source_validation_mae_bpm": score})
            if best is None or score < best[0]:
                best = (score, alpha, probe)
    del x_train, x_val
    assert best is not None
    return scaler, best[2], best[1], trials


def predict(embeddings: np.ndarray, indices: np.ndarray, scaler: StandardScaler,
            probe: Ridge, *, batch_size: int = 4096) -> np.ndarray:
    output = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        x = np.asarray(embeddings[indices[start:start + batch_size]], dtype=np.float64)
        output[start:start + len(x)] = probe.predict(scaler.transform(x))
    return output


def result_rows(frame: pd.DataFrame, mask: np.ndarray, predictions: np.ndarray,
                *, protocol: str, fold: int | None = None) -> pd.DataFrame:
    rows = frame.loc[mask, ["dataset", "record_id", "subject_id", "window_index",
                            "condition_id", "hr_bpm", "embedding_index"]].copy()
    rows.rename(columns={"hr_bpm": "reference_hr_bpm"}, inplace=True)
    rows["predicted_hr_bpm"] = predictions
    rows["protocol"] = protocol
    if fold is not None:
        rows["test_fold"] = fold
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embed-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    frame = manifest()
    embeddings = build_embeddings(frame, batch_size=args.batch_size)
    if args.embed_only:
        return
    labels = frame.hr_bpm.to_numpy(dtype=np.float64)
    within_roles = pd.read_csv(WITHIN_ROLES)
    lodo_roles = pd.read_csv(LODO_ROLES)
    within = frame.merge(within_roles, on=["dataset", "subject_id"], validate="many_to_one")
    lodo = frame.merge(lodo_roles, on=["dataset", "subject_id"], validate="many_to_one")
    if len(within) != len(frame) or len(lodo) != len(frame):
        raise RuntimeError("Incomplete subject-role join")
    # Joins must preserve manifest order so row indices continue to index the cache.
    if not np.array_equal(within.embedding_index, frame.embedding_index) or not np.array_equal(lodo.embedding_index, frame.embedding_index):
        raise RuntimeError("Subject-role join reordered windows")

    all_predictions, run_rows, validation_rows = [], [], []
    for dataset in DATASETS:
        dataset_mask = within.dataset.eq(dataset).to_numpy()
        for fold in range(3):
            role = within[f"within3_test_fold_{fold}_role"].to_numpy()
            train_mask = dataset_mask & (role == "train")
            val_mask = dataset_mask & (role == "validation")
            test_mask = dataset_mask & (role == "test")
            train_ids, val_ids, test_ids = (np.flatnonzero(mask) for mask in (train_mask, val_mask, test_mask))
            train_subjects = set(frame.loc[train_mask, "subject_id"])
            val_subjects = set(frame.loc[val_mask, "subject_id"])
            test_subjects = set(frame.loc[test_mask, "subject_id"])
            if (train_subjects & val_subjects) or (train_subjects & test_subjects) or (val_subjects & test_subjects):
                raise RuntimeError("Within-dataset subject leakage")
            scaler, probe, alpha, trials = fit_source_only(embeddings, train_ids, val_ids, labels)
            predictions = predict(embeddings, test_ids, scaler, probe)
            metrics = hr_metrics(labels[test_ids], predictions)
            run_rows.append({"protocol": "within", "dataset": dataset, "test_fold": fold,
                             "train_windows": len(train_ids), "validation_windows": len(val_ids),
                             "test_windows": len(test_ids), "alpha": alpha, **metrics})
            validation_rows.append({"protocol": "within", "dataset": dataset, "test_fold": fold,
                                    "trials": trials})
            all_predictions.append(result_rows(frame, test_mask, predictions, protocol="within", fold=fold))
            print(f"Within {dataset} fold {fold}: MAE {metrics['mae_bpm']:.3f}", flush=True)

    within_predictions = pd.concat(all_predictions, ignore_index=True)
    if len(within_predictions) != len(frame) or within_predictions.embedding_index.duplicated().any():
        raise RuntimeError("Each within-dataset window must be tested exactly once")
    within_predictions.to_csv(OUT / "within_test_predictions.csv", index=False)
    lodo_predictions = []
    for target in DATASETS:
        role = lodo[f"lodo_{target}_role"].to_numpy()
        train_mask = role == "source_train"
        val_mask = role == "source_validation"
        test_mask = role == "target_test"
        if not (frame.loc[test_mask, "dataset"] == target).all():
            raise RuntimeError("LODO target role mismatch")
        if (frame.loc[train_mask | val_mask, "dataset"] == target).any():
            raise RuntimeError("Target dataset leaked into source")
        train_ids, val_ids, test_ids = (np.flatnonzero(mask) for mask in (train_mask, val_mask, test_mask))
        scaler, probe, alpha, trials = fit_source_only(embeddings, train_ids, val_ids, labels)
        predictions = predict(embeddings, test_ids, scaler, probe)
        metrics = hr_metrics(labels[test_ids], predictions)
        run_rows.append({"protocol": "lodo", "dataset": target, "test_fold": None,
                         "train_windows": len(train_ids), "validation_windows": len(val_ids),
                         "test_windows": len(test_ids), "alpha": alpha, **metrics})
        validation_rows.append({"protocol": "lodo", "dataset": target, "trials": trials})
        lodo_predictions.append(result_rows(frame, test_mask, predictions, protocol="lodo"))
        print(f"LODO target {target}: MAE {metrics['mae_bpm']:.3f}", flush=True)
    lodo_predictions = pd.concat(lodo_predictions, ignore_index=True)
    if len(lodo_predictions) != len(frame) or lodo_predictions.embedding_index.duplicated().any():
        raise RuntimeError("Every LODO target window must be tested exactly once")
    lodo_predictions.to_csv(OUT / "lodo_target_predictions.csv", index=False)
    pd.DataFrame(run_rows).to_csv(OUT / "run_metrics.csv", index=False)
    (OUT / "source_validation_trials.json").write_text(json.dumps(validation_rows, indent=2) + "\n", encoding="utf-8")

    summary = []
    for dataset in DATASETS:
        within_rows = within_predictions.loc[within_predictions.dataset.eq(dataset)]
        lodo_rows = lodo_predictions.loc[lodo_predictions.dataset.eq(dataset)]
        within_metrics = hr_metrics(within_rows.reference_hr_bpm, within_rows.predicted_hr_bpm)
        lodo_metrics = hr_metrics(lodo_rows.reference_hr_bpm, lodo_rows.predicted_hr_bpm)
        summary.append({"dataset": dataset, "windows": len(within_rows),
                        "within_mae_bpm": within_metrics["mae_bpm"],
                        "lodo_mae_bpm": lodo_metrics["mae_bpm"],
                        "gap_bpm": lodo_metrics["mae_bpm"] - within_metrics["mae_bpm"],
                        **{f"within_{key}": value for key, value in within_metrics.items() if key != "mae_bpm"},
                        **{f"lodo_{key}": value for key, value in lodo_metrics.items() if key != "mae_bpm"}})
    pd.DataFrame(summary).to_csv(OUT / "dataset_summary.csv", index=False)
    print(pd.DataFrame(summary)[["dataset", "windows", "within_mae_bpm", "lodo_mae_bpm", "gap_bpm"]].to_string(index=False))


if __name__ == "__main__":
    main()
