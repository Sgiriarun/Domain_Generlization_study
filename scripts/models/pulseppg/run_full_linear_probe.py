#!/usr/bin/env python3
"""Full-data frozen Pulse-PPG within-dataset and LODO Ridge benchmark.

The encoder is run once over every Phase-7 window. All heads use the frozen
TimePPG subject roles; target data are never used to fit, scale or tune LODO.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "external/pulseppg"))
sys.path.insert(0, str(ROOT / "src"))
from pulseppg.nets.ResNet1D.ResNet1D_Net import Net  # noqa: E402
from rq1_hr.evaluation import hr_metrics  # noqa: E402


MANIFEST = ROOT / "reports/phase7_frozen_dataset/main_window_manifest.csv"
WITHIN_ROLES = ROOT / "reports/phase9_timeppg/three_fold_subject_roles.csv"
LODO_ROLES = ROOT / "reports/phase7_frozen_dataset/subject_splits.csv"
CHECKPOINT = ROOT / "artifacts/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl"
CACHE = ROOT / "artifacts/pulseppg/phase12_embeddings"
OUTPUT = ROOT / "reports/phase12_pulseppg/02_frozen_linear_probe"
DATASETS = ("BIDMC", "WESAD", "PTT-PPG", "PPG-DaLiA")
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_frame() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST, low_memory=False)
    key = ["dataset", "record_id", "subject_id", "window_index"]
    if frame.duplicated(key).any() or len(frame) != 136625:
        raise RuntimeError("Frozen primary manifest is not the expected 136,625 unique windows")
    if set(frame.dataset) != set(DATASETS):
        raise RuntimeError("Unexpected dataset names")
    if not frame.structurally_usable.astype(bool).all():
        raise RuntimeError("Manifest contains an unusable input")
    return frame


@lru_cache(maxsize=12)
def signal_array(relative_path: str) -> np.ndarray:
    return np.load(ROOT / relative_path, mmap_mode="r", allow_pickle=False)


def batch_windows(frame: pd.DataFrame) -> np.ndarray:
    array = np.empty((len(frame), 512), dtype=np.float64)
    for i, row in enumerate(frame.itertuples(index=False)):
        values = signal_array(row.signal_path)[int(row.start_sample_64hz):int(row.end_sample_64hz), int(row.channel_index)]
        if len(values) != 512:
            raise RuntimeError("Incorrect frozen window length")
        array[i] = values
    if not np.isfinite(array).all():
        raise RuntimeError("Nonfinite raw signal")
    # No fitting: fixed 64 -> 50 Hz conversion, then independent window z-score.
    array = resample_poly(array, 25, 32, axis=1)
    std = np.std(array, axis=1, keepdims=True)
    if np.any(std <= np.finfo(np.float32).eps):
        raise RuntimeError("Constant PPG window")
    return np.asarray((array - np.mean(array, axis=1, keepdims=True)) / std, dtype=np.float32)


def load_encoder(device: torch.device) -> Net:
    model = Net(in_channels=1, base_filters=128, kernel_size=11, stride=2,
                groups=1, n_block=12, finalpool="max")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["net"], strict=True)
    model.to(device).eval()
    return model


def extract(frame: pd.DataFrame, batch_size: int, device: torch.device, resume: bool) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    matrix_path = CACHE / "embeddings_8s_50hz.npy"
    metadata_path = CACHE / "metadata.json"
    if metadata_path.exists():
        raise FileExistsError("Complete embedding cache already exists; use --stage evaluate")
    if matrix_path.exists() and not resume:
        raise FileExistsError("Incomplete embedding cache exists; rerun with --resume after verifying its origin")
    model = load_encoder(device)
    if resume:
        matrix = np.load(matrix_path, mmap_mode="r+", allow_pickle=False)
        if matrix.shape != (len(frame), 512):
            raise RuntimeError("Incomplete cache has wrong shape")
        # Recovery is safe for a cache produced by this script: each completed
        # batch is assigned at once; unwritten preallocated rows remain zero.
        written = np.any(matrix != 0, axis=1)
        first_unwritten = int(np.flatnonzero(~written)[0]) if not written.all() else len(frame)
        if not written[:first_unwritten].all() or written[first_unwritten:].any() or first_unwritten % batch_size:
            raise RuntimeError("Interrupted cache is not a contiguous sequence of complete batches")
        start_row = first_unwritten
        print(f"Resuming verified contiguous cache prefix at {start_row:,} rows", flush=True)
    else:
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(frame), 512))
        start_row = 0
    try:
        for start in range(start_row, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            windows = batch_windows(frame.iloc[start:stop])
            with torch.inference_mode():
                values = model(torch.from_numpy(windows[:, None, :]).to(device)).cpu().numpy()
            if values.shape != (stop - start, 512) or not np.isfinite(values).all():
                raise RuntimeError("Invalid encoder output")
            matrix[start:stop] = values
            if start == start_row or stop == len(frame) or stop % 8192 < batch_size:
                print(f"Embedded {stop:,}/{len(frame):,} windows", flush=True)
        matrix.flush()
        metadata_path.write_text(json.dumps({
            "status": "complete", "rows": len(frame), "dimensions": 512,
            "manifest_sha256": sha256(MANIFEST), "checkpoint_sha256": sha256(CHECKPOINT),
            "preprocessing": "64 Hz 0.5–4 Hz frozen PPG -> polyphase 50 Hz -> per-window z-score",
            "matrix": matrix_path.name,
        }, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        matrix.flush()
        # No completion metadata is written; an interrupted cache is invalid.
        raise


def cached_features(expected_rows: int) -> np.ndarray:
    metadata_path = CACHE / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("No complete embedding cache; run --stage extract")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] != "complete" or metadata["rows"] != expected_rows:
        raise RuntimeError("Embedding cache dimensions do not match manifest")
    if metadata["manifest_sha256"] != sha256(MANIFEST) or metadata["checkpoint_sha256"] != sha256(CHECKPOINT):
        raise RuntimeError("Embedding cache provenance does not match current inputs")
    matrix = np.load(CACHE / metadata["matrix"], mmap_mode="r", allow_pickle=False)
    if matrix.shape != (expected_rows, 512):
        raise RuntimeError("Invalid embedding shape")
    return matrix


def joined_roles(frame: pd.DataFrame, roles_path: Path, columns: list[str]) -> pd.DataFrame:
    roles = pd.read_csv(roles_path)[["dataset", "subject_id", *columns]]
    if roles.duplicated(["dataset", "subject_id"]).any():
        raise RuntimeError("Duplicate subject in frozen roles")
    result = frame[["dataset", "subject_id"]].merge(
        roles, on=["dataset", "subject_id"], how="left", validate="many_to_one", sort=False
    )
    if len(result) != len(frame) or result[columns].isna().any().any():
        raise RuntimeError("Missing subject role")
    return result


def fit_head(features: np.ndarray, labels: np.ndarray, train: np.ndarray, val: np.ndarray,
             domains: np.ndarray) -> tuple[StandardScaler, Ridge, float, dict[str, float]]:
    if not train.any() or not val.any() or np.any(train & val):
        raise RuntimeError("Invalid train/validation roles")
    scaler = StandardScaler().fit(features[train])
    x_train = scaler.transform(features[train])
    x_val = scaler.transform(features[val])
    # Each source dataset has equal influence on alpha selection; all its
    # eligible windows are still used in the fit (no subsampling or weighting).
    selected = None
    scores = {}
    for alpha in ALPHAS:
        head = Ridge(alpha=alpha, solver="cholesky").fit(x_train, labels[train])
        predicted = head.predict(x_val)
        source_maes = [float(np.mean(np.abs(predicted[domains[val] == dataset] - labels[val][domains[val] == dataset])))
                       for dataset in sorted(set(domains[val]))]
        score = float(np.mean(source_maes))
        scores[str(alpha)] = score
        if selected is None or score < selected[0]:
            selected = (score, alpha, head)
    assert selected is not None
    return scaler, selected[2], float(selected[1]), scores


def predictions(frame: pd.DataFrame, indices: np.ndarray, predicted: np.ndarray,
                *, fold: int | None = None) -> pd.DataFrame:
    columns = ["dataset", "record_id", "subject_id", "window_index", "condition_id",
               "condition_name", "hr_bpm", "hr_quality_flag"]
    result = frame.loc[indices, columns].copy()
    result = result.rename(columns={"hr_bpm": "reference_hr_bpm"})
    result["predicted_hr_bpm"] = predicted
    result["error_bpm"] = predicted - result.reference_hr_bpm.to_numpy()
    result["absolute_error_bpm"] = np.abs(result.error_bpm)
    if fold is not None:
        result["test_fold"] = fold
    return result


def subject_summary(frame: pd.DataFrame) -> tuple[float, list[float], pd.DataFrame]:
    subject = frame.groupby("subject_id", sort=True).agg(
        windows=("absolute_error_bpm", "size"), mae_bpm=("absolute_error_bpm", "mean")
    ).reset_index()
    values = subject.mae_bpm.to_numpy(float)
    rng = np.random.default_rng(17)
    boot = np.mean(rng.choice(values, size=(10000, len(values)), replace=True), axis=1)
    return float(np.mean(values)), np.quantile(boot, [.025, .975]).tolist(), subject


def paired_timeppg(pulse: pd.DataFrame, dataset: str, *, within: bool) -> dict:
    slug = dataset.lower().replace("-", "_")
    if within:
        paths = [ROOT / "reports/phase9_timeppg/within_dataset_3fold" / slug /
                 f"test_fold_{fold}/seed_17/test_predictions.csv" for fold in range(3)]
    else:
        paths = [ROOT / "reports/phase9_timeppg/lodo_full" /
                 f"target_{slug}/seed_17/target_test_predictions.csv"]
    key = ["dataset", "record_id", "subject_id", "window_index"]
    prior = pd.concat([pd.read_csv(path, usecols=key + ["reference_hr_bpm", "predicted_hr_bpm"])
                       for path in paths], ignore_index=True)
    if prior.duplicated(key).any():
        raise RuntimeError("Duplicate TimePPG comparison window")
    prior = prior.rename(columns={"reference_hr_bpm": "timeppg_reference_hr_bpm",
                                  "predicted_hr_bpm": "timeppg_predicted_hr_bpm"})
    paired = pulse.merge(prior, on=key, how="left", validate="one_to_one")
    if len(paired) != len(pulse) or paired.timeppg_predicted_hr_bpm.isna().any():
        raise RuntimeError("TimePPG comparison does not cover identical windows")
    if np.max(np.abs(paired.reference_hr_bpm - paired.timeppg_reference_hr_bpm)) > 0.001:
        raise RuntimeError("Reference HR differs from prior TimePPG result")
    return hr_metrics(paired.reference_hr_bpm, paired.timeppg_predicted_hr_bpm)


def evaluate_within(frame: pd.DataFrame, features: np.ndarray, labels: np.ndarray,
                    roles: pd.DataFrame) -> dict[str, dict]:
    summaries = {}
    domains = frame.dataset.to_numpy()
    for dataset in DATASETS:
        base = domains == dataset
        parts, fold_metrics = [], []
        for fold in range(3):
            role = roles[f"within3_test_fold_{fold}_role"].to_numpy()
            train, val, test = base & (role == "train"), base & (role == "validation"), base & (role == "test")
            if np.any(train & test) or np.any(val & test):
                raise RuntimeError("Within-subject split overlap")
            scaler, head, alpha, scores = fit_head(features, labels, train, val, domains)
            indices = np.flatnonzero(test)
            predicted = head.predict(scaler.transform(features[indices]))
            part = predictions(frame, indices, predicted, fold=fold)
            parts.append(part)
            fold_metrics.append({"fold": fold, "alpha": alpha, "train_windows": int(train.sum()),
                                 "validation_windows": int(val.sum()), "test_windows": int(test.sum()),
                                 "validation_macro_mae_by_alpha": scores, **hr_metrics(labels[test], predicted)})
        combined = pd.concat(parts, ignore_index=True)
        if len(combined) != int(base.sum()) or combined.duplicated(["dataset", "record_id", "subject_id", "window_index"]).any():
            raise RuntimeError("Each within-dataset window must be tested exactly once")
        output = OUTPUT / "within_dataset" / dataset.lower().replace("-", "_")
        output.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output / "all_test_predictions.csv", index=False)
        pd.DataFrame(fold_metrics).to_csv(output / "fold_metrics.csv", index=False)
        macro, ci, subject = subject_summary(combined)
        subject.to_csv(output / "subject_metrics.csv", index=False)
        summary = {"dataset": dataset, "protocol": "three subject-disjoint outer folds",
                   "each_window_tested_once": True, "unique_subjects": int(subject.subject_id.nunique()),
                   "pooled": hr_metrics(combined.reference_hr_bpm, combined.predicted_hr_bpm),
                   "timeppg_same_window_pooled": paired_timeppg(combined, dataset, within=True),
                   "subject_macro_mae_bpm": macro, "subject_bootstrap_95_ci_bpm": ci,
                   "folds": fold_metrics}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries[dataset] = summary
        print(f"Within {dataset}: MAE={summary['pooled']['mae_bpm']:.3f}", flush=True)
    return summaries


def evaluate_lodo(frame: pd.DataFrame, features: np.ndarray, labels: np.ndarray,
                  roles: pd.DataFrame) -> dict[str, dict]:
    summaries = {}
    domains = frame.dataset.to_numpy()
    for target in DATASETS:
        role = roles[f"lodo_{target}_role"].to_numpy()
        train, val, test = role == "source_train", role == "source_validation", role == "target_test"
        if np.any((train | val) & (domains == target)) or not np.array_equal(test, domains == target):
            raise RuntimeError("Held-out target was used before final evaluation")
        scaler, head, alpha, scores = fit_head(features, labels, train, val, domains)
        indices = np.flatnonzero(test)
        predicted = head.predict(scaler.transform(features[indices]))
        result = predictions(frame, indices, predicted)
        output = OUTPUT / "lodo" / f"target_{target.lower().replace('-', '_')}"
        output.mkdir(parents=True, exist_ok=True)
        result.to_csv(output / "target_test_predictions.csv", index=False)
        macro, ci, subject = subject_summary(result)
        subject.to_csv(output / "subject_metrics.csv", index=False)
        summary = {"target": target, "protocol": "full-data train-three/test-one LODO",
                   "source_datasets": sorted(set(domains[train])), "target_used_for_fit_or_selection": False,
                   "alpha": alpha, "validation_macro_mae_by_alpha": scores,
                   "train_windows": int(train.sum()), "validation_windows": int(val.sum()),
                   "target_test_windows": int(test.sum()),
                   "train_windows_by_dataset": pd.Series(domains[train]).value_counts().sort_index().to_dict(),
                   "validation_windows_by_dataset": pd.Series(domains[val]).value_counts().sort_index().to_dict(),
                   "pooled": hr_metrics(result.reference_hr_bpm, result.predicted_hr_bpm),
                   "timeppg_same_window_pooled": paired_timeppg(result, target, within=False),
                   "subject_macro_mae_bpm": macro, "subject_bootstrap_95_ci_bpm": ci}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries[target] = summary
        print(f"LODO {target}: MAE={summary['pooled']['mae_bpm']:.3f}", flush=True)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "evaluate", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted extraction made by this script")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    device = torch.device(device_name)
    print(f"Encoder extraction device: {device_name}", flush=True)
    frame = manifest_frame()
    if args.stage in ("extract", "all"):
        extract(frame, args.batch_size, device, args.resume)
    if args.stage in ("evaluate", "all"):
        features = cached_features(len(frame))
        labels = frame.hr_bpm.to_numpy(float)
        within_columns = [f"within3_test_fold_{fold}_role" for fold in range(3)]
        lodo_columns = [f"lodo_{dataset}_role" for dataset in DATASETS]
        within_roles = joined_roles(frame, WITHIN_ROLES, within_columns)
        lodo_roles = joined_roles(frame, LODO_ROLES, lodo_columns)
        within = evaluate_within(frame, features, labels, within_roles)
        lodo = evaluate_lodo(frame, features, labels, lodo_roles)
        summary = []
        for target in DATASETS:
            familiar = within[target]["pooled"]
            unseen = lodo[target]["pooled"]
            prior_familiar = within[target]["timeppg_same_window_pooled"]
            prior_unseen = lodo[target]["timeppg_same_window_pooled"]
            row = {"target": target, "generalisation_gap_bpm": unseen["mae_bpm"] - familiar["mae_bpm"],
                   "timeppg_generalisation_gap_bpm": prior_unseen["mae_bpm"] - prior_familiar["mae_bpm"],
                   "within_subject_macro_mae_bpm": within[target]["subject_macro_mae_bpm"],
                   "lodo_subject_macro_mae_bpm": lodo[target]["subject_macro_mae_bpm"]}
            for prefix, metrics in (("within", familiar), ("lodo", unseen),
                                    ("timeppg_within", prior_familiar), ("timeppg_lodo", prior_unseen)):
                row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
            summary.append(row)
        pd.DataFrame(summary).to_csv(OUTPUT / "comparison_summary.csv", index=False)
        (OUTPUT / "run_provenance.json").write_text(json.dumps({
            "status": "complete", "manifest_sha256": sha256(MANIFEST),
            "within_roles_sha256": sha256(WITHIN_ROLES), "lodo_roles_sha256": sha256(LODO_ROLES),
            "checkpoint_sha256": sha256(CHECKPOINT), "encoder_frozen": True,
            "all_eligible_windows_used": True, "source_label_selection_only": True,
            "preprocessing": "64 Hz filtered PPG -> polyphase 50 Hz -> per-window z-score",
            "head": "Ridge; StandardScaler fit on train; alpha chosen by source-validation dataset-macro MAE",
            "alpha_grid": ALPHAS,
        }, indent=2) + "\n", encoding="utf-8")
        print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
