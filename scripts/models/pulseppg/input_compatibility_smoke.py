#!/usr/bin/env python3
"""Numerical Pulse-PPG input audit; no HR training or target-based selection."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "external/pulseppg"))
from pulseppg.nets.ResNet1D.ResNet1D_Net import Net  # noqa: E402


MANIFEST = ROOT / "reports/phase7_frozen_dataset/main_window_manifest.csv"
CHECKPOINT = ROOT / "artifacts/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl"
OUTPUT = ROOT / "reports/phase12_pulseppg/01_input_compatibility/smoke_metrics.json"
DATASETS = ("BIDMC", "WESAD", "PTT-PPG", "PPG-DaLiA")


def selected_rows(max_per_dataset: int = 24) -> dict[str, list[dict[str, str]]]:
    """Deterministic, label-blind sample spanning subjects and recordings."""
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {name: {} for name in DATASETS}
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] in grouped:
                grouped[row["dataset"]].setdefault(row["subject_id"], []).append(row)
    selected = {}
    for name, subjects in grouped.items():
        subject_ids = sorted(subjects)
        rows = []
        cursor = 0
        while len(rows) < max_per_dataset:
            added = False
            for subject_id in subject_ids:
                candidate_rows = subjects[subject_id]
                # 8-s spaced starts minimise near-duplicate overlapping windows.
                position = cursor * 4
                if position < len(candidate_rows):
                    rows.append(candidate_rows[position])
                    added = True
                    if len(rows) == max_per_dataset:
                        break
            if not added:
                break
            cursor += 1
        selected[name] = rows
    return selected


def input_array(row: dict[str, str], seconds: int) -> np.ndarray | None:
    signal_path = ROOT / row["signal_path"]
    signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
    start = int(row["start_sample_64hz"])
    stop = start + seconds * 64
    if stop > signal.shape[0]:
        return None
    values = np.asarray(signal[start:stop, int(row["channel_index"])], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        return None
    # Frozen Phase-5 signal is already filtered at 0.5–4 Hz and sampled at 64 Hz.
    # Convert the clock to the encoder's reported 50 Hz; do not use target stats.
    values = resample_poly(values, 25, 32)
    if values.size != seconds * 50:
        raise AssertionError(f"Unexpected {seconds}-s resample length: {values.size}")
    std = float(np.std(values))
    if std <= np.finfo(np.float32).eps:
        return None
    return np.asarray((values - np.mean(values)) / std, dtype=np.float32)


def embed(model: Net, values: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        tensor = torch.from_numpy(values[None, None, :])
        return model(tensor).cpu().numpy()[0]


def describe(embeddings: list[np.ndarray]) -> dict[str, float | int]:
    matrix = np.stack(embeddings)
    norms = np.linalg.norm(matrix, axis=1)
    unit = matrix / np.maximum(norms[:, None], 1e-12)
    similarities = (unit @ unit.T)[np.triu_indices(len(unit), k=1)]
    return {
        "n": len(matrix),
        "dimensions": int(matrix.shape[1]),
        "finite_fraction": float(np.mean(np.isfinite(matrix))),
        "embedding_norm_median": float(np.median(norms)),
        "feature_std_median": float(np.median(np.std(matrix, axis=0))),
        "pairwise_cosine_median": float(np.median(similarities)),
        "pairwise_cosine_p95": float(np.percentile(similarities, 95)),
        "near_duplicate_pairs_cosine_gt_0_9999": int(np.sum(similarities > 0.9999)),
    }


def main() -> None:
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = Net(in_channels=1, base_filters=128, kernel_size=11, stride=2,
                groups=1, n_block=12, finalpool="max")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["net"], strict=True)
    model.eval()

    result = {
        "purpose": "Label-blind numerical compatibility only; not HR performance or model selection",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "preprocessing": "Frozen 64-Hz 0.5–4-Hz PPG -> resample_poly(25,32) to 50 Hz -> per-input z-score",
        "sample_rule": "Up to 24 manifest windows per dataset, subject-round-robin, 8-s spaced within each subject",
        "datasets": {},
    }
    for name, rows in selected_rows().items():
        short, long = [], []
        repeat_max_abs_diff = 0.0
        for row in rows:
            values_8 = input_array(row, 8)
            if values_8 is None:
                continue
            embedding_8 = embed(model, values_8)
            repeat_max_abs_diff = max(repeat_max_abs_diff, float(np.max(np.abs(embedding_8 - embed(model, values_8)))))
            short.append(embedding_8)
            values_32 = input_array(row, 32)
            if values_32 is not None:
                long.append(embed(model, values_32))
        if len(short) < 2 or len(long) < 2:
            raise RuntimeError(f"Insufficient usable inputs for {name}")
        result["datasets"][name] = {
            "eight_seconds": describe(short),
            "thirty_two_seconds": describe(long),
            "deterministic_repeat_max_abs_diff": repeat_max_abs_diff,
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
