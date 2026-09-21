#!/usr/bin/env python3
"""Prepare continuous PPG and link quality metrics to frozen Phase 4 windows."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from rq1_hr.data.loaders import load_bidmc, load_ppg_dalia, load_ptt_ppg, load_wesad
from rq1_hr.preprocessing.ppg import measure_ppg_window, prepare_ppg


ROOTS = {
    "BIDMC": Path("datasets/raw/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0"),
    "PPG-DaLiA": Path("datasets/raw/ppg_dalia/data/PPG_FieldStudy"),
    "PTT-PPG": Path("datasets/raw/ptt_ppg"),
    "WESAD": Path("datasets/raw/wesad/WESAD"),
}
LOADERS = {
    "BIDMC": load_bidmc,
    "PPG-DaLiA": load_ppg_dalia,
    "PTT-PPG": load_ptt_ppg,
    "WESAD": load_wesad,
}


def source_condition(record, start_s: float, fallback: object) -> tuple[object, object]:
    """Use the modal source label in the same 8-second interval when present."""
    if record.conditions is None:
        return pd.NA, fallback
    left = int(round(start_s * record.conditions.fs_hz))
    right = int(round((start_s + 8.0) * record.conditions.fs_hz))
    values = np.asarray(record.conditions.values[left:right])
    values = values[np.isfinite(values)].astype(np.int64, copy=False)
    if values.size == 0:
        return pd.NA, fallback
    labels, counts = np.unique(values, return_counts=True)
    label = int(labels[np.argmax(counts)])
    name = record.conditions.value_map.get(label, str(label))
    return label, name


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        shown = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(shown) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("reports/phase4k_canonical_hr/canonical_window_hr.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase5_ppg"))
    args = parser.parse_args()
    labels = pd.read_csv(args.labels, low_memory=False)
    signal_root = args.output_dir / "signals"
    signal_root.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    recording_rows: list[dict[str, object]] = []
    width = 8 * 64

    for (dataset, record_id), windows in labels.groupby(["dataset", "record_id"], sort=False):
        print(f"Processing {dataset}:{record_id}", flush=True)
        record = LOADERS[dataset](ROOTS[dataset], str(record_id))
        processed = prepare_ppg(record.ppg.values, record.ppg.fs_hz)
        dataset_dir = signal_root / dataset.lower().replace("-", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        signal_path = dataset_dir / f"{record_id}.npy"
        np.save(signal_path, processed, allow_pickle=False)

        recording_rows.append({
            "dataset": dataset, "record_id": record_id, "subject_id": record.subject_id,
            "native_fs_hz": record.ppg.fs_hz, "target_fs_hz": 64.0,
            "native_samples": record.ppg.n_samples, "processed_samples": processed.shape[0],
            "channels": processed.shape[1], "channel_names": "|".join(record.ppg.channel_names),
            "signal_path": str(signal_path),
        })
        for row in windows.itertuples(index=False):
            left = int(round(row.window_start_s * 64))
            right = left + width
            if right > processed.shape[0]:
                raise RuntimeError(f"window exceeds processed signal: {dataset}:{record_id}:{row.window_index}")
            condition_id, condition_name = source_condition(record, row.window_start_s, row.condition)
            for channel_index, channel_name in enumerate(record.ppg.channel_names):
                quality = measure_ppg_window(processed[left:right, channel_index], 64)
                metric_rows.append({
                    "dataset": dataset, "record_id": record_id, "subject_id": row.subject_id,
                    "window_index": row.window_index, "window_start_s": row.window_start_s,
                    "start_sample_64hz": left, "end_sample_64hz": right,
                    "condition_id": condition_id, "condition_name": condition_name,
                    "channel_index": channel_index, "channel_name": channel_name,
                    "hr_bpm": row.hr_bpm, "hr_quality_flag": row.quality_flag,
                    **{name: getattr(quality, name) for name in quality.__dataclass_fields__},
                })
        del record, processed
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    recordings = pd.DataFrame(recording_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "window_channel_quality.csv", index=False)
    recordings.to_csv(args.output_dir / "processed_recordings.csv", index=False)
    summary = metrics.groupby("dataset", sort=False).agg(
        records=("record_id", "nunique"), subjects=("subject_id", "nunique"),
        window_channels=("window_index", "size"), structurally_usable=("structurally_usable", "sum"),
        median_std=("standard_deviation", "median"), median_flat_fraction=("flat_difference_fraction", "median"),
        median_spectral_concentration=("spectral_concentration", "median"),
    ).reset_index()
    summary.to_csv(args.output_dir / "dataset_quality_summary.csv", index=False)
    distributions = metrics.groupby("dataset", sort=False).agg(
        std_q05=("standard_deviation", lambda x: x.quantile(0.05)),
        std_median=("standard_deviation", "median"),
        std_q95=("standard_deviation", lambda x: x.quantile(0.95)),
        concentration_q05=("spectral_concentration", lambda x: x.quantile(0.05)),
        concentration_median=("spectral_concentration", "median"),
        concentration_q95=("spectral_concentration", lambda x: x.quantile(0.95)),
    ).reset_index()
    distributions.to_csv(args.output_dir / "quality_distributions.csv", index=False)
    readme = "\n".join([
        "# Phase 5: common PPG preparation", "",
        "## Frozen processing", "",
        "1. Load the complete native PPG recording.",
        "2. Resample to 64 Hz using polyphase anti-alias filtering.",
        "3. Apply a fourth-order 0.5–4 Hz Butterworth band-pass in zero phase.",
        "4. Reference each 8-second window by exact 64 Hz sample indices (512 samples).",
        "5. Preserve every source PPG channel; PTT-PPG therefore remains six-channel.",
        "6. Record descriptive quality evidence without arbitrary motion rejection.", "",
        "The 0.5–4 Hz band corresponds approximately to 30–240 beats/min and surrounds the frozen 35–220 bpm HR range. Filtering the complete recording, rather than each window separately, reduces artificial boundary effects. No amplitude normalization is applied here because amplitude/device differences are part of the distribution-shift question; model-time normalization can be evaluated later as an explicit experiment.", "",
        "## Structural results", "", markdown(summary), "",
        "## Descriptive distributions", "", markdown(distributions), "",
        "All window-channels are structurally usable. This does **not** mean that all are clean: spectral concentration, variability, flat differences, and edge-value fractions are retained for distribution analysis and later justified sensitivity rules. The amplitude statistics must not be compared as physical units across devices without normalization.", "",
        "`signals/` stores each processed continuous recording once. `window_channel_quality.csv` provides its channel and `[start_sample_64hz, end_sample_64hz)` slice, HR label, condition, and quality measures.", "",
    ])
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
