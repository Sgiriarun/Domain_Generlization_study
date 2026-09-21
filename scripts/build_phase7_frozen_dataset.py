#!/usr/bin/env python3
"""Freeze RQ1 model manifests and subject-independent evaluation splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rq1_hr.data.windows import (
    PRIMARY_CHANNELS,
    PTT_SITE_METADATA,
    deterministic_subject_folds,
    zscore_window,
)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def add_signal_paths(windows: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    paths = recordings[["dataset", "record_id", "signal_path", "target_fs_hz"]]
    result = windows.merge(paths, on=["dataset", "record_id"], how="left", validate="many_to_one")
    if result["signal_path"].isna().any():
        raise RuntimeError("some windows have no processed signal path")
    return result


def build_main_manifest(windows: pd.DataFrame) -> pd.DataFrame:
    expected = windows["dataset"].map(PRIMARY_CHANNELS)
    if expected.isna().any():
        unknown = sorted(windows.loc[expected.isna(), "dataset"].unique())
        raise RuntimeError(f"no primary-channel policy for {unknown}")
    main = windows.loc[windows["channel_name"].eq(expected)].copy()
    main["input_normalization"] = "per_window_zscore"
    main["primary_analysis"] = True
    main["sensitivity_exclude_detector_warning"] = main["hr_quality_flag"].ne("pass")
    return main


def build_ptt_site_manifest(windows: pd.DataFrame) -> pd.DataFrame:
    site = windows.loc[windows["dataset"].eq("PTT-PPG")].copy()
    metadata = site["channel_name"].map(PTT_SITE_METADATA)
    if metadata.isna().any():
        unknown = sorted(site.loc[metadata.isna(), "channel_name"].unique())
        raise RuntimeError(f"unknown PTT channels: {unknown}")
    site["sensor_site"] = metadata.map(lambda value: value[0])
    site["site_pair"] = metadata.map(lambda value: value[1])
    site["input_normalization"] = "per_window_zscore"
    return site


def build_subject_splits(main: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in main.groupby("dataset", sort=True):
        fold_map = deterministic_subject_folds(group["subject_id"], n_folds=n_folds, seed=seed)
        for subject_id, fold in sorted(fold_map.items()):
            rows.append({"dataset": dataset, "subject_id": subject_id, "within_dataset_fold": fold})
    subjects = pd.DataFrame(rows)
    # Fold zero is a fixed source-validation group. It is never used to tune on
    # the held-out target; every target subject remains test-only.
    for target in sorted(main["dataset"].unique()):
        subjects[f"lodo_{target}_role"] = np.where(
            subjects["dataset"].eq(target),
            "target_test",
            np.where(subjects["within_dataset_fold"].eq(0), "source_validation", "source_train"),
        )
    return subjects


def validate_slices(manifest: pd.DataFrame) -> None:
    """Check every referenced slice and exercise the final normalization rule."""
    for signal_path, rows in manifest.groupby("signal_path", sort=False):
        signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        for row in rows.itertuples(index=False):
            if row.end_sample_64hz - row.start_sample_64hz != 512:
                raise RuntimeError(f"non-512-sample window at {row.dataset}:{row.record_id}:{row.window_index}")
            if row.end_sample_64hz > signal.shape[0] or row.channel_index >= signal.shape[1]:
                raise RuntimeError(f"invalid signal slice at {row.dataset}:{row.record_id}:{row.window_index}")
            zscore_window(signal[row.start_sample_64hz:row.end_sample_64hz, row.channel_index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-dir", type=Path, default=Path("reports/phase5_ppg"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase7_frozen_dataset"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    windows = pd.read_csv(args.phase5_dir / "window_channel_quality.csv", low_memory=False)
    recordings = pd.read_csv(args.phase5_dir / "processed_recordings.csv")
    accepted = windows.loc[windows["structurally_usable"].astype(bool)].copy()
    accepted = add_signal_paths(accepted, recordings)
    main = build_main_manifest(accepted)
    site = build_ptt_site_manifest(accepted)
    splits = build_subject_splits(main, args.folds, args.seed)

    validate_slices(main)
    expected_site_rows = main.loc[main["dataset"].eq("PTT-PPG")].shape[0] * 6
    if site.shape[0] != expected_site_rows:
        raise RuntimeError("PTT site manifest is not six aligned channels per primary window")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main.to_csv(args.output_dir / "main_window_manifest.csv", index=False)
    site.to_csv(args.output_dir / "ptt_site_window_manifest.csv", index=False)
    splits.to_csv(args.output_dir / "subject_splits.csv", index=False)

    summary = main.groupby("dataset", sort=False).agg(
        subjects=("subject_id", "nunique"), records=("record_id", "nunique"),
        accepted_windows=("window_index", "size"),
        detector_warning_windows=("sensitivity_exclude_detector_warning", "sum"),
    ).reset_index()
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    split_summary = []
    for target in sorted(main["dataset"].unique()):
        column = f"lodo_{target}_role"
        counts = splits.groupby(["dataset", column]).size()
        for (dataset, role), count in counts.items():
            split_summary.append({"held_out_dataset": target, "dataset": dataset, "role": role, "subjects": count})
    pd.DataFrame(split_summary).to_csv(args.output_dir / "lodo_split_summary.csv", index=False)

    readme = f"""# Phase 7: frozen modelling dataset and splits

## Decision aligned with the RQ1 documents

- Primary task: single-channel PPG to ECG-derived HR.
- Every input is an 8-second, 512-sample window at 64 Hz with a 2-second step.
- PTT-PPG uses `pleth_1` only in the primary four-dataset benchmark.
- `ptt_site_window_manifest.csv` preserves all six aligned PTT channels for the secondary site-shift analysis.
- Model input normalization is per-window z-scoring and must happen after slicing.
- The primary analysis retains detector-disagreement review windows. The manifest marks them so a documented sensitivity analysis can exclude them.
- No motion-quality threshold was invented after viewing target data; only structural failures are rejected.

## Primary dataset

{markdown(summary)}

## Leakage control

- `subject_splits.csv` assigns complete subjects, never windows, to five deterministic folds using seed {args.seed}.
- Within-dataset evaluation rotates the test fold; training-time validation must be selected only from the remaining subjects.
- For each leave-one-dataset-out experiment, every held-out-dataset subject is `target_test`.
- Source fold 0 is `source_validation`; all other source folds are `source_train`.
- Target subjects are not used for training, normalization fitting, early stopping, or hyperparameter selection.

## Files

- `main_window_manifest.csv`: frozen single-channel input references for RQ1.
- `ptt_site_window_manifest.csv`: six-channel paired-site secondary analysis.
- `subject_splits.csv`: reusable subject-level split assignments for RQ1–RQ3.
- `dataset_summary.csv` and `lodo_split_summary.csv`: audit tables.

The manifests reference immutable Phase 5 `.npy` recordings using `[start_sample_64hz, end_sample_64hz)` and `channel_index`; they do not duplicate overlapping signal windows.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

