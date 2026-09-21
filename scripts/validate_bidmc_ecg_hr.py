#!/usr/bin/env python3
"""Validate BIDMC ECG-derived HR against bedside monitor HR."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import numpy as np
import pandas as pd

from rq1_hr.data.loaders import list_bidmc_records, load_bidmc
from rq1_hr.preprocessing import (
    detect_rpeaks_emrich2023,
    detect_rpeaks_sleepecg,
    detect_rpeaks_xqrs,
    window_hr_from_rpeaks,
)


def monitor_window_mean(values: np.ndarray, fs_hz: float, starts: np.ndarray, window_s: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    output = np.full(starts.size, np.nan)
    counts = np.zeros(starts.size, dtype=np.int64)
    times = np.arange(values.size) / fs_hz
    for index, start in enumerate(starts):
        inside = (times >= start) & (times < start + window_s) & np.isfinite(values)
        counts[index] = np.count_nonzero(inside)
        if counts[index]:
            output[index] = float(np.mean(values[inside]))
    return output, counts


def metrics(reference: pd.Series, estimate: pd.Series) -> dict[str, float | int]:
    keep = np.isfinite(reference) & np.isfinite(estimate)
    error = estimate[keep].to_numpy() - reference[keep].to_numpy()
    return {
        "matched_windows": int(keep.sum()),
        "mae_bpm": float(np.mean(np.abs(error))),
        "rmse_bpm": float(np.sqrt(np.mean(error * error))),
        "bias_bpm": float(np.mean(error)),
        "correlation": float(np.corrcoef(reference[keep], estimate[keep])[0, 1]),
        "windows_over_5_bpm": int(np.count_nonzero(np.abs(error) > 5)),
        "windows_over_10_bpm": int(np.count_nonzero(np.abs(error) > 10)),
        "windows_over_20_bpm": int(np.count_nonzero(np.abs(error) > 20)),
    }


def markdown(frame: pd.DataFrame) -> str:
    def show(value: object) -> str:
        if pd.isna(value):
            return ""
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(show(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4i_bidmc_hr"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detectors = (
        ("emrich2023", detect_rpeaks_emrich2023),
        ("xqrs", detect_rpeaks_xqrs),
        ("sleepecg", detect_rpeaks_sleepecg),
    )
    frames: list[pd.DataFrame] = []
    peak_rows: list[dict[str, object]] = []
    for record_id in list_bidmc_records(args.root):
        print(f"Processing {record_id}", flush=True)
        record = load_bidmc(args.root, record_id)
        frame: pd.DataFrame | None = None
        for detector_name, detector in detectors:
            peaks = detector(record.ecg.values[:, 0], record.ecg.fs_hz)
            hr = window_hr_from_rpeaks(peaks, record.ecg.fs_hz, record.ppg.duration_s)
            if frame is None:
                monitor_hr, monitor_count = monitor_window_mean(record.provided_hr.values, record.provided_hr.fs_hz, hr.starts_s)
                frame = pd.DataFrame({
                    "dataset": "BIDMC", "record_id": record.record_id,
                    "subject_id": record.subject_id, "window_index": np.arange(hr.starts_s.size),
                    "window_start_s": hr.starts_s, "window_end_s": hr.starts_s + 8,
                    "monitor_hr_bpm": monitor_hr, "monitor_finite_samples": monitor_count,
                })
            frame[f"{detector_name}_hr_bpm"] = hr.values_bpm
            frame[f"{detector_name}_valid_rr"] = hr.valid_rr_count
            frame[f"{detector_name}_invalid_rr"] = hr.invalid_rr_count
            peak_rows.append({
                "record_id": record.record_id, "subject_id": record.subject_id,
                "detector": detector_name, "detected_peaks": int(peaks.size),
                "windows": int(hr.starts_s.size),
                "windows_with_hr": int(np.count_nonzero(np.isfinite(hr.values_bpm))),
                "invalid_rr_in_windows": int(hr.invalid_rr_count.sum()),
            })
        assert frame is not None
        frame["emrich_xqrs_absolute_difference_bpm"] = (frame.emrich2023_hr_bpm - frame.xqrs_hr_bpm).abs()
        frame["emrich_sleepecg_absolute_difference_bpm"] = (frame.emrich2023_hr_bpm - frame.sleepecg_hr_bpm).abs()
        frames.append(frame)
        del record, frame, peaks, hr
        gc.collect()

    windows = pd.concat(frames, ignore_index=True)
    detector_rows = []
    for detector_name, _ in detectors:
        result = metrics(windows.monitor_hr_bpm, windows[f"{detector_name}_hr_bpm"])
        detector_rows.append({"detector": detector_name, **result})
    comparison = pd.DataFrame(detector_rows)
    record_rows = []
    for record_id, group in windows.groupby("record_id", sort=False):
        for detector_name, _ in detectors:
            result = metrics(group.monitor_hr_bpm, group[f"{detector_name}_hr_bpm"])
            record_rows.append({"record_id": record_id, "subject_id": group.subject_id.iloc[0], "detector": detector_name, **result})
    records = pd.DataFrame(record_rows)
    peaks = pd.DataFrame(peak_rows)

    windows.to_csv(args.output_dir / "window_comparison.csv", index=False)
    comparison.to_csv(args.output_dir / "detector_comparison.csv", index=False)
    records.to_csv(args.output_dir / "record_comparison.csv", index=False)
    peaks.to_csv(args.output_dir / "detector_counts.csv", index=False)
    worst = records[records.detector == "emrich2023"].nlargest(10, "mae_bpm")
    report = "\n".join([
        "# BIDMC ECG-derived HR validation", "", "## Method", "",
        "Frozen Emrich 2023, XQRS, and SleepECG detectors are applied to lead II ECG at 125 Hz. Detector HR uses the common 8-second/2-second-step RR pipeline. The comparison reference is the arithmetic mean of finite 1 Hz monitor-HR samples occurring in the same 8-second interval.", "",
        "The bedside monitor may use undocumented smoothing, update timing, and artifact handling. Therefore monitor disagreement is not identical to R-peak annotation error.", "",
        "## Overall detector-to-monitor comparison", "", markdown(comparison), "",
        "## BIDMC records with highest Emrich MAE", "", markdown(worst), "",
        "## Interpretation boundary", "",
        "This test assesses agreement with the available clinical monitor HR and consistency between detector methods. It does not establish manually annotated peak accuracy. Missing monitor values are excluded only from the corresponding comparisons and remain counted in the window file.", "",
    ])
    (args.output_dir / "validation_report.md").write_text(report, encoding="utf-8")
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
