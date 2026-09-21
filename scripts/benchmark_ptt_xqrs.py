#!/usr/bin/env python3
"""External validation of candidate detectors on annotated PTT-PPG ECG."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from rq1_hr.data.loaders import list_ptt_ppg_records, load_ptt_ppg
from rq1_hr.preprocessing import (
    detect_rpeaks_emrich2023,
    detect_rpeaks_sleepecg,
    detect_rpeaks_xqrs,
    match_rpeaks,
    window_hr_from_rpeaks,
)


def hr_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    keep = np.isfinite(reference) & np.isfinite(estimate)
    error = estimate[keep] - reference[keep]
    return {
        "reference_windows": int(np.count_nonzero(np.isfinite(reference))),
        "matched_windows": int(keep.sum()),
        "coverage": float(keep.sum() / np.count_nonzero(np.isfinite(reference))),
        "hr_mae_bpm": float(np.mean(np.abs(error))),
        "hr_rmse_bpm": float(np.sqrt(np.mean(error * error))),
        "hr_bias_bpm": float(np.mean(error)),
        "hr_correlation": float(np.corrcoef(reference[keep], estimate[keep])[0, 1]),
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


def aggregate(group: pd.DataFrame, windows: pd.DataFrame, label: str) -> dict[str, object]:
    detector = str(group.detector.iloc[0])
    selected = windows[(windows.detector == detector) & windows.record_id.isin(group.record_id)]
    reference = selected.reference_hr_bpm.to_numpy()
    estimate = selected.detected_hr_bpm.to_numpy()
    matched = int(group.matched_peaks.sum())
    detected = int(group.detected_peaks.sum())
    supplied = int(group.reference_peaks.sum())
    precision = matched / detected
    recall = matched / supplied
    return {
        "detector": detector,
        "group": label,
        "records": len(group),
        "reference_peaks": supplied,
        "detected_peaks": detected,
        "peak_precision_micro": precision,
        "peak_recall_micro": recall,
        "peak_f1_micro": 2 * precision * recall / (precision + recall),
        "peak_f1_macro": float(group.peak_f1.mean()),
        **hr_metrics(reference, estimate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/ptt_ppg"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4f_ptt_detector_comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    record_rows: list[dict[str, object]] = []
    window_rows: list[pd.DataFrame] = []
    detectors = (
        ("xqrs", detect_rpeaks_xqrs),
        ("emrich2023", detect_rpeaks_emrich2023),
        ("sleepecg", detect_rpeaks_sleepecg),
    )
    for record_id in list_ptt_ppg_records(args.root):
        print(f"Validating {record_id}", flush=True)
        record = load_ptt_ppg(args.root, record_id)
        reference_hr = window_hr_from_rpeaks(
            record.rpeaks.sample_indices, record.rpeaks.fs_hz, record.ppg.duration_s
        )
        for detector_name, detector in detectors:
            detected = detector(record.ecg.values[:, 0], record.ecg.fs_hz)
            peak = match_rpeaks(record.rpeaks.sample_indices, detected, record.ecg.fs_hz)
            detected_hr = window_hr_from_rpeaks(detected, record.ecg.fs_hz, record.ppg.duration_s)
            metrics = hr_metrics(reference_hr.values_bpm, detected_hr.values_bpm)
            record_rows.append({
                "record_id": record.record_id, "subject_id": record.subject_id,
                "activity": record.recording_condition, "detector": detector_name,
                "reference_peaks": peak.reference_count, "detected_peaks": peak.detected_count,
                "matched_peaks": peak.matched_count, "peak_precision": peak.precision,
                "peak_recall": peak.recall, "peak_f1": peak.f1,
                "peak_median_timing_error_ms": peak.median_absolute_timing_error_ms,
                **metrics,
            })
            window_rows.append(pd.DataFrame({
                "record_id": record.record_id, "subject_id": record.subject_id,
                "activity": record.recording_condition, "detector": detector_name,
                "window_index": np.arange(reference_hr.starts_s.size),
                "window_start_s": reference_hr.starts_s,
                "reference_hr_bpm": reference_hr.values_bpm,
                "detected_hr_bpm": detected_hr.values_bpm,
                "reference_valid_rr": reference_hr.valid_rr_count,
                "detected_valid_rr": detected_hr.valid_rr_count,
                "detected_invalid_rr": detected_hr.invalid_rr_count,
            }))
        del record, reference_hr, detected, detected_hr
        gc.collect()

    records = pd.DataFrame(record_rows)
    windows = pd.concat(window_rows, ignore_index=True)
    summaries = []
    for detector_name, detector_records in records.groupby("detector", sort=False):
        summaries.append(aggregate(detector_records, windows, "all"))
        for activity, group in detector_records.groupby("activity", sort=True):
            summaries.append(aggregate(group, windows, str(activity)))
    summary = pd.DataFrame(summaries)
    records.to_csv(args.output_dir / "record_metrics.csv", index=False)
    windows.to_csv(args.output_dir / "window_comparison.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    worst = records.sort_values(["detector", "hr_mae_bpm"], ascending=[True, False]).groupby("detector").head(5)[[
        "detector", "record_id", "activity", "peak_precision", "peak_recall", "peak_f1",
        "hr_mae_bpm", "hr_rmse_bpm", "windows_over_10_bpm", "coverage",
    ]]
    report = "\n".join([
        "# External validation of ECG detector candidates on PTT-PPG", "",
        "## Protocol", "",
        "WFDB XQRS, NeuroKit2 Emrich 2023, and SleepECG are applied unchanged to all 66 PTT-PPG ECG records after comparison on PPG-DaLiA. No PTT annotations are used for tuning. Detected peaks are compared with manually verified peaks using one-to-one ±100 ms matching.", "",
        "Both annotation routes use the same 8-second windows, 2-second step, RR-midpoint assignment, 35–220 bpm interval range, minimum four valid RR intervals, and arithmetic mean HR.", "",
        "## Overall and activity results", "", markdown(summary), "",
        "## Records with highest HR MAE", "", markdown(worst), "",
        "## Interpretation", "",
        "This is the external detector test because PTT-PPG differs from the development dataset in ECG hardware, sampling frequency, attachment, and activity protocol. Selection must be based on consistency across both datasets using peak accuracy, window HR error, large-error counts, coverage, runtime, and reproducibility.", "",
    ])
    (args.output_dir / "validation_report.md").write_text(report, encoding="utf-8")
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
