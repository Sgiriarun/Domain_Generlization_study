#!/usr/bin/env python3
"""Compare four reproducible ECG R-peak detectors on PPG-DaLiA."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from rq1_hr.data.loaders import list_ppg_dalia_records, load_ppg_dalia
from rq1_hr.preprocessing import (
    detect_rpeaks,
    detect_rpeaks_emrich2023,
    detect_rpeaks_sleepecg,
    detect_rpeaks_xqrs,
    match_rpeaks,
    window_hr_from_rpeaks,
)


def metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    keep = np.isfinite(reference) & np.isfinite(estimate)
    error = estimate[keep] - reference[keep]
    return {
        "matched_windows": int(keep.sum()),
        "coverage": float(keep.mean()),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/ppg_dalia/data/PPG_FieldStudy"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4e_dalia_detector_comparison"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_rows: list[dict[str, object]] = []
    window_rows: list[pd.DataFrame] = []
    detectors = (
        ("development", detect_rpeaks),
        ("xqrs", detect_rpeaks_xqrs),
        ("emrich2023", detect_rpeaks_emrich2023),
        ("sleepecg", detect_rpeaks_sleepecg),
    )
    for subject in list_ppg_dalia_records(args.root):
        print(f"Benchmarking {subject}", flush=True)
        record = load_ppg_dalia(args.root, subject)
        published = np.asarray(record.provided_hr.values, dtype=np.float64)
        subject_windows = {"subject_id": subject, "window_index": np.arange(published.size), "published_hr_bpm": published}
        for detector_name, detector in detectors:
            started = perf_counter()
            peaks = detector(record.ecg.values[:, 0], record.ecg.fs_hz)
            runtime_s = perf_counter() - started
            peak = match_rpeaks(record.rpeaks.sample_indices, peaks, record.ecg.fs_hz)
            hr = window_hr_from_rpeaks(peaks, record.ecg.fs_hz, record.ppg.duration_s)
            result = metrics(published, hr.values_bpm)
            subject_rows.append({
                "subject_id": subject,
                "detector": detector_name,
                "reference_peaks": peak.reference_count,
                "detected_peaks": peak.detected_count,
                "peak_precision": peak.precision,
                "peak_recall": peak.recall,
                "peak_f1": peak.f1,
                "peak_median_timing_error_ms": peak.median_absolute_timing_error_ms,
                "runtime_s": runtime_s,
                **result,
            })
            subject_windows[f"{detector_name}_hr_bpm"] = hr.values_bpm
            subject_windows[f"{detector_name}_valid_rr"] = hr.valid_rr_count
            subject_windows[f"{detector_name}_invalid_rr"] = hr.invalid_rr_count
        window_rows.append(pd.DataFrame(subject_windows))
        del record
        gc.collect()

    subjects = pd.DataFrame(subject_rows)
    windows = pd.concat(window_rows, ignore_index=True)
    overall_rows = []
    for detector_name, _ in detectors:
        detector_subjects = subjects[subjects.detector == detector_name]
        result = metrics(windows.published_hr_bpm.to_numpy(), windows[f"{detector_name}_hr_bpm"].to_numpy())
        overall_rows.append({
            "detector": detector_name,
            "peak_precision_macro": detector_subjects.peak_precision.mean(),
            "peak_recall_macro": detector_subjects.peak_recall.mean(),
            "peak_f1_macro": detector_subjects.peak_f1.mean(),
            "peak_timing_error_ms_macro": detector_subjects.peak_median_timing_error_ms.mean(),
            "runtime_s_total": detector_subjects.runtime_s.sum(),
            **result,
        })
    overall = pd.DataFrame(overall_rows)
    subjects.to_csv(args.output_dir / "subject_comparison.csv", index=False)
    windows.to_csv(args.output_dir / "window_comparison.csv", index=False)
    overall.to_csv(args.output_dir / "overall_comparison.csv", index=False)

    report = "\n".join([
        "# PPG-DaLiA ECG detector comparison", "",
        "## Overall comparison", "", markdown(overall), "",
        "All detectors use one-to-one annotation matching with ±100 ms tolerance and the same frozen RR-to-window-HR calculation. XQRS uses learning with default WFDB configuration; NeuroKit2 uses Emrich 2023/FastNVG without artifact correction; SleepECG uses its default compiled adaptive Pan–Tompkins implementation. No detector-specific threshold was tuned in this comparison.", "",
        "## Decision rule", "",
        "Choose using clinically relevant HR MAE/RMSE and large-error counts together with peak F1, coverage, runtime, reproducibility, and external PTT-PPG evidence. This is development comparison on PPG-DaLiA and cannot replace external validation.", "",
    ])
    (args.output_dir / "benchmark_report.md").write_text(report, encoding="utf-8")
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
