#!/usr/bin/env python3
"""Generate WESAD ECG-derived HR and annotation-free quality diagnostics."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import numpy as np
import pandas as pd

from rq1_hr.data.loaders import list_wesad_records, load_wesad
from rq1_hr.preprocessing import (
    detect_rpeaks_emrich2023,
    detect_rpeaks_sleepecg,
    detect_rpeaks_xqrs,
    window_hr_from_rpeaks,
)


CONDITION_NAMES = {
    0: "not_defined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "not_defined",
    6: "not_defined",
    7: "not_defined",
}


def modal_labels(values: np.ndarray, fs_hz: float, starts: np.ndarray, window_s: float) -> np.ndarray:
    labels = np.asarray(values).reshape(-1).astype(np.int64)
    width = int(round(window_s * fs_hz))
    result = np.full(starts.size, -1, dtype=np.int64)
    for index, start in enumerate(starts):
        left = int(round(start * fs_hz))
        segment = labels[left : left + width]
        if segment.size:
            counts = np.bincount(segment[segment >= 0])
            if counts.size:
                result[index] = int(np.argmax(counts))
    return result


def markdown(frame: pd.DataFrame) -> str:
    def show(value: object) -> str:
        if pd.isna(value):
            return ""
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(show(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def agreement(a: pd.Series, b: pd.Series, name: str) -> dict[str, object]:
    keep = np.isfinite(a) & np.isfinite(b)
    difference = np.abs(a[keep].to_numpy() - b[keep].to_numpy())
    return {
        "comparison": name,
        "matched_windows": int(keep.sum()),
        "median_absolute_difference_bpm": float(np.median(difference)),
        "mean_absolute_difference_bpm": float(np.mean(difference)),
        "p95_absolute_difference_bpm": float(np.quantile(difference, 0.95)),
        "windows_over_5_bpm": int(np.count_nonzero(difference > 5)),
        "windows_over_10_bpm": int(np.count_nonzero(difference > 10)),
        "correlation": float(np.corrcoef(a[keep], b[keep])[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/wesad/WESAD"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4g_wesad_hr"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    peak_rows: list[dict[str, object]] = []
    detectors = (
        ("emrich2023", detect_rpeaks_emrich2023),
        ("xqrs", detect_rpeaks_xqrs),
        ("sleepecg", detect_rpeaks_sleepecg),
    )
    for subject in list_wesad_records(args.root):
        print(f"Processing {subject}", flush=True)
        record = load_wesad(args.root, subject)
        frame: pd.DataFrame | None = None
        for detector_name, detector in detectors:
            peaks = detector(record.ecg.values[:, 0], record.ecg.fs_hz)
            hr = window_hr_from_rpeaks(peaks, record.ecg.fs_hz, record.ppg.duration_s)
            if frame is None:
                labels = modal_labels(record.conditions.values, record.conditions.fs_hz, hr.starts_s, 8.0)
                frame = pd.DataFrame({
                    "dataset": "WESAD", "subject_id": subject,
                    "window_index": np.arange(hr.starts_s.size),
                    "window_start_s": hr.starts_s, "window_end_s": hr.starts_s + 8,
                    "condition_id": labels,
                    "condition_name": [CONDITION_NAMES.get(int(label), "unknown") for label in labels],
                })
            frame[f"{detector_name}_hr_bpm"] = hr.values_bpm
            frame[f"{detector_name}_candidate_rr"] = hr.candidate_rr_count
            frame[f"{detector_name}_valid_rr"] = hr.valid_rr_count
            frame[f"{detector_name}_invalid_rr"] = hr.invalid_rr_count
            peak_rows.append({
                "subject_id": subject, "detector": detector_name,
                "detected_peak_count": int(peaks.size),
                "windows": int(hr.starts_s.size),
                "windows_with_hr": int(np.count_nonzero(np.isfinite(hr.values_bpm))),
                "total_invalid_rr_in_windows": int(hr.invalid_rr_count.sum()),
            })
        assert frame is not None
        frame["emrich_xqrs_absolute_difference_bpm"] = (frame.emrich2023_hr_bpm - frame.xqrs_hr_bpm).abs()
        frame["emrich_sleepecg_absolute_difference_bpm"] = (frame.emrich2023_hr_bpm - frame.sleepecg_hr_bpm).abs()
        frame["detector_disagreement_over_5_bpm"] = (
            (frame.emrich_xqrs_absolute_difference_bpm > 5)
            | (frame.emrich_sleepecg_absolute_difference_bpm > 5)
        )
        frame["detector_disagreement_over_10_bpm"] = (
            (frame.emrich_xqrs_absolute_difference_bpm > 10)
            | (frame.emrich_sleepecg_absolute_difference_bpm > 10)
        )
        frame["emrich_hr_available"] = np.isfinite(frame.emrich2023_hr_bpm)
        frames.append(frame)
        del record, frame, peaks, hr
        gc.collect()

    windows = pd.concat(frames, ignore_index=True)
    peaks = pd.DataFrame(peak_rows)
    subject_summary = windows.groupby("subject_id", sort=False).agg(
        windows=("window_index", "size"),
        emrich_hr_coverage=("emrich_hr_available", "mean"),
        emrich_hr_median_bpm=("emrich2023_hr_bpm", "median"),
        emrich_hr_min_bpm=("emrich2023_hr_bpm", "min"),
        emrich_hr_max_bpm=("emrich2023_hr_bpm", "max"),
        emrich_invalid_rr=("emrich2023_invalid_rr", "sum"),
        disagreement_over_5_bpm=("detector_disagreement_over_5_bpm", "sum"),
        disagreement_over_10_bpm=("detector_disagreement_over_10_bpm", "sum"),
    ).reset_index()
    condition_summary = windows.groupby(["condition_id", "condition_name"], sort=True).agg(
        windows=("window_index", "size"),
        subjects=("subject_id", "nunique"),
        emrich_hr_median_bpm=("emrich2023_hr_bpm", "median"),
        emrich_hr_q05_bpm=("emrich2023_hr_bpm", lambda x: x.quantile(0.05)),
        emrich_hr_q95_bpm=("emrich2023_hr_bpm", lambda x: x.quantile(0.95)),
        emrich_hr_coverage=("emrich_hr_available", "mean"),
        disagreement_over_5_bpm=("detector_disagreement_over_5_bpm", "sum"),
        disagreement_over_10_bpm=("detector_disagreement_over_10_bpm", "sum"),
    ).reset_index()
    detector_agreement = pd.DataFrame([
        agreement(windows.emrich2023_hr_bpm, windows.xqrs_hr_bpm, "Emrich 2023 vs XQRS"),
        agreement(windows.emrich2023_hr_bpm, windows.sleepecg_hr_bpm, "Emrich 2023 vs SleepECG"),
        agreement(windows.xqrs_hr_bpm, windows.sleepecg_hr_bpm, "XQRS vs SleepECG"),
    ])

    windows.to_csv(args.output_dir / "window_hr_and_quality.csv", index=False)
    peaks.to_csv(args.output_dir / "subject_detector_counts.csv", index=False)
    subject_summary.to_csv(args.output_dir / "subject_summary.csv", index=False)
    condition_summary.to_csv(args.output_dir / "condition_summary.csv", index=False)
    detector_agreement.to_csv(args.output_dir / "detector_agreement.csv", index=False)

    report = "\n".join([
        "# WESAD ECG-derived HR and detector-quality report", "",
        "## Method", "",
        "NeuroKit2 Emrich 2023/FastNVG is applied with its frozen default implementation to 700 Hz WESAD chest ECG. HR uses 8-second windows, 2-second step, RR-midpoint assignment, 35–220 bpm interval filtering, at least four valid RR intervals, and arithmetic mean.", "",
        "WESAD has no supplied R-peaks or HR. Therefore this report does not claim accuracy. XQRS and SleepECG are independent comparison detectors; disagreement flags identify windows needing review but do not prove which detector is correct.", "",
        "## Detector agreement", "", markdown(detector_agreement), "",
        "## Subject summary", "", markdown(subject_summary), "",
        "## Protocol-condition summary", "", markdown(condition_summary), "",
        "## Interpretation rules", "",
        "- Emrich HR is the selected candidate reference.",
        "- Missing HR means fewer than four valid RR intervals.",
        "- Pairwise disagreement above 5 or 10 bpm is retained as a quality flag, not automatically rejected.",
        "- Protocol labels 0, 5, 6, and 7 remain `not_defined` according to the source label scheme and should not be interpreted as named experimental conditions.",
        "- Final acceptance requires visual review of representative agreement and disagreement windows and later BIDMC monitor-HR validation.", "",
    ])
    (args.output_dir / "validation_report.md").write_text(report, encoding="utf-8")
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
