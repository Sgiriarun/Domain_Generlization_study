#!/usr/bin/env python3
"""Validate supplied-peak and detected-peak HR paths on PPG-DaLiA."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from rq1_hr.data.loaders import list_ppg_dalia_records, load_ppg_dalia
from rq1_hr.preprocessing import detect_rpeaks, match_rpeaks, window_hr_from_rpeaks


def agreement(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    count = int(np.count_nonzero(valid))
    if count == 0:
        return {"matched_windows": 0, "mae_bpm": np.nan, "rmse_bpm": np.nan, "bias_bpm": np.nan, "correlation": np.nan}
    error = estimate[valid] - reference[valid]
    correlation = float(np.corrcoef(reference[valid], estimate[valid])[0, 1]) if count > 1 else np.nan
    return {
        "matched_windows": count,
        "mae_bpm": float(np.mean(np.abs(error))),
        "rmse_bpm": float(np.sqrt(np.mean(error * error))),
        "bias_bpm": float(np.mean(error)),
        "correlation": correlation,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    def show(value: object) -> str:
        if pd.isna(value):
            return ""
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(show(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/ppg_dalia/data/PPG_FieldStudy"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4a_ppg_dalia"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_rows: list[dict[str, object]] = []
    window_frames: list[pd.DataFrame] = []
    for subject in list_ppg_dalia_records(args.root):
        print(f"Validating {subject}", flush=True)
        record = load_ppg_dalia(args.root, subject)
        supplied = window_hr_from_rpeaks(
            record.rpeaks.sample_indices, record.rpeaks.fs_hz, record.ppg.duration_s
        )
        detected_indices = detect_rpeaks(record.ecg.values[:, 0], record.ecg.fs_hz)
        detected = window_hr_from_rpeaks(
            detected_indices, record.ecg.fs_hz, record.ppg.duration_s
        )
        peak_metrics = match_rpeaks(
            record.rpeaks.sample_indices, detected_indices, record.ecg.fs_hz
        )
        published = np.asarray(record.provided_hr.values, dtype=np.float64)
        if published.size != supplied.values_bpm.size:
            raise ValueError(
                f"{subject}: {published.size} supplied labels but "
                f"{supplied.values_bpm.size} documented windows"
            )

        supplied_metrics = agreement(published, supplied.values_bpm)
        detected_metrics = agreement(published, detected.values_bpm)
        subject_rows.append({
            "subject_id": subject,
            "published_windows": published.size,
            "provided_peak_hr_coverage": supplied_metrics["matched_windows"] / published.size,
            "provided_peak_hr_mae_bpm": supplied_metrics["mae_bpm"],
            "provided_peak_hr_rmse_bpm": supplied_metrics["rmse_bpm"],
            "provided_peak_hr_bias_bpm": supplied_metrics["bias_bpm"],
            "provided_peak_hr_correlation": supplied_metrics["correlation"],
            "detected_peak_count": peak_metrics.detected_count,
            "reference_peak_count": peak_metrics.reference_count,
            "peak_precision": peak_metrics.precision,
            "peak_recall": peak_metrics.recall,
            "peak_f1": peak_metrics.f1,
            "peak_median_timing_error_ms": peak_metrics.median_absolute_timing_error_ms,
            "detected_peak_hr_coverage": detected_metrics["matched_windows"] / published.size,
            "detected_peak_hr_mae_bpm": detected_metrics["mae_bpm"],
            "detected_peak_hr_rmse_bpm": detected_metrics["rmse_bpm"],
            "detected_peak_hr_bias_bpm": detected_metrics["bias_bpm"],
            "detected_peak_hr_correlation": detected_metrics["correlation"],
        })
        window_frames.append(pd.DataFrame({
            "subject_id": subject,
            "window_index": np.arange(published.size),
            "window_start_s": supplied.starts_s,
            "window_end_s": supplied.starts_s + 8,
            "published_hr_bpm": published,
            "provided_peak_hr_bpm": supplied.values_bpm,
            "provided_peak_valid_rr": supplied.valid_rr_count,
            "provided_peak_invalid_rr": supplied.invalid_rr_count,
            "detected_peak_hr_bpm": detected.values_bpm,
            "detected_peak_valid_rr": detected.valid_rr_count,
            "detected_peak_invalid_rr": detected.invalid_rr_count,
        }))
        del record, supplied, detected, detected_indices
        gc.collect()

    subjects = pd.DataFrame(subject_rows)
    windows = pd.concat(window_frames, ignore_index=True)
    overall_rows = []
    for route, column in (("provided R-peaks", "provided_peak_hr_bpm"), ("detected R-peaks", "detected_peak_hr_bpm")):
        metrics = agreement(windows["published_hr_bpm"].to_numpy(), windows[column].to_numpy())
        overall_rows.append({
            "route": route,
            **metrics,
            "coverage": metrics["matched_windows"] / len(windows),
        })
    overall = pd.DataFrame(overall_rows)

    subjects.to_csv(args.output_dir / "subject_metrics.csv", index=False)
    windows.to_csv(args.output_dir / "window_comparison.csv", index=False)
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)

    peak_macro = subjects[["peak_precision", "peak_recall", "peak_f1", "peak_median_timing_error_ms"]].mean().to_frame("macro_mean").reset_index(names="metric")
    report = "\n".join([
        "# Phase 4A — PPG-DaLiA ECG-to-HR validation", "",
        "## Design", "",
        "Two paths are evaluated independently. Path 1 uses dataset-supplied R-peaks and therefore tests RR conversion, window placement, physiological filtering, and aggregation. Path 2 detects peaks from raw ECG and tests the complete deployable reference pipeline.", "",
        "Published labels are aligned using the documented 8-second windows starting every 2 seconds. RR intervals are assigned by their midpoint; intervals implying HR outside 35–220 bpm are excluded; at least four valid RR intervals are required; remaining beat-level HR values are averaged.", "",
        "## Overall HR agreement with published labels", "", markdown_table(overall), "",
        "## ECG detector annotation agreement", "", markdown_table(peak_macro), "",
        "Peak matching is one-to-one with a fixed ±100 ms tolerance. The development detector uses a fixed 5–25 Hz ECG bandpass, derivative-energy integration over 120 ms, a median-plus-12-MAD threshold, a 250 ms candidate separation, and ±100 ms polarity-independent refinement. The same parameters are used for every subject.", "",
        "These are development-set results, not an unbiased external estimate: PPG-DaLiA annotations were available while the detector design was being checked. The supplied-peak HR calculation is considered validated; the raw-ECG detector remains provisional because occasional false peaks produce a much larger RMSE than MAE, especially for S10.", "",
        "## Interpretation boundary", "",
        "Strong agreement on PPG-DaLiA validates this implementation for RespiBAN ECG in this dataset. It does not by itself prove validity for BIDMC or WESAD. WESAD uses the same chest-device family and should be checked next by waveform review and detector quality diagnostics; BIDMC requires separate validation against monitor HR because supplied R-peaks are unavailable.", "",
        "Subject-level results are in `subject_metrics.csv`; every window and its quality counts are in `window_comparison.csv`.", "",
    ])
    (args.output_dir / "validation_report.md").write_text(report, encoding="utf-8")
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
