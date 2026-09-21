#!/usr/bin/env python3
"""Combine validated Phase 4 outputs into one canonical HR-label table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rq1_hr.preprocessing.hr_quality import classify_hr_window


def markdown(frame: pd.DataFrame) -> str:
    """Render a small report table without an optional pandas dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.3f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def finish(frame: pd.DataFrame) -> pd.DataFrame:
    decisions = [
        classify_hr_window(row.hr_bpm, row.valid_rr_count, detector_disagreement_bpm=row.max_detector_disagreement_bpm)
        for row in frame.itertuples(index=False)
    ]
    frame["include_primary"] = [x.include_primary for x in decisions]
    frame["review_recommended"] = [x.review_recommended for x in decisions]
    frame["quality_flag"] = [x.quality_flag for x in decisions]
    return frame


def dalia(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    return pd.DataFrame({
        "dataset": "PPG-DaLiA", "record_id": x.subject_id, "subject_id": x.subject_id,
        "window_index": x.window_index, "window_start_s": x.window_start_s,
        "window_end_s": x.window_end_s, "condition": pd.NA,
        "hr_bpm": x.provided_peak_hr_bpm, "hr_source": "supplied_rpeaks",
        "valid_rr_count": x.provided_peak_valid_rr, "invalid_rr_count": x.provided_peak_invalid_rr,
        "max_detector_disagreement_bpm": np.nan, "comparison_hr_bpm": x.published_hr_bpm,
        "comparison_source": "published_window_hr",
    })


def ptt(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    # Reference HR is identical on the three detector rows; retain one row per window.
    x = x[x.detector.eq("emrich2023")].copy()
    return pd.DataFrame({
        "dataset": "PTT-PPG", "record_id": x.record_id, "subject_id": x.subject_id,
        "window_index": x.window_index, "window_start_s": x.window_start_s,
        "window_end_s": x.window_start_s + 8.0, "condition": x.activity,
        "hr_bpm": x.reference_hr_bpm, "hr_source": "supplied_rpeaks",
        # The detector benchmark did not export the supplied annotation's
        # invalid-RR count, so preserve that distinction as missing, not zero.
        "valid_rr_count": x.reference_valid_rr, "invalid_rr_count": np.nan,
        "max_detector_disagreement_bpm": np.nan, "comparison_hr_bpm": x.detected_hr_bpm,
        "comparison_source": "emrich2023_ecg",
    })


def wesad(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    maximum = x[["emrich_xqrs_absolute_difference_bpm", "emrich_sleepecg_absolute_difference_bpm"]].max(axis=1)
    return pd.DataFrame({
        "dataset": "WESAD", "record_id": x.subject_id, "subject_id": x.subject_id,
        "window_index": x.window_index, "window_start_s": x.window_start_s,
        "window_end_s": x.window_end_s, "condition": x.condition_name,
        "hr_bpm": x.emrich2023_hr_bpm, "hr_source": "emrich2023_ecg",
        "valid_rr_count": x.emrich2023_valid_rr, "invalid_rr_count": x.emrich2023_invalid_rr,
        "max_detector_disagreement_bpm": maximum, "comparison_hr_bpm": np.nan,
        "comparison_source": "detector_consensus_only",
    })


def bidmc(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    maximum = x[["emrich_xqrs_difference_bpm", "emrich_sleepecg_difference_bpm"]].max(axis=1)
    return pd.DataFrame({
        "dataset": "BIDMC", "record_id": x.record_id, "subject_id": x.subject_id,
        "window_index": x.window_index, "window_start_s": x.window_start_s,
        "window_end_s": x.window_end_s, "condition": "clinical_icu",
        "hr_bpm": x.emrich2023_hr_bpm, "hr_source": "emrich2023_ecg",
        "valid_rr_count": x.emrich2023_valid_rr, "invalid_rr_count": x.emrich2023_invalid_rr,
        "max_detector_disagreement_bpm": maximum, "comparison_hr_bpm": x.monitor_hr_bpm,
        "comparison_source": "bedside_monitor_hr",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4k_canonical_hr"))
    args = parser.parse_args()
    root = args.reports_root
    tables = [
        dalia(root / "phase4a_ppg_dalia/window_comparison.csv"),
        ptt(root / "phase4f_ptt_detector_comparison/window_comparison.csv"),
        wesad(root / "phase4g_wesad_hr/window_hr_and_quality.csv"),
        bidmc(root / "phase4i_bidmc_hr/window_hr_and_quality.csv"),
    ]
    result = finish(pd.concat(tables, ignore_index=True))
    if result.duplicated(["dataset", "record_id", "window_index"]).any():
        raise RuntimeError("canonical window keys are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "canonical_window_hr.csv", index=False)
    summary = result.groupby("dataset", sort=False).agg(
        records=("record_id", "nunique"), subjects=("subject_id", "nunique"),
        windows=("window_index", "size"), included_primary=("include_primary", "sum"),
        review_recommended=("review_recommended", "sum"),
    ).reset_index()
    summary["primary_coverage_percent"] = 100 * summary.included_primary / summary.windows
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)

    flags = result.groupby(["dataset", "quality_flag"], sort=False).size().rename("windows").reset_index()
    flags.to_csv(args.output_dir / "quality_flag_summary.csv", index=False)
    lines = [
        "# Frozen Phase 4 canonical HR labels", "",
        "## Decision", "",
        "- Supplied R-peaks are the primary source for PPG-DaLiA and PTT-PPG.",
        "- Emrich 2023 ECG detections are the primary source for WESAD and BIDMC.",
        "- HR uses 8-second windows, 2-second steps, RR-midpoint assignment, 35–220 bpm limits, and at least four valid RR intervals.",
        "- Detector disagreement above 10 bpm is retained as a review flag, not automatically excluded.",
        "- BIDMC monitor HR and published PPG-DaLiA HR are comparison fields, not replacements for the selected ECG reference.", "",
        "## Dataset summary", "", markdown(summary), "",
        "## Quality flags", "", markdown(flags), "",
        "The table preserves `subject_id` for subject-wise splitting and `record_id` for provenance. No PPG filtering, resampling, or model input construction occurs in Phase 4.", "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {len(result):,} canonical windows to {args.output_dir}")


if __name__ == "__main__":
    main()
