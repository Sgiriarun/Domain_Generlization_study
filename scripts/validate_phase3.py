#!/usr/bin/env python3
"""Validate every Phase 2 record and produce the Phase 3 audit reports."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Callable

import pandas as pd

from rq1_hr.data.loaders import (
    list_bidmc_records,
    list_ppg_dalia_records,
    list_ptt_ppg_records,
    list_wesad_records,
    load_bidmc,
    load_ppg_dalia,
    load_ptt_ppg,
    load_wesad,
)
from rq1_hr.data.schema import PPGRecord
from rq1_hr.data.validation import validate_record


Loader = Callable[[Path, str], PPGRecord]
Lister = Callable[[Path], tuple[str, ...]]


def dataset_jobs(raw_root: Path) -> tuple[tuple[str, Path, Lister, Loader], ...]:
    return (
        ("BIDMC", raw_root / "bidmc/bidmc-ppg-and-respiration-dataset-1.0.0", list_bidmc_records, load_bidmc),
        ("PPG-DaLiA", raw_root / "ppg_dalia/data/PPG_FieldStudy", list_ppg_dalia_records, load_ppg_dalia),
        ("PTT-PPG", raw_root / "ptt_ppg", list_ptt_ppg_records, load_ptt_ppg),
        ("WESAD", raw_root / "wesad/WESAD", list_wesad_records, load_wesad),
    )


def audit_all(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    issues: list[dict[str, str]] = []
    for dataset, root, lister, loader in dataset_jobs(raw_root):
        record_ids = lister(root)
        print(f"{dataset}: validating {len(record_ids)} records", flush=True)
        for number, record_id in enumerate(record_ids, start=1):
            try:
                record = loader(root, record_id)
                row, record_issues = validate_record(record)
                rows.append(row)
                issues.extend(record_issues)
                del record
            except Exception as exc:  # keep auditing after an isolated bad record
                rows.append({
                    "dataset": dataset, "record_id": record_id, "subject_id": "",
                    "load_success": False, "validation_status": "error", "issue_count": 1,
                })
                issues.append({
                    "dataset": dataset, "record_id": record_id, "subject_id": "",
                    "severity": "error", "check": "loader", "message": f"{type(exc).__name__}: {exc}",
                })
            gc.collect()
            if number % 10 == 0 or number == len(record_ids):
                print(f"  {number}/{len(record_ids)}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(issues, columns=["dataset", "record_id", "subject_id", "severity", "check", "message"])


def dataset_summary(records: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for dataset, group in records.groupby("dataset", sort=False):
        dataset_issues = issues[issues["dataset"] == dataset]
        summaries.append({
            "dataset": dataset,
            "records": len(group),
            "loaded_records": int(group["load_success"].fillna(False).sum()),
            "subjects": group.loc[group["load_success"] == True, "subject_id"].nunique(),  # noqa: E712
            "total_ppg_hours": group["ppg_duration_s"].sum() / 3600,
            "ppg_nonfinite_values": int(group["ppg_nonfinite_count"].fillna(0).sum()),
            "ecg_nonfinite_values": int(group["ecg_nonfinite_count"].fillna(0).sum()),
            "records_with_rpeaks": int(group["has_rpeaks"].fillna(False).sum()),
            "records_with_provided_hr": int(group["has_provided_hr"].fillna(False).sum()),
            "pass_records": int((group["validation_status"] == "pass").sum()),
            "warning_records": int((group["validation_status"] == "warning").sum()),
            "error_records": int((group["validation_status"] == "error").sum()),
            "warnings": int((dataset_issues["severity"] == "warning").sum()),
            "errors": int((dataset_issues["severity"] == "error").sum()),
        })
    return pd.DataFrame(summaries)


def dataframe_to_markdown(frame: pd.DataFrame, float_digits: int = 3) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""
    def display(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.{float_digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in values) + " |")
    return "\n".join(lines)


def markdown_report(records: pd.DataFrame, summary: pd.DataFrame, issues: pd.DataFrame) -> str:
    total = len(records)
    loaded = int(records["load_success"].fillna(False).sum())
    errors = int((issues["severity"] == "error").sum())
    warnings = int((issues["severity"] == "warning").sum())
    lines = [
        "# Phase 3 — Complete raw-record validation", "",
        "## Result", "",
        f"Loaded **{loaded}/{total}** discovered recordings. The audit found **{errors} errors** and **{warnings} warnings**.", "",
        "An error means the WP1 loading gate is not satisfied. A warning identifies a real source-data characteristic that must be explained or handled later; Phase 3 does not alter it.", "",
        "## Dataset summary", "",
        dataframe_to_markdown(summary), "",
        "## Availability of reference information", "",
        "`provided_hr` means a dataset-supplied HR series. It is retained for provenance and comparison, but it is not automatically the common RQ1 ECG-derived target.", "",
        "| Dataset | PPG fs | ECG fs | PPG channels | R-peaks supplied | HR series supplied |", "|---|---:|---:|---:|:---:|:---:|",
        "| BIDMC | 125 Hz | 125 Hz | 1 | No | Yes |",
        "| PPG-DaLiA | 64 Hz | 700 Hz | 1 | Yes | Yes |",
        "| PTT-PPG | 500 Hz | 500 Hz | 6 | Yes | No |",
        "| WESAD | 64 Hz | 700 Hz | 1 | No | No |", "",
        "## Validation rules", "",
        "- Dataset-specific PPG/ECG sampling rates and PPG channel names must match the documented source format.",
        "- PPG and ECG durations may differ by no more than one sample period.",
        "- Non-finite values are counted and reported, never repaired here.",
        "- Supplied R-peaks must satisfy the schema; RR intervals implying HR outside 35–220 bpm are flagged for review, not deleted.",
        "- BIDMC uses the source MIMIC patient as the leakage-control identity.", "",
        "## Issues", "",
    ]
    if issues.empty:
        lines.append("No validation issues were detected.")
    else:
        counts = issues.groupby(["severity", "check"], sort=True).size().reset_index(name="count")
        lines.extend([dataframe_to_markdown(counts), "", "See `validation_issues.csv` for record-level details."])
    lines.extend(["", "## Scope boundary", "", "This report validates loading and raw structure only. It does not filter, resample, derive ECG HR, create windows, or reject low-quality samples.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, issues = audit_all(args.raw_root)
    summary = dataset_summary(records, issues)
    records.to_csv(args.output_dir / "recording_validation.csv", index=False)
    summary.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    issues.to_csv(args.output_dir / "validation_issues.csv", index=False)
    (args.output_dir / "validation_report.md").write_text(
        markdown_report(records, summary, issues), encoding="utf-8"
    )
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
