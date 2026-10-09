"""Raw-record validation used before any RQ1 preprocessing.

The checks in this module measure and report source-data properties. They never
repair values, reject samples, detect ECG peaks, or derive HR targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rq1_hr.data.schema import PPGRecord, SampledSignal


@dataclass(frozen=True, slots=True)
class DatasetExpectation:
    ppg_fs_hz: float
    ecg_fs_hz: float
    ppg_channels: tuple[str, ...]
    has_rpeaks: bool
    has_provided_hr: bool


EXPECTATIONS: dict[str, DatasetExpectation] = {
    "BIDMC": DatasetExpectation(125, 125, ("PLETH",), False, True),
    "PPG-DaLiA": DatasetExpectation(64, 700, ("BVP",), True, True),
    "PTT-PPG": DatasetExpectation(
        500, 500, tuple(f"pleth_{i}" for i in range(1, 7)), True, False
    ),
    "WESAD": DatasetExpectation(64, 700, ("BVP",), False, False),
}


def _max_true_run(mask: np.ndarray) -> int:
    """Return the longest consecutive run containing any invalid channel."""
    row_mask = np.any(mask, axis=1)
    if not np.any(row_mask):
        return 0
    padded = np.pad(row_mask.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2]))


def signal_quality(signal: SampledSignal, prefix: str) -> dict[str, Any]:
    """Calculate structural and missingness statistics for one signal."""
    values = np.asarray(signal.values)
    nonfinite = ~np.isfinite(values)
    nonfinite_count = int(np.count_nonzero(nonfinite))
    finite_count = values.size - nonfinite_count
    return {
        f"{prefix}_samples": signal.n_samples,
        f"{prefix}_channels": "|".join(signal.channel_names),
        f"{prefix}_n_channels": signal.n_channels,
        f"{prefix}_fs_hz": signal.fs_hz,
        f"{prefix}_duration_s": signal.duration_s,
        f"{prefix}_nonfinite_count": nonfinite_count,
        f"{prefix}_nonfinite_fraction": nonfinite_count / values.size,
        f"{prefix}_max_nonfinite_run_samples": _max_true_run(nonfinite),
        f"{prefix}_finite_min": float(np.min(values[np.isfinite(values)])) if finite_count else np.nan,
        f"{prefix}_finite_max": float(np.max(values[np.isfinite(values)])) if finite_count else np.nan,
    }


def validate_record(record: PPGRecord) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate one loaded record and return metrics plus explicit issues."""
    expected = EXPECTATIONS[record.dataset]
    row: dict[str, Any] = {
        "dataset": record.dataset,
        "record_id": record.record_id,
        "subject_id": record.subject_id,
        "recording_condition": record.recording_condition or "",
        "load_success": True,
        "has_ecg": record.ecg is not None,
        "has_rpeaks": record.rpeaks is not None,
        "has_provided_hr": record.provided_hr is not None,
        "has_conditions": record.conditions is not None,
    }
    row.update(signal_quality(record.ppg, "ppg"))
    issues: list[dict[str, str]] = []

    def issue(severity: str, check: str, message: str) -> None:
        issues.append({
            "dataset": record.dataset,
            "record_id": record.record_id,
            "subject_id": record.subject_id,
            "severity": severity,
            "check": check,
            "message": message,
        })

    if record.ecg is None:
        issue("error", "ecg_present", "ECG signal is missing")
        row.update({"ecg_samples": np.nan, "ecg_fs_hz": np.nan, "ecg_duration_s": np.nan})
        row["ppg_ecg_duration_difference_s"] = np.nan
    else:
        row.update(signal_quality(record.ecg, "ecg"))
        duration_difference = record.ppg.duration_s - record.ecg.duration_s
        row["ppg_ecg_duration_difference_s"] = duration_difference
        tolerance = max(1 / record.ppg.fs_hz, 1 / record.ecg.fs_hz) + 1e-9
        if abs(duration_difference) > tolerance:
            issue(
                "error", "ppg_ecg_alignment",
                f"PPG and ECG durations differ by {duration_difference:.6f} s",
            )

    if not np.isclose(record.ppg.fs_hz, expected.ppg_fs_hz):
        issue("error", "ppg_sampling_rate", f"expected {expected.ppg_fs_hz:g} Hz, found {record.ppg.fs_hz:g} Hz")
    if record.ppg.channel_names != expected.ppg_channels:
        issue("error", "ppg_channels", f"expected {expected.ppg_channels}, found {record.ppg.channel_names}")
    if record.ecg is not None and not np.isclose(record.ecg.fs_hz, expected.ecg_fs_hz):
        issue("error", "ecg_sampling_rate", f"expected {expected.ecg_fs_hz:g} Hz, found {record.ecg.fs_hz:g} Hz")
    if (record.rpeaks is not None) != expected.has_rpeaks:
        issue("error", "rpeak_availability", f"expected has_rpeaks={expected.has_rpeaks}")
    if (record.provided_hr is not None) != expected.has_provided_hr:
        issue("error", "provided_hr_availability", f"expected has_provided_hr={expected.has_provided_hr}")

    if row["ppg_nonfinite_count"]:
        issue("warning", "ppg_nonfinite", f"{row['ppg_nonfinite_count']} non-finite PPG values")
    if record.ecg is not None and row["ecg_nonfinite_count"]:
        issue("warning", "ecg_nonfinite", f"{row['ecg_nonfinite_count']} non-finite ECG values")

    row.update({
        "rpeak_count": 0,
        "rr_interval_count": 0,
        "rr_implied_hr_median_bpm": np.nan,
        "rr_implied_hr_below_35_count": 0,
        "rr_implied_hr_above_220_count": 0,
    })
    if record.rpeaks is not None:
        indices = record.rpeaks.sample_indices
        row["rpeak_count"] = int(indices.size)
        if indices.size < 2:
            issue("warning", "rpeak_count", "fewer than two annotated R-peaks")
        else:
            rr_s = np.diff(indices) / record.rpeaks.fs_hz
            implied_hr = 60.0 / rr_s
            row["rr_interval_count"] = int(rr_s.size)
            row["rr_implied_hr_median_bpm"] = float(np.median(implied_hr))
            row["rr_implied_hr_below_35_count"] = int(np.count_nonzero(implied_hr < 35))
            row["rr_implied_hr_above_220_count"] = int(np.count_nonzero(implied_hr > 220))
            outside = row["rr_implied_hr_below_35_count"] + row["rr_implied_hr_above_220_count"]
            if outside:
                issue(
                    "warning", "rpeak_rr_plausibility",
                    f"{outside}/{rr_s.size} annotated RR intervals imply HR outside 35–220 bpm",
                )

    source_duplicate_count = int(record.metadata.get("source_rpeak_duplicate_count", 0))
    row["source_rpeak_duplicate_count"] = source_duplicate_count
    if source_duplicate_count:
        issue(
            "warning", "source_rpeak_duplicates",
            f"{source_duplicate_count} exact duplicate source R-peak indices were collapsed",
        )

    if record.provided_hr is not None:
        hr = np.asarray(record.provided_hr.values)
        finite = np.isfinite(hr)
        row.update({
            "provided_hr_samples": int(hr.size),
            "provided_hr_fs_hz": record.provided_hr.fs_hz,
            "provided_hr_duration_s": record.provided_hr.duration_s,
            "provided_hr_nonfinite_count": int(np.count_nonzero(~finite)),
            "provided_hr_below_35_count": int(np.count_nonzero(finite & (hr < 35))),
            "provided_hr_above_220_count": int(np.count_nonzero(finite & (hr > 220))),
        })
        if row["provided_hr_nonfinite_count"]:
            issue("warning", "provided_hr_nonfinite", f"{row['provided_hr_nonfinite_count']} non-finite provided-HR values")

    auxiliary_nonfinite = 0
    for signal in record.extra_signals.values():
        auxiliary_nonfinite += int(np.count_nonzero(~np.isfinite(signal.values)))
    row["auxiliary_signal_count"] = len(record.extra_signals)
    row["auxiliary_nonfinite_count"] = auxiliary_nonfinite
    if auxiliary_nonfinite:
        issue("warning", "auxiliary_nonfinite", f"{auxiliary_nonfinite} non-finite auxiliary values")

    severities = {item["severity"] for item in issues}
    row["validation_status"] = "error" if "error" in severities else "warning" if "warning" in severities else "pass"
    row["issue_count"] = len(issues)
    return row, issues
