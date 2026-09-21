#!/usr/bin/env python3
"""Build the canonical recording-level manifest for RQ1 Task 1."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
OUTPUT = ROOT / "datasets" / "metadata" / "recordings.csv"
FIELDS = [
    "dataset", "record_id", "subject_id", "condition", "population",
    "device", "body_site", "source_format", "signal_path", "ppg_channel",
    "ppg_fs_hz", "ecg_channel", "ecg_fs_hz", "reference_hr", "annotation_path",
]


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def tags(text: str) -> dict[str, str]:
    return dict(re.findall(r"<([^>]+)>:\s*([^<]+?)(?=\s*<|$)", text))


def ppg_dalia_rows() -> list[dict[str, str]]:
    base = RAW / "ppg_dalia" / "data" / "PPG_FieldStudy"
    rows = []
    for number in range(1, 16):
        subject = f"S{number}"
        path = base / subject / f"{subject}.pkl"
        rows.append({
            "dataset": "ppg_dalia", "record_id": subject, "subject_id": f"ppg_dalia_{subject}",
            "condition": "activities 0-8 (within continuous session)", "population": "healthy adults",
            "device": "Empatica E4 + RespiBAN", "body_site": "wrist PPG; chest ECG",
            "source_format": "pickle", "signal_path": relative(path), "ppg_channel": "signal/wrist/BVP",
            "ppg_fs_hz": "64", "ecg_channel": "signal/chest/ECG", "ecg_fs_hz": "700",
            "reference_hr": "provided ECG-derived label at approximately 0.5 Hz; rpeaks also provided",
            "annotation_path": relative(path),
        })
    return rows


def wesad_rows() -> list[dict[str, str]]:
    base = RAW / "wesad" / "WESAD"
    rows = []
    for number in list(range(2, 12)) + list(range(13, 18)):
        subject = f"S{number}"
        path = base / subject / f"{subject}.pkl"
        rows.append({
            "dataset": "wesad", "record_id": subject, "subject_id": f"wesad_{subject}",
            "condition": "protocol labels 0,1,2,3,4,6,7", "population": "healthy adults",
            "device": "Empatica E4 + RespiBAN", "body_site": "wrist PPG; chest ECG",
            "source_format": "pickle", "signal_path": relative(path), "ppg_channel": "signal/wrist/BVP",
            "ppg_fs_hz": "64", "ecg_channel": "signal/chest/ECG", "ecg_fs_hz": "700",
            "reference_hr": "derive from chest ECG R peaks", "annotation_path": relative(path),
        })
    return rows


def ptt_rows() -> list[dict[str, str]]:
    base = RAW / "ptt_ppg"
    rows = []
    for number in range(1, 23):
        for condition in ("sit", "walk", "run"):
            record = f"s{number}_{condition}"
            header = base / f"{record}.hea"
            meta = tags(header.read_text(errors="replace"))
            rows.append({
                "dataset": "ptt_ppg", "record_id": record, "subject_id": f"ptt_ppg_s{number}",
                "condition": condition, "population": "healthy adults",
                "device": "custom MAX30101 + AD8232", "body_site": "left index finger PPG; ECG electrodes",
                "source_format": "WFDB", "signal_path": relative(base / f"{record}.dat"),
                "ppg_channel": "pleth_3 (primary distal green)", "ppg_fs_hz": "500",
                "ecg_channel": "ecg", "ecg_fs_hz": "500",
                "reference_hr": "derive from manually verified ECG R peaks",
                "annotation_path": relative(base / f"{record}.atr"),
            })
            # Metadata parsing is deliberately exercised even though demographics
            # will live in the separate subject manifest.
            if meta.get("activity") != condition:
                raise ValueError(f"Activity mismatch in {header}")
    return rows


def bidmc_rows() -> list[dict[str, str]]:
    base = RAW / "bidmc" / "bidmc-ppg-and-respiration-dataset-1.0.0"
    rows = []
    for number in range(1, 54):
        record = f"bidmc{number:02d}"
        header = base / f"{record}.hea"
        text = header.read_text(errors="replace")
        source = re.search(r"/matched/([^/]+)/", text)
        if not source:
            raise ValueError(f"No MIMIC source subject in {header}")
        rows.append({
            "dataset": "bidmc", "record_id": record, "subject_id": f"bidmc_{source.group(1)}",
            "condition": "ICU recording", "population": "critically ill adults",
            "device": "clinical bedside monitor", "body_site": "pulse oximeter PPG; ECG leads",
            "source_format": "WFDB", "signal_path": relative(base / f"{record}.dat"),
            "ppg_channel": "PLETH", "ppg_fs_hz": "125", "ecg_channel": "II", "ecg_fs_hz": "125",
            "reference_hr": "derive from ECG for common protocol; monitor HR at 1 Hz available",
            "annotation_path": relative(base / f"{record}n.dat"),
        })
    return rows


def main() -> None:
    rows = ppg_dalia_rows() + wesad_rows() + ptt_rows() + bidmc_rows()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    unique_subjects = len({(r["dataset"], r["subject_id"]) for r in rows})
    print(f"Wrote {len(rows)} recordings across {unique_subjects} dataset-scoped subjects to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
