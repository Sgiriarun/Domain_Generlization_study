#!/usr/bin/env python3
"""Phase 4B: explain large PPG-DaLiA ECG-to-HR errors without retuning."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from rq1_hr.data.loaders import list_ppg_dalia_records, load_ppg_dalia
from rq1_hr.preprocessing import detect_rpeaks, match_rpeaks


WINDOW_S = 8.0


def activity_for_windows(values: np.ndarray, fs_hz: float, starts: np.ndarray) -> np.ndarray:
    """Assign the modal source activity ID within each complete window."""
    flat = np.asarray(values).reshape(-1)
    result = np.full(starts.size, -1, dtype=np.int64)
    width = int(round(WINDOW_S * fs_hz))
    for index, start in enumerate(starts):
        left = int(round(start * fs_hz))
        segment = flat[left : left + width].astype(np.int64)
        if segment.size:
            counts = np.bincount(segment[segment >= 0])
            if counts.size:
                result[index] = int(np.argmax(counts))
    return result


def local_peak_counts(reference: np.ndarray, detected: np.ndarray, fs_hz: float, starts: np.ndarray) -> tuple[np.ndarray, ...]:
    outputs: list[list[int]] = [[], [], [], []]
    for start in starts:
        left, right = int(round(start * fs_hz)), int(round((start + WINDOW_S) * fs_hz))
        ref = reference[(reference >= left) & (reference < right)]
        det = detected[(detected >= left) & (detected < right)]
        matched = match_rpeaks(ref, det, fs_hz).matched_count
        for values, value in zip(outputs, (ref.size, det.size, ref.size - matched, det.size - matched)):
            values.append(int(value))
    return tuple(np.asarray(values, dtype=np.int64) for values in outputs)


def markdown_table(frame: pd.DataFrame) -> str:
    def show(value: object) -> str:
        if pd.isna(value):
            return ""
        return f"{value:.3f}" if isinstance(value, float) else str(value)
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(show(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def select_examples(windows: pd.DataFrame) -> pd.DataFrame:
    selected = [windows.nlargest(6, "detected_absolute_error_bpm").assign(example_group="worst_overall")]
    selected.append(windows[windows.subject_id == "S10"].nlargest(4, "detected_absolute_error_bpm").assign(example_group="worst_s10"))
    moderate = windows[(windows.detected_absolute_error_bpm >= 5) & (windows.detected_absolute_error_bpm < 10)]
    if not moderate.empty:
        selected.append(moderate.nlargest(2, "detected_absolute_error_bpm").assign(example_group="moderate_5_to_10"))
    selected.append(windows.nsmallest(2, "detected_absolute_error_bpm").assign(example_group="accurate"))
    return pd.concat(selected).drop_duplicates(["subject_id", "window_index"]).reset_index(drop=True)


def plot_example(record, detected: np.ndarray, filtered_full: np.ndarray, row: pd.Series, output: Path) -> None:
    fs = record.ecg.fs_hz
    start, end = float(row.window_start_s), float(row.window_end_s)
    left, right = int(round(start * fs)), int(round(end * fs))
    raw = record.ecg.values[left:right, 0]
    filtered = filtered_full[left:right]
    time = start + np.arange(raw.size) / fs
    supplied = record.rpeaks.sample_indices
    supplied = supplied[(supplied >= left) & (supplied < right)] / fs
    local_detected = detected[(detected >= left) & (detected < right)] / fs

    acc = record.extra_signals["wrist_acc"]
    acc_left, acc_right = int(round(start * acc.fs_hz)), int(round(end * acc.fs_hz))
    acc_values = acc.values[acc_left:acc_right]
    acc_time = start + np.arange(acc_values.shape[0]) / acc.fs_hz

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.25, 0.7]})
    axes[0].plot(time, raw, color="0.2", linewidth=0.7)
    axes[0].set_ylabel("Raw ECG")
    axes[1].plot(time, filtered, color="black", linewidth=0.8)
    for peak in supplied:
        axes[1].axvline(peak, color="#238b45", linewidth=0.9, alpha=0.8)
    for peak in local_detected:
        axes[1].axvline(peak, color="#cb181d", linewidth=0.8, linestyle="--", alpha=0.8)
    axes[1].plot([], [], color="#238b45", label="supplied R-peak")
    axes[1].plot([], [], color="#cb181d", linestyle="--", label="detected R-peak")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].set_ylabel("Filtered ECG")
    axes[2].plot(acc_time, np.linalg.norm(acc_values, axis=1), color="0.25", linewidth=0.8)
    axes[2].set_ylabel("Wrist ACC\nmagnitude")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(
        f"{row.subject_id}, window {int(row.window_index)}, activity ID {int(row.activity_id)} | "
        f"published {row.published_hr_bpm:.1f}, detected {row.detected_peak_hr_bpm:.1f}, "
        f"error {row.detected_absolute_error_bpm:.1f} bpm | missed {int(row.missed_peaks)}, extra {int(row.extra_peaks)}",
        fontsize=10,
    )
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/ppg_dalia/data/PPG_FieldStudy"))
    parser.add_argument("--phase4a-dir", type=Path, default=Path("reports/phase4a_ppg_dalia"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4b_dalia_errors"))
    args = parser.parse_args()
    plot_dir = args.output_dir / "diagnostic_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.phase4a_dir / "window_comparison.csv")
    enriched: list[pd.DataFrame] = []
    for subject in list_ppg_dalia_records(args.root):
        print(f"Analysing {subject}", flush=True)
        record = load_ppg_dalia(args.root, subject)
        detected = detect_rpeaks(record.ecg.values[:, 0], record.ecg.fs_hz)
        frame = base[base.subject_id == subject].copy()
        starts = frame.window_start_s.to_numpy()
        frame["activity_id"] = activity_for_windows(record.conditions.values, record.conditions.fs_hz, starts)
        ref_n, det_n, missed, extra = local_peak_counts(record.rpeaks.sample_indices, detected, record.ecg.fs_hz, starts)
        frame["supplied_peaks_in_window"] = ref_n
        frame["detected_peaks_in_window"] = det_n
        frame["missed_peaks"] = missed
        frame["extra_peaks"] = extra
        frame["detected_error_bpm"] = frame.detected_peak_hr_bpm - frame.published_hr_bpm
        frame["detected_absolute_error_bpm"] = frame.detected_error_bpm.abs()
        enriched.append(frame)
        del record, detected
        gc.collect()

    windows = pd.concat(enriched, ignore_index=True)
    windows["error_band"] = pd.cut(windows.detected_absolute_error_bpm, bins=[-np.inf, 2, 5, 10, 20, np.inf], labels=["≤2", "2–5", "5–10", "10–20", ">20"])
    subject_summary = windows.groupby("subject_id", sort=False).agg(
        windows=("window_index", "size"), mae_bpm=("detected_absolute_error_bpm", "mean"),
        median_ae_bpm=("detected_absolute_error_bpm", "median"), p95_ae_bpm=("detected_absolute_error_bpm", lambda x: x.quantile(0.95)),
        max_ae_bpm=("detected_absolute_error_bpm", "max"), windows_over_5_bpm=("detected_absolute_error_bpm", lambda x: int((x > 5).sum())),
        windows_over_10_bpm=("detected_absolute_error_bpm", lambda x: int((x > 10).sum())), total_missed_peaks=("missed_peaks", "sum"), total_extra_peaks=("extra_peaks", "sum"),
    ).reset_index()
    activity_summary = windows.groupby("activity_id", sort=True).agg(
        windows=("window_index", "size"), mae_bpm=("detected_absolute_error_bpm", "mean"), p95_ae_bpm=("detected_absolute_error_bpm", lambda x: x.quantile(0.95)),
        windows_over_10_bpm=("detected_absolute_error_bpm", lambda x: int((x > 10).sum())), missed_peaks=("missed_peaks", "sum"), extra_peaks=("extra_peaks", "sum"),
    ).reset_index()
    error_bands = windows.error_band.value_counts(sort=False).rename_axis("absolute_error_bpm").reset_index(name="windows")
    error_bands["fraction"] = error_bands.windows / len(windows)
    examples = select_examples(windows)

    windows.to_csv(args.output_dir / "window_error_analysis.csv", index=False)
    subject_summary.to_csv(args.output_dir / "subject_error_summary.csv", index=False)
    activity_summary.to_csv(args.output_dir / "activity_error_summary.csv", index=False)
    error_bands.to_csv(args.output_dir / "error_band_summary.csv", index=False)
    examples.to_csv(args.output_dir / "selected_examples.csv", index=False)

    for subject, rows in examples.groupby("subject_id"):
        record = load_ppg_dalia(args.root, subject)
        detected = detect_rpeaks(record.ecg.values[:, 0], record.ecg.fs_hz)
        filtered_full = sosfiltfilt(
            butter(3, [5, 25], btype="bandpass", fs=record.ecg.fs_hz, output="sos"),
            record.ecg.values[:, 0],
        )
        for _, row in rows.iterrows():
            name = f"{row.example_group}_{subject}_w{int(row.window_index):05d}.png"
            plot_example(record, detected, filtered_full, row, plot_dir / name)
        del record, detected, filtered_full
        gc.collect()

    report = "\n".join([
        "# Phase 4B — PPG-DaLiA ECG detector error analysis", "", "## Error distribution", "", markdown_table(error_bands), "",
        "Because windows overlap by 6 seconds, neighbouring window errors are correlated and must not be interpreted as independent failures.", "",
        "## Subjects with highest MAE", "", markdown_table(subject_summary.nlargest(5, "mae_bpm")), "",
        "## Activity IDs with highest MAE", "", markdown_table(activity_summary.nlargest(5, "mae_bpm")), "",
        "Activity is assigned as the modal source activity ID in each 8-second window. IDs are retained instead of attaching undocumented names.", "",
        "## Observed failure pattern", "",
        "Most windows are accurate: 88.5% have absolute error at or below 2 bpm. However, 6.1% exceed 10 bpm. The errors are episodic rather than uniformly poor; for example, S10 has median absolute error 0.18 bpm but 95th-percentile error 27.3 bpm.", "",
        "In the inspected worst windows, extra detections dominate. The detector responds to large non-R deflections between supplied beats, which shortens RR intervals and can approximately double estimated HR. The S10 examples also show strong ECG clipping/distortion. These observations justify investigating morphology/RR-consistency rejection or a validated adaptive detector; they do not justify changing a threshold solely to improve this dataset.", "",
        "Activity ID 3 has the largest aggregate MAE, followed by IDs 2 and 7. This is an association with protocol segments, not evidence that motion caused every error.", "",
        "## How to read the diagnostic plots", "",
        "Green solid lines are supplied R-peaks; red dashed lines are our detections. A red-only line is typically an extra peak and a green-only line a missed peak. Wrist ACC magnitude provides motion context but does not prove that motion caused an error.", "",
        "## Scope", "",
        "This phase diagnoses the frozen development detector. It does not change detector parameters or delete windows. Selected examples are listed in `selected_examples.csv`, and plots are under `diagnostic_plots/`.", "",
    ])
    (args.output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
