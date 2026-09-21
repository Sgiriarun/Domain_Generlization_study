#!/usr/bin/env python3
"""Inspect the largest WESAD S2 ECG-detector disagreement episodes."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from rq1_hr.data.loaders import load_wesad
from rq1_hr.preprocessing import detect_rpeaks_emrich2023, detect_rpeaks_sleepecg, detect_rpeaks_xqrs


ROOT = Path("datasets/raw/wesad/WESAD")
INPUT = Path("reports/phase4g_wesad_hr/window_hr_and_quality.csv")
OUTPUT = Path("reports/phase4h_wesad_s2_review")


def main() -> None:
    plot_dir = OUTPUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(INPUT)
    s2 = windows[windows.subject_id == "S2"].copy()
    s2["maximum_pairwise_difference_bpm"] = s2[[
        "emrich_xqrs_absolute_difference_bpm", "emrich_sleepecg_absolute_difference_bpm"
    ]].max(axis=1)
    # Keep one representative per overlapping 8-second episode.
    candidates = s2.sort_values("maximum_pairwise_difference_bpm", ascending=False)
    chosen = []
    for _, row in candidates.iterrows():
        if all(abs(row.window_start_s - previous.window_start_s) >= 8 for previous in chosen):
            chosen.append(row)
        if len(chosen) == 10:
            break
    selected = pd.DataFrame(chosen)

    record = load_wesad(ROOT, "S2")
    ecg = record.ecg.values[:, 0]
    fs = record.ecg.fs_hz
    filtered = sosfiltfilt(butter(3, [5, 25], btype="bandpass", fs=fs, output="sos"), ecg)
    detectors = {
        "Emrich 2023": detect_rpeaks_emrich2023(ecg, fs),
        "XQRS": detect_rpeaks_xqrs(ecg, fs),
        "SleepECG": detect_rpeaks_sleepecg(ecg, fs),
    }
    colors = {"Emrich 2023": "#d7301f", "XQRS": "#238b45", "SleepECG": "#2171b5"}
    for _, row in selected.iterrows():
        start, end = float(row.window_start_s), float(row.window_end_s)
        left, right = int(start * fs), int(end * fs)
        time = start + np.arange(right - left) / fs
        fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
        axes[0].plot(time, ecg[left:right], color="0.25", lw=0.7)
        axes[0].set_ylabel("Raw ECG")
        axes[1].plot(time, filtered[left:right], color="black", lw=0.7)
        for name, peaks in detectors.items():
            local = peaks[(peaks >= left) & (peaks < right)] / fs
            for peak in local:
                axes[1].axvline(peak, color=colors[name], lw=0.8, alpha=0.7)
            axes[1].plot([], [], color=colors[name], label=name)
        axes[1].legend(ncol=3, fontsize=8)
        axes[1].set_ylabel("Filtered ECG")
        axes[1].set_xlabel("Time (s)")
        fig.suptitle(
            f"WESAD S2 window {int(row.window_index)}, condition {row.condition_name} ({int(row.condition_id)}) | "
            f"Emrich {row.emrich2023_hr_bpm:.1f}, XQRS {row.xqrs_hr_bpm:.1f}, SleepECG {row.sleepecg_hr_bpm:.1f} bpm"
        )
        fig.tight_layout()
        fig.savefig(plot_dir / f"S2_w{int(row.window_index):05d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    selected.to_csv(OUTPUT / "selected_disagreement_windows.csv", index=False)
    summary = s2.groupby(["condition_id", "condition_name"]).agg(
        windows=("window_index", "size"),
        disagreements_over_5=("detector_disagreement_over_5_bpm", "sum"),
        disagreements_over_10=("detector_disagreement_over_10_bpm", "sum"),
        maximum_difference_bpm=("maximum_pairwise_difference_bpm", "max"),
    ).reset_index()
    summary.to_csv(OUTPUT / "condition_disagreement_summary.csv", index=False)
    report = f"""# WESAD S2 detector-disagreement review

S2 contains 267 of WESAD's 316 windows with detector disagreement above 10 bpm.
Because 8-second windows overlap every 2 seconds, ten non-overlapping
representative episodes were selected rather than treating every window as an
independent failure.

The plots show raw ECG, the same 5–25 Hz diagnostic filtering, and peaks from
all three detectors. They support visual classification of clipping, baseline
distortion, extra detections, and missed detections. With no supplied WESAD
annotations, the plots cannot establish which detector is correct. These
windows remain quality-flagged rather than automatically deleted or relabelled.

See `selected_disagreement_windows.csv`, `condition_disagreement_summary.csv`,
and `plots/`.
"""
    (OUTPUT / "review.md").write_text(report, encoding="utf-8")
    print(f"Reports written to {OUTPUT}")


if __name__ == "__main__":
    main()
