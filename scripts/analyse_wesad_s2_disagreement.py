#!/usr/bin/env python3
"""Create diagnostic plots for the strongest WESAD S2 detector disagreements."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rq1_hr.data.loaders import load_wesad
from rq1_hr.preprocessing import (
    detect_rpeaks_emrich2023,
    detect_rpeaks_sleepecg,
    detect_rpeaks_xqrs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/raw/wesad/WESAD"))
    parser.add_argument("--window-file", type=Path, default=Path("reports/phase4g_wesad_hr/window_hr_and_quality.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase4h_wesad_s2_review"))
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(args.window_file)
    s2 = windows[windows.subject_id == "S2"].copy()
    s2["maximum_pairwise_difference_bpm"] = s2[[
        "emrich_xqrs_absolute_difference_bpm",
        "emrich_sleepecg_absolute_difference_bpm",
    ]].max(axis=1)
    selected = s2.nlargest(args.examples, "maximum_pairwise_difference_bpm").copy()
    selected.to_csv(args.output_dir / "selected_windows.csv", index=False)

    record = load_wesad(args.root, "S2")
    fs = record.ecg.fs_hz
    ecg = record.ecg.values[:, 0]
    peaks = {
        "Emrich 2023": detect_rpeaks_emrich2023(ecg, fs),
        "XQRS": detect_rpeaks_xqrs(ecg, fs),
        "SleepECG": detect_rpeaks_sleepecg(ecg, fs),
    }
    styles = {
        "Emrich 2023": ("#cb181d", "-"),
        "XQRS": ("#2171b5", "--"),
        "SleepECG": ("#238b45", ":"),
    }
    acc = record.extra_signals["wrist_acc"]
    for _, row in selected.iterrows():
        start, end = float(row.window_start_s), float(row.window_end_s)
        left, right = int(round(start * fs)), int(round(end * fs))
        time = start + np.arange(right - left) / fs
        acc_left, acc_right = int(round(start * acc.fs_hz)), int(round(end * acc.fs_hz))
        acc_time = start + np.arange(acc_right - acc_left) / acc.fs_hz
        acc_magnitude = np.linalg.norm(acc.values[acc_left:acc_right], axis=1)

        fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 0.7]})
        axes[0].plot(time, ecg[left:right], color="0.15", linewidth=0.75)
        for name, indices in peaks.items():
            local = indices[(indices >= left) & (indices < right)] / fs
            color, linestyle = styles[name]
            for peak in local:
                axes[0].axvline(peak, color=color, linestyle=linestyle, linewidth=0.9, alpha=0.8)
            axes[0].plot([], [], color=color, linestyle=linestyle, label=name)
        axes[0].legend(loc="upper right", ncol=3, fontsize=8)
        axes[0].set_ylabel("Raw ECG")
        axes[1].plot(acc_time, acc_magnitude, color="0.25", linewidth=0.8)
        axes[1].set_ylabel("Wrist ACC\nmagnitude")
        axes[1].set_xlabel("Time (s)")
        fig.suptitle(
            f"WESAD S2 window {int(row.window_index)}; condition {int(row.condition_id)} ({row.condition_name})\n"
            f"HR: Emrich {row.emrich2023_hr_bpm:.1f}, XQRS {row.xqrs_hr_bpm:.1f}, SleepECG {row.sleepecg_hr_bpm:.1f} bpm",
            fontsize=10,
        )
        for axis in axes:
            axis.grid(alpha=0.18)
        fig.tight_layout()
        fig.savefig(plot_dir / f"S2_w{int(row.window_index):05d}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    lines = [
        "# WESAD S2 detector-disagreement review", "",
        f"The {len(selected)} windows with the largest Emrich disagreement were selected before visual inspection.", "",
        "This is an annotation-free review: agreement between two detectors does not prove correctness. Plots show raw ECG, wrist-ACC magnitude, and peak locations from all three methods. No peaks or HR windows are modified by this script.", "",
        "See `selected_windows.csv` and the `plots/` directory.", "",
    ]
    (args.output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
