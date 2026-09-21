#!/usr/bin/env python3
"""Diagnose difficult BIDMC ECG-derived versus monitor-HR episodes."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from rq1_hr.data.loaders import load_bidmc
from rq1_hr.preprocessing import detect_rpeaks_emrich2023, detect_rpeaks_sleepecg, detect_rpeaks_xqrs


ROOT = Path("datasets/raw/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0")
INPUT = Path("reports/phase4i_bidmc_hr/window_hr_and_quality.csv")
OUTPUT = Path("reports/phase4j_bidmc_review")
RECORDS = ("bidmc41", "bidmc45", "bidmc26", "bidmc53")


def choose_separated(frame: pd.DataFrame, count: int = 4) -> pd.DataFrame:
    ranked = frame.sort_values("emrich_monitor_absolute_error_bpm", ascending=False)
    chosen = []
    for _, row in ranked.iterrows():
        if all(abs(row.window_start_s - old.window_start_s) >= 8 for old in chosen):
            chosen.append(row)
        if len(chosen) == count:
            break
    return pd.DataFrame(chosen)


def lag_summary(windows: pd.DataFrame) -> pd.DataFrame:
    """Descriptive sensitivity to monitor-series shifts; not parameter tuning."""
    rows = []
    for record_id, group in windows.groupby("record_id", sort=False):
        group = group.sort_values("window_start_s")
        derived = group.emrich2023_hr_bpm.to_numpy()
        monitor = group.monitor_hr_bpm.to_numpy()
        for lag_s in range(-10, 11):
            shifted = np.interp(
                group.window_start_s.to_numpy() + lag_s,
                group.window_start_s.to_numpy(), monitor,
                left=np.nan, right=np.nan,
            )
            keep = np.isfinite(derived) & np.isfinite(shifted)
            error = derived[keep] - shifted[keep]
            rows.append({
                "record_id": record_id, "monitor_shift_s": lag_s,
                "windows": int(keep.sum()), "mae_bpm": float(np.mean(np.abs(error))),
                "bias_bpm": float(np.mean(error)),
                "correlation": float(np.corrcoef(derived[keep], shifted[keep])[0, 1]),
            })
    return pd.DataFrame(rows)


def plot_episode(record, peaks: dict[str, np.ndarray], row: pd.Series, output: Path) -> None:
    start, end = float(row.window_start_s), float(row.window_end_s)
    fs = record.ecg.fs_hz
    left, right = int(start * fs), int(end * fs)
    ecg = record.ecg.values[:, 0]
    filtered = sosfiltfilt(butter(3, [5, 25], btype="bandpass", fs=fs, output="sos"), ecg)
    ppg = record.ppg.values[left:right, 0]
    time = start + np.arange(right-left) / fs
    colors = {"Emrich 2023":"#d7301f", "XQRS":"#238b45", "SleepECG":"#2171b5"}
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=False, gridspec_kw={"height_ratios":[1,1.2,0.8,0.9]})
    axes[0].plot(time, ecg[left:right], color="0.25", lw=.7); axes[0].set_ylabel("Raw ECG")
    axes[1].plot(time, filtered[left:right], color="black", lw=.7)
    for name, values in peaks.items():
        local = values[(values>=left)&(values<right)]/fs
        for peak in local: axes[1].axvline(peak,color=colors[name],lw=.8,alpha=.7)
        axes[1].plot([],[],color=colors[name],label=name)
    axes[1].legend(ncol=3,fontsize=8); axes[1].set_ylabel("Filtered ECG")
    axes[2].plot(time,ppg,color="0.15",lw=.8); axes[2].set_ylabel("PLETH")
    monitor_t=np.arange(record.provided_hr.values.size)/record.provided_hr.fs_hz
    mask=(monitor_t>=max(0,start-10))&(monitor_t<=min(record.ppg.duration_s,end+10))
    axes[3].plot(monitor_t[mask],record.provided_hr.values[mask],"o-",ms=2,lw=.8,color="0.25",label="monitor HR")
    for name,column,color in (("Emrich","emrich2023_hr_bpm","#d7301f"),("XQRS","xqrs_hr_bpm","#238b45"),("SleepECG","sleepecg_hr_bpm","#2171b5")):
        axes[3].hlines(row[column],start,end,color=color,lw=2,label=name)
    axes[3].axvspan(start,end,color="0.7",alpha=.15); axes[3].legend(ncol=4,fontsize=8); axes[3].set_ylabel("HR (bpm)"); axes[3].set_xlabel("Time (s)")
    fig.suptitle(f"{row.record_id} window {int(row.window_index)} | monitor {row.monitor_hr_bpm:.1f}, Emrich {row.emrich2023_hr_bpm:.1f}, error {row.emrich_monitor_absolute_error_bpm:.1f} bpm")
    for axis in axes: axis.grid(alpha=.18)
    fig.tight_layout(); fig.savefig(output,dpi=160,bbox_inches="tight"); plt.close(fig)


def main() -> None:
    plot_dir=OUTPUT/"plots"; plot_dir.mkdir(parents=True,exist_ok=True)
    windows=pd.read_csv(INPUT); selected_records=windows[windows.record_id.isin(RECORDS)].copy()
    selected_records["emrich_monitor_absolute_error_bpm"]=(selected_records.emrich2023_hr_bpm-selected_records.monitor_hr_bpm).abs()
    selected=pd.concat([choose_separated(group) for _,group in selected_records.groupby("record_id",sort=False)],ignore_index=True)
    selected.to_csv(OUTPUT/"selected_episodes.csv",index=False)
    lags=lag_summary(selected_records); lags.to_csv(OUTPUT/"monitor_lag_sensitivity.csv",index=False)
    best_lags=lags.loc[lags.groupby("record_id").mae_bpm.idxmin()].reset_index(drop=True); best_lags.to_csv(OUTPUT/"best_descriptive_lag_by_record.csv",index=False)
    for rid,rows in selected.groupby("record_id"):
        record=load_bidmc(ROOT,rid); ecg=record.ecg.values[:,0]; fs=record.ecg.fs_hz
        peaks={"Emrich 2023":detect_rpeaks_emrich2023(ecg,fs),"XQRS":detect_rpeaks_xqrs(ecg,fs),"SleepECG":detect_rpeaks_sleepecg(ecg,fs)}
        for _,row in rows.iterrows(): plot_episode(record,peaks,row,plot_dir/f"{rid}_w{int(row.window_index):04d}.png")
    report="""# BIDMC difficult-record review

Four high-error records were reviewed using separated 8-second episodes. Each
plot shows raw and filtered lead-II ECG, peaks from three detectors, PLETH, and
the 1 Hz monitor HR around the window.

`monitor_lag_sensitivity.csv` tests shifts from -10 to +10 seconds only as a
descriptive check for monitor delay. The best shift is not adopted as a fitted
parameter because monitor processing and latency are undocumented and differ by
record.

Interpretation must separate: detector disagreement, agreement of all detectors
on ECG beats, ECG/PPG signal distortion, and mismatch with monitor HR. Monitor
HR is an independent clinical comparator, not R-peak ground truth. No record or
window is removed by this review.
"""
    (OUTPUT/"review.md").write_text(report,encoding="utf-8"); print(f"Reports written to {OUTPUT}")


if __name__=="__main__": main()
