#!/usr/bin/env python3
"""Generate BIDMC ECG-derived HR and compare it with bedside-monitor HR."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import numpy as np
import pandas as pd

from rq1_hr.data.loaders import list_bidmc_records, load_bidmc
from rq1_hr.preprocessing import (
    detect_rpeaks_emrich2023,
    detect_rpeaks_sleepecg,
    detect_rpeaks_xqrs,
    window_hr_from_rpeaks,
)


def monitor_window_mean(values: np.ndarray, fs_hz: float, starts: np.ndarray, window_s: float = 8) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(starts.size, np.nan)
    counts = np.zeros(starts.size, dtype=np.int64)
    for index, start in enumerate(starts):
        left, right = int(round(start * fs_hz)), int(round((start + window_s) * fs_hz))
        segment = np.asarray(values[left:right], dtype=float)
        finite = segment[np.isfinite(segment)]
        counts[index] = finite.size
        if finite.size:
            result[index] = float(np.mean(finite))
    return result, counts


def metrics(reference: pd.Series, estimate: pd.Series, name: str) -> dict[str, object]:
    keep = np.isfinite(reference) & np.isfinite(estimate)
    error = estimate[keep].to_numpy() - reference[keep].to_numpy()
    return {
        "comparison": name, "windows": int(keep.sum()),
        "mae_bpm": float(np.mean(np.abs(error))),
        "rmse_bpm": float(np.sqrt(np.mean(error * error))),
        "bias_bpm": float(np.mean(error)),
        "correlation": float(np.corrcoef(reference[keep], estimate[keep])[0, 1]),
        "over_5_bpm": int(np.count_nonzero(np.abs(error) > 5)),
        "over_10_bpm": int(np.count_nonzero(np.abs(error) > 10)),
        "over_20_bpm": int(np.count_nonzero(np.abs(error) > 20)),
    }


def md(frame: pd.DataFrame) -> str:
    def show(x):
        if pd.isna(x): return ""
        return f"{x:.4f}" if isinstance(x, float) else str(x)
    cols=list(frame.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join("---" for _ in cols)+" |"]
    lines += ["| "+" | ".join(show(x) for x in row)+" |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("datasets/raw/bidmc/bidmc-ppg-and-respiration-dataset-1.0.0"))
    parser.add_argument("--output-dir",type=Path,default=Path("reports/phase4i_bidmc_hr"))
    args=parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    frames=[]
    detectors=(("emrich2023",detect_rpeaks_emrich2023),("xqrs",detect_rpeaks_xqrs),("sleepecg",detect_rpeaks_sleepecg))
    for rid in list_bidmc_records(args.root):
        print(f"Processing {rid}",flush=True); record=load_bidmc(args.root,rid); frame=None
        for name,detector in detectors:
            peaks=detector(record.ecg.values[:,0],record.ecg.fs_hz)
            hr=window_hr_from_rpeaks(peaks,record.ecg.fs_hz,record.ppg.duration_s)
            if frame is None:
                monitor,monitor_n=monitor_window_mean(record.provided_hr.values,record.provided_hr.fs_hz,hr.starts_s)
                frame=pd.DataFrame({"dataset":"BIDMC","record_id":rid,"subject_id":record.subject_id,"window_index":np.arange(hr.starts_s.size),"window_start_s":hr.starts_s,"window_end_s":hr.starts_s+8,"monitor_hr_bpm":monitor,"monitor_finite_samples":monitor_n})
            frame[f"{name}_hr_bpm"]=hr.values_bpm
            frame[f"{name}_valid_rr"]=hr.valid_rr_count
            frame[f"{name}_invalid_rr"]=hr.invalid_rr_count
        frame["emrich_xqrs_difference_bpm"]=(frame.emrich2023_hr_bpm-frame.xqrs_hr_bpm).abs()
        frame["emrich_sleepecg_difference_bpm"]=(frame.emrich2023_hr_bpm-frame.sleepecg_hr_bpm).abs()
        frame["detector_disagreement_over_10_bpm"]=(frame.emrich_xqrs_difference_bpm>10)|(frame.emrich_sleepecg_difference_bpm>10)
        frames.append(frame); del record,frame,peaks,hr; gc.collect()
    windows=pd.concat(frames,ignore_index=True)
    overall=pd.DataFrame([metrics(windows.monitor_hr_bpm,windows[f"{name}_hr_bpm"],f"monitor vs {name}") for name,_ in detectors])
    record_rows=[]
    for rid,group in windows.groupby("record_id",sort=False):
        row={"record_id":rid,"subject_id":group.subject_id.iloc[0],"windows":len(group),"monitor_coverage":float(np.isfinite(group.monitor_hr_bpm).mean()),"detector_disagreement_over_10_bpm":int(group.detector_disagreement_over_10_bpm.sum())}
        for name,_ in detectors:
            result=metrics(group.monitor_hr_bpm,group[f"{name}_hr_bpm"],name)
            row[f"{name}_mae_bpm"]=result["mae_bpm"]; row[f"{name}_rmse_bpm"]=result["rmse_bpm"]; row[f"{name}_correlation"]=result["correlation"]
        record_rows.append(row)
    records=pd.DataFrame(record_rows)
    windows.to_csv(args.output_dir/"window_hr_and_quality.csv",index=False); overall.to_csv(args.output_dir/"overall_comparison.csv",index=False); records.to_csv(args.output_dir/"record_summary.csv",index=False)
    report="\n".join(["# BIDMC ECG-derived HR validation","","## Method","","Frozen Emrich 2023, XQRS, and SleepECG detectors are applied to lead II. Their 8-second/2-second-step HR is compared with the mean of finite 1 Hz bedside-monitor HR values in the same interval.","","The monitor series is an independent clinical comparison but not sample-level R-peak ground truth. Monitor smoothing, update timing, and proprietary processing can contribute to disagreement.","","## Overall comparison","",md(overall),"","## Records with highest Emrich-versus-monitor MAE","",md(records.nlargest(10,"emrich2023_mae_bpm")[["record_id","subject_id","monitor_coverage","emrich2023_mae_bpm","emrich2023_rmse_bpm","emrich2023_correlation","detector_disagreement_over_10_bpm"]]),"","All window values and quality counts are retained in `window_hr_and_quality.csv`. No window is removed by this validation report.",""])
    (args.output_dir/"validation_report.md").write_text(report,encoding="utf-8"); print(f"Reports written to {args.output_dir}")


if __name__=="__main__": main()
