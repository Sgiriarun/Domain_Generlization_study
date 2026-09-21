#!/usr/bin/env python3
"""Run the frozen non-learning spectral HR baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rq1_hr.baselines import estimate_hr_spectral
from rq1_hr.evaluation import hr_metrics


COLORS = {"PPG-DaLiA": "#0072B2", "PTT-PPG": "#D55E00", "WESAD": "#009E73", "BIDMC": "#CC79A7"}


def predict_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_path, windows in manifest.groupby("signal_path", sort=False):
        signal = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        for row in windows.itertuples(index=False):
            estimate = estimate_hr_spectral(
                signal[row.start_sample_64hz:row.end_sample_64hz, row.channel_index], 64.0
            )
            rows.append({
                "dataset": row.dataset, "record_id": row.record_id, "subject_id": row.subject_id,
                "window_index": row.window_index, "window_start_s": row.window_start_s,
                "condition_name": row.condition_name, "channel_name": row.channel_name,
                "reference_hr_bpm": row.hr_bpm, "predicted_hr_bpm": estimate.hr_bpm,
                "error_bpm": estimate.hr_bpm - row.hr_bpm,
                "absolute_error_bpm": abs(estimate.hr_bpm - row.hr_bpm),
                "spectral_peak_power_fraction": estimate.peak_power_fraction,
                "hr_quality_flag": row.hr_quality_flag,
            })
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    grouper = columns[0] if len(columns) == 1 else columns
    for keys, group in predictions.groupby(grouper, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(dict(zip(columns, keys)) | hr_metrics(group.reference_hr_bpm, group.predicted_hr_bpm))
    return pd.DataFrame(rows)


def plot_dataset_errors(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    names = summary.dataset.tolist()
    ax.bar(names, summary.mae_bpm, color=[COLORS[name] for name in names], edgecolor="black", linewidth=0.7)
    for index, value in enumerate(summary.mae_bpm):
        ax.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("Frequency-domain HR baseline — primary channel")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_bland_altman(predictions: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False, sharey=False)
    for ax, (dataset, group) in zip(axes.flat, predictions.groupby("dataset", sort=False)):
        mean_hr = (group.reference_hr_bpm + group.predicted_hr_bpm) / 2
        error = group.predicted_hr_bpm - group.reference_hr_bpm
        metrics = hr_metrics(group.reference_hr_bpm, group.predicted_hr_bpm)
        # Plot a deterministic subset only for legibility; metrics use all windows.
        shown = np.arange(0, len(group), max(1, len(group) // 3000))
        ax.scatter(mean_hr.iloc[shown], error.iloc[shown], s=5, alpha=0.22, color=COLORS[dataset], rasterized=True)
        ax.axhline(metrics["bland_altman_bias_bpm"], color="black", linewidth=1.2)
        ax.axhline(metrics["bland_altman_lower_loa_bpm"], color="#666666", linestyle="--")
        ax.axhline(metrics["bland_altman_upper_loa_bpm"], color="#666666", linestyle="--")
        ax.set_title(dataset)
        ax.set_xlabel("Mean of reference and estimate (bpm)")
        ax.set_ylabel("Estimate − reference (bpm)")
    fig.suptitle("Frequency-domain baseline: Bland–Altman analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.select_dtypes(include="number"):
        if column != "windows": shown[column] = shown[column].map(lambda value: f"{value:.3f}")
    lines = ["| " + " | ".join(shown.columns) + " |", "| " + " | ".join("---" for _ in shown.columns) + " |"]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False, name=None)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase7-dir", type=Path, default=Path("reports/phase7_frozen_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase8_spectral_baseline"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"
    figures.mkdir(exist_ok=True)

    main_manifest = pd.read_csv(args.phase7_dir / "main_window_manifest.csv", low_memory=False)
    predictions = predict_manifest(main_manifest)
    predictions.to_csv(args.output_dir / "window_predictions.csv", index=False)
    dataset = grouped_metrics(predictions, ["dataset"])
    subject = grouped_metrics(predictions, ["dataset", "subject_id"])
    condition = grouped_metrics(predictions, ["dataset", "condition_name"])
    dataset.to_csv(args.output_dir / "dataset_metrics.csv", index=False)
    subject.to_csv(args.output_dir / "subject_metrics.csv", index=False)
    condition.to_csv(args.output_dir / "condition_metrics.csv", index=False)

    ptt_manifest = pd.read_csv(args.phase7_dir / "ptt_site_window_manifest.csv", low_memory=False)
    ptt_predictions = predict_manifest(ptt_manifest)
    ptt_predictions = ptt_predictions.merge(
        ptt_manifest[["record_id", "window_index", "channel_name", "sensor_site", "site_pair"]],
        on=["record_id", "window_index", "channel_name"], how="left", validate="one_to_one",
    )
    ptt_predictions.to_csv(args.output_dir / "ptt_site_window_predictions.csv", index=False)
    ptt_metrics = grouped_metrics(ptt_predictions, ["channel_name", "sensor_site", "site_pair"])
    ptt_metrics.to_csv(args.output_dir / "ptt_channel_metrics.csv", index=False)

    plot_dataset_errors(dataset, figures / "dataset_mae.png")
    plot_bland_altman(predictions, figures / "bland_altman.png")
    readme = f"""# Phase 8: conventional frequency-domain HR baseline

## Frozen estimator

For each filtered 8-second primary-channel PPG window, subtract its mean, apply a Hann taper, calculate a 4096-point zero-padded FFT, select the strongest spectral component between 35 and 220 bpm, and convert frequency to HR using `bpm = Hz × 60`.

The estimator has no training data and no dataset-specific tuning. Zero-padding produces a finer numerical frequency grid but does not create new physiological information. Motion and harmonic peaks can still be mistaken for HR; that limitation is part of this baseline.

## Primary-channel results

{markdown(dataset)}

## PTT six-channel secondary results

{markdown(ptt_metrics)}

Metrics are descriptive across overlapping windows. `subject_metrics.csv` is the correct basis for later subject-level uncertainty and comparisons. The spectral baseline does not itself have a leave-one-dataset-out training stage because it learns no parameters; it is applied unchanged to every dataset. The learned CNN will use the frozen Phase 7 LODO splits.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(dataset.to_string(index=False))
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()

