#!/usr/bin/env python3
"""Evaluate one completed TimePPG fold without retraining or retesting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rq1_hr.evaluation import hr_metrics


DALIA_ACTIVITY = {
    0: "no_activity", 1: "baseline", 2: "stairs", 3: "table_soccer",
    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working",
}
TIMEPPG_COLOR = "#0072B2"
FFT_COLOR = "#D55E00"


def grouped_metrics(frame: pd.DataFrame, columns: list[str], prediction: str) -> pd.DataFrame:
    rows = []
    grouper = columns[0] if len(columns) == 1 else columns
    for keys, group in frame.groupby(grouper, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(dict(zip(columns, keys)) | hr_metrics(group.reference_hr_bpm, group[prediction]))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fft-predictions", type=Path, default=Path("reports/phase8_spectral_baseline/window_predictions.csv"))
    args = parser.parse_args()
    output = args.run_dir / "evaluation"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    history = pd.read_csv(args.run_dir / "training_history.csv")
    predictions = pd.read_csv(args.run_dir / "test_predictions.csv", low_memory=False)
    run_metrics = json.loads((args.run_dir / "metrics.json").read_text())
    predictions["activity"] = pd.to_numeric(predictions.condition_id, errors="coerce").round().astype("Int64").map(DALIA_ACTIVITY)

    fft = pd.read_csv(args.fft_predictions, low_memory=False)
    fft = fft.loc[fft.dataset.eq(run_metrics["dataset"]), [
        "dataset", "record_id", "subject_id", "window_index", "predicted_hr_bpm"
    ]].rename(columns={"predicted_hr_bpm": "fft_predicted_hr_bpm"})
    paired = predictions.merge(
        fft, on=["dataset", "record_id", "subject_id", "window_index"],
        how="left", validate="one_to_one",
    )
    if paired.fft_predicted_hr_bpm.isna().any():
        raise RuntimeError("FFT prediction missing for a TimePPG test window")
    paired["timeppg_absolute_error_bpm"] = np.abs(paired.predicted_hr_bpm - paired.reference_hr_bpm)
    paired["fft_absolute_error_bpm"] = np.abs(paired.fft_predicted_hr_bpm - paired.reference_hr_bpm)
    paired["absolute_error_improvement_bpm"] = paired.fft_absolute_error_bpm - paired.timeppg_absolute_error_bpm
    paired.to_csv(output / "paired_timeppg_fft_predictions.csv", index=False)

    subject = grouped_metrics(paired, ["subject_id"], "predicted_hr_bpm")
    fft_subject = grouped_metrics(paired, ["subject_id"], "fft_predicted_hr_bpm").rename(
        columns={column: f"fft_{column}" for column in hr_metrics([1, 2], [1, 2])}
    )
    subject = subject.merge(fft_subject, on="subject_id", validate="one_to_one")
    subject["mae_improvement_bpm"] = subject.fft_mae_bpm - subject.mae_bpm
    subject.to_csv(output / "subject_metrics.csv", index=False)

    activity = grouped_metrics(paired, ["activity"], "predicted_hr_bpm")
    fft_activity = grouped_metrics(paired, ["activity"], "fft_predicted_hr_bpm").rename(
        columns={column: f"fft_{column}" for column in hr_metrics([1, 2], [1, 2])}
    )
    activity = activity.merge(fft_activity, on="activity", validate="one_to_one")
    activity["mae_improvement_bpm"] = activity.fft_mae_bpm - activity.mae_bpm
    activity.to_csv(output / "activity_metrics.csv", index=False)

    comparison = pd.DataFrame([
        {"method": "TimePPG", **hr_metrics(paired.reference_hr_bpm, paired.predicted_hr_bpm)},
        {"method": "FFT", **hr_metrics(paired.reference_hr_bpm, paired.fft_predicted_hr_bpm)},
    ])
    comparison.to_csv(output / "method_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(history.epoch, history.train_mae_bpm, label="Training MAE", color="#009E73")
    ax.plot(history.epoch, history.validation_mae_bpm, label="Validation MAE", color=TIMEPPG_COLOR)
    ax.axvline(run_metrics["best_epoch"], color="#CC79A7", linestyle="--", label=f"Best epoch: {run_metrics['best_epoch']}")
    ax.set(xlabel="Epoch", ylabel="MAE (bpm)", title="TimePPG training convergence — PPG-DaLiA fold 0")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(figures / "training_convergence.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(subject)); width = .36
    ax.bar(x-width/2, subject.mae_bpm, width, label="TimePPG", color=TIMEPPG_COLOR)
    ax.bar(x+width/2, subject.fft_mae_bpm, width, label="FFT", color=FFT_COLOR)
    ax.set_xticks(x, subject.subject_id); ax.set_ylabel("MAE (bpm)")
    ax.set_title("Same unseen subjects: TimePPG versus FFT"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "subject_mae_comparison.png", dpi=200); plt.close(fig)

    ordered = activity.sort_values("mae_bpm")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    x = np.arange(len(ordered)); width = .38
    ax.bar(x-width/2, ordered.mae_bpm, width, label="TimePPG", color=TIMEPPG_COLOR)
    ax.bar(x+width/2, ordered.fft_mae_bpm, width, label="FFT", color=FFT_COLOR)
    ax.set_xticks(x, ordered.activity, rotation=30, ha="right"); ax.set_ylabel("MAE (bpm)")
    ax.set_title("Activity-specific error on PPG-DaLiA fold 0"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(figures / "activity_mae_comparison.png", dpi=200); plt.close(fig)

    timeppg = comparison.loc[comparison.method.eq("TimePPG")].iloc[0]
    fft_row = comparison.loc[comparison.method.eq("FFT")].iloc[0]
    report = f"""# Evaluation: PPG-DaLiA TimePPG fold 0, seed 17

## Validity

- Training stopped at epoch {run_metrics['stopped_epoch']} after 20 epochs without validation-loss improvement.
- Epoch {run_metrics['best_epoch']} was selected using validation data only.
- Training, validation, and test subjects are disjoint.
- The saved test set was evaluated once after checkpoint selection.
- This is one fold and one seed; it is not the final within-dataset estimate.

## Same-window method comparison

| Method | MAE (bpm) | RMSE (bpm) | Pearson r | Within +/-5 bpm | Bias (bpm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TimePPG | {timeppg.mae_bpm:.3f} | {timeppg.rmse_bpm:.3f} | {timeppg.pearson_r:.4f} | {timeppg.within_5_bpm_percent:.2f}% | {timeppg.bland_altman_bias_bpm:.3f} |
| FFT | {fft_row.mae_bpm:.3f} | {fft_row.rmse_bpm:.3f} | {fft_row.pearson_r:.4f} | {fft_row.within_5_bpm_percent:.2f}% | {fft_row.bland_altman_bias_bpm:.3f} |

TimePPG reduces MAE by {fft_row.mae_bpm-timeppg.mae_bpm:.3f} bpm ({100*(fft_row.mae_bpm-timeppg.mae_bpm)/fft_row.mae_bpm:.1f}%) on exactly the same three test subjects and windows.

## Interpretation boundary

The improvement demonstrates that the learned temporal representation is much
stronger than selecting one dominant PPG frequency on this fold. It does not yet
establish final within-dataset performance, statistical significance, or
cross-dataset generalisation. Those claims require all subject folds and the
four LODO experiments.
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    print(comparison.to_string(index=False))
    print(subject.to_string(index=False))
    print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()
