#!/usr/bin/env python3
"""Aggregate completed unseen-subject TimePPG folds for one dataset."""

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


def metrics_by(frame: pd.DataFrame, column: str, prediction: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(column, sort=True, dropna=False):
        rows.append({column: value, **hr_metrics(group.reference_hr_bpm, group[prediction])})
    return pd.DataFrame(rows)


def bootstrap_subject_macro_ci(subject_mae: np.ndarray, seed: int = 17) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(subject_mae, size=(10_000, len(subject_mae)), replace=True), axis=1)
    return float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("reports/phase9_timeppg/within_dataset/ppg_dalia"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fft-predictions", type=Path, default=Path("reports/phase8_spectral_baseline/window_predictions.csv"))
    args = parser.parse_args()
    output = args.run_root / f"aggregate_seed_{args.seed}"
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    frames, fold_rows = [], []
    all_test_subjects = []
    dataset = None
    for fold in range(args.folds):
        run = args.run_root / f"test_fold_{fold}" / f"seed_{args.seed}"
        metrics = json.loads((run / "metrics.json").read_text())
        if metrics["status"] != "complete" or metrics["test_fold"] != fold:
            raise RuntimeError(f"fold {fold} is not a valid completed run")
        dataset = metrics["dataset"] if dataset is None else dataset
        if metrics["dataset"] != dataset or metrics["subject_overlap"] or metrics["test_evaluations"] != 1:
            raise RuntimeError(f"fold {fold} provenance failed")
        all_test_subjects.extend(metrics["subjects"]["test"])
        fold_rows.append({key: value for key, value in metrics.items() if key not in {"subjects"}})
        frame = pd.read_csv(run / "test_predictions.csv", low_memory=False)
        frame["test_fold"] = fold
        frames.append(frame)
    if len(all_test_subjects) != len(set(all_test_subjects)):
        raise RuntimeError("a subject occurs in more than one test fold")

    predictions = pd.concat(frames, ignore_index=True)
    key = ["dataset", "record_id", "subject_id", "window_index"]
    if predictions.duplicated(key).any():
        raise RuntimeError("duplicate test window across folds")
    predictions["activity"] = pd.to_numeric(predictions.condition_id, errors="coerce").round().astype("Int64").map(DALIA_ACTIVITY)
    predictions.to_csv(output / "all_test_predictions.csv", index=False)

    fft = pd.read_csv(args.fft_predictions, low_memory=False)
    fft = fft.loc[fft.dataset.eq(dataset), key + ["predicted_hr_bpm"]].rename(columns={"predicted_hr_bpm": "fft_predicted_hr_bpm"})
    paired = predictions.merge(fft, on=key, how="left", validate="one_to_one")
    if paired.fft_predicted_hr_bpm.isna().any():
        raise RuntimeError("missing paired FFT predictions")

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    subject = metrics_by(paired, "subject_id", "predicted_hr_bpm")
    fft_subject = metrics_by(paired, "subject_id", "fft_predicted_hr_bpm").rename(
        columns={column: f"fft_{column}" for column in hr_metrics([1, 2], [1, 2])}
    )
    subject = subject.merge(fft_subject, on="subject_id", validate="one_to_one")
    subject["mae_improvement_bpm"] = subject.fft_mae_bpm - subject.mae_bpm
    subject.to_csv(output / "subject_metrics.csv", index=False)
    activity = metrics_by(paired, "activity", "predicted_hr_bpm")
    activity.to_csv(output / "activity_metrics.csv", index=False)

    pooled_timeppg = hr_metrics(paired.reference_hr_bpm, paired.predicted_hr_bpm)
    pooled_fft = hr_metrics(paired.reference_hr_bpm, paired.fft_predicted_hr_bpm)
    macro_mae = float(subject.mae_bpm.mean())
    macro_sd = float(subject.mae_bpm.std(ddof=1))
    ci_low, ci_high = bootstrap_subject_macro_ci(subject.mae_bpm.to_numpy(), args.seed)
    summary = {
        "dataset": dataset, "seed": args.seed, "folds": args.folds,
        "unique_test_subjects": int(subject.subject_id.nunique()),
        "unique_test_windows": int(len(paired)),
        "each_subject_tested_once": True,
        "timeppg_pooled": pooled_timeppg,
        "fft_pooled_same_windows": pooled_fft,
        "timeppg_subject_macro_mae_bpm": macro_mae,
        "timeppg_subject_mae_sd_bpm": macro_sd,
        "timeppg_subject_bootstrap_95_ci_bpm": [ci_low, ci_high],
        "fft_subject_macro_mae_bpm": float(subject.fft_mae_bpm.mean()),
        "fold_mae_mean_bpm": float(fold_metrics.mae_bpm.mean()),
        "fold_mae_sd_bpm": float(fold_metrics.mae_bpm.std(ddof=1)),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ordered = subject.sort_values("mae_bpm")
    x = np.arange(len(ordered)); width = .38
    ax.bar(x-width/2, ordered.mae_bpm, width, label="TimePPG", color="#0072B2")
    ax.bar(x+width/2, ordered.fft_mae_bpm, width, label="FFT", color="#D55E00")
    ax.set_xticks(x, ordered.subject_id, rotation=45); ax.set_ylabel("MAE (bpm)")
    ax.set_title("PPG-DaLiA: unseen-subject MAE across all five folds")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(figures / "all_subject_mae.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(fold_metrics.test_fold.astype(str), fold_metrics.mae_bpm, color="#009E73")
    ax.axhline(fold_metrics.mae_bpm.mean(), color="black", linestyle="--", label="Fold mean")
    ax.set(xlabel="Test fold", ylabel="MAE (bpm)", title="TimePPG performance across subject folds")
    ax.legend(); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    fig.savefig(figures / "fold_mae.png", dpi=200); plt.close(fig)

    hardest = activity.sort_values("mae_bpm", ascending=False).iloc[0]
    report = f"""# PPG-DaLiA five-fold TimePPG result — seed {args.seed}

## Integrity

- All {args.folds} runs completed with disjoint training, validation, and test subjects.
- All {len(subject)} subjects appear in exactly one test fold.
- The combined test predictions contain {len(paired):,} unique windows, matching the frozen PPG-DaLiA manifest.
- Checkpoints were selected by validation loss; each test fold was evaluated once.

## Results

- Subject-macro MAE: **{macro_mae:.3f} +/- {macro_sd:.3f} bpm** across {len(subject)} subjects.
- Subject-bootstrap 95% CI for macro MAE: **{ci_low:.3f} to {ci_high:.3f} bpm**.
- Pooled-window MAE: **{pooled_timeppg['mae_bpm']:.3f} bpm**; RMSE: **{pooled_timeppg['rmse_bpm']:.3f} bpm**.
- Pooled Pearson r: **{pooled_timeppg['pearson_r']:.4f}**; within +/-5 bpm: **{pooled_timeppg['within_5_bpm_percent']:.2f}%**.
- Same-window FFT pooled MAE: **{pooled_fft['mae_bpm']:.3f} bpm**.
- TimePPG pooled MAE reduction relative to FFT: **{pooled_fft['mae_bpm']-pooled_timeppg['mae_bpm']:.3f} bpm ({100*(pooled_fft['mae_bpm']-pooled_timeppg['mae_bpm'])/pooled_fft['mae_bpm']:.1f}%)**.
- Hardest pooled activity: **{hardest.activity} ({hardest.mae_bpm:.3f} bpm MAE)**.

## Correct interpretation

This is the complete five-fold within-dataset PPG-DaLiA result for one random
seed. It supports unseen-subject generalisation within PPG-DaLiA and establishes
the familiar-domain reference for its later LODO generalisation gap. It does not
demonstrate transfer to a different dataset/device. Seed sensitivity and the
four leave-one-dataset-out experiments remain outstanding.
"""
    (output / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to {output}")


if __name__ == "__main__":
    main()
