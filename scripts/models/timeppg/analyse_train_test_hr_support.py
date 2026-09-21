#!/usr/bin/env python3
"""Compare the HR-label support used for within-dataset and full-LODO tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("reports/phase9_timeppg")
FROZEN = Path("reports/phase7_frozen_dataset/main_window_manifest.csv")
OUT = Path("reports/phase10_lodo_failure_analysis")
SEED = 17
TARGETS = {
    "bidmc": "BIDMC",
    "wesad": "WESAD",
    "ptt_ppg": "PTT-PPG",
    "ppg_dalia": "PPG-DaLiA",
}
DEVICE = {
    "BIDMC": "clinical PLETH, ICU, 125 Hz",
    "WESAD": "Empatica E4 wrist BVP, 64 Hz",
    "PTT-PPG": "MAX30101 finger pleth_1, 500→64 Hz",
    "PPG-DaLiA": "Empatica E4 wrist BVP, 64 Hz",
}
BINS = np.arange(35, 225, 5)


def describe(values: pd.Series, **identity) -> dict:
    x = values.dropna().to_numpy(float)
    return {
        **identity,
        "windows": len(x),
        "mean_hr_bpm": np.mean(x),
        "median_hr_bpm": np.median(x),
        "sd_hr_bpm": np.std(x),
        "q01_hr_bpm": np.quantile(x, .01),
        "q05_hr_bpm": np.quantile(x, .05),
        "q95_hr_bpm": np.quantile(x, .95),
        "q99_hr_bpm": np.quantile(x, .99),
        "below_60_percent": 100 * np.mean(x < 60),
        "above_120_percent": 100 * np.mean(x >= 120),
        "above_140_percent": 100 * np.mean(x >= 140),
    }


def density(values: pd.Series) -> np.ndarray:
    counts, _ = np.histogram(values.to_numpy(float), bins=BINS)
    return counts / counts.sum() if counts.sum() else counts.astype(float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    frozen = pd.read_csv(FROZEN, low_memory=False)
    frozen = frozen.loc[frozen.primary_analysis.astype(bool)].copy()
    centres = (BINS[:-1] + BINS[1:]) / 2
    summaries, coverage_rows = [], []
    fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharex=True)

    for row, (key, target) in enumerate(TARGETS.items()):
        target_frozen = frozen.loc[frozen.dataset.eq(target)]
        within_train_densities, within_test_densities = [], []

        # Each within model has its own subject-disjoint train and test set.
        for fold in range(3):
            metric_path = ROOT / "within_dataset_3fold" / key / f"test_fold_{fold}" / f"seed_{SEED}" / "metrics.json"
            metrics = json.loads(metric_path.read_text())
            train_subjects = set(map(str, metrics["subjects"]["train"]))
            test_subjects = set(map(str, metrics["subjects"]["test"]))
            train = target_frozen.loc[target_frozen.subject_id.astype(str).isin(train_subjects)]
            test = target_frozen.loc[target_frozen.subject_id.astype(str).isin(test_subjects)]
            within_train_densities.append(density(train.hr_bpm))
            within_test_densities.append(density(test.hr_bpm))
            summaries.append(describe(train.hr_bpm, target=target, experiment="within", role="train", fold=fold,
                                      datasets=target, subjects=train.subject_id.nunique()))
            summaries.append(describe(test.hr_bpm, target=target, experiment="within", role="test", fold=fold,
                                      datasets=target, subjects=test.subject_id.nunique()))

        train_stack = np.vstack(within_train_densities)
        test_stack = np.vstack(within_test_densities)
        ax = axes[row, 0]
        ax.fill_between(centres, train_stack.min(0), train_stack.max(0), color="#2A9D8F", alpha=.18)
        ax.plot(centres, train_stack.mean(0), color="#2A9D8F", label="Train: mean across 3 folds")
        ax.plot(centres, test_stack.mean(0), color="#132A3A", linestyle="--", label="Test: mean across 3 folds")
        ax.set_title(f"{target} — within dataset")

        # Full LODO uses the saved, actual source-training manifest.
        source_path = ROOT / "lodo_full" / f"target_{key}" / f"seed_{SEED}" / "sampled_source_train_manifest.csv"
        source = pd.read_csv(source_path, low_memory=False)
        prediction_path = ROOT / "lodo_full" / f"target_{key}" / f"seed_{SEED}" / "target_test_predictions.csv"
        target_test = pd.read_csv(prediction_path, usecols=["subject_id", "reference_hr_bpm"])
        source_names = ", ".join(sorted(source.dataset.unique()))
        summaries.append(describe(source.hr_bpm, target=target, experiment="LODO", role="train", fold="all",
                                  datasets=source_names, subjects=(source.dataset.astype(str) + ":" + source.subject_id.astype(str)).nunique()))
        summaries.append(describe(target_test.reference_hr_bpm, target=target, experiment="LODO", role="test", fold="all",
                                  datasets=target, subjects=target_test.subject_id.nunique()))

        ax = axes[row, 1]
        ax.plot(centres, density(source.hr_bpm), color="#E76F51", linewidth=2, label="Combined source train")
        for source_name, group in source.groupby("dataset"):
            ax.plot(centres, density(group.hr_bpm), linewidth=1, alpha=.45, label=str(source_name))
        ax.plot(centres, density(target_test.reference_hr_bpm), color="#132A3A", linestyle="--", linewidth=2,
                label="Untouched target test")
        ax.set_title(f"{target} held out — LODO")
        device_box = "TRAIN\n" + "\n".join(
            f"{name}: {DEVICE[name]}" for name in sorted(source.dataset.unique())
        ) + f"\n\nTEST\n{target}: {DEVICE[target]}"
        ax.text(
            .985, .52, device_box,
            transform=ax.transAxes, ha="right", va="center", fontsize=7.2,
            linespacing=1.25,
            bbox={"boxstyle": "round,pad=.55", "facecolor": "white",
                  "edgecolor": "#9FB1BA", "alpha": .94},
        )

        lo, hi = np.quantile(source.hr_bpm, [.01, .99])
        target_hr = target_test.reference_hr_bpm.to_numpy(float)
        coverage_rows.append({
            "target": target,
            "lodo_source_datasets": source_names,
            "source_q01_hr_bpm": lo,
            "source_q99_hr_bpm": hi,
            "target_below_source_q01_percent": 100 * np.mean(target_hr < lo),
            "target_above_source_q99_percent": 100 * np.mean(target_hr > hi),
            "target_outside_source_1_99_percent": 100 * np.mean((target_hr < lo) | (target_hr > hi)),
        })

        for current in axes[row]:
            current.axvline(80, color="grey", linewidth=.7, linestyle=":")
            current.axvline(120, color="grey", linewidth=.7, linestyle=":")
            current.set_ylabel("Proportion of windows")
            current.grid(alpha=.18)

    for ax in axes[-1]:
        ax.set_xlabel("ECG-derived reference HR (bpm)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Training label support versus the HR labels encountered at test time", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, .98])
    fig.savefig(figures / "train_test_hr_label_distributions.png", dpi=220)
    plt.close(fig)

    pd.DataFrame(summaries).to_csv(OUT / "train_test_hr_distribution_summary.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(OUT / "lodo_target_tail_coverage.csv", index=False)
    print(pd.DataFrame(coverage_rows).to_string(index=False))


if __name__ == "__main__":
    main()
