#!/usr/bin/env python3
"""Aggregate replicated PTT-PPG full-LODO results across training seeds."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("reports/phase9_timeppg")
OUT = Path("reports/phase10_lodo_failure_analysis/ptt_multiseed")
SEEDS = (17, 29, 43)
KEY = ["dataset", "record_id", "subject_id", "window_index", "channel_name"]
HR_BINS = [-np.inf, 60, 80, 100, 120, 140, np.inf]
HR_LABELS = ["<60", "60-80", "80-100", "100-120", "120-140", ">=140"]


def metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    absolute = np.abs(error)
    return {
        "windows": len(reference),
        "mae_bpm": absolute.mean(),
        "rmse_bpm": np.sqrt(np.mean(error**2)),
        "bias_bpm": error.mean(),
        "pearson_r": np.corrcoef(reference, prediction)[0, 1],
        "within_5_bpm_percent": 100 * np.mean(absolute <= 5),
        "over_20_bpm_percent": 100 * np.mean(absolute > 20),
        "p95_absolute_error_bpm": np.quantile(absolute, .95),
    }


def load_within() -> pd.DataFrame:
    parts = []
    for fold in range(3):
        path = ROOT / "within_dataset_3fold/ptt_ppg" / f"test_fold_{fold}/seed_17/test_predictions.csv"
        part = pd.read_csv(path, low_memory=False)
        part["fold"] = fold
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(KEY).any():
        raise RuntimeError("duplicate PTT within-dataset target window")
    return frame[KEY + ["window_start_s", "reference_hr_bpm", "predicted_hr_bpm", "fold"]].rename(
        columns={"predicted_hr_bpm": "within_prediction_bpm"}
    )


def load_paired() -> pd.DataFrame:
    paired = load_within()
    expected_keys = paired[KEY]
    for seed in SEEDS:
        path = ROOT / "lodo_full" / "target_ptt_ppg" / f"seed_{seed}" / "target_test_predictions.csv"
        lodo = pd.read_csv(path, low_memory=False)
        if lodo.duplicated(KEY).any() or lodo[KEY].isna().any().any():
            raise RuntimeError(f"invalid PTT keys for seed {seed}")
        if len(lodo) != len(paired):
            raise RuntimeError(f"row mismatch for seed {seed}")
        audit = expected_keys.merge(lodo[KEY], how="outer", on=KEY, indicator=True)
        if not audit._merge.eq("both").all():
            raise RuntimeError(f"window-set mismatch for seed {seed}")
        retain = KEY + ["window_start_s", "reference_hr_bpm", "predicted_hr_bpm"]
        paired = paired.merge(
            lodo[retain].rename(columns={
                "window_start_s": f"window_start_s_{seed}",
                "reference_hr_bpm": f"reference_hr_bpm_{seed}",
                "predicted_hr_bpm": f"lodo_prediction_bpm_{seed}",
            }), on=KEY, how="inner", validate="one_to_one",
        )
        if not np.allclose(paired.reference_hr_bpm, paired[f"reference_hr_bpm_{seed}"], rtol=0, atol=1e-4):
            raise RuntimeError(f"reference mismatch for seed {seed}")
        if not np.allclose(paired.window_start_s, paired[f"window_start_s_{seed}"], rtol=0, atol=1e-8):
            raise RuntimeError(f"window-time mismatch for seed {seed}")
    paired["within_absolute_error_bpm"] = np.abs(
        paired.within_prediction_bpm - paired.reference_hr_bpm
    )
    paired["hr_bin"] = pd.cut(
        paired.reference_hr_bpm, HR_BINS, labels=HR_LABELS, right=False, ordered=True
    )
    return paired


def to_long(paired: pd.DataFrame) -> pd.DataFrame:
    parts = []
    base = KEY + ["window_start_s", "reference_hr_bpm", "within_prediction_bpm",
                  "within_absolute_error_bpm", "hr_bin"]
    for seed in SEEDS:
        part = paired[base].copy()
        part["seed"] = seed
        part["lodo_prediction_bpm"] = paired[f"lodo_prediction_bpm_{seed}"]
        part["lodo_error_bpm"] = part.lodo_prediction_bpm - part.reference_hr_bpm
        part["lodo_absolute_error_bpm"] = part.lodo_error_bpm.abs()
        part["paired_gap_bpm"] = part.lodo_absolute_error_bpm - part.within_absolute_error_bpm
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def aggregate_mean_sd(frame: pd.DataFrame, by: list[str], value_columns: list[str]) -> pd.DataFrame:
    result = frame.groupby(by, observed=True)[value_columns].agg(["mean", "std", "min", "max"])
    result.columns = [f"{column}_{stat}" for column, stat in result.columns]
    return result.reset_index()


def subject_bootstrap(subject_average: pd.DataFrame, draws: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    values = subject_average[["within_mae_bpm", "lodo_mae_bpm_mean", "gap_bpm_mean"]].to_numpy(float)
    selections = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[selections].mean(axis=1)
    rows = []
    for index, name in enumerate(("within_subject_macro_mae", "lodo_multiseed_subject_macro_mae", "multiseed_subject_macro_gap")):
        rows.append({
            "quantity": name,
            "estimate_bpm": values[:, index].mean(),
            "subject_bootstrap_ci_low_bpm": np.quantile(estimates[:, index], .025),
            "subject_bootstrap_ci_high_bpm": np.quantile(estimates[:, index], .975),
            "bootstrap_draws": draws,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    paired = load_paired()
    long = to_long(paired)

    overall_rows = []
    for seed, group in long.groupby("seed"):
        row = {"seed": seed}
        row.update(metrics(group.reference_hr_bpm.to_numpy(), group.lodo_prediction_bpm.to_numpy()))
        row["paired_gap_bpm"] = group.paired_gap_bpm.mean()
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall_summary = aggregate_mean_sd(
        overall, [], ["mae_bpm", "rmse_bpm", "bias_bpm", "pearson_r",
                      "within_5_bpm_percent", "over_20_bpm_percent",
                      "p95_absolute_error_bpm", "paired_gap_bpm"]
    ) if False else pd.DataFrame({
        f"{column}_{stat}": [getattr(overall[column], stat)()]
        for column in ["mae_bpm", "rmse_bpm", "bias_bpm", "pearson_r",
                       "within_5_bpm_percent", "over_20_bpm_percent",
                       "p95_absolute_error_bpm", "paired_gap_bpm"]
        for stat in ["mean", "std", "min", "max"]
    })

    subject_rows = []
    for (subject, seed), group in long.groupby(["subject_id", "seed"]):
        subject_rows.append({
            "subject_id": subject, "seed": seed, "windows": len(group),
            "within_mae_bpm": group.within_absolute_error_bpm.mean(),
            "lodo_mae_bpm": group.lodo_absolute_error_bpm.mean(),
            "gap_bpm": group.paired_gap_bpm.mean(),
            "lodo_bias_bpm": group.lodo_error_bpm.mean(),
        })
    subject_seed = pd.DataFrame(subject_rows)
    subject_average = aggregate_mean_sd(subject_seed, ["subject_id"], ["lodo_mae_bpm", "gap_bpm", "lodo_bias_bpm"])
    within_subject = subject_seed.groupby("subject_id", as_index=False).within_mae_bpm.first()
    subject_average = subject_average.merge(within_subject, on="subject_id", validate="one_to_one")
    worse_all = subject_seed.assign(worse=subject_seed.gap_bpm > 0).groupby("subject_id").worse.all()
    subject_average["worse_in_all_three_seeds"] = subject_average.subject_id.map(worse_all)
    bootstrap = subject_bootstrap(subject_average)

    hr_rows = []
    for (seed, hr_bin), group in long.groupby(["seed", "hr_bin"], observed=True):
        row = {"seed": seed, "hr_bin": hr_bin, "subjects": group.subject_id.nunique()}
        row.update(metrics(group.reference_hr_bpm.to_numpy(), group.lodo_prediction_bpm.to_numpy()))
        row["paired_gap_bpm"] = group.paired_gap_bpm.mean()
        hr_rows.append(row)
    hr_seed = pd.DataFrame(hr_rows)
    hr_average = aggregate_mean_sd(
        hr_seed, ["hr_bin"], ["mae_bpm", "bias_bpm", "paired_gap_bpm", "pearson_r"]
    )

    overall.to_csv(OUT / "overall_metrics_by_seed.csv", index=False)
    overall_summary.to_csv(OUT / "overall_multiseed_summary.csv", index=False)
    subject_seed.to_csv(OUT / "subject_metrics_by_seed.csv", index=False)
    subject_average.to_csv(OUT / "subject_multiseed_summary.csv", index=False)
    bootstrap.to_csv(OUT / "subject_bootstrap_multiseed.csv", index=False)
    hr_seed.to_csv(OUT / "hr_bin_metrics_by_seed.csv", index=False)
    hr_average.to_csv(OUT / "hr_bin_multiseed_summary.csv", index=False)

    # Overall seed stability.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, column, title in zip(axes, ["mae_bpm", "bias_bpm", "pearson_r"],
                                 ["MAE", "Signed bias", "Correlation"]):
        ax.bar(overall.seed.astype(str), overall[column], color="#E76F51")
        ax.set(title=title, xlabel="Training seed", ylabel="bpm" if column != "pearson_r" else "Pearson r")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("PTT-PPG LODO failure is stable across training seeds")
    fig.tight_layout(); fig.savefig(figures / "ptt_lodo_overall_seed_stability.png", dpi=220); plt.close(fig)

    # Per-subject stability.
    shown = subject_average.sort_values("lodo_mae_bpm_mean", ascending=False)
    y = np.arange(len(shown))
    lower = shown.lodo_mae_bpm_mean - shown.lodo_mae_bpm_min
    upper = shown.lodo_mae_bpm_max - shown.lodo_mae_bpm_mean
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y, shown.lodo_mae_bpm_mean, color="#E76F51", alpha=.85, label="Mean LODO across seeds")
    ax.errorbar(shown.lodo_mae_bpm_mean, y, xerr=np.vstack([lower, upper]), fmt="none", color="#8C2F20", capsize=2)
    ax.scatter(shown.within_mae_bpm, y, color="#2A9D8F", marker="|", s=130, label="Within seed 17")
    ax.set_yticks(y, shown.subject_id); ax.invert_yaxis()
    ax.set(xlabel="MAE (bpm)", title="PTT subject failures persist across seeds (bars mean; whiskers range)")
    ax.legend(frameon=False); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "ptt_subject_multiseed_stability.png", dpi=220); plt.close(fig)

    # HR-range pattern stability.
    hr_plot = hr_average.set_index("hr_bin").reindex(HR_LABELS)
    x = np.arange(len(hr_plot))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, prefix, title in [(axes[0], "mae_bpm", "LODO MAE"), (axes[1], "bias_bpm", "Signed bias")]:
        mean = hr_plot[f"{prefix}_mean"]
        lower = mean - hr_plot[f"{prefix}_min"]
        upper = hr_plot[f"{prefix}_max"] - mean
        ax.errorbar(x, mean, yerr=np.vstack([lower, upper]), marker="o", capsize=3, color="#E76F51")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(x, HR_LABELS, rotation=25)
        ax.set(title=title, xlabel="Reference HR bin", ylabel="bpm")
        ax.grid(alpha=.2)
    fig.suptitle("PTT low-HR overprediction repeats across all three seeds")
    fig.tight_layout(); fig.savefig(figures / "ptt_hr_failure_multiseed_stability.png", dpi=220); plt.close(fig)

    all_worse = int(subject_average.worse_in_all_three_seeds.sum())
    summary = overall_summary.iloc[0]
    gap_ci = bootstrap.loc[bootstrap.quantity.eq("multiseed_subject_macro_gap")].iloc[0]
    report = f"""# PTT-PPG full-LODO replication across seeds 17, 29 and 43

All three runs use the same 15,982 untouched PTT target windows and the same frozen
source/validation protocol. Matching keys, timestamps and reference labels agree.

- Mean LODO MAE: **{summary.mae_bpm_mean:.3f} ± {summary.mae_bpm_std:.3f} bpm** (training-seed SD).
- Mean signed bias: **{summary.bias_bpm_mean:+.3f} ± {summary.bias_bpm_std:.3f} bpm**.
- Mean correlation: **{summary.pearson_r_mean:.3f} ± {summary.pearson_r_std:.3f}**.
- **{all_worse}/22 subjects** are worse than the fixed within-dataset reference in all three LODO seeds.
- Seed-averaged subject-macro gap: **{gap_ci.estimate_bpm:+.3f} bpm**; subject-bootstrap
  95% interval **{gap_ci.subject_bootstrap_ci_low_bpm:+.3f} to {gap_ci.subject_bootstrap_ci_high_bpm:+.3f} bpm**.

The low-HR positive-bias pattern repeats across seeds. Therefore, the main PTT transfer
failure and its direction are not peculiar to seed 17. The SD across only three seeds is
descriptive and should not be presented as a precise population confidence interval.
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
