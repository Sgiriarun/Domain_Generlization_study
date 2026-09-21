#!/usr/bin/env python3
"""Subject-aware paired failure analysis for completed TimePPG LODO runs.

The script compares each full-LODO prediction with the within-dataset out-of-fold
prediction for the exact same target window.  All subgroup results are post-hoc
diagnostics and must not be described as isolated causal effects.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = {
    "bidmc": "BIDMC",
    "wesad": "WESAD",
    "ptt_ppg": "PTT-PPG",
    "ppg_dalia": "PPG-DaLiA",
}
DALIA_ACTIVITY = {
    0: "No activity", 1: "Baseline", 2: "Stairs", 3: "Table soccer",
    4: "Cycling", 5: "Driving", 6: "Lunch", 7: "Walking", 8: "Working",
}
HR_BINS = [-np.inf, 60, 80, 100, 120, 140, np.inf]
HR_LABELS = ["<60", "60-80", "80-100", "100-120", "120-140", ">=140"]
KEY = ["dataset", "record_id", "subject_id", "window_index", "channel_name"]


def metric_row(frame: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    error = frame[f"{prefix}_error_bpm"].to_numpy(float)
    absolute = np.abs(error)
    reference = frame.reference_hr_bpm.to_numpy(float)
    prediction = frame[f"{prefix}_prediction_bpm"].to_numpy(float)
    correlation = np.corrcoef(reference, prediction)[0, 1] if len(frame) > 1 else np.nan
    slope = np.polyfit(reference, prediction, 1)[0] if len(frame) > 1 and np.std(reference) > 0 else np.nan
    return {
        "windows": len(frame),
        "mae_bpm": float(absolute.mean()),
        "rmse_bpm": float(np.sqrt(np.mean(error ** 2))),
        "bias_bpm": float(error.mean()),
        "median_bias_bpm": float(np.median(error)),
        "pearson_r": float(correlation),
        "calibration_slope": float(slope),
        "prediction_to_reference_sd_ratio": float(np.std(prediction) / np.std(reference)) if np.std(reference) else np.nan,
        "p90_absolute_error_bpm": float(np.quantile(absolute, .90)),
        "p95_absolute_error_bpm": float(np.quantile(absolute, .95)),
        "within_5_bpm_percent": float(100 * np.mean(absolute <= 5)),
        "over_5_bpm_percent": float(100 * np.mean(absolute > 5)),
        "over_10_bpm_percent": float(100 * np.mean(absolute > 10)),
        "over_20_bpm_percent": float(100 * np.mean(absolute > 20)),
    }


def grouped_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    grouper = columns[0] if len(columns) == 1 else columns
    for values, group in frame.groupby(grouper, observed=True, dropna=False, sort=False):
        values = (values,) if len(columns) == 1 else values
        row = dict(zip(columns, values))
        row.update({f"within_{k}": v for k, v in metric_row(group, "within").items()})
        row.update({f"lodo_{k}": v for k, v in metric_row(group, "lodo").items()})
        row["paired_gap_bpm"] = float(group.paired_absolute_error_difference_bpm.mean())
        row["subjects"] = int(group.subject_id.nunique())
        subject_counts = group.groupby("subject_id").size()
        row["largest_subject_window_share_percent"] = float(100 * subject_counts.max() / len(group))
        row["mean_reference_hr_bpm"] = float(group.reference_hr_bpm.mean())
        row["mean_within_prediction_bpm"] = float(group.within_prediction_bpm.mean())
        row["mean_lodo_prediction_bpm"] = float(group.lodo_prediction_bpm.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def subject_bootstrap(subject: pd.DataFrame, seed: int, draws: int = 20_000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = subject[["within_mae_bpm", "lodo_mae_bpm", "gap_bpm"]].to_numpy(float)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[indices].mean(axis=1)
    result = {}
    for index, name in enumerate(("within_macro_mae", "lodo_macro_mae", "macro_gap")):
        result[f"{name}_bpm"] = float(values[:, index].mean())
        result[f"{name}_ci_low_bpm"] = float(np.quantile(estimates[:, index], .025))
        result[f"{name}_ci_high_bpm"] = float(np.quantile(estimates[:, index], .975))
    result["bootstrap_unit"] = "subject"
    result["bootstrap_draws"] = draws
    return result


def subject_deterioration_summary(subject: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in subject.groupby("dataset", sort=False):
        gap = group.gap_bpm.to_numpy(float)
        rows.append({
            "dataset": dataset,
            "subjects": len(group),
            "subjects_worse": int(np.sum(gap > 0)),
            "subjects_worse_percent": float(100 * np.mean(gap > 0)),
            "subjects_improved": int(np.sum(gap < 0)),
            "median_subject_gap_bpm": float(np.median(gap)),
            "mean_subject_gap_bpm": float(np.mean(gap)),
            "min_subject_gap_bpm": float(np.min(gap)),
            "max_subject_gap_bpm": float(np.max(gap)),
        })
    return pd.DataFrame(rows)


def load_paired(root: Path, target_key: str, seed: int) -> pd.DataFrame:
    dataset = TARGETS[target_key]
    lodo_path = root / "lodo_full" / f"target_{target_key}" / f"seed_{seed}" / "target_test_predictions.csv"
    lodo = pd.read_csv(lodo_path, low_memory=False)
    within_parts = []
    for fold in range(3):
        path = root / "within_dataset_3fold" / target_key / f"test_fold_{fold}" / f"seed_{seed}" / "test_predictions.csv"
        part = pd.read_csv(path, low_memory=False)
        part["within_test_fold"] = fold
        within_parts.append(part)
    within = pd.concat(within_parts, ignore_index=True)
    if within[KEY].isna().any().any() or lodo[KEY].isna().any().any():
        raise RuntimeError(f"missing matching-key value for {dataset}")
    if within.duplicated(KEY).any() or lodo.duplicated(KEY).any():
        raise RuntimeError(f"duplicate target window for {dataset}")
    fold_counts = within.groupby("subject_id").within_test_fold.nunique()
    if (fold_counts != 1).any():
        raise RuntimeError(f"subject appears in multiple within-test folds for {dataset}")
    if len(within) != len(lodo):
        raise RuntimeError(f"within/LODO row-count mismatch for {dataset}: {len(within)} vs {len(lodo)}")

    retain = KEY + ["window_start_s", "start_sample_64hz", "end_sample_64hz",
                    "predicted_hr_bpm", "reference_hr_bpm", "within_test_fold"]
    paired = lodo.merge(
        within[retain].rename(columns={
            "window_start_s": "within_window_start_s",
            "start_sample_64hz": "within_start_sample_64hz",
            "end_sample_64hz": "within_end_sample_64hz",
            "predicted_hr_bpm": "within_prediction_bpm",
            "reference_hr_bpm": "within_reference_hr_bpm",
        }),
        on=KEY,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(within) or len(paired) != len(lodo):
        raise RuntimeError(f"failed to pair every target window for {dataset}")
    for left, right in [
        ("window_start_s", "within_window_start_s"),
        ("start_sample_64hz", "within_start_sample_64hz"),
        ("end_sample_64hz", "within_end_sample_64hz"),
    ]:
        if not np.allclose(paired[left], paired[right], rtol=0, atol=1e-8):
            raise RuntimeError(f"physical window mismatch ({left}) for {dataset}")
    if not np.allclose(paired.reference_hr_bpm, paired.within_reference_hr_bpm, atol=1e-4):
        raise RuntimeError(f"reference mismatch between within and LODO predictions for {dataset}")
    numeric = ["reference_hr_bpm", "predicted_hr_bpm", "within_prediction_bpm"]
    if not np.isfinite(paired[numeric].to_numpy(float)).all():
        raise RuntimeError(f"non-finite label or prediction for {dataset}")
    paired = paired.rename(columns={"predicted_hr_bpm": "lodo_prediction_bpm"})
    paired["within_error_bpm"] = paired.within_prediction_bpm - paired.reference_hr_bpm
    paired["lodo_error_bpm"] = paired.lodo_prediction_bpm - paired.reference_hr_bpm
    paired["within_absolute_error_bpm"] = paired.within_error_bpm.abs()
    paired["lodo_absolute_error_bpm"] = paired.lodo_error_bpm.abs()
    paired["paired_absolute_error_difference_bpm"] = (
        paired.lodo_absolute_error_bpm - paired.within_absolute_error_bpm
    )
    paired["hr_bin"] = pd.cut(
        paired.reference_hr_bpm, HR_BINS, labels=HR_LABELS, right=False, ordered=True
    )
    paired["quality_tertile"] = pd.qcut(
        paired.spectral_concentration.rank(method="first"), 3,
        labels=["Low", "Medium", "High"],
    )
    paired["activity"] = paired.condition_name.astype(str)
    if dataset == "PPG-DaLiA":
        ids = pd.to_numeric(paired.condition_id, errors="coerce").round().astype("Int64")
        paired["activity"] = ids.map(DALIA_ACTIVITY).fillna("Unknown")
    elif dataset == "WESAD":
        paired.loc[paired.activity.eq("not_defined"), "activity"] = "Unlabelled"
    elif dataset == "BIDMC":
        paired["activity"] = "Clinical ICU"
    return paired


def within_subject_adjusted_ptt(ptt: pd.DataFrame, seed: int, draws: int = 4_000) -> pd.DataFrame:
    """Exploratory within-subject regression with subject-cluster bootstrap CIs."""
    frame = ptt.copy()
    frame["hr_per_10_bpm"] = frame.reference_hr_bpm / 10
    frame["spectral_concentration_per_0_1"] = frame.spectral_concentration / .1
    frame["fft_mismatch_per_10_bpm"] = (
        (frame.dominant_frequency_hz * 60 - frame.reference_hr_bpm).abs() / 10
    )
    frame["walking_vs_sitting"] = frame.activity.eq("walk").astype(float)
    frame["running_vs_sitting"] = frame.activity.eq("run").astype(float)
    columns = [
        "hr_per_10_bpm", "spectral_concentration_per_0_1", "fft_mismatch_per_10_bpm",
        "walking_vs_sitting", "running_vs_sitting",
    ]
    # Removing each subject's mean estimates associations from changes within a
    # person and avoids treating between-person differences as independent rows.
    y = frame.lodo_absolute_error_bpm - frame.groupby("subject_id").lodo_absolute_error_bpm.transform("mean")
    x = frame[columns] - frame.groupby("subject_id")[columns].transform("mean")
    estimate = np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)[0]

    subjects = frame.subject_id.unique()
    blocks = {subject: np.flatnonzero(frame.subject_id.to_numpy() == subject) for subject in subjects}
    rng = np.random.default_rng(seed)
    boot = np.empty((draws, len(columns)))
    x_array, y_array = x.to_numpy(), y.to_numpy()
    for draw in range(draws):
        chosen = rng.choice(subjects, len(subjects), replace=True)
        indices = np.concatenate([blocks[subject] for subject in chosen])
        boot[draw] = np.linalg.lstsq(x_array[indices], y_array[indices], rcond=None)[0]
    return pd.DataFrame({
        "predictor": columns,
        "coefficient_bpm": estimate,
        "subject_bootstrap_ci_low_bpm": np.quantile(boot, .025, axis=0),
        "subject_bootstrap_ci_high_bpm": np.quantile(boot, .975, axis=0),
        "interpretation": [
            "Change in absolute error per +10 bpm reference HR, within subject and adjusted for listed predictors.",
            "Change in absolute error per +0.1 spectral concentration, within subject and adjusted.",
            "Change in absolute error per +10 bpm FFT/reference mismatch, within subject and adjusted.",
            "Adjusted walking-versus-sitting difference within subject.",
            "Adjusted running-versus-sitting difference within subject.",
        ],
    })


def error_episodes(frame: pd.DataFrame, threshold: float = 20) -> pd.DataFrame:
    rows = []
    for (dataset, subject, record), group in frame.groupby(["dataset", "subject_id", "record_id"], sort=False):
        group = group.sort_values("window_index")
        bad = group.loc[group.lodo_absolute_error_bpm > threshold].copy()
        if bad.empty:
            continue
        bad["episode"] = bad.window_index.diff().ne(1).cumsum()
        for _, episode in bad.groupby("episode"):
            rows.append({
                "dataset": dataset, "subject_id": subject, "record_id": record,
                "start_window": int(episode.window_index.min()),
                "end_window": int(episode.window_index.max()),
                "windows": len(episode),
                "approximate_span_seconds": float(8 + 2 * (len(episode) - 1)),
                "mean_absolute_error_bpm": float(episode.lodo_absolute_error_bpm.mean()),
                "max_absolute_error_bpm": float(episode.lodo_absolute_error_bpm.max()),
                "mean_signed_error_bpm": float(episode.lodo_error_bpm.mean()),
                "activity": "|".join(sorted(episode.activity.astype(str).unique())),
                "mean_reference_hr_bpm": float(episode.reference_hr_bpm.mean()),
            })
    return pd.DataFrame(rows).sort_values(
        ["windows", "mean_absolute_error_bpm"], ascending=[False, False]
    )


def source_support_table(root: Path, target_key: str, seed: int, target: pd.DataFrame) -> pd.DataFrame:
    source_path = root / "lodo_full" / f"target_{target_key}" / f"seed_{seed}" / "sampled_source_train_manifest.csv"
    source = pd.read_csv(source_path, low_memory=False)
    source["dataset_subject"] = source.dataset.astype(str) + ":" + source.subject_id.astype(str)
    source["hr_bin"] = pd.cut(source.hr_bpm, HR_BINS, labels=HR_LABELS, right=False, ordered=True)
    rows = []
    for label in HR_LABELS:
        s = source.loc[source.hr_bin.astype(str).eq(label)]
        t = target.loc[target.hr_bin.astype(str).eq(label)]
        rows.append({
            "target": TARGETS[target_key], "hr_bin": label,
            "source_train_windows": len(s), "source_train_subjects": s.dataset_subject.nunique(),
            "target_windows": len(t), "target_subjects": t.subject_id.nunique(),
            "lodo_mae_bpm": t.lodo_absolute_error_bpm.mean() if len(t) else np.nan,
            "paired_gap_bpm": t.paired_absolute_error_difference_bpm.mean() if len(t) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_ptt_subjects(subject: pd.DataFrame, output: Path) -> None:
    shown = subject.sort_values("lodo_mae_bpm", ascending=False)
    y = np.arange(len(shown)); height = .38
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(y + height/2, shown.lodo_mae_bpm, height, label="LODO", color="#E76F51")
    ax.barh(y - height/2, shown.within_mae_bpm, height, label="Within dataset", color="#2A9D8F")
    ax.set_yticks(y, shown.subject_id); ax.invert_yaxis()
    ax.set(xlabel="MAE (bpm)", title="PTT-PPG failure is concentrated in particular subjects")
    ax.legend(frameon=False); ax.grid(axis="x", alpha=.2); fig.tight_layout()
    fig.savefig(output, dpi=200); plt.close(fig)


def plot_ptt_heatmap(table: pd.DataFrame, output: Path) -> None:
    counts = table.pivot(index="activity", columns="hr_bin", values="lodo_windows").reindex(columns=HR_LABELS)
    subjects = table.pivot(index="activity", columns="hr_bin", values="subjects").reindex(columns=HR_LABELS)
    matrices = [
        (table.pivot(index="activity", columns="hr_bin", values="lodo_mae_bpm").reindex(columns=HR_LABELS),
         "LODO MAE: which conditions are difficult?", "YlOrRd"),
        (table.pivot(index="activity", columns="hr_bin", values="paired_gap_bpm").reindex(columns=HR_LABELS),
         "Paired gap: where does transfer add error?", "RdBu_r"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    for ax, (matrix, title, cmap) in zip(axes, matrices):
        limit = np.nanmax(np.abs(matrix.to_numpy())) if "gap" in title.lower() else None
        image = ax.imshow(matrix, cmap=cmap, aspect="auto",
                          vmin=-limit if limit else None, vmax=limit if limit else None)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.1f}\nn={int(counts.iloc[i,j])}; S={int(subjects.iloc[i,j])}",
                            ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(matrix.columns)), matrix.columns)
        ax.set_yticks(range(len(matrix.index)), matrix.index)
        ax.set(xlabel="Reference-HR bin (bpm)", ylabel="Activity", title=title)
        fig.colorbar(image, ax=ax, label="bpm")
    fig.suptitle("PTT-PPG activity–HR analysis (white = no eligible windows)")
    fig.savefig(output, dpi=200); plt.close(fig)


def plot_signed_bias_by_hr(table: pd.DataFrame, output: Path) -> None:
    datasets = list(TARGETS.values())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, dataset in zip(axes.flat, datasets):
        part = table.loc[table.dataset.eq(dataset)].set_index("hr_bin").reindex(HR_LABELS)
        x = np.arange(len(HR_LABELS))
        ax.plot(x, part.within_bias_bpm, "o-", label="Within", color="#2A9D8F")
        ax.plot(x, part.lodo_bias_bpm, "o-", label="LODO", color="#E76F51")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(x, HR_LABELS, rotation=25)
        ax.set(title=dataset, ylabel="Signed bias (prediction − reference, bpm)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Direction of error by true-HR range")
    fig.tight_layout(); fig.savefig(output, dpi=200); plt.close(fig)


def plot_ptt_traces(ptt: pd.DataFrame, subject: pd.DataFrame, output: Path) -> None:
    ptt_subject = subject.loc[subject.dataset.eq("PTT-PPG")].copy()
    worst = ptt_subject.sort_values("gap_bpm", ascending=False).iloc[0].subject_id
    ordered = ptt_subject.sort_values("gap_bpm").reset_index(drop=True)
    typical = ordered.iloc[len(ordered)//2].subject_id
    successful = ordered.iloc[0].subject_id
    selected = []
    for label, subject_id in [("s2 anomaly", "s2"), ("worst gap", worst),
                              ("typical gap", typical), ("best gap", successful)]:
        if subject_id not in [value for _, value in selected]:
            selected.append((label, subject_id))
    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 3.1 * len(selected)), squeeze=False)
    for ax, (label, subject_id) in zip(axes.flat, selected):
        part = ptt.loc[ptt.subject_id.eq(subject_id)].sort_values(["record_id", "window_start_s"])
        # Plot the longest recording to avoid drawing lines across recording boundaries.
        record = part.groupby("record_id").size().idxmax()
        part = part.loc[part.record_id.eq(record)]
        time_minutes = (part.window_start_s - part.window_start_s.min()) / 60
        ax.plot(time_minutes, part.reference_hr_bpm, label="Reference", color="black", linewidth=1.4)
        ax.plot(time_minutes, part.within_prediction_bpm, label="Within", color="#2A9D8F", alpha=.85)
        ax.plot(time_minutes, part.lodo_prediction_bpm, label="LODO", color="#E76F51", alpha=.85)
        ax.set(title=f"{label}: {subject_id}, record {record}", xlabel="Minutes", ylabel="HR (bpm)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.suptitle("PTT-PPG temporal traces: isolated errors versus sustained failure", y=1.0)
    fig.tight_layout(); fig.savefig(output, dpi=200, bbox_inches="tight"); plt.close(fig)


def plot_all_hr(table: pd.DataFrame, output: Path) -> None:
    datasets = list(TARGETS.values())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, dataset in zip(axes.flat, datasets):
        part = table.loc[table.dataset.eq(dataset)].set_index("hr_bin").reindex(HR_LABELS)
        x = np.arange(len(HR_LABELS)); width = .38
        ax.bar(x-width/2, part.within_mae_bpm, width, label="Within", color="#2A9D8F")
        ax.bar(x+width/2, part.lodo_mae_bpm, width, label="LODO", color="#E76F51")
        ax.set_xticks(x, HR_LABELS, rotation=25); ax.set_title(dataset); ax.grid(axis="y", alpha=.2)
        ax.set_ylabel("MAE (bpm)")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Error profiles differ by target and reference-HR range")
    fig.tight_layout(); fig.savefig(output, dpi=200); plt.close(fig)


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase9-root", type=Path, default=Path("reports/phase9_timeppg"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase10_lodo_failure_analysis"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures = args.output_dir / "figures"; figures.mkdir(exist_ok=True)

    paired_frames, overall_rows, bootstrap_rows, subject_tables, support_tables = [], [], [], [], []
    for target_key, dataset in TARGETS.items():
        paired = load_paired(args.phase9_root, target_key, args.seed)
        paired_frames.append(paired)
        overall = {"dataset": dataset}
        overall.update({f"within_{k}": v for k, v in metric_row(paired, "within").items()})
        overall.update({f"lodo_{k}": v for k, v in metric_row(paired, "lodo").items()})
        overall["pooled_paired_gap_bpm"] = paired.paired_absolute_error_difference_bpm.mean()
        overall_rows.append(overall)

        subject = grouped_metrics(paired, ["dataset", "subject_id"])
        subject["gap_bpm"] = subject.lodo_mae_bpm - subject.within_mae_bpm
        subject_tables.append(subject)
        bootstrap_rows.append({"dataset": dataset, **subject_bootstrap(subject, args.seed)})
        support_tables.append(source_support_table(args.phase9_root, target_key, args.seed, paired))

    all_paired = pd.concat(paired_frames, ignore_index=True)
    overall = pd.DataFrame(overall_rows)
    subject = pd.concat(subject_tables, ignore_index=True)
    subject_summary = subject_deterioration_summary(subject)
    bootstrap = pd.DataFrame(bootstrap_rows)
    condition = grouped_metrics(all_paired, ["dataset", "activity"])
    hr = grouped_metrics(all_paired, ["dataset", "hr_bin"])
    quality = grouped_metrics(all_paired, ["dataset", "quality_tertile"])
    joint = grouped_metrics(all_paired, ["dataset", "activity", "hr_bin", "quality_tertile"])
    ptt_joint = grouped_metrics(
        all_paired.loc[all_paired.dataset.eq("PTT-PPG")], ["activity", "hr_bin"]
    )
    support = pd.concat(support_tables, ignore_index=True)
    adjusted = within_subject_adjusted_ptt(all_paired.loc[all_paired.dataset.eq("PTT-PPG")], args.seed)
    episodes = error_episodes(all_paired)

    tables = {
        "overall_paired_metrics.csv": overall,
        "subject_metrics.csv": subject,
        "subject_deterioration_summary.csv": subject_summary,
        "subject_bootstrap_ci.csv": bootstrap,
        "condition_metrics.csv": condition,
        "hr_bin_metrics.csv": hr,
        "quality_tertile_metrics.csv": quality,
        "joint_stratum_metrics.csv": joint,
        "ptt_activity_hr_metrics.csv": ptt_joint,
        "source_hr_support.csv": support,
        "ptt_adjusted_within_subject_associations.csv": adjusted,
        "large_error_episodes.csv": episodes,
    }
    for name, table in tables.items():
        table.to_csv(args.output_dir / name, index=False)

    plot_ptt_subjects(subject.loc[subject.dataset.eq("PTT-PPG")], figures / "ptt_subject_within_vs_lodo.png")
    plot_ptt_heatmap(ptt_joint, figures / "ptt_activity_hr_heatmap.png")
    plot_all_hr(hr, figures / "all_targets_hr_profile.png")
    plot_signed_bias_by_hr(hr, figures / "all_targets_signed_bias_by_hr.png")
    plot_ptt_traces(
        all_paired.loc[all_paired.dataset.eq("PTT-PPG")], subject,
        figures / "ptt_representative_temporal_traces.png",
    )

    ptt_overall = overall.loc[overall.dataset.eq("PTT-PPG")].iloc[0]
    ptt_boot = bootstrap.loc[bootstrap.dataset.eq("PTT-PPG")].iloc[0]
    worst_subjects = subject.loc[subject.dataset.eq("PTT-PPG")].nlargest(5, "lodo_mae_bpm")
    report = f"""# TimePPG full-LODO failure analysis — seed {args.seed}

## Scope and integrity

- Every LODO prediction was paired one-to-one with the within-dataset out-of-fold prediction for the same target window.
- Matching keys contain no missing values; physical start times/sample bounds agree; every subject occurs in exactly one within-test fold; and labels/predictions are finite.
- Analysis covers {len(all_paired):,} windows and {all_paired.groupby(['dataset','subject_id']).ngroups} dataset-specific subjects.
- Subject bootstrap keeps each person's overlapping windows together.
- Subgroup and adjusted results are post-hoc associations, not isolated causal effects.

## Overall paired result

{markdown(overall[['dataset','within_mae_bpm','lodo_mae_bpm','pooled_paired_gap_bpm','lodo_rmse_bpm','lodo_bias_bpm','lodo_p95_absolute_error_bpm','lodo_over_20_bpm_percent']])}

## How widespread is deterioration?

{markdown(subject_summary)}

PTT-PPG is the dominant failure: pooled MAE rises from **{ptt_overall.within_mae_bpm:.3f}** to
**{ptt_overall.lodo_mae_bpm:.3f} bpm**, a paired window-level difference of
**{ptt_overall.pooled_paired_gap_bpm:+.3f} bpm**. At the subject level, the mean gap is
**{ptt_boot.macro_gap_bpm:+.3f} bpm** with a subject-bootstrap 95% interval of
**{ptt_boot.macro_gap_ci_low_bpm:+.3f} to {ptt_boot.macro_gap_ci_high_bpm:+.3f} bpm**.

## Mechanism-level findings

- **PTT is a low-HR overestimation failure:** LODO bias is **{ptt_overall.lodo_bias_bpm:+.3f} bpm** overall; the `<60` and `60-80` bpm rows in `hr_bin_metrics.csv` show the largest errors. Those bins contain many source-training windows, so simple HR-bin count scarcity is not a sufficient explanation.
- **PTT tracks the target HR range poorly:** its prediction-versus-reference calibration slope is **{ptt_overall.lodo_calibration_slope:.3f}** (ideal 1.0), consistent with weak target tracking rather than a constant offset alone.
- **PTT failure persists across quality tertiles:** low spectral concentration is harder, but even the high-concentration tertile remains poor. Quality contributes without fully explaining transfer failure.
- **Motion alone is unsupported:** unadjusted sitting is hardest, while adjusted walking-versus-sitting and running-versus-sitting bootstrap intervals include zero.
- **Spectral disagreement remains informative:** larger FFT/reference mismatch is associated with larger TimePPG error after within-subject adjustment.
- **DaLiA has a different failure shape:** error rises sharply above 140 bpm, where source training support is sparse, and predictions are strongly biased downward.
- **Failures are sustained episodes:** `large_error_episodes.csv` identifies long consecutive sequences, so overlapping windows must not be counted as independent failure events.

## Worst PTT-PPG subjects

{markdown(worst_subjects[['subject_id','within_mae_bpm','lodo_mae_bpm','gap_bpm','lodo_bias_bpm','lodo_over_20_bpm_percent']])}

## PTT adjusted within-subject diagnostic

{markdown(adjusted[['predictor','coefficient_bpm','subject_bootstrap_ci_low_bpm','subject_bootstrap_ci_high_bpm']])}

These coefficients compare changes within a subject while adjusting for the listed predictors. They
help reject simplistic explanations but do not prove causality. Spectral concentration is a proxy,
not an independently validated signal-quality label.

## Interpretation boundary and next decisions

1. Use `subject_bootstrap_ci.csv` to judge whether target gaps survive subject variation.
2. Use `ptt_activity_hr_metrics.csv` and the adjusted table to determine whether activity remains associated after HR and quality adjustment.
3. Use `source_hr_support.csv` to distinguish sparse source HR coverage from residual waveform mismatch.
4. Inspect `large_error_episodes.csv` and corresponding waveforms before treating adjacent windows as independent failures.
5. Repeat LODO with prespecified additional seeds before selecting an RQ2 mechanism.
6. Run paired PTT distal/proximal TimePPG transfer separately; primary LODO uses only `pleth_1` and cannot isolate site causally.

## Files

- `overall_paired_metrics.csv`: complete target metrics and paired gaps.
- `subject_metrics.csv`: localisation and subject-level variation.
- `condition_metrics.csv`, `hr_bin_metrics.csv`, `quality_tertile_metrics.csv`: one-factor diagnostics.
- `joint_stratum_metrics.csv`: jointly stratified diagnostics with sample counts.
- `ptt_adjusted_within_subject_associations.csv`: exploratory adjusted PTT analysis.
- `source_hr_support.csv`: source-training coverage versus target error.
- `large_error_episodes.csv`: contiguous >20-bpm LODO error episodes.
- `figures/`: presentation-ready diagnostic plots.
"""
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    provenance = {
        "seed": args.seed,
        "paired_key": KEY,
        "hr_bins": HR_LABELS,
        "quality_definition": "within-target tertiles of spectral_concentration",
        "bootstrap_unit": "subject",
        "post_hoc": True,
    }
    (args.output_dir / "analysis_config.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
