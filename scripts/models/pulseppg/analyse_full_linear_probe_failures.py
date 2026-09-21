#!/usr/bin/env python3
"""Post-hoc, same-window localisation of frozen Pulse-PPG HR errors.

This script never fits or changes a model. All analyses are descriptive;
target labels are used only after the frozen evaluations have completed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PRED = ROOT / "reports/phase12_pulseppg/02_frozen_linear_probe"
MANIFEST = ROOT / "reports/phase7_frozen_dataset/main_window_manifest.csv"
ROLES = ROOT / "reports/phase7_frozen_dataset/subject_splits.csv"
TPPG = ROOT / "reports/phase9_timeppg/lodo_full"
TPPG_WITHIN = ROOT / "reports/phase9_timeppg/within_dataset_3fold"
OUT = ROOT / "reports/phase12_pulseppg/04_failure_analysis"
KEY = ["dataset", "record_id", "subject_id", "window_index"]
DATASETS = {"BIDMC": "bidmc", "WESAD": "wesad", "PTT-PPG": "ptt_ppg", "PPG-DaLiA": "ppg_dalia"}
EDGES = [-np.inf, 60, 80, 100, 120, 140, np.inf]
LABELS = ["<60", "60-80", "80-100", "100-120", "120-140", ">=140"]


def unique(frame: pd.DataFrame, name: str) -> None:
    if frame.duplicated(KEY).any():
        raise ValueError(f"Duplicate window keys in {name}")


def metrics(group: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "windows": len(group),
        "subjects": group.subject_for_count.nunique(),
        "reference_mean_bpm": group.reference_hr_bpm.mean(),
        "within_mae_bpm": group.within_abs.mean(),
        "lodo_mae_bpm": group.lodo_abs.mean(),
        "paired_gap_bpm": (group.lodo_abs - group.within_abs).mean(),
        "lodo_bias_bpm": group.lodo_error.mean(),
        "timeppg_within_mae_bpm": group.timeppg_within_abs.mean(),
        "timeppg_lodo_mae_bpm": group.timeppg_abs.mean(),
        "pulse_minus_timeppg_bpm": (group.lodo_abs - group.timeppg_abs).mean(),
    })


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, usecols=KEY + ["spectral_concentration", "dominant_frequency_hz", "hr_quality_flag", "condition_name", "primary_analysis"], dtype={"condition_name": "string"}, low_memory=False)
    manifest = manifest.loc[manifest.primary_analysis.astype(bool)].drop(columns="primary_analysis")
    unique(manifest, "manifest")
    roles = pd.read_csv(ROLES)
    frames = []
    for dataset, slug in DATASETS.items():
        base = PRED / "within_dataset" / slug / "all_test_predictions.csv"
        lodo = PRED / "lodo" / f"target_{slug}" / "target_test_predictions.csv"
        timeppg = TPPG / f"target_{slug}" / "seed_17" / "target_test_predictions.csv"
        timeppg_within_paths = [
            TPPG_WITHIN / slug / f"test_fold_{fold}" / "seed_17" / "test_predictions.csv"
            for fold in range(3)
        ]
        w = pd.read_csv(base, usecols=KEY + ["reference_hr_bpm", "predicted_hr_bpm"])
        l = pd.read_csv(lodo, usecols=KEY + ["reference_hr_bpm", "predicted_hr_bpm"])
        t = pd.read_csv(timeppg, usecols=KEY + ["reference_hr_bpm", "predicted_hr_bpm"])
        tw = pd.concat([
            pd.read_csv(path, usecols=KEY + ["reference_hr_bpm", "predicted_hr_bpm"])
            for path in timeppg_within_paths
        ], ignore_index=True)
        for name, frame in [("within", w), ("lodo", l), ("timeppg_lodo", t), ("timeppg_within", tw)]:
            unique(frame, f"{dataset} {name}")
        if not (len(w) == len(l) == len(t) == len(tw)):
            raise ValueError(f"Different target-window counts for {dataset}")
        frame = w.rename(columns={"reference_hr_bpm": "reference_hr_bpm", "predicted_hr_bpm": "within_pred"})
        frame = frame.merge(l.rename(columns={"reference_hr_bpm": "lodo_reference", "predicted_hr_bpm": "lodo_pred"}), on=KEY, validate="one_to_one")
        frame = frame.merge(t.rename(columns={"reference_hr_bpm": "timeppg_reference", "predicted_hr_bpm": "timeppg_pred"}), on=KEY, validate="one_to_one")
        frame = frame.merge(tw.rename(columns={"reference_hr_bpm": "timeppg_within_reference", "predicted_hr_bpm": "timeppg_within_pred"}), on=KEY, validate="one_to_one")
        if len(frame) != len(w) or max(
            (frame.reference_hr_bpm - frame.lodo_reference).abs().max(),
            (frame.reference_hr_bpm - frame.timeppg_reference).abs().max(),
            (frame.reference_hr_bpm - frame.timeppg_within_reference).abs().max(),
        ) > .001:
            raise ValueError(f"Mismatched windows or ECG targets for {dataset}")
        frames.append(frame.drop(columns=["lodo_reference", "timeppg_reference", "timeppg_within_reference"]))
    data = pd.concat(frames, ignore_index=True)
    data = data.merge(manifest, on=KEY, validate="one_to_one")
    if len(data) != len(manifest) or len(data) != 136625:
        raise ValueError("The paired analysis did not cover all 136,625 primary windows")
    data["subject_for_count"] = data.subject_id
    for name, col in [
        ("within", "within_pred"), ("lodo", "lodo_pred"),
        ("timeppg", "timeppg_pred"), ("timeppg_within", "timeppg_within_pred"),
    ]:
        data[f"{name}_error"] = data[col] - data.reference_hr_bpm
        data[f"{name}_abs"] = data[f"{name}_error"].abs()
    data["hr_bin"] = pd.cut(data.reference_hr_bpm, EDGES, labels=LABELS, right=False)
    data["hr_below_80"] = np.where(data.reference_hr_bpm < 80, "<80", ">=80")
    # Target-specific tertiles are descriptive signal proxies, not clinical SQI labels.
    data["spectral_tertile"] = data.groupby("dataset").spectral_concentration.transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "middle", "high"])
    )
    for name, cols in [
        ("target", ["dataset"]), ("hr_bins", ["dataset", "hr_bin"]),
        ("subjects", ["dataset", "subject_id"]),
        ("conditions", ["dataset", "condition_name"]),
        ("spectral_tertiles", ["dataset", "spectral_tertile"]),
        ("ptt_activity_hr", ["dataset", "condition_name", "hr_below_80"]),
    ]:
        source = data.loc[data.dataset.eq("PTT-PPG")] if name == "ptt_activity_hr" else data
        table = source.groupby(cols, observed=True, dropna=False).apply(metrics, include_groups=False).reset_index()
        table.to_csv(OUT / f"{name}_metrics.csv", index=False)
    # Count source-training HR support for each LODO target from the frozen roles.
    source = pd.read_csv(MANIFEST, usecols=["dataset", "subject_id", "hr_bpm", "primary_analysis"])
    source = source.loc[source.primary_analysis.astype(bool)].drop(columns="primary_analysis")
    supports = []
    for target in DATASETS:
        role = f"lodo_{target}_role"
        joined = source.merge(roles[["dataset", "subject_id", role]], on=["dataset", "subject_id"], validate="many_to_one")
        train = joined.loc[joined[role].eq("source_train")].copy()
        train["source_person"] = train.dataset.astype(str) + ":" + train.subject_id.astype(str)
        train["hr_bin"] = pd.cut(train.hr_bpm, EDGES, labels=LABELS, right=False)
        tab = train.groupby("hr_bin", observed=True).agg(source_train_windows=("hr_bpm", "size"), source_train_subjects=("source_person", "nunique")).reset_index()
        tab.insert(0, "target", target)
        supports.append(tab)
    pd.concat(supports).to_csv(OUT / "source_hr_support.csv", index=False)
    # Summary slopes quantify range compression; no training or calibration is performed.
    slopes = []
    for dataset, group in data.groupby("dataset"):
        for model in ["within", "lodo", "timeppg_within", "timeppg"]:
            slope, intercept = np.polyfit(group.reference_hr_bpm, group[f"{model}_pred"], 1)
            slopes.append({"dataset": dataset, "model": model, "slope_pred_vs_reference": slope,
                           "intercept_bpm": intercept, "reference_sd_bpm": group.reference_hr_bpm.std(),
                           "prediction_sd_bpm": group[f"{model}_pred"].std()})
    pd.DataFrame(slopes).to_csv(OUT / "prediction_range.csv", index=False)
    plot(data)
    print(f"Saved paired failure analysis for {len(data):,} windows to {OUT.relative_to(ROOT)}")


def plot(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        subset = data.loc[data.dataset.eq(dataset)]
        agg = subset.groupby("hr_bin", observed=True).agg(
            n=("lodo_abs", "size"), within=("within_abs", "mean"), lodo=("lodo_abs", "mean"),
            timeppg_within=("timeppg_within_abs", "mean"),
            timeppg=("timeppg_abs", "mean"), bias=("lodo_error", "mean")
        ).reindex(LABELS)
        x = np.arange(len(LABELS))
        ax.plot(x, agg.within, "o--", color="#2A9D8F", label="Pulse-PPG within")
        ax.plot(x, agg.lodo, "o-", color="#2A9D8F", label="Pulse-PPG LODO")
        ax.plot(x, agg.timeppg_within, "s--", color="#E76F51", label="TimePPG within")
        ax.plot(x, agg.timeppg, "s-", color="#E76F51", label="TimePPG LODO")
        ax.set_xticks(x, [f"{b}\n(n={int(n)})" if pd.notna(n) else b for b, n in zip(LABELS, agg.n)], fontsize=8)
        ax.set_title(dataset)
        ax.set_ylabel("MAE (bpm)")
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.06), ncol=4)
    fig.suptitle("Same-window HR-range failure: model × evaluation protocol", y=1.12)
    fig.savefig(OUT / "hr_range_failure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Overall model-by-protocol comparison.
    order = list(DATASETS)
    overall = data.groupby("dataset").agg(
        pulse_within=("within_abs", "mean"), pulse_lodo=("lodo_abs", "mean"),
        timeppg_within=("timeppg_within_abs", "mean"),
        timeppg_lodo=("timeppg_abs", "mean"),
    ).reindex(order)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = np.arange(len(order)); width = .19
    series = [
        ("Pulse-PPG within", "pulse_within", "#8FD3C7", "//"),
        ("Pulse-PPG LODO", "pulse_lodo", "#2A9D8F", None),
        ("TimePPG within", "timeppg_within", "#F3AE98", "//"),
        ("TimePPG LODO", "timeppg_lodo", "#E76F51", None),
    ]
    for index, (label, column, colour, hatch) in enumerate(series):
        bars = ax.bar(x + (index - 1.5) * width, overall[column], width,
                      label=label, color=colour, hatch=hatch, edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x, order); ax.set_ylabel("MAE (bpm)")
    ax.set_title("Overall accuracy: familiar versus unseen-dataset testing")
    ax.legend(ncol=2, frameon=False); ax.grid(axis="y", alpha=.2)
    fig.savefig(OUT / "overall_model_protocol_comparison.png", dpi=180)
    plt.close(fig)

    # Signed error shows whether each transferred model predicts too high or too low.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, dataset in zip(axes.flat, order):
        subset = data.loc[data.dataset.eq(dataset)]
        agg = subset.groupby("hr_bin", observed=True).agg(
            n=("lodo_error", "size"), pulse_bias=("lodo_error", "mean"),
            timeppg_bias=("timeppg_error", "mean"),
        ).reindex(LABELS)
        x = np.arange(len(LABELS))
        ax.axhline(0, color="#61717D", linewidth=1)
        ax.plot(x, agg.pulse_bias, "o-", color="#2A9D8F", label="Pulse-PPG LODO")
        ax.plot(x, agg.timeppg_bias, "s-", color="#E76F51", label="TimePPG LODO")
        ax.set_xticks(x, [f"{b}\n(n={int(n)})" if pd.notna(n) else b
                          for b, n in zip(LABELS, agg.n)], fontsize=8)
        ax.set_title(dataset); ax.set_ylabel("Signed bias (bpm)"); ax.grid(axis="y", alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.04), ncol=2)
    fig.suptitle("LODO bias by HR range: positive = too high; negative = too low", y=1.09)
    fig.savefig(OUT / "lodo_signed_bias_by_hr.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # A subject-level view prevents pooled averages from hiding prevalence.
    subject_metrics = data.groupby(["dataset", "subject_id"]).apply(
        lambda g: pd.Series({
            "pulse_gap": (g.lodo_abs - g.within_abs).mean(),
            "timeppg_gap": (g.timeppg_abs - g.timeppg_within_abs).mean(),
        }), include_groups=False,
    ).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, dataset in zip(axes.flat, order):
        part = subject_metrics.loc[subject_metrics.dataset.eq(dataset)].sort_values("pulse_gap")
        x = np.arange(len(part))
        ax.axhline(0, color="#61717D", linewidth=1)
        ax.scatter(x - .12, part.pulse_gap, s=25, color="#2A9D8F", label="Pulse-PPG")
        ax.scatter(x + .12, part.timeppg_gap, s=25, color="#E76F51", label="TimePPG")
        ax.set_title(f"{dataset} (n={len(part)} people)")
        ax.set_xlabel("People ordered by Pulse-PPG gap"); ax.set_ylabel("LODO − within MAE (bpm)")
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=2)
    fig.suptitle("Subject-level transfer penalty: above zero means LODO became worse", y=1.07)
    fig.savefig(OUT / "subject_transfer_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # PTT mechanism summary: separate low-HR prevalence from the signal proxy.
    ptt = data.loc[data.dataset.eq("PTT-PPG")]
    activity = ptt.groupby(["condition_name", "hr_below_80"], observed=True).agg(
        mae=("lodo_abs", "mean"), n=("lodo_abs", "size")
    ).reset_index()
    quality = ptt.groupby("spectral_tertile", observed=True).agg(
        within=("within_abs", "mean"), lodo=("lodo_abs", "mean"), n=("lodo_abs", "size")
    ).reindex(["low", "middle", "high"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    activities = [name for name in ["sit", "walk", "run"] if name in set(activity.condition_name)]
    x = np.arange(len(activities)); width = .36
    for offset, hr_group, colour in [(-width/2, "<80", "#E76F51"), (width/2, ">=80", "#2A9D8F")]:
        vals = [activity.loc[(activity.condition_name.eq(a)) & (activity.hr_below_80.eq(hr_group)), "mae"].iloc[0] for a in activities]
        bars = axes[0].bar(x + offset, vals, width, label=hr_group, color=colour)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
    axes[0].set_xticks(x, activities); axes[0].set_ylabel("Pulse-PPG LODO MAE (bpm)")
    axes[0].set_title("PTT activity × HR range"); axes[0].legend(title="Reference HR")
    qx = np.arange(3)
    axes[1].bar(qx - width/2, quality.within, width, label="Within", color="#8FD3C7")
    axes[1].bar(qx + width/2, quality.lodo, width, label="LODO", color="#2A9D8F")
    axes[1].set_xticks(qx, [f"{q}\n(n={int(n)})" for q, n in zip(quality.index, quality.n)])
    axes[1].set_ylabel("MAE (bpm)"); axes[1].set_title("PTT spectral-concentration strata")
    axes[1].legend(); axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("PTT failure persists across activities and in the highest spectral-concentration group")
    fig.savefig(OUT / "ptt_failure_checks.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    run()
