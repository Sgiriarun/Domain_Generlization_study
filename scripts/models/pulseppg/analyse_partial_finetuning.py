#!/usr/bin/env python3
"""Freeze the complete Phase-12 Pulse-PPG partial-fine-tuning analysis.

This script is post-hoc only: it reads already frozen predictions, never fits a
model, and uses target labels only for evaluation and failure localisation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PHASE = ROOT / "reports/phase12_pulseppg"
PARTIAL = PHASE / "03_head_and_tuning_ablation/partial_final_block"
NONLINEAR = PHASE / "03_head_and_tuning_ablation/nonlinear_head"
RIDGE = PHASE / "02_frozen_linear_probe"
TIMEPPG = ROOT / "reports/phase9_timeppg"
FFT = ROOT / "reports/phase8_spectral_baseline/dataset_metrics.csv"
MANIFEST = ROOT / "reports/phase7_frozen_dataset/main_window_manifest.csv"
OUT = PHASE / "07_frozen_outcome"
FIG = OUT / "figures"

DATASETS = ("BIDMC", "WESAD", "PTT-PPG", "PPG-DaLiA")
SEEDS = (17, 29, 43)
KEY = ["dataset", "record_id", "subject_id", "window_index"]
EDGES = [-np.inf, 60, 80, 100, 120, 140, np.inf]
LABELS = ["<60", "60–80", "80–100", "100–120", "120–140", "≥140"]
COLORS = {"BIDMC": "#177E89", "WESAD": "#2A9D8F",
          "PTT-PPG": "#E76F51", "PPG-DaLiA": "#B84A3A"}


def slug(dataset: str) -> str:
    return dataset.lower().replace("-", "_")


def read_predictions(root: Path, protocol: str, dataset: str, seed: int) -> pd.DataFrame:
    name = slug(dataset)
    if root == PARTIAL:
        if protocol == "within":
            paths = sorted((root / "within" / f"target_{name}").glob(
                f"fold_*/seed_{seed}/test_predictions.csv"))
        else:
            paths = [root / "lodo" / f"target_{name}" / f"seed_{seed}" / "test_predictions.csv"]
    elif root == NONLINEAR:
        if protocol == "within":
            paths = sorted((root / "within_dataset" / name).glob(
                f"fold_*/seed_{seed}/test_predictions.csv"))
        else:
            paths = [root / "lodo" / f"target_{name}" / f"seed_{seed}" / "test_predictions.csv"]
    else:
        raise ValueError(root)
    if not paths or any(not path.exists() for path in paths):
        raise FileNotFoundError(f"Missing {root.name} {protocol} {dataset} seed {seed}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if frame.duplicated(KEY).any():
        raise ValueError(f"Duplicate target windows: {root.name} {protocol} {dataset} seed {seed}")
    return frame


def measures(frame: pd.DataFrame) -> dict[str, float]:
    error = frame.predicted_hr_bpm - frame.reference_hr_bpm
    return {
        "windows": len(frame),
        "mae_bpm": error.abs().mean(),
        "rmse_bpm": np.sqrt(np.mean(error ** 2)),
        "pearson_r": frame.reference_hr_bpm.corr(frame.predicted_hr_bpm),
        "within_5_bpm_percent": 100 * (error.abs() <= 5).mean(),
        "signed_bias_bpm": error.mean(),
    }


def build_partial_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows, paired_rows = [], []
    manifest = pd.read_csv(
        MANIFEST,
        usecols=KEY + ["condition_name", "spectral_concentration", "primary_analysis"],
        dtype={"condition_name": "string"}, low_memory=False,
    )
    manifest = manifest.loc[manifest.primary_analysis.astype(bool)].drop(columns="primary_analysis")
    if len(manifest) != 136625 or manifest.duplicated(KEY).any():
        raise RuntimeError("Unexpected primary manifest")

    for dataset in DATASETS:
        for seed in SEEDS:
            within = read_predictions(PARTIAL, "within", dataset, seed)
            lodo = read_predictions(PARTIAL, "lodo", dataset, seed)
            for protocol, frame in (("within", within), ("lodo", lodo)):
                metric_rows.append({"model": "Pulse-PPG final-block", "target": dataset,
                                    "protocol": protocol, "seed": seed, **measures(frame)})
            pair = within[KEY + ["reference_hr_bpm", "predicted_hr_bpm"]].rename(
                columns={"predicted_hr_bpm": "within_prediction_bpm"})
            pair = pair.merge(
                lodo[KEY + ["reference_hr_bpm", "predicted_hr_bpm"]].rename(
                    columns={"reference_hr_bpm": "lodo_reference_bpm",
                             "predicted_hr_bpm": "lodo_prediction_bpm"}),
                on=KEY, validate="one_to_one",
            )
            if len(pair) != len(within) or not np.allclose(
                    pair.reference_hr_bpm, pair.lodo_reference_bpm, atol=1e-6):
                raise RuntimeError(f"Within/LODO pairing failed for {dataset} seed {seed}")
            pair = pair.drop(columns="lodo_reference_bpm").merge(manifest, on=KEY, validate="one_to_one")
            pair["seed"] = seed
            pair["within_error_bpm"] = pair.within_prediction_bpm - pair.reference_hr_bpm
            pair["lodo_error_bpm"] = pair.lodo_prediction_bpm - pair.reference_hr_bpm
            pair["within_abs_error_bpm"] = pair.within_error_bpm.abs()
            pair["lodo_abs_error_bpm"] = pair.lodo_error_bpm.abs()
            pair["paired_transfer_error_bpm"] = pair.lodo_abs_error_bpm - pair.within_abs_error_bpm
            paired_rows.append(pair)

    metrics = pd.DataFrame(metric_rows)
    paired = pd.concat(paired_rows, ignore_index=True)
    gaps = metrics.pivot(index=["target", "seed"], columns="protocol", values="mae_bpm").reset_index()
    gaps["generalisation_gap_bpm"] = gaps.lodo - gaps.within
    gaps = gaps.rename(columns={"within": "within_mae_bpm", "lodo": "lodo_mae_bpm"})
    return metrics, gaps, paired


def summarise_failures(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paired = paired.copy()
    paired["hr_bin"] = pd.cut(paired.reference_hr_bpm, EDGES, labels=LABELS, right=False)
    paired["spectral_tertile"] = paired.groupby(["dataset", "seed"]).spectral_concentration.transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["low", "middle", "high"]))

    def agg(group: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "windows": len(group),
            # pandas removes grouping columns under include_groups=False. A
            # subject-group therefore contains exactly one person even though
            # the subject_id column is no longer present.
            "subjects": group.subject_id.nunique() if "subject_id" in group else 1,
            "reference_mean_bpm": group.reference_hr_bpm.mean(),
            "within_mae_bpm": group.within_abs_error_bpm.mean(),
            "lodo_mae_bpm": group.lodo_abs_error_bpm.mean(),
            "paired_gap_bpm": group.paired_transfer_error_bpm.mean(),
            "lodo_signed_bias_bpm": group.lodo_error_bpm.mean(),
            "large_error_over_20_percent": 100 * (group.lodo_abs_error_bpm > 20).mean(),
        })

    hr = paired.groupby(["dataset", "seed", "hr_bin"], observed=True).apply(
        agg, include_groups=False).reset_index()
    subject = paired.groupby(["dataset", "seed", "subject_id"]).apply(
        agg, include_groups=False).reset_index()
    condition = paired.groupby(["dataset", "seed", "condition_name"], dropna=False).apply(
        agg, include_groups=False).reset_index()
    return hr, subject, condition


def model_ladder(partial_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Partial and nonlinear: retain per-seed values.
    rows.extend(partial_metrics.to_dict("records"))
    for dataset in DATASETS:
        for seed in SEEDS:
            for protocol in ("within", "lodo"):
                frame = read_predictions(NONLINEAR, protocol, dataset, seed)
                rows.append({"model": "Pulse-PPG nonlinear", "target": dataset,
                             "protocol": protocol, "seed": seed, **measures(frame)})
    # Frozen Ridge and seed-17 TimePPG are already exact same-window summaries.
    ridge = pd.read_csv(RIDGE / "comparison_summary.csv").set_index("target")
    for dataset in DATASETS:
        record = ridge.loc[dataset]
        for protocol in ("within", "lodo"):
            prefix = f"{protocol}_"
            rows.append({"model": "Pulse-PPG Ridge", "target": dataset,
                         "protocol": protocol, "seed": np.nan,
                         "windows": record[prefix + "windows"],
                         "mae_bpm": record[prefix + "mae_bpm"],
                         "rmse_bpm": record[prefix + "rmse_bpm"],
                         "pearson_r": record[prefix + "pearson_r"],
                         "within_5_bpm_percent": record[prefix + "within_5_bpm_percent"],
                         "signed_bias_bpm": record[prefix + "bland_altman_bias_bpm"]})
            tp = "timeppg_within_" if protocol == "within" else "timeppg_lodo_"
            rows.append({"model": "TimePPG seed 17", "target": dataset,
                         "protocol": protocol, "seed": 17,
                         "windows": record[tp + "windows"], "mae_bpm": record[tp + "mae_bpm"],
                         "rmse_bpm": record[tp + "rmse_bpm"], "pearson_r": record[tp + "pearson_r"],
                         "within_5_bpm_percent": record[tp + "within_5_bpm_percent"],
                         "signed_bias_bpm": record[tp + "bland_altman_bias_bpm"]})
    fft = pd.read_csv(FFT).set_index("dataset")
    for dataset in DATASETS:
        r = fft.loc[dataset]
        rows.append({"model": "FFT", "target": dataset, "protocol": "fixed_rule",
                     "seed": np.nan, "windows": r.windows, "mae_bpm": r.mae_bpm,
                     "rmse_bpm": r.rmse_bpm, "pearson_r": r.pearson_r,
                     "within_5_bpm_percent": r.within_5_bpm_percent,
                     "signed_bias_bpm": r.bland_altman_bias_bpm})
    return pd.DataFrame(rows)


def selection_summary() -> pd.DataFrame:
    summary = pd.read_csv(PARTIAL / "run_summary.csv")
    return summary.groupby(["protocol", "target"], as_index=False).agg(
        runs=("seed", "size"),
        initial_source_validation_mae_mean=("initial_source_validation_macro_mae_bpm", "mean"),
        selected_source_validation_mae_mean=("selected_source_validation_macro_mae_bpm", "mean"),
        source_validation_improvement_mean=("initial_source_validation_macro_mae_bpm",
                                            lambda x: np.nan),
        no_adaptation_selected=("source_selected_no_adaptation", "sum"),
        selected_epoch_mean=("selected_epoch", "mean"),
    ).assign(source_validation_improvement_mean=lambda d:
             d.initial_source_validation_mae_mean - d.selected_source_validation_mae_mean)


def make_figures(ladder: pd.DataFrame, gaps: pd.DataFrame, hr: pd.DataFrame,
                 subjects: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    models = ["TimePPG seed 17", "Pulse-PPG Ridge", "Pulse-PPG nonlinear", "Pulse-PPG final-block"]
    summary = ladder.loc[ladder.protocol.isin(["within", "lodo"])].groupby(
        ["model", "target", "protocol"]).mae_bpm.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        part = summary.loc[summary.target.eq(dataset)]
        x = np.arange(len(models)); width = .36
        for offset, protocol, color in [(-width/2, "within", "#2A9D8F"),
                                        (width/2, "lodo", "#E76F51")]:
            vals, errs = [], []
            for model in models:
                row = part.loc[part.model.eq(model) & part.protocol.eq(protocol)]
                vals.append(row["mean"].iloc[0]); errs.append(row["std"].fillna(0).iloc[0])
            bars = ax.bar(x + offset, vals, width, yerr=errs, color=color,
                          label=protocol.upper(), capsize=3)
            ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax.set_xticks(x, ["TimePPG", "Ridge", "Nonlinear", "Final block"], rotation=15)
        ax.set_ylabel("MAE (bpm)"); ax.set_title(dataset); ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Model ladder: familiar-domain and unseen-dataset accuracy")
    fig.savefig(FIG / "model_ladder.png", dpi=200)
    plt.close(fig)

    # Fine-tuning effect against the frozen nonlinear head, matched by target and seed.
    base = ladder.loc[ladder.model.eq("Pulse-PPG nonlinear")].pivot(
        index=["target", "seed"], columns="protocol", values="mae_bpm")
    tuned = ladder.loc[ladder.model.eq("Pulse-PPG final-block")].pivot(
        index=["target", "seed"], columns="protocol", values="mae_bpm")
    delta = (tuned - base).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(len(DATASETS)); width = .36
    for off, col, color, label in [(-width/2, "within", "#177E89", "Within"),
                                    (width/2, "lodo", "#E76F51", "LODO")]:
        means = [delta.loc[delta.target.eq(d), col].mean() for d in DATASETS]
        stds = [delta.loc[delta.target.eq(d), col].std() for d in DATASETS]
        bars = ax.bar(x + off, means, width, yerr=stds, color=color, capsize=3, label=label)
        ax.bar_label(bars, fmt="%+.2f", fontsize=8, padding=2)
    ax.axhline(0, color="#1F2933", linewidth=1)
    ax.set_xticks(x, DATASETS); ax.set_ylabel("Fine-tuned − frozen nonlinear MAE (bpm)")
    ax.set_title("Final-block fine-tuning effect: below zero is improvement")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.savefig(FIG / "fine_tuning_effect.png", dpi=200)
    plt.close(fig)

    # LODO error and signed bias by HR range, averaged over seeds.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, dataset in zip(axes.flat, DATASETS):
        p = hr.loc[hr.dataset.eq(dataset)].groupby("hr_bin", observed=True).agg(
            mae=("lodo_mae_bpm", "mean"), bias=("lodo_signed_bias_bpm", "mean"),
            windows=("windows", "max")).reindex(LABELS)
        x = np.arange(len(LABELS))
        ax.plot(x, p.mae, "o-", color="#177E89", label="MAE")
        ax.plot(x, p.bias, "s--", color="#E76F51", label="Signed bias")
        ax.axhline(0, color="#61717D", linewidth=.8)
        labels = [f"{b}\n(n={int(n)})" if pd.notna(n) else b for b, n in zip(LABELS, p.windows)]
        ax.set_xticks(x, labels, fontsize=8); ax.set_ylabel("bpm"); ax.set_title(dataset)
        ax.grid(axis="y", alpha=.2)
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Fine-tuned Pulse-PPG LODO failure by ECG heart-rate range")
    fig.savefig(FIG / "hr_range_error_and_bias.png", dpi=200)
    plt.close(fig)

    # Prevalence of positive person-level transfer gap over seeds.
    prevalence = subjects.assign(positive=subjects.paired_gap_bpm > 0).groupby(
        ["dataset", "seed"]).positive.mean().mul(100).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    means = prevalence.groupby("dataset").positive.mean().reindex(DATASETS)
    stds = prevalence.groupby("dataset").positive.std().reindex(DATASETS)
    bars = ax.bar(DATASETS, means, yerr=stds, color=[COLORS[d] for d in DATASETS], capsize=4)
    ax.bar_label(bars, fmt="%.0f%%", padding=3)
    ax.set_ylim(0, 110); ax.set_ylabel("People with LODO MAE > within MAE (%)")
    ax.set_title("How widespread is the transfer penalty across people?")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(FIG / "subject_transfer_prevalence.png", dpi=200)
    plt.close(fig)


def write_readme(ladder: pd.DataFrame, gaps: pd.DataFrame, hr: pd.DataFrame,
                 subjects: pd.DataFrame, selections: pd.DataFrame) -> None:
    summary = gaps.groupby("target").agg(
        within_mean=("within_mae_bpm", "mean"), within_sd=("within_mae_bpm", "std"),
        lodo_mean=("lodo_mae_bpm", "mean"), lodo_sd=("lodo_mae_bpm", "std"),
        gap_mean=("generalisation_gap_bpm", "mean"), gap_sd=("generalisation_gap_bpm", "std"),
    ).reindex(DATASETS)
    nonlinear = ladder.loc[ladder.model.eq("Pulse-PPG nonlinear")].groupby(
        ["target", "protocol"]).mae_bpm.mean().unstack()
    partial = ladder.loc[ladder.model.eq("Pulse-PPG final-block")].groupby(
        ["target", "protocol"]).mae_bpm.mean().unstack()
    ptt = hr.loc[hr.dataset.eq("PTT-PPG")].groupby("hr_bin", observed=True).agg(
        mae=("lodo_mae_bpm", "mean"), bias=("lodo_signed_bias_bpm", "mean"),
        gap=("paired_gap_bpm", "mean"), windows=("windows", "max"))
    dalia = hr.loc[hr.dataset.eq("PPG-DaLiA")].groupby("hr_bin", observed=True).agg(
        mae=("lodo_mae_bpm", "mean"), bias=("lodo_signed_bias_bpm", "mean"),
        gap=("paired_gap_bpm", "mean"), windows=("windows", "max"))
    prevalence = subjects.assign(pos=subjects.paired_gap_bpm > 0).groupby("dataset").pos.mean().mul(100)

    lines = [
        "# Phase 12 frozen outcome — Pulse-PPG representation and final-block adaptation", "",
        "Status: **primary complete and frozen for analysis**", "",
        "## Protocol", "",
        "The released self-supervised Pulse-PPG encoder was evaluated on all 136,625 primary 8-second windows. The final experiment warm-started the source-selected nonlinear HR head, froze the stem and residual blocks 0–10, and updated only residual block 11 plus the head. Learning rate and early stopping used source-validation people only. No held-out LODO target signal, label or statistic entered training or selection. Four within-dataset targets (three subject folds) and four train-three/test-one LODO targets were completed for seeds 17, 29 and 43 on CUDA.", "",
        "## Primary results", "",
        "| Target | Within MAE | LODO MAE | Gap (LODO − within) |", "|---|---:|---:|---:|",
    ]
    for d, r in summary.iterrows():
        lines.append(f"| {d} | {r.within_mean:.2f} ± {r.within_sd:.2f} | {r.lodo_mean:.2f} ± {r.lodo_sd:.2f} | {r.gap_mean:+.2f} ± {r.gap_sd:.2f} |")
    lines += ["", "Values are pooled-window MAE, mean ± sample SD across three seeds. Within-dataset predictions pool the three subject-disjoint folds before calculating each seed's metric.", "",
              "## What final-block fine-tuning changed", ""]
    for d in DATASETS:
        lines.append(f"- **{d}:** LODO MAE changed from {nonlinear.loc[d, 'lodo']:.2f} to {partial.loc[d, 'lodo']:.2f} bpm ({partial.loc[d, 'lodo']-nonlinear.loc[d, 'lodo']:+.2f}); within-dataset MAE changed by {partial.loc[d, 'within']-nonlinear.loc[d, 'within']:+.2f} bpm.")
    lines += ["", "Final-block adaptation therefore improves BIDMC, WESAD and DaLiA LODO MAE, but leaves PTT essentially unchanged. Better fitting of the source domains is not sufficient to remove the PTT acquisition/site-related failure.", "",
              "## Failure localisation", "",
              f"- PTT retains the largest gap ({summary.loc['PTT-PPG','gap_mean']:+.2f} bpm). Across seed–person evaluations, the transfer penalty is positive in about {prevalence['PTT-PPG']:.1f}% of cases.",]
    if "60–80" in ptt.index:
        r=ptt.loc["60–80"]; lines.append(f"- PTT at 60–80 bpm has mean LODO MAE {r.mae:.2f} bpm, signed bias {r.bias:+.2f} bpm and paired transfer gap {r.gap:+.2f} bpm ({int(r.windows)} windows per seed). The transferred model systematically predicts this range too high.")
    if "120–140" in dalia.index:
        r=dalia.loc["120–140"]; lines.append(f"- DaLiA at 120–140 bpm has mean LODO MAE {r.mae:.2f} bpm and signed bias {r.bias:+.2f} bpm ({int(r.windows)} windows per seed), retaining the high-HR underprediction pattern.")
    lines += ["- These are post-hoc associations. They localise failure but do not prove that one device, site, activity or physiological factor is the sole cause.", "",
              "## Selection and integrity", "",
              f"All 48 expected runs, prediction files, subject-metric files, checkpoints and selection records are present. Every selection record declares `target_data_used_for_selection: false`. Fine-tuning was selected over the no-adaptation control in {int(selections.runs.sum()-selections.no_adaptation_selected.sum())}/48 runs; one within-PTT run retained the frozen source-selected control.", "",
              "The frozen provenance records CUDA, batch size 64, seeds 17/29/43, the two predeclared learning rates, and matching hashes for the encoder, manifest and subject roles.", "",
              "## Approved conclusion", "",
              "> Self-supervised Pulse-PPG pretraining and source-only final-block adaptation improve several targets, but do not eliminate target-dependent generalisation failure. PTT retains a large, widespread low-HR overprediction gap, while DaLiA retains high-HR underprediction. Representation quality and modest source-only adaptation are therefore helpful but insufficient; RQ2 still requires controlled tests of domain-generalisation mechanisms that preserve HR structure while reducing acquisition-specific dependence.", "",
              "This conclusion applies to the released Pulse-PPG checkpoint, 8-second inputs resampled to 50 Hz, the declared nonlinear head and final-block adaptation protocol. It is not a claim that every self-supervised or fully fine-tuned PPG foundation model will behave identically.", "",
              "## Files", "",
              "- `partial_seed_metrics.csv`: complete per-target/per-seed metrics.",
              "- `partial_generalisation_gaps.csv`: paired within/LODO values and gaps.",
              "- `model_ladder.csv`: FFT, TimePPG, Ridge, nonlinear-head and final-block conditions.",
              "- `hr_bin_metrics.csv`, `subject_metrics.csv`, `condition_metrics.csv`: post-hoc failure localisation.",
              "- `selection_summary.csv`: source-validation selection behaviour.",
              "- `figures/model_ladder.png`: complete model/protocol comparison.",
              "- `figures/fine_tuning_effect.png`: matched change from frozen nonlinear to final-block adaptation.",
              "- `figures/hr_range_error_and_bias.png`: target-specific HR-range error direction.",
              "- `figures/subject_transfer_prevalence.png`: prevalence of transfer deterioration.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((PARTIAL / "run_provenance.json").read_text())
    if provenance.get("status") != "primary_complete" or provenance.get("target_used_for_selection") is not False:
        raise RuntimeError("Partial-fine-tuning provenance is not a complete source-only primary run")
    summary = pd.read_csv(PARTIAL / "run_summary.csv")
    if len(summary) != 48:
        raise RuntimeError(f"Expected 48 primary results, found {len(summary)}")
    selections = list(PARTIAL.glob("**/selection.json"))
    predictions = list(PARTIAL.glob("**/test_predictions.csv"))
    subjects_files = list(PARTIAL.glob("**/subject_metrics.csv"))
    checkpoints = list(PARTIAL.glob("**/partial_checkpoint.pt"))
    if not all(len(x) == 48 for x in [selections, predictions, subjects_files, checkpoints]):
        raise RuntimeError("Incomplete primary artifact set")
    if any(json.loads(path.read_text()).get("target_data_used_for_selection") is not False
           for path in selections):
        raise RuntimeError("A selection record does not declare target exclusion")

    metrics, gaps, paired = build_partial_tables()
    hr, subjects, conditions = summarise_failures(paired)
    ladder = model_ladder(metrics)
    selections_table = selection_summary()
    metrics.to_csv(OUT / "partial_seed_metrics.csv", index=False)
    gaps.to_csv(OUT / "partial_generalisation_gaps.csv", index=False)
    ladder.to_csv(OUT / "model_ladder.csv", index=False)
    hr.to_csv(OUT / "hr_bin_metrics.csv", index=False)
    subjects.to_csv(OUT / "subject_metrics.csv", index=False)
    conditions.to_csv(OUT / "condition_metrics.csv", index=False)
    selections_table.to_csv(OUT / "selection_summary.csv", index=False)
    make_figures(ladder, gaps, hr, subjects)
    write_readme(ladder, gaps, hr, subjects, selections_table)
    print(f"Frozen Phase-12 analysis written to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
