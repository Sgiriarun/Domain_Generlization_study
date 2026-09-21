#!/usr/bin/env python3
"""Phase 11: paired PTT pleth_1-to-pleth_4 TimePPG site-transfer test.

Each completed pleth_1 within-dataset checkpoint is evaluated on pleth_1 and
its synchronized paired pleth_4 channel for exactly the same held-out windows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rq1_hr.evaluation import hr_metrics
from rq1_hr.models.timeppg import ManifestPPGDataset, TimePPGBigPPGOnly, predict


SITE_MANIFEST = Path("reports/phase7_frozen_dataset/ptt_site_window_manifest.csv")
WITHIN_ROOT = Path("reports/phase9_timeppg/within_dataset_3fold/ptt_ppg")
OUT = Path("reports/phase11_ptt_site_transfer")
SEED = 17
SOURCE_CHANNEL = "pleth_1"
TARGET_CHANNEL = "pleth_4"
PAIR_KEY = ["dataset", "record_id", "subject_id", "window_index"]
HR_BINS = [-np.inf, 60, 80, 100, 120, 140, np.inf]
HR_LABELS = ["<60", "60-80", "80-100", "100-120", "120-140", ">=140"]


def loader(frame: pd.DataFrame) -> DataLoader:
    return DataLoader(ManifestPPGDataset(frame), batch_size=256, shuffle=False, num_workers=0)


def evaluate(model: torch.nn.Module, frame: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    references, predictions, row_indices = predict(model, loader(frame), device=device)
    result = frame.iloc[row_indices][PAIR_KEY + ["window_start_s", "condition_name", "hr_bpm"]].copy()
    result["reference_hr_bpm"] = references
    result["prediction_bpm"] = predictions
    result["error_bpm"] = predictions - references
    result["absolute_error_bpm"] = np.abs(result.error_bpm)
    return result


def metric_record(frame: pd.DataFrame, label: str, **identity) -> dict:
    values = hr_metrics(frame.reference_hr_bpm.to_numpy(), frame[f"{label}_prediction_bpm"].to_numpy())
    error = frame[f"{label}_error_bpm"].to_numpy()
    return {
        **identity, "windows": len(frame), **values,
        "p95_absolute_error_bpm": np.quantile(np.abs(error), .95),
        "over_20_bpm_percent": 100 * np.mean(np.abs(error) > 20),
    }


def subject_bootstrap(subject: pd.DataFrame, draws: int = 20_000) -> dict:
    rng = np.random.default_rng(SEED)
    values = subject.site_gap_bpm.to_numpy()
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "subjects": len(values),
        "subject_macro_site_gap_bpm": values.mean(),
        "ci_low_bpm": np.quantile(sampled, .025),
        "ci_high_bpm": np.quantile(sampled, .975),
        "subjects_proximal_worse": int(np.sum(values > 0)),
        "subjects_proximal_worse_percent": 100 * np.mean(values > 0),
        "bootstrap_draws": draws,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    manifest = pd.read_csv(SITE_MANIFEST, low_memory=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    fold_pairs = []

    for fold in range(3):
        run = WITHIN_ROOT / f"test_fold_{fold}" / f"seed_{SEED}"
        metadata = json.loads((run / "metrics.json").read_text())
        test_subjects = set(map(str, metadata["subjects"]["test"]))
        source = manifest.loc[
            manifest.subject_id.astype(str).isin(test_subjects) & manifest.channel_name.eq(SOURCE_CHANNEL)
        ].reset_index(drop=True)
        target = manifest.loc[
            manifest.subject_id.astype(str).isin(test_subjects) & manifest.channel_name.eq(TARGET_CHANNEL)
        ].reset_index(drop=True)
        if source.duplicated(PAIR_KEY).any() or target.duplicated(PAIR_KEY).any():
            raise RuntimeError(f"duplicate synchronized window in fold {fold}")
        audit = source[PAIR_KEY].merge(target[PAIR_KEY], on=PAIR_KEY, how="outer", indicator=True)
        if len(source) != len(target) or not audit._merge.eq("both").all():
            raise RuntimeError(f"unpaired site windows in fold {fold}")

        model = TimePPGBigPPGOnly(512).to(device)
        saved = torch.load(run / "best_checkpoint.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        source_prediction = evaluate(model, source, device).rename(columns={
            "prediction_bpm": "distal_prediction_bpm", "error_bpm": "distal_error_bpm",
            "absolute_error_bpm": "distal_absolute_error_bpm",
        })
        target_prediction = evaluate(model, target, device)[PAIR_KEY + [
            "window_start_s", "reference_hr_bpm", "prediction_bpm", "error_bpm", "absolute_error_bpm"
        ]].rename(columns={
            "window_start_s": "proximal_window_start_s",
            "reference_hr_bpm": "proximal_reference_hr_bpm",
            "prediction_bpm": "proximal_prediction_bpm", "error_bpm": "proximal_error_bpm",
            "absolute_error_bpm": "proximal_absolute_error_bpm",
        })
        paired = source_prediction.merge(target_prediction, on=PAIR_KEY, validate="one_to_one")
        if not np.allclose(paired.window_start_s, paired.proximal_window_start_s, rtol=0, atol=1e-8):
            raise RuntimeError(f"site timestamp mismatch in fold {fold}")
        if not np.allclose(paired.reference_hr_bpm, paired.proximal_reference_hr_bpm, rtol=0, atol=1e-4):
            raise RuntimeError(f"site label mismatch in fold {fold}")
        paired["fold"] = fold
        paired["site_gap_bpm"] = paired.proximal_absolute_error_bpm - paired.distal_absolute_error_bpm
        paired["activity"] = paired.condition_name.astype(str)
        paired["hr_bin"] = pd.cut(paired.reference_hr_bpm, HR_BINS, labels=HR_LABELS, right=False)
        fold_pairs.append(paired)

    paired = pd.concat(fold_pairs, ignore_index=True)
    if paired.duplicated(PAIR_KEY).any() or len(paired) != 15_982:
        raise RuntimeError("pooled folds do not cover every PTT window exactly once")
    paired.to_csv(OUT / "paired_window_predictions.csv", index=False)

    metric_rows = []
    for fold, group in paired.groupby("fold"):
        metric_rows += [metric_record(group, "distal", fold=fold, site="distal pleth_1"),
                        metric_record(group, "proximal", fold=fold, site="proximal pleth_4")]
    metric_rows += [metric_record(paired, "distal", fold="pooled", site="distal pleth_1"),
                    metric_record(paired, "proximal", fold="pooled", site="proximal pleth_4")]
    metrics_table = pd.DataFrame(metric_rows)
    metrics_table.to_csv(OUT / "site_metrics.csv", index=False)

    subject_rows = []
    for subject, group in paired.groupby("subject_id"):
        subject_rows.append({
            "subject_id": subject, "windows": len(group),
            "distal_mae_bpm": group.distal_absolute_error_bpm.mean(),
            "proximal_mae_bpm": group.proximal_absolute_error_bpm.mean(),
            "site_gap_bpm": group.site_gap_bpm.mean(),
            "distal_bias_bpm": group.distal_error_bpm.mean(),
            "proximal_bias_bpm": group.proximal_error_bpm.mean(),
        })
    subject = pd.DataFrame(subject_rows)
    subject.to_csv(OUT / "subject_site_metrics.csv", index=False)
    bootstrap = subject_bootstrap(subject)
    pd.DataFrame([bootstrap]).to_csv(OUT / "subject_bootstrap.csv", index=False)

    strata_rows = []
    for columns in [["activity"], ["hr_bin"], ["activity", "hr_bin"]]:
        for values, group in paired.groupby(columns, observed=True):
            if not isinstance(values, tuple): values = (values,)
            row = {column: value for column, value in zip(columns, values)}
            row.update({
                "stratification": "_x_".join(columns), "windows": len(group),
                "subjects": group.subject_id.nunique(),
                "distal_mae_bpm": group.distal_absolute_error_bpm.mean(),
                "proximal_mae_bpm": group.proximal_absolute_error_bpm.mean(),
                "site_gap_bpm": group.site_gap_bpm.mean(),
            })
            strata_rows.append(row)
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(OUT / "stratified_site_metrics.csv", index=False)

    # Overall and fold comparison.
    plot = metrics_table.loc[metrics_table.fold.ne("pooled")].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(3); width = .36
    distal = plot.loc[plot.site.eq("distal pleth_1")].sort_values("fold")
    proximal = plot.loc[plot.site.eq("proximal pleth_4")].sort_values("fold")
    ax.bar(x-width/2, distal.mae_bpm, width, label="Distal pleth_1 (familiar site)", color="#2A9D8F")
    ax.bar(x+width/2, proximal.mae_bpm, width, label="Proximal pleth_4 (unseen site)", color="#E76F51")
    ax.set_xticks(x, [f"Fold {i}" for i in x]); ax.set_ylabel("MAE (bpm)")
    ax.set_title("Same pleth_1-trained model, same held-out windows, different sensor site")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(figures / "fold_site_transfer_mae.png", dpi=220); plt.close(fig)

    # Subject-paired comparison.
    shown = subject.sort_values("site_gap_bpm", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 8))
    for _, row in shown.iterrows():
        color = "#E76F51" if row.site_gap_bpm > 0 else "#2A9D8F"
        ax.plot([0, 1], [row.distal_mae_bpm, row.proximal_mae_bpm], color=color, alpha=.65)
    ax.scatter(np.zeros(len(shown)), shown.distal_mae_bpm, color="#2A9D8F")
    ax.scatter(np.ones(len(shown)), shown.proximal_mae_bpm, color="#E76F51")
    ax.set_xticks([0, 1], ["Distal pleth_1\ntrained/evaluated", "Proximal pleth_4\nunseen-site evaluation"])
    ax.set_ylabel("Subject MAE (bpm)"); ax.set_title("Paired site change for every PTT subject")
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(figures / "subject_paired_site_change.png", dpi=220); plt.close(fig)

    # HR-range localisation of familiar-site error and added site-transfer error.
    by_hr = strata.loc[strata.stratification.eq("hr_bin")].set_index("hr_bin").reindex(HR_LABELS)
    x = np.arange(len(by_hr)); width = .36
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(x-width/2, by_hr.distal_mae_bpm, width, label="Distal familiar site", color="#2A9D8F")
    axes[0].bar(x+width/2, by_hr.proximal_mae_bpm, width, label="Proximal unseen site", color="#E76F51")
    axes[0].set(title="Total error at each site", ylabel="MAE (bpm)")
    axes[0].legend(frameon=False)
    colors = np.where(by_hr.site_gap_bpm >= 0, "#E76F51", "#2A9D8F")
    axes[1].bar(x, by_hr.site_gap_bpm, color=colors)
    axes[1].axhline(0, color="black", linewidth=.8)
    axes[1].set(title="Extra error after distal → proximal transfer", ylabel="Proximal MAE − distal MAE (bpm)")
    axes[1].set_ylim(min(-6, by_hr.site_gap_bpm.min() - 1.5), by_hr.site_gap_bpm.max() + 2.2)
    for ax in axes:
        ax.set_xticks(x, HR_LABELS, rotation=25)
        ax.set_xlabel("ECG-derived reference HR bin")
        ax.grid(axis="y", alpha=.2)
    for index, row in enumerate(by_hr.itertuples()):
        axes[1].text(index, row.site_gap_bpm + (.45 if row.site_gap_bpm >= 0 else -.45),
                     f"n={int(row.windows)}\nS={int(row.subjects)}", ha="center",
                     va="bottom" if row.site_gap_bpm >= 0 else "top", fontsize=8)
    fig.suptitle("PTT site sensitivity depends on the true-HR range")
    fig.tight_layout(); fig.savefig(figures / "hr_range_site_transfer.png", dpi=220); plt.close(fig)

    pooled = metrics_table.loc[metrics_table.fold.eq("pooled")].set_index("site")
    distal_mae = pooled.loc["distal pleth_1", "mae_bpm"]
    proximal_mae = pooled.loc["proximal pleth_4", "mae_bpm"]
    report = f"""# Phase 11: controlled PTT sensor-site transfer — seed {SEED}

## Design

- The completed three-fold TimePPG models were trained on distal `pleth_1`.
- Each model was evaluated on familiar distal `pleth_1` and synchronized paired
  proximal `pleth_4` for the exact same held-out subjects and 8-second windows.
- Subject, activity, recording time, reference HR, split and model checkpoint are
  held fixed. The changed input is sensor placement/channel within pair 1/4.
- All 15,982 primary PTT windows are covered once across the three test folds.

## Result

- Distal familiar-site MAE: **{distal_mae:.3f} bpm**.
- Proximal unseen-site MAE: **{proximal_mae:.3f} bpm**.
- Pooled window-level site gap: **{proximal_mae-distal_mae:+.3f} bpm**.
- Subject-macro site gap: **{bootstrap['subject_macro_site_gap_bpm']:+.3f} bpm**,
  subject-bootstrap 95% interval **{bootstrap['ci_low_bpm']:+.3f} to {bootstrap['ci_high_bpm']:+.3f} bpm**.
- Proximal is worse for **{bootstrap['subjects_proximal_worse']}/22 subjects**.

## Interpretation boundary

This is a controlled one-direction transfer diagnostic: a `pleth_1`-trained model
is challenged with paired `pleth_4`. It provides stronger evidence that placement/
channel shift changes model usability, but it is not yet a symmetric site experiment.
Training on `pleth_4` and reversing the test direction is required next. Wavelength
names remain neutral because the local documentation versions conflict, although the
1/4 pairing and distal/proximal placement are documented.
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
