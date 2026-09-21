#!/usr/bin/env python3
"""Evaluate proximal-trained PTT models on familiar pleth_4 and distal pleth_1."""

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


MANIFEST = Path("reports/phase7_frozen_dataset/ptt_site_window_manifest.csv")
MODEL_ROOT = Path("reports/phase11_ptt_site_transfer/proximal_training/ptt_ppg")
OUT = Path("reports/phase11_ptt_site_transfer/reverse_proximal_to_distal")
KEY = ["dataset", "record_id", "subject_id", "window_index"]
SEED = 17


def infer(model, frame, device):
    data = DataLoader(ManifestPPGDataset(frame), batch_size=256, shuffle=False, num_workers=0)
    reference, prediction, indices = predict(model, data, device=device)
    result = frame.iloc[indices][KEY + ["window_start_s", "condition_name"]].copy()
    result["reference_hr_bpm"] = reference
    result["prediction_bpm"] = prediction
    result["error_bpm"] = prediction - reference
    result["absolute_error_bpm"] = np.abs(result.error_bpm)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    folds = []
    for fold in range(3):
        run = MODEL_ROOT / f"test_fold_{fold}" / f"seed_{SEED}"
        if not (run / "metrics.json").is_file():
            raise RuntimeError(f"proximal training fold {fold} is not complete: run run_ptt_proximal_training.sh")
        meta = json.loads((run / "metrics.json").read_text())
        subjects = set(map(str, meta["subjects"]["test"]))
        familiar = manifest.loc[manifest.subject_id.astype(str).isin(subjects) & manifest.channel_name.eq("pleth_4")].reset_index(drop=True)
        transfer = manifest.loc[manifest.subject_id.astype(str).isin(subjects) & manifest.channel_name.eq("pleth_1")].reset_index(drop=True)
        model = TimePPGBigPPGOnly(512).to(device)
        model.load_state_dict(torch.load(run / "best_checkpoint.pt", map_location=device, weights_only=False)["model_state"])
        a = infer(model, familiar, device).rename(columns={
            "prediction_bpm": "proximal_prediction_bpm", "error_bpm": "proximal_error_bpm",
            "absolute_error_bpm": "proximal_absolute_error_bpm",
        })
        b = infer(model, transfer, device)[KEY + ["window_start_s", "reference_hr_bpm", "prediction_bpm", "error_bpm", "absolute_error_bpm"]].rename(columns={
            "window_start_s": "distal_window_start_s", "reference_hr_bpm": "distal_reference_hr_bpm",
            "prediction_bpm": "distal_prediction_bpm", "error_bpm": "distal_error_bpm",
            "absolute_error_bpm": "distal_absolute_error_bpm",
        })
        paired = a.merge(b, on=KEY, validate="one_to_one")
        if len(paired) != len(a) or len(a) != len(b): raise RuntimeError("site-window pairing failed")
        if not np.allclose(paired.reference_hr_bpm, paired.distal_reference_hr_bpm, rtol=0, atol=1e-4):
            raise RuntimeError("site reference labels differ")
        paired["fold"] = fold
        paired["reverse_site_gap_bpm"] = paired.distal_absolute_error_bpm - paired.proximal_absolute_error_bpm
        folds.append(paired)
    paired = pd.concat(folds, ignore_index=True)
    if len(paired) != 15_982 or paired.duplicated(KEY).any(): raise RuntimeError("incomplete pooled reverse test")
    paired.to_csv(OUT / "paired_window_predictions.csv", index=False)

    reference = paired.reference_hr_bpm.to_numpy()
    rows = []
    for site, column in [("proximal pleth_4 familiar", "proximal_prediction_bpm"),
                         ("distal pleth_1 transfer", "distal_prediction_bpm")]:
        rows.append({"site": site, **hr_metrics(reference, paired[column].to_numpy())})
    metric = pd.DataFrame(rows); metric.to_csv(OUT / "site_metrics.csv", index=False)
    subject = paired.groupby("subject_id").agg(
        windows=("window_index", "size"),
        proximal_mae_bpm=("proximal_absolute_error_bpm", "mean"),
        distal_mae_bpm=("distal_absolute_error_bpm", "mean"),
        reverse_site_gap_bpm=("reverse_site_gap_bpm", "mean"),
    ).reset_index()
    subject.to_csv(OUT / "subject_site_metrics.csv", index=False)
    rng = np.random.default_rng(SEED); gaps = subject.reverse_site_gap_bpm.to_numpy()
    boot = gaps[rng.integers(0, len(gaps), size=(20_000, len(gaps)))].mean(1)

    fig, ax = plt.subplots(figsize=(8, 7))
    for _, row in subject.iterrows():
        ax.plot([0, 1], [row.proximal_mae_bpm, row.distal_mae_bpm],
                color="#E76F51" if row.reverse_site_gap_bpm > 0 else "#2A9D8F", alpha=.65)
    ax.set_xticks([0, 1], ["Proximal pleth_4\nfamiliar", "Distal pleth_1\nunseen-site transfer"])
    ax.set_ylabel("Subject MAE (bpm)"); ax.set_title("Reverse PTT site transfer for the same held-out subjects")
    ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(figures / "subject_reverse_site_change.png", dpi=220); plt.close(fig)

    familiar_mae = metric.iloc[0].mae_bpm; transfer_mae = metric.iloc[1].mae_bpm
    report = f"""# Phase 11 reverse PTT sensor-site transfer — seed {SEED}

- Proximal `pleth_4` familiar-site MAE: **{familiar_mae:.3f} bpm**.
- Distal `pleth_1` unseen-site MAE: **{transfer_mae:.3f} bpm**.
- Reverse pooled site gap: **{transfer_mae-familiar_mae:+.3f} bpm**.
- Subject-macro reverse gap: **{gaps.mean():+.3f} bpm**, subject-bootstrap 95%
  interval **{np.quantile(boot,.025):+.3f} to {np.quantile(boot,.975):+.3f} bpm**.
- Distal transfer is worse for **{np.sum(gaps>0)}/22 subjects**.

This reverse direction must be interpreted together with the existing distal-to-proximal
result. Agreement in both directions supports general site sensitivity; improvement when
moving to distal supports intrinsically greater usability of the distal channel.
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
