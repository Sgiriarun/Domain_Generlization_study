#!/usr/bin/env python3
"""Evaluate and freeze the bidirectional TimePPG result for all three PTT pairs."""

from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path("reports/phase11_ptt_site_transfer")
OUT = ROOT / "pair_sensitivity"
MANIFEST = Path("reports/phase7_frozen_dataset/ptt_site_window_manifest.csv")
KEY = ["dataset", "record_id", "subject_id", "window_index"]
PAIRS = {
    "pair_1_4": ("pleth_1", "pleth_4"),
    "pair_2_5": ("pleth_2", "pleth_5"),
    "pair_3_6": ("pleth_3", "pleth_6"),
}


def model_root(channel: str) -> Path:
    if channel == "pleth_1":
        return Path("reports/phase9_timeppg/within_dataset_3fold/ptt_ppg")
    if channel == "pleth_4":
        return ROOT / "proximal_training/ptt_ppg"
    return OUT / f"training/{channel}/ptt_ppg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer(model: torch.nn.Module, frame: pd.DataFrame, device: torch.device) -> np.ndarray:
    loader = DataLoader(ManifestPPGDataset(frame), batch_size=256, shuffle=False, num_workers=0)
    references, predictions, indices = predict(model, loader, device=device)
    if not np.array_equal(indices, np.arange(len(frame))):
        raise RuntimeError("inference changed manifest row order")
    if not np.allclose(references, frame.hr_bpm.to_numpy(), rtol=0, atol=1e-4):
        raise RuntimeError("inference labels differ from frozen manifest")
    return predictions


def bootstrap(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"; figures.mkdir(exist_ok=True)
    # Do not spend time evaluating a partial experiment or write a misleading
    # partial freeze. Every pair, direction and fold must exist first.
    missing = []
    for distal_channel, proximal_channel in PAIRS.values():
        for channel in (distal_channel, proximal_channel):
            for fold in range(3):
                run = model_root(channel) / f"test_fold_{fold}/seed_{args.seed}"
                for filename in ("metrics.json", "best_checkpoint.pt"):
                    if not (run / filename).is_file():
                        missing.append(str(run / filename))
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:12])
        raise RuntimeError(f"cannot freeze an incomplete all-pair experiment; missing:\n{preview}")

    manifest = pd.read_csv(MANIFEST, low_memory=False)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    all_windows, metric_rows, hashes = [], [], {}

    for pair, (distal_channel, proximal_channel) in PAIRS.items():
        pair_parts = []
        for fold in range(3):
            base = manifest.loc[manifest.channel_name.eq(distal_channel)].copy()
            run_meta = model_root(distal_channel) / f"test_fold_{fold}/seed_{args.seed}/metrics.json"
            if not run_meta.is_file():
                raise RuntimeError(f"missing completed run: {run_meta}")
            subjects = set(map(str, json.loads(run_meta.read_text())["subjects"]["test"]))
            distal = base.loc[base.subject_id.astype(str).isin(subjects)].sort_values(KEY).reset_index(drop=True)
            proximal = manifest.loc[
                manifest.channel_name.eq(proximal_channel)
                & manifest.subject_id.astype(str).isin(subjects)
            ].sort_values(KEY).reset_index(drop=True)
            if len(distal) != len(proximal) or not distal[KEY].equals(proximal[KEY]):
                raise RuntimeError(f"unpaired windows for {pair}, fold {fold}")
            if not np.allclose(distal.hr_bpm, proximal.hr_bpm, rtol=0, atol=1e-4):
                raise RuntimeError(f"reference mismatch for {pair}, fold {fold}")

            result = distal[KEY + ["window_start_s", "condition_name", "hr_bpm"]].copy()
            result["pair"] = pair; result["fold"] = fold
            for training_channel, training_site in [(distal_channel, "distal"), (proximal_channel, "proximal")]:
                checkpoint = model_root(training_channel) / f"test_fold_{fold}/seed_{args.seed}/best_checkpoint.pt"
                if not checkpoint.is_file():
                    raise RuntimeError(f"missing checkpoint: {checkpoint}")
                hashes[str(checkpoint)] = sha256(checkpoint)
                model = TimePPGBigPPGOnly(512).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
                result[f"distal_prediction_trained_{training_site}_bpm"] = infer(model, distal, device)
                result[f"proximal_prediction_trained_{training_site}_bpm"] = infer(model, proximal, device)
            for training_site in ["distal", "proximal"]:
                for test_site in ["distal", "proximal"]:
                    pred = result[f"{test_site}_prediction_trained_{training_site}_bpm"]
                    result[f"{test_site}_absolute_error_trained_{training_site}_bpm"] = np.abs(pred - result.hr_bpm)
            result["proximal_penalty_trained_distal_bpm"] = (
                result.proximal_absolute_error_trained_distal_bpm - result.distal_absolute_error_trained_distal_bpm
            )
            result["proximal_penalty_trained_proximal_bpm"] = (
                result.proximal_absolute_error_trained_proximal_bpm - result.distal_absolute_error_trained_proximal_bpm
            )
            result["bidirectional_proximal_penalty_bpm"] = result[[
                "proximal_penalty_trained_distal_bpm", "proximal_penalty_trained_proximal_bpm"
            ]].mean(axis=1)
            pair_parts.append(result)

        paired = pd.concat(pair_parts, ignore_index=True)
        if len(paired) != 15_982 or paired.duplicated(KEY).any():
            raise RuntimeError(f"{pair} does not cover every window exactly once")
        all_windows.append(paired)
        for training_site in ["distal", "proximal"]:
            for test_site in ["distal", "proximal"]:
                prediction = paired[f"{test_site}_prediction_trained_{training_site}_bpm"].to_numpy()
                metric_rows.append({
                    "pair": pair, "train_site": training_site, "test_site": test_site,
                    "train_channel": distal_channel if training_site == "distal" else proximal_channel,
                    "test_channel": distal_channel if test_site == "distal" else proximal_channel,
                    "windows": len(paired), **hr_metrics(paired.hr_bpm.to_numpy(), prediction),
                })

    windows = pd.concat(all_windows, ignore_index=True)
    windows.to_csv(OUT / "all_pair_window_contrasts.csv", index=False)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT / "all_pair_train_test_metrics.csv", index=False)
    subject = windows.groupby(["pair", "subject_id"]).agg(
        windows=("window_index", "size"),
        trained_distal_penalty_bpm=("proximal_penalty_trained_distal_bpm", "mean"),
        trained_proximal_penalty_bpm=("proximal_penalty_trained_proximal_bpm", "mean"),
        bidirectional_proximal_penalty_bpm=("bidirectional_proximal_penalty_bpm", "mean"),
    ).reset_index()
    subject.to_csv(OUT / "all_pair_subject_contrasts.csv", index=False)

    summaries = []
    for pair, group in subject.groupby("pair", sort=False):
        values = group.bidirectional_proximal_penalty_bpm.to_numpy()
        low, high = bootstrap(values, args.seed)
        summaries.append({
            "pair": pair, "subjects": len(group), "subject_macro_proximal_penalty_bpm": values.mean(),
            "ci_low_bpm": low, "ci_high_bpm": high,
            "subjects_positive": int(np.sum(values > 0)),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "frozen_all_pair_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(summary)); means = summary.subject_macro_proximal_penalty_bpm.to_numpy()
    errors = np.vstack([means-summary.ci_low_bpm, summary.ci_high_bpm-means])
    colors = np.where(means >= 0, "#E76F51", "#2A9D8F")
    ax.bar(x, means, color=colors, width=.58)
    ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="#132A3A", capsize=5, linewidth=1.5)
    ax.axhline(0, color="#132A3A", linewidth=.9)
    ax.set_xticks(x, [value.replace("pair_", "Pair ").replace("_", "/") for value in summary.pair])
    ax.set_ylabel("Proximal MAE − distal MAE (bpm)")
    ax.set_title("Does the proximal-input disadvantage repeat across PTT channel pairs?")
    ax.grid(axis="y", alpha=.2)
    for index, row in summary.iterrows():
        ax.text(index, row.subject_macro_proximal_penalty_bpm + .18,
                f"{row.subject_macro_proximal_penalty_bpm:+.2f} bpm\n{int(row.subjects_positive)}/22 subjects",
                ha="center", va="bottom", fontweight="bold")
    fig.tight_layout(); fig.savefig(figures / "all_pair_proximal_penalty.png", dpi=220); plt.close(fig)

    frozen = {
        "status": "frozen", "seed": args.seed,
        "scope": "TimePPG-Big PPG-only; PTT pairs 1/4, 2/5 and 3/6; frozen three-fold subjects",
        "windows_per_pair": 15_982, "subjects_per_pair": 22,
        "interpretation": (
            "Positive values mean proximal input has higher absolute HR error than synchronized distal input, "
            "averaged across both training directions. Placement and physical channel remain jointly changed."
        ),
        "pair_results": summary.to_dict(orient="records"),
        "checkpoint_sha256": hashes,
    }
    (OUT / "frozen_all_pair_outcome.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    lines = [
        "# Frozen PTT all-pair site-sensitivity outcome", "",
        "Each comparison uses the same held-out person, activity, recording time, 8-second window and ECG-derived HR.", "",
    ]
    for row in summary.itertuples():
        lines.append(
            f"- {row.pair}: proximal penalty **{row.subject_macro_proximal_penalty_bpm:+.3f} bpm** "
            f"(subject-bootstrap 95% interval {row.ci_low_bpm:+.3f} to {row.ci_high_bpm:+.3f}); "
            f"positive for **{row.subjects_positive}/22 subjects**."
        )
    lines += ["", "Positive means proximal input produced more error. This is controlled placement/channel evidence, not a universal anatomical-site claim."]
    (OUT / "FROZEN_ALL_PAIR_OUTCOME.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
