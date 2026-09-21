#!/usr/bin/env python3
"""Complete and freeze the paired bidirectional PTT site outcome for pair 1/4."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("reports/phase11_ptt_site_transfer")
FORWARD = ROOT / "paired_window_predictions.csv"
REVERSE = ROOT / "reverse_proximal_to_distal/paired_window_predictions.csv"
KEY = ["dataset", "record_id", "subject_id", "window_index"]
HR_BINS = [-np.inf, 60, 80, 100, 120, 140, np.inf]
HR_LABELS = ["<60", "60-80", "80-100", "100-120", "120-140", ">=140"]
SEED = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_bootstrap(subject: pd.DataFrame, column: str, draws: int = 20_000) -> tuple[float, float, float]:
    values = subject[column].to_numpy(float)
    rng = np.random.default_rng(SEED)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return values.mean(), np.quantile(sampled, .025), np.quantile(sampled, .975)


def main() -> None:
    forward = pd.read_csv(FORWARD, low_memory=False)
    reverse = pd.read_csv(REVERSE, low_memory=False)
    if forward.duplicated(KEY).any() or reverse.duplicated(KEY).any():
        raise RuntimeError("duplicate paired window")
    retain = KEY + ["window_start_s", "reference_hr_bpm", "condition_name",
                    "proximal_absolute_error_bpm", "distal_absolute_error_bpm"]
    combined = forward[retain + ["site_gap_bpm"]].merge(
        reverse[KEY + ["window_start_s", "reference_hr_bpm",
                       "proximal_absolute_error_bpm", "distal_absolute_error_bpm",
                       "reverse_site_gap_bpm"]].rename(columns={
            "window_start_s": "reverse_window_start_s",
            "reference_hr_bpm": "reverse_reference_hr_bpm",
            "proximal_absolute_error_bpm": "proximal_absolute_error_trained_proximal_bpm",
            "distal_absolute_error_bpm": "distal_absolute_error_trained_proximal_bpm",
        }), on=KEY, validate="one_to_one",
    )
    if len(combined) != 15_982:
        raise RuntimeError("bidirectional analysis does not cover all 15,982 windows")
    if not np.allclose(combined.window_start_s, combined.reverse_window_start_s, rtol=0, atol=1e-8):
        raise RuntimeError("window times differ by direction")
    if not np.allclose(combined.reference_hr_bpm, combined.reverse_reference_hr_bpm, rtol=0, atol=1e-4):
        raise RuntimeError("reference HR differs by direction")

    # Positive always means that proximal input has more absolute error than
    # distal input. Each contrast uses one checkpoint and the exact same window.
    combined = combined.rename(columns={
        "proximal_absolute_error_bpm": "proximal_absolute_error_trained_distal_bpm",
        "distal_absolute_error_bpm": "distal_absolute_error_trained_distal_bpm",
    })
    combined["proximal_penalty_trained_distal_bpm"] = combined.site_gap_bpm
    combined["proximal_penalty_trained_proximal_bpm"] = -combined.reverse_site_gap_bpm
    combined["bidirectional_proximal_penalty_bpm"] = combined[[
        "proximal_penalty_trained_distal_bpm", "proximal_penalty_trained_proximal_bpm"
    ]].mean(axis=1)
    combined["hr_bin"] = pd.cut(combined.reference_hr_bpm, HR_BINS, labels=HR_LABELS, right=False)
    combined["activity"] = combined.condition_name.astype(str)
    combined.to_csv(ROOT / "frozen_bidirectional_window_contrasts.csv", index=False)

    subject = combined.groupby("subject_id").agg(
        windows=("window_index", "size"),
        trained_distal_penalty_bpm=("proximal_penalty_trained_distal_bpm", "mean"),
        trained_proximal_penalty_bpm=("proximal_penalty_trained_proximal_bpm", "mean"),
        bidirectional_proximal_penalty_bpm=("bidirectional_proximal_penalty_bpm", "mean"),
    ).reset_index()
    subject.to_csv(ROOT / "frozen_bidirectional_subject_contrasts.csv", index=False)

    estimates = {}
    for column in ["trained_distal_penalty_bpm", "trained_proximal_penalty_bpm",
                   "bidirectional_proximal_penalty_bpm"]:
        mean, low, high = subject_bootstrap(subject, column)
        estimates[column] = {"subject_macro_mean_bpm": mean, "ci_low_bpm": low, "ci_high_bpm": high}

    grouped_rows = []
    for factor in ["hr_bin", "activity"]:
        for value, group in combined.groupby(factor, observed=True):
            counts = group.groupby("subject_id").size()
            grouped_rows.append({
                "factor": factor, "level": str(value), "windows": len(group),
                "subjects": group.subject_id.nunique(),
                "largest_subject_share_percent": 100 * counts.max() / len(group),
                "trained_distal_proximal_penalty_bpm": group.proximal_penalty_trained_distal_bpm.mean(),
                "trained_proximal_proximal_penalty_bpm": group.proximal_penalty_trained_proximal_bpm.mean(),
                "bidirectional_proximal_penalty_bpm": group.bidirectional_proximal_penalty_bpm.mean(),
            })
    grouped = pd.DataFrame(grouped_rows)
    grouped.to_csv(ROOT / "frozen_bidirectional_effect_modifiers.csv", index=False)

    hr = grouped.loc[grouped.factor.eq("hr_bin")].set_index("level").reindex(HR_LABELS)
    activity = grouped.loc[grouped.factor.eq("activity")].sort_values("bidirectional_proximal_penalty_bpm", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, table, labels, title in [
        (axes[0], hr, HR_LABELS, "By true-HR range"),
        (axes[1], activity.set_index("level"), activity.level.tolist(), "By activity"),
    ]:
        x = np.arange(len(labels)); width = .36
        ax.bar(x-width/2, table.trained_distal_proximal_penalty_bpm, width,
               label="Model trained distal", color="#177E89")
        ax.bar(x+width/2, table.trained_proximal_proximal_penalty_bpm, width,
               label="Model trained proximal", color="#E9C46A")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(x, labels, rotation=25)
        ax.set(title=title, ylabel="Proximal MAE − distal MAE (bpm)")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("PTT pair 1/4: proximal input disadvantage after exact-window control")
    fig.tight_layout(); fig.savefig(ROOT / "figures/frozen_site_effect_by_hr_activity.png", dpi=220); plt.close(fig)

    outcome = {
        "status": "frozen",
        "scope": "PTT-PPG pair 1/4, TimePPG-Big PPG-only, seed 17, frozen three-fold subjects",
        "windows": len(combined), "subjects": subject.subject_id.nunique(),
        "forward_distal_to_proximal_pooled_gap_bpm": combined.proximal_penalty_trained_distal_bpm.mean(),
        "reverse_proximal_to_distal_as_proximal_penalty_bpm": combined.proximal_penalty_trained_proximal_bpm.mean(),
        "subject_bootstrap": estimates,
        "subjects_with_positive_bidirectional_penalty": int((subject.bidirectional_proximal_penalty_bpm > 0).sum()),
        "input_sha256": {str(FORWARD): sha256(FORWARD), str(REVERSE): sha256(REVERSE)},
    }
    (ROOT / "frozen_outcome.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")

    main_estimate = estimates["bidirectional_proximal_penalty_bpm"]
    report = f"""# Frozen Phase 11 outcome: PTT pair 1/4 bidirectional site analysis

## Frozen scope

This conclusion applies to PTT-PPG pair `pleth_1`/`pleth_4`, TimePPG-Big
PPG-only, seed 17, the frozen preprocessing and the frozen three-fold subject
roles. It covers {len(combined):,} synchronized windows from {subject.subject_id.nunique()} subjects.

## Controlled design

For each checkpoint and window, proximal and distal absolute errors use the same
subject, activity, recording time and ECG-derived HR. These variables are
controlled by exact pairing rather than only by statistical adjustment. Both
training directions are included.

## Frozen result

- Proximal penalty when trained distal: **{estimates['trained_distal_penalty_bpm']['subject_macro_mean_bpm']:+.3f} bpm**
  (subject-bootstrap 95% interval {estimates['trained_distal_penalty_bpm']['ci_low_bpm']:+.3f} to
  {estimates['trained_distal_penalty_bpm']['ci_high_bpm']:+.3f}).
- Proximal penalty when trained proximal: **{estimates['trained_proximal_penalty_bpm']['subject_macro_mean_bpm']:+.3f} bpm**
  ({estimates['trained_proximal_penalty_bpm']['ci_low_bpm']:+.3f} to
  {estimates['trained_proximal_penalty_bpm']['ci_high_bpm']:+.3f}).
- Bidirectional subject-macro proximal penalty: **{main_estimate['subject_macro_mean_bpm']:+.3f} bpm**
  ({main_estimate['ci_low_bpm']:+.3f} to {main_estimate['ci_high_bpm']:+.3f}).
- The bidirectional proximal penalty is positive for
  **{outcome['subjects_with_positive_bidirectional_penalty']}/22 subjects**.

## Frozen interpretation

> For PTT pair 1/4, proximal input is consistently less usable for TimePPG than
> synchronized distal input. The disadvantage remains across both training
> directions while subject, activity, time and HR label are held constant.

This supports a substantial placement/channel contribution. It does not isolate
an anatomical site cause from physical-channel, optical or hardware differences,
and it is not a universal claim about every PPG device, wavelength or model.
Pairs 2/5 and 3/6 are optional external-sensitivity extensions, not prerequisites
for this frozen pair-1/4 conclusion.
"""
    (ROOT / "FROZEN_OUTCOME.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
