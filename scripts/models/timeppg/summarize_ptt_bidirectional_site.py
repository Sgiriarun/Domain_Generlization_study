#!/usr/bin/env python3
"""Create a clear combined summary of both Phase-11 PTT site directions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("reports/phase11_ptt_site_transfer")


def main() -> None:
    forward = pd.read_csv(ROOT / "subject_site_metrics.csv")
    reverse = pd.read_csv(ROOT / "reverse_proximal_to_distal/subject_site_metrics.csv")
    subject = forward[["subject_id", "windows", "distal_mae_bpm", "proximal_mae_bpm", "site_gap_bpm"]].rename(columns={
        "distal_mae_bpm": "train_distal_test_distal_mae_bpm",
        "proximal_mae_bpm": "train_distal_test_proximal_mae_bpm",
        "site_gap_bpm": "proximal_minus_distal_when_trained_distal_bpm",
    }).merge(
        reverse[["subject_id", "proximal_mae_bpm", "distal_mae_bpm", "reverse_site_gap_bpm"]].rename(columns={
            "proximal_mae_bpm": "train_proximal_test_proximal_mae_bpm",
            "distal_mae_bpm": "train_proximal_test_distal_mae_bpm",
            "reverse_site_gap_bpm": "distal_minus_proximal_when_trained_proximal_bpm",
        }), on="subject_id", validate="one_to_one",
    )
    subject["proximal_minus_distal_when_trained_proximal_bpm"] = -subject.distal_minus_proximal_when_trained_proximal_bpm
    subject["mean_proximal_disadvantage_bpm"] = subject[[
        "proximal_minus_distal_when_trained_distal_bpm",
        "proximal_minus_distal_when_trained_proximal_bpm",
    ]].mean(axis=1)
    subject.to_csv(ROOT / "bidirectional_subject_summary.csv", index=False)

    overall = pd.DataFrame({
        "case": ["Train distal\nTest distal", "Train distal\nTest proximal",
                 "Train proximal\nTest proximal", "Train proximal\nTest distal"],
        "mae_bpm": [
            np.average(subject.train_distal_test_distal_mae_bpm, weights=subject.windows),
            np.average(subject.train_distal_test_proximal_mae_bpm, weights=subject.windows),
            np.average(subject.train_proximal_test_proximal_mae_bpm, weights=subject.windows),
            np.average(subject.train_proximal_test_distal_mae_bpm, weights=subject.windows),
        ],
        "test_site": ["Distal", "Proximal", "Proximal", "Distal"],
    })
    overall.to_csv(ROOT / "bidirectional_overall_summary.csv", index=False)

    shown = subject.sort_values("mean_proximal_disadvantage_bpm", ascending=False)
    y = np.arange(len(shown)); height = .36
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={"width_ratios": [1, 1.65]})
    colors = ["#2A9D8F" if site == "Distal" else "#E76F51" for site in overall.test_site]
    axes[0].bar(np.arange(4), overall.mae_bpm, color=colors)
    axes[0].set_xticks(np.arange(4), overall.case, rotation=15, ha="right")
    axes[0].set_ylabel("MAE (bpm)"); axes[0].set_title("Overall: distal input is easier")
    for index, value in enumerate(overall.mae_bpm):
        axes[0].text(index, value + .25, f"{value:.2f}", ha="center", fontweight="bold")
    axes[0].grid(axis="y", alpha=.2)

    axes[1].barh(y-height/2, shown.proximal_minus_distal_when_trained_distal_bpm,
                 height, label="Model trained distal", color="#177E89")
    axes[1].barh(y+height/2, shown.proximal_minus_distal_when_trained_proximal_bpm,
                 height, label="Model trained proximal", color="#E9C46A")
    axes[1].axvline(0, color="black", linewidth=.9)
    axes[1].set_yticks(y, shown.subject_id); axes[1].invert_yaxis()
    axes[1].set_xlabel("Proximal MAE − distal MAE (bpm)")
    axes[1].set_title("Each subject: positive means proximal was harder")
    axes[1].legend(frameon=False); axes[1].grid(axis="x", alpha=.2)
    fig.suptitle("Bidirectional PTT sensor-site result")
    fig.tight_layout(); fig.savefig(ROOT / "figures/bidirectional_site_summary.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
