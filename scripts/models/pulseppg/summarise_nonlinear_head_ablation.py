#!/usr/bin/env python3
"""Create frozen tables and figures for the Phase-12 nonlinear-head ablation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
LINEAR = ROOT / "reports/phase12_pulseppg/02_frozen_linear_probe/comparison_summary.csv"
NONLINEAR = ROOT / "reports/phase12_pulseppg/03_head_and_tuning_ablation/nonlinear_head/run_summary.csv"
OUT = ROOT / "reports/phase12_pulseppg/03_head_and_tuning_ablation/nonlinear_head"
ORDER = ["BIDMC", "WESAD", "PTT-PPG", "PPG-DaLiA"]


def main() -> None:
    linear = pd.read_csv(LINEAR).set_index("target").reindex(ORDER)
    nonlinear = pd.read_csv(NONLINEAR)
    if set(nonlinear.seed) != {17, 29, 43}:
        raise RuntimeError("The frozen nonlinear comparison requires seeds 17, 29 and 43")
    rows = []
    for target in ORDER:
        item = {"target": target}
        for protocol in ("within", "lodo"):
            values = nonlinear.loc[(nonlinear.target == target) & (nonlinear.protocol == protocol), "mae_bpm"]
            item[f"nonlinear_{protocol}_mae_mean_bpm"] = values.mean()
            item[f"nonlinear_{protocol}_mae_sd_bpm"] = values.std(ddof=1)
            item[f"nonlinear_{protocol}_mae_min_bpm"] = values.min()
            item[f"nonlinear_{protocol}_mae_max_bpm"] = values.max()
            item[f"linear_{protocol}_mae_bpm"] = linear.loc[target, f"{protocol}_mae_bpm"]
            item[f"timeppg_{protocol}_mae_bpm"] = linear.loc[target, f"timeppg_{protocol}_mae_bpm"]
        gaps = []
        for seed in sorted(set(nonlinear.seed)):
            familiar = nonlinear.loc[(nonlinear.target == target) & (nonlinear.protocol == "within") &
                                      (nonlinear.seed == seed), "mae_bpm"].iloc[0]
            unseen = nonlinear.loc[(nonlinear.target == target) & (nonlinear.protocol == "lodo") &
                                    (nonlinear.seed == seed), "mae_bpm"].iloc[0]
            gaps.append(unseen - familiar)
        item["nonlinear_gap_mean_bpm"] = np.mean(gaps)
        item["nonlinear_gap_sd_bpm"] = np.std(gaps, ddof=1)
        item["linear_gap_bpm"] = linear.loc[target, "generalisation_gap_bpm"]
        item["timeppg_gap_bpm"] = linear.loc[target, "timeppg_generalisation_gap_bpm"]
        item["nonlinear_minus_linear_lodo_bpm"] = (
            item["nonlinear_lodo_mae_mean_bpm"] - item["linear_lodo_mae_bpm"]
        )
        rows.append(item)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "frozen_comparison.csv", index=False)
    plot(summary)
    print(summary.to_string(index=False))


def plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    x = np.arange(len(summary)); width = .25
    colours = ["#7B8794", "#2A9D8F", "#E76F51"]
    labels = ["Pulse linear", "Pulse nonlinear", "TimePPG"]
    for ax, protocol, title in zip(axes[:2], ("within", "lodo"),
                                   ("Familiar-domain MAE", "Unseen-dataset LODO MAE")):
        values = [summary[f"linear_{protocol}_mae_bpm"],
                  summary[f"nonlinear_{protocol}_mae_mean_bpm"],
                  summary[f"timeppg_{protocol}_mae_bpm"]]
        for index, (label, colour, value) in enumerate(zip(labels, colours, values)):
            error = summary[f"nonlinear_{protocol}_mae_sd_bpm"] if label == "Pulse nonlinear" else None
            bars = ax.bar(x + (index - 1) * width, value, width, label=label, color=colour,
                          yerr=error, capsize=3)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.set_xticks(x, summary.target, rotation=15); ax.set_ylabel("MAE (bpm)")
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
    gap_values = [summary.linear_gap_bpm, summary.nonlinear_gap_mean_bpm, summary.timeppg_gap_bpm]
    for index, (label, colour, value) in enumerate(zip(labels, colours, gap_values)):
        error = summary.nonlinear_gap_sd_bpm if label == "Pulse nonlinear" else None
        bars = axes[2].bar(x + (index - 1) * width, value, width, label=label, color=colour,
                           yerr=error, capsize=3)
        axes[2].bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    axes[2].axhline(0, color="#1F2933", linewidth=1)
    axes[2].set_xticks(x, summary.target, rotation=15); axes[2].set_ylabel("LODO − within MAE (bpm)")
    axes[2].set_title("Generalisation gap"); axes[2].grid(axis="y", alpha=.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, 1.07))
    fig.suptitle("Frozen Pulse-PPG nonlinear-head ablation (mean ± SD across seeds 17, 29, 43)", y=1.13)
    fig.savefig(OUT / "nonlinear_head_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
