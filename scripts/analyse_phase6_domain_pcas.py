#!/usr/bin/env python3
"""Create separate, controlled PCA views of heterogeneous PPG domains."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from analyse_phase6_distribution_shift import SHAPE_FEATURES, extract_features


PHASE5 = Path("reports/phase5_ppg")
OUTPUT = Path("reports/phase6_distribution_shift")
FIGURES = OUTPUT / "figures"
DALIA_LABELS = {
    0: "No activity", 1: "Baseline", 2: "Stairs", 3: "Table soccer",
    4: "Cycling", 5: "Driving", 6: "Lunch", 7: "Walking", 8: "Working",
}
WESAD_LABELS = {1: "Baseline", 2: "Stress", 3: "Amusement", 4: "Meditation"}
ENVIRONMENT = {
    "PPG-DaLiA": "Daily life — PPG-DaLiA",
    "PTT-PPG": "Controlled activity — PTT-PPG",
    "WESAD": "Laboratory protocol — WESAD",
    "BIDMC": "Clinical ICU — BIDMC",
}
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#332288")


def hr_balanced(frame: pd.DataFrame, group: str, *, per_cell: int = 120, seed: int = 17) -> pd.DataFrame:
    """Balance group membership inside 10-bpm bins to reduce HR confounding."""
    x = frame.copy()
    x["hr_bin"] = pd.cut(x.hr_bpm, np.arange(35, 226, 10), right=False)
    pieces = []
    for _, cell in x.dropna(subset=["hr_bin"]).groupby("hr_bin", observed=True):
        counts = cell.groupby(group).size()
        if len(counts) < 2:
            continue
        n = min(int(counts.min()), per_cell)
        if n < 10:
            continue
        pieces.extend(part.sample(n, random_state=seed) for _, part in cell.groupby(group, sort=False))
    return pd.concat(pieces, ignore_index=True)


def pca_coordinates(frame: pd.DataFrame, columns: tuple[str, ...] = SHAPE_FEATURES) -> tuple[np.ndarray, np.ndarray]:
    x = StandardScaler().fit_transform(frame[list(columns)])
    model = PCA(n_components=2, random_state=17)
    return model.fit_transform(x), model.explained_variance_ratio_


def plot_groups(frame: pd.DataFrame, group: str, title: str, path: Path,
                columns: tuple[str, ...] = SHAPE_FEATURES,
                source_note: str = "") -> None:
    xy, explained = pca_coordinates(frame, columns)
    groups = sorted(frame[group].unique())
    fig, ax = plt.subplots(figsize=(10, 7))
    for index, name in enumerate(groups):
        keep = frame[group].eq(name).to_numpy()
        ax.scatter(xy[keep, 0], xy[keep, 1], s=11, alpha=.28,
                   color=PALETTE[index % len(PALETTE)], edgecolors="none", label=f"{name} (n={keep.sum():,})")
    ax.set(xlabel=f"PCA 1 ({explained[0]*100:.1f}%)", ylabel=f"PCA 2 ({explained[1]*100:.1f}%)", title=title)
    ax.legend(markerscale=1.8, fontsize=9); ax.grid(alpha=.15)
    if source_note:
        fig.text(
            0.5, 0.012, f"Domain source: {source_note}", ha="center", va="bottom",
            fontsize=10, color="#333333",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f3f3f3", "edgecolor": "#bdbdbd"},
        )
    fig.tight_layout(rect=(0, .055 if source_note else 0, 1, 1))
    fig.savefig(path, dpi=180); plt.close(fig)


def sampled_features(metrics: pd.DataFrame, recordings: pd.DataFrame, dataset: str,
                     label_map: dict[int, str], cap_per_subject_condition: int = 80) -> pd.DataFrame:
    x = metrics.loc[metrics.dataset.eq(dataset) & metrics.channel_index.eq(0)
                    & (metrics.window_index % 4 == 0) & metrics.condition_id.isin(label_map)].copy()
    pieces = []
    for _, group in x.groupby(["subject_id", "condition_id"], sort=False):
        pieces.append(group.sample(min(cap_per_subject_condition, len(group)), random_state=17))
    selected = pd.concat(pieces, ignore_index=True)
    result = extract_features(selected, recordings)
    result["domain"] = selected.condition_id.map(label_map).to_numpy()
    return result


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(PHASE5 / "window_channel_quality.csv", low_memory=False)
    recordings = pd.read_csv(PHASE5 / "processed_recordings.csv")

    dalia = sampled_features(metrics, recordings, "PPG-DaLiA", DALIA_LABELS)
    dalia = hr_balanced(dalia, "domain", per_cell=80)
    plot_groups(dalia, "domain", "PPG-DaLiA: daily-life activity domains\nHR-bin balanced, z-normalized PPG features",
                FIGURES / "dalia_activity_pca.png",
                source_note="PPG-DaLiA only — Empatica E4 wrist PPG")

    ptt_map = {"sit": "Sitting", "walk": "Walking", "run": "Running"}
    ptt_metrics = metrics.loc[metrics.dataset.eq("PTT-PPG") & metrics.channel_index.eq(0)
                              & (metrics.window_index % 4 == 0)].copy()
    pieces = [group.sample(min(150, len(group)), random_state=17)
              for _, group in ptt_metrics.groupby(["subject_id", "condition_name"], sort=False)]
    ptt_selected = pd.concat(pieces, ignore_index=True)
    ptt = extract_features(ptt_selected, recordings)
    ptt["domain"] = ptt.condition_name.map(ptt_map)
    ptt = hr_balanced(ptt, "domain")
    plot_groups(ptt, "domain", "PTT-PPG: physical-activity domains (pleth_1)\nHR-bin balanced, z-normalized PPG features",
                FIGURES / "ptt_activity_pca.png",
                source_note="PTT-PPG only — MAX30101, pleth_1, distal left index finger")

    wesad = sampled_features(metrics, recordings, "WESAD", WESAD_LABELS)
    wesad = hr_balanced(wesad, "domain")
    plot_groups(wesad, "domain", "WESAD: stress-protocol domains\nHR-bin balanced, z-normalized PPG features",
                FIGURES / "wesad_stress_condition_pca.png",
                source_note="WESAD only — Empatica E4 wrist PPG")

    base = pd.read_csv(OUTPUT / "hr_matched_features.csv")
    base["environment"] = base.dataset.map(ENVIRONMENT)
    plot_groups(base, "environment", "Recording-environment domains\nHR-matched, z-normalized PPG features; environment is confounded with dataset/device",
                FIGURES / "environment_domain_pca.png",
                source_note="Daily life=PPG-DaLiA | Controlled activity=PTT-PPG | Laboratory=WESAD | Clinical ICU=BIDMC")

    quality_source = metrics.loc[(metrics.channel_index == 0) & (metrics.window_index % 4 == 0)].copy()
    quality_source["quality_domain"] = quality_source.groupby("dataset").spectral_concentration.transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=("Low concentration", "Medium concentration", "High concentration")))
    pieces = [group.sample(min(100, len(group)), random_state=17)
              for _, group in quality_source.groupby(["dataset", "subject_id", "quality_domain"], observed=True, sort=False)]
    quality_selected = pd.concat(pieces, ignore_index=True)
    quality = extract_features(quality_selected, recordings)
    quality["quality_domain"] = quality_selected.quality_domain.astype(str).to_numpy()
    quality_columns = tuple(x for x in SHAPE_FEATURES if x not in {"spectral_concentration", "spectral_entropy", "dominant_frequency_hz"})
    plot_groups(quality, "quality_domain", "Signal-quality domains across datasets\nWithin-dataset spectral-concentration tertiles; defining spectral features excluded from PCA",
                FIGURES / "signal_quality_domain_pca.png", quality_columns,
                source_note="PPG-DaLiA + PTT-PPG (pleth_1) + WESAD + BIDMC")

    for name, frame in (("dalia_activity", dalia), ("ptt_activity", ptt), ("wesad_stress", wesad),
                        ("environment", base), ("signal_quality", quality)):
        frame.to_csv(OUTPUT / f"{name}_pca_features.csv", index=False)
    summary = pd.concat([
        frame.groupby(key).size().rename("windows").reset_index().assign(analysis=name)
        for name, frame, key in (("dalia_activity", dalia, "domain"), ("ptt_activity", ptt, "domain"),
                                 ("wesad_stress", wesad, "domain"), ("environment", base, "environment"),
                                 ("signal_quality", quality, "quality_domain"))
    ], ignore_index=True)
    summary.to_csv(OUTPUT / "domain_pca_sample_counts.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
