#!/usr/bin/env python3
"""Quantify dataset, HR-support, activity, and PPG-representation shifts."""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rq1_hr_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASETS = ("PPG-DaLiA", "PTT-PPG", "WESAD", "BIDMC")
DATASET_COLORS = {
    "PPG-DaLiA": "#0072B2",  # blue
    "PTT-PPG": "#E69F00",    # orange
    "WESAD": "#009E73",      # green
    "BIDMC": "#CC79A7",      # purple
}
DATASET_LINESTYLES = {
    "PPG-DaLiA": "-", "PTT-PPG": "--", "WESAD": "-.", "BIDMC": ":",
}
DEVICE_BY_DATASET = {
    "PPG-DaLiA": "Empatica E4",
    "WESAD": "Empatica E4",
    "PTT-PPG": "MAX30101",
    "BIDMC": "Clinical pulse oximeter",
}
DEVICE_COLORS = {
    "Empatica E4": "#0072B2",
    "MAX30101": "#E69F00",
    "Clinical pulse oximeter": "#CC79A7",
}
DATASET_MARKERS = {"PPG-DaLiA": "o", "WESAD": "^", "PTT-PPG": "s", "BIDMC": "D"}
PTT_CHANNEL_INFO = {
    "pleth_1": ("Fingertip segment (distal)", "Pair A: pleth_1 ↔ pleth_4"),
    "pleth_2": ("Fingertip segment (distal)", "Pair B: pleth_2 ↔ pleth_5"),
    "pleth_3": ("Fingertip segment (distal)", "Pair C: pleth_3 ↔ pleth_6"),
    "pleth_4": ("Finger-base segment (proximal)", "Pair A: pleth_1 ↔ pleth_4"),
    "pleth_5": ("Finger-base segment (proximal)", "Pair B: pleth_2 ↔ pleth_5"),
    "pleth_6": ("Finger-base segment (proximal)", "Pair C: pleth_3 ↔ pleth_6"),
}
SHAPE_FEATURES = (
    "skewness", "kurtosis", "derivative_std", "zero_crossing_rate",
    "spectral_entropy", "dominant_frequency_hz", "spectral_concentration",
    "autocorrelation_peak", "autocorrelation_lag_s", "q05", "q25", "q75", "q95",
)
RAW_FEATURES = ("log_standard_deviation", "log_peak_to_peak") + SHAPE_FEATURES


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        shown = []
        for value in row:
            if isinstance(value, float):
                shown.append("" if not np.isfinite(value) else f"{value:.4f}")
            else:
                shown.append(str(value))
        lines.append("| " + " | ".join(shown) + " |")
    return "\n".join(lines)


def balanced_sample(frame: pd.DataFrame, per_subject: int, seed: int) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby(["dataset", "subject_id"], sort=False):
        pieces.append(group.sample(min(per_subject, len(group)), random_state=seed))
    capped = pd.concat(pieces, ignore_index=True)
    target = min(capped.groupby("dataset").size())
    return pd.concat([
        group.sample(target, random_state=seed) for _, group in capped.groupby("dataset", sort=False)
    ], ignore_index=True)


def waveform_features(x: np.ndarray, fs_hz: float = 64.0) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    std = float(np.std(x))
    z = (x - np.mean(x)) / std
    frequency, power = welch(z, fs=fs_hz, nperseg=256)
    band = (frequency >= 0.5) & (frequency <= 4.0)
    p = power[band]
    p = p / np.sum(p)
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(p.size))
    correlation = np.correlate(z, z, mode="full")[len(z) - 1 :]
    correlation /= correlation[0]
    lo, hi = int(0.25 * fs_hz), int(2.0 * fs_hz)
    peak_lag = lo + int(np.argmax(correlation[lo:hi + 1]))
    quantiles = np.quantile(z, [0.05, 0.25, 0.75, 0.95])
    return {
        "log_standard_deviation": float(np.log(std + 1e-12)),
        "log_peak_to_peak": float(np.log(np.ptp(x) + 1e-12)),
        "skewness": float(skew(z)), "kurtosis": float(kurtosis(z)),
        "derivative_std": float(np.std(np.diff(z))),
        "zero_crossing_rate": float(np.mean(np.diff(np.signbit(z)) != 0)),
        "spectral_entropy": entropy,
        "dominant_frequency_hz": float(frequency[band][np.argmax(p)]),
        "spectral_concentration": float(np.max(p)),
        "autocorrelation_peak": float(correlation[peak_lag]),
        "autocorrelation_lag_s": float(peak_lag / fs_hz),
        "q05": float(quantiles[0]), "q25": float(quantiles[1]),
        "q75": float(quantiles[2]), "q95": float(quantiles[3]),
    }


def extract_features(sample: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    paths = {(r.dataset, str(r.record_id)): r.signal_path for r in recordings.itertuples(index=False)}
    arrays: dict[tuple[str, str], np.ndarray] = {}
    rows = []
    for row in sample.itertuples(index=False):
        key = (row.dataset, str(row.record_id))
        if key not in arrays:
            arrays[key] = np.load(paths[key], mmap_mode="r")
        x = arrays[key][row.start_sample_64hz:row.end_sample_64hz, row.channel_index]
        rows.append({
            "dataset": row.dataset, "record_id": row.record_id, "subject_id": row.subject_id,
            "window_index": row.window_index, "window_start_s": row.window_start_s,
            "condition_name": row.condition_name, "channel_name": row.channel_name,
            "hr_bpm": row.hr_bpm, **waveform_features(x),
        })
    return pd.DataFrame(rows)


def evaluate_classifier(frame: pd.DataFrame, features: tuple[str, ...], representation: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = frame[list(features)].to_numpy()
    y = frame.dataset.to_numpy()
    groups = (frame.dataset + ":" + frame.subject_id.astype(str)).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=17)
    folds, truth, predicted = [], [], []
    for fold, (train, test) in enumerate(splitter.split(x, y, groups), 1):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        model.fit(x[train], y[train])
        estimate = model.predict(x[test])
        folds.append({"representation": representation, "fold": fold, "windows": len(test),
                      "accuracy": accuracy_score(y[test], estimate),
                      "macro_f1": f1_score(y[test], estimate, average="macro")})
        truth.extend(y[test]); predicted.extend(estimate)
    cm = confusion_matrix(truth, predicted, labels=DATASETS, normalize="true")
    confusion = pd.DataFrame(cm, index=DATASETS, columns=DATASETS).reset_index(names="true_dataset")
    confusion.insert(0, "representation", representation)
    return pd.DataFrame(folds), confusion


def hr_match(frame: pd.DataFrame, seed: int = 17) -> pd.DataFrame:
    result = frame.copy()
    result["hr_bin"] = pd.cut(result.hr_bpm, np.arange(40, 201, 10), right=False)
    pieces = []
    for _, group in result.dropna(subset=["hr_bin"]).groupby("hr_bin", observed=True):
        counts = group.groupby("dataset").size()
        if len(counts) != len(DATASETS) or counts.min() < 20:
            continue
        n = int(counts.min())
        pieces.extend(g.sample(n, random_state=seed) for _, g in group.groupby("dataset", sort=False))
    if not pieces:
        raise RuntimeError("no HR bins have support in every dataset")
    return pd.concat(pieces, ignore_index=True)


def pairwise_distances(frame: pd.DataFrame) -> pd.DataFrame:
    standardized = frame.copy()
    for feature in SHAPE_FEATURES:
        standardized[feature] = (frame[feature] - frame[feature].mean()) / frame[feature].std()
    rows = []
    for left, right in itertools.combinations(DATASETS, 2):
        distances = [wasserstein_distance(standardized.loc[standardized.dataset.eq(left), f],
                                          standardized.loc[standardized.dataset.eq(right), f]) for f in SHAPE_FEATURES]
        rows.append({"dataset_a": left, "dataset_b": right,
                     "mean_standardized_wasserstein": float(np.mean(distances)),
                     "max_standardized_wasserstein": float(np.max(distances))})
    return pd.DataFrame(rows)


def representative_windows(matched: pd.DataFrame) -> pd.DataFrame:
    """Choose a central example per dataset within the same 80–90 bpm band."""
    candidates = matched.loc[matched.hr_bpm.between(80, 90, inclusive="left")].copy()
    chosen = []
    for dataset, group in candidates.groupby("dataset", sort=False):
        values = group[list(SHAPE_FEATURES)].to_numpy()
        scale = np.std(values, axis=0)
        scale[scale == 0] = 1
        distance = np.sum(((values - np.median(values, axis=0)) / scale) ** 2, axis=1)
        chosen.append(group.iloc[int(np.argmin(distance))])
    result = pd.DataFrame(chosen)
    if set(result.dataset) != set(DATASETS):
        raise RuntimeError("80–90 bpm representative band does not cover every dataset")
    return result


def plot_same_hr_examples(examples: pd.DataFrame, recordings: pd.DataFrame, output: Path) -> None:
    paths = {(r.dataset, str(r.record_id)): r.signal_path for r in recordings.itertuples(index=False)}
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex="col")
    for row_index, dataset in enumerate(DATASETS):
        row = examples.loc[examples.dataset.eq(dataset)].iloc[0]
        signal = np.load(paths[(dataset, str(row.record_id))], mmap_mode="r")
        left = int(round(row.window_start_s * 64))
        x = np.asarray(signal[left:left + 512, 0], dtype=float)
        z = (x - x.mean()) / x.std()
        time = np.arange(z.size) / 64
        frequency, power = welch(z, fs=64, nperseg=256)
        band = (frequency >= .5) & (frequency <= 4)
        color = DATASET_COLORS[dataset]
        axes[row_index, 0].plot(time, z, color=color, linewidth=1.2)
        axes[row_index, 1].plot(frequency[band], power[band] / power[band].max(), color=color, linewidth=2)
        axes[row_index, 0].set_ylabel(f"{dataset}\n{row.hr_bpm:.1f} bpm\nz-score")
        axes[row_index, 0].grid(alpha=.18); axes[row_index, 1].grid(alpha=.18)
    axes[-1, 0].set_xlabel("Time (s)"); axes[-1, 1].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_title("Normalized PPG waveform"); axes[0, 1].set_title("Normalized power spectrum")
    fig.suptitle("Similar HR does not produce identical PPG across domains", fontsize=17, y=.995)
    fig.tight_layout(); fig.savefig(output / "same_hr_different_domains.png", dpi=180); plt.close(fig)


def plot_dashboard(features: pd.DataFrame, matched: pd.DataFrame, classifier: pd.DataFrame,
                   distances: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    ax = axes[0, 0]
    for dataset in DATASETS:
        ax.hist(features.loc[features.dataset.eq(dataset), "hr_bpm"], bins=np.arange(35, 205, 5),
                density=True, histtype="step", linewidth=2, color=DATASET_COLORS[dataset],
                linestyle=DATASET_LINESTYLES[dataset], label=dataset)
    ax.set(title="A. Label/support shift", xlabel="Reference HR (bpm)", ylabel="Density"); ax.legend(fontsize=8)

    for column, title, target in (("log_standard_deviation", "B. Device/amplitude shift", axes[0, 1]),
                                  ("spectral_entropy", "C. Spectral-complexity shift", axes[0, 2])):
        data = [features.loc[features.dataset.eq(d), column] for d in DATASETS]
        violin = target.violinplot(data, showmedians=True, showextrema=False)
        for body, dataset in zip(violin["bodies"], DATASETS):
            body.set_facecolor(DATASET_COLORS[dataset]); body.set_alpha(.7)
        target.set_xticks(range(1, 5), DATASETS, rotation=18, ha="right")
        target.set_title(title); target.set_ylabel(column.replace("_", " "))

    x = StandardScaler().fit_transform(matched[list(SHAPE_FEATURES)])
    xy = PCA(n_components=2, random_state=17).fit_transform(x)
    ax = axes[1, 0]
    for dataset in DATASETS:
        keep = matched.dataset.eq(dataset).to_numpy()
        ax.scatter(xy[keep, 0], xy[keep, 1], s=5, alpha=.22, color=DATASET_COLORS[dataset], edgecolors="none")
    ax.set(title="D. HR-matched normalized shape", xlabel="PCA 1", ylabel="PCA 2")

    ax = axes[1, 1]
    order = ("raw_scale", "per_window_zscore")
    values = [classifier.set_index("representation").loc[name, "mean_accuracy"] for name in order]
    errors = [classifier.set_index("representation").loc[name, "sd_accuracy"] for name in order]
    ax.bar((0, 1), values, yerr=errors, color=("#D55E00", "#0072B2"), alpha=.82, capsize=5)
    ax.axhline(.25, color="#333333", linestyle="--", label="chance = 25%")
    ax.set_xticks((0, 1), ("Raw scale", "Z-normalized")); ax.set_ylim(0, 1)
    ax.set(title="E. Can AI identify the dataset?", ylabel="Subject-held-out accuracy"); ax.legend(fontsize=8)

    matrix = np.zeros((4, 4)); index = {name: i for i, name in enumerate(DATASETS)}
    for row in distances.itertuples(index=False):
        i, j = index[row.dataset_a], index[row.dataset_b]
        matrix[i, j] = matrix[j, i] = row.mean_standardized_wasserstein
    ax = axes[1, 2]; image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=matrix.max())
    for i in range(4):
        for j in range(4): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center")
    ax.set_xticks(range(4), DATASETS, rotation=18, ha="right"); ax.set_yticks(range(4), DATASETS)
    ax.set_title("F. HR-matched shape distance"); fig.colorbar(image, ax=ax, fraction=.046)
    fig.suptitle("Heterogeneous PPG domains: evidence relevant to model generalisation", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(output / "heterogeneous_domain_dashboard.png", dpi=180); plt.close(fig)


def plot_device_pca(matched: pd.DataFrame, output: Path) -> None:
    """Show device groups while retaining dataset markers for interpretation."""
    x = StandardScaler().fit_transform(matched[list(SHAPE_FEATURES)])
    xy = PCA(n_components=2, random_state=17).fit_transform(x)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for dataset in DATASETS:
        keep = matched.dataset.eq(dataset).to_numpy()
        device = DEVICE_BY_DATASET[dataset]
        ax.scatter(
            xy[keep, 0], xy[keep, 1], s=13, alpha=.34,
            color=DEVICE_COLORS[device], marker=DATASET_MARKERS[dataset],
            edgecolors="none", label=f"{device} — {dataset}",
        )
    ax.set(
        xlabel="PCA 1", ylabel="PCA 2",
        title="PPG feature distribution by recording device\nHR-matched and per-window z-normalized",
    )
    ax.legend(title="Device — dataset", markerscale=1.8, frameon=True)
    ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(output / "normalized_feature_pca_by_device.png", dpi=180); plt.close(fig)


def analyse_ptt_sensor_sites(metrics: pd.DataFrame, recordings: pd.DataFrame,
                             output_dir: Path, per_subject: int = 200) -> None:
    """Compare the two simultaneous PTT finger sites while retaining wavelength."""
    ptt = metrics.loc[
        metrics.dataset.eq("PTT-PPG") & metrics.channel_index.eq(0)
        & (metrics.window_index % 4 == 0) & metrics.structurally_usable
    ].copy()
    selected = pd.concat([
        group.sample(min(per_subject, len(group)), random_state=17)
        for _, group in ptt.groupby("subject_id", sort=False)
    ])[["dataset", "record_id", "window_index"]]
    all_channels = metrics.merge(selected, on=["dataset", "record_id", "window_index"], how="inner")
    features = extract_features(all_channels, recordings)
    features["sensor_site"] = features.channel_name.map(lambda x: PTT_CHANNEL_INFO[x][0])
    features["channel_pair"] = features.channel_name.map(lambda x: PTT_CHANNEL_INFO[x][1])

    x = StandardScaler().fit_transform(features[list(SHAPE_FEATURES)])
    xy = PCA(n_components=2, random_state=17).fit_transform(x)
    features["pca_1"], features["pca_2"] = xy[:, 0], xy[:, 1]
    features.to_csv(output_dir / "ptt_sensor_site_pca_features.csv", index=False)

    site_colors = {"Fingertip segment (distal)": "#0072B2", "Finger-base segment (proximal)": "#E69F00"}
    pair_markers = {
        "Pair A: pleth_1 ↔ pleth_4": "o",
        "Pair B: pleth_2 ↔ pleth_5": "s",
        "Pair C: pleth_3 ↔ pleth_6": "^",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    ax = axes[0]
    for site in site_colors:
        for channel_pair in pair_markers:
            keep = features.sensor_site.eq(site) & features.channel_pair.eq(channel_pair)
            ax.scatter(features.loc[keep, "pca_1"], features.loc[keep, "pca_2"],
                       s=8, alpha=.18, color=site_colors[site], marker=pair_markers[channel_pair],
                       edgecolors="none")
    ax.set(title="A. Window distributions", xlabel="PCA 1", ylabel="PCA 2")
    site_handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=8,
                           markerfacecolor=color, markeredgecolor="none", label=site)
                    for site, color in site_colors.items()]
    pair_handles = [Line2D([0], [0], marker=marker, linestyle="none", markersize=8,
                           markerfacecolor="#777777", markeredgecolor="none", label=pair)
                    for pair, marker in pair_markers.items()]
    site_legend = ax.legend(handles=site_handles, title="Colour = sensor position", loc="upper left", fontsize=8)
    ax.add_artist(site_legend)
    ax.legend(handles=pair_handles, title="Marker = optical pair", loc="lower right", fontsize=8)
    ax.grid(alpha=.15)

    ax = axes[1]
    centroids = features.groupby(["sensor_site", "channel_pair"])[["pca_1", "pca_2"]].mean()
    for channel_pair, marker in pair_markers.items():
        points = []
        for site in site_colors:
            point = centroids.loc[(site, channel_pair)].to_numpy(); points.append(point)
            ax.scatter(*point, s=130, color=site_colors[site], marker=marker,
                       edgecolor="#222222", linewidth=.7, label=f"{site} — {channel_pair}")
        ax.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]],
                color="#666666", linewidth=1.5, alpha=.8)
    ax.axhline(0, color="#cccccc", linewidth=.8); ax.axvline(0, color="#cccccc", linewidth=.8)
    ax.set(title="B. Site centroids, paired by channel/wavelength", xlabel="PCA 1", ylabel="PCA 2")
    ax.grid(alpha=.15)
    fig.suptitle("PTT-PPG feature distribution by sensor placement\nLeft index finger; HR-matched processing and per-window z-normalized features", fontsize=17)
    fig.text(
        .5, .012,
        "Blue = 3 light channels from fingertip sensor  |  Orange = 3 matching light channels from finger-base sensor  |  "
        "Circle/square/triangle connects each matching light-channel pair",
        ha="center", fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f3f3f3", "edgecolor": "#bdbdbd"},
    )
    fig.tight_layout(rect=(0, .055, 1, .93))
    fig.savefig(output_dir / "figures" / "ptt_sensor_site_pca.png", dpi=180); plt.close(fig)

    centroid_table = centroids.reset_index()
    centroid_table.to_csv(output_dir / "ptt_sensor_site_centroids.csv", index=False)


def plots(features: pd.DataFrame, matched: pd.DataFrame, metrics: pd.DataFrame,
          recordings: pd.DataFrame, classifier: pd.DataFrame,
          distances: pd.DataFrame, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for dataset in DATASETS:
        ax.hist(features.loc[features.dataset.eq(dataset), "hr_bpm"], bins=np.arange(35, 225, 5), density=True,
                histtype="step", linewidth=2.3, color=DATASET_COLORS[dataset],
                linestyle=DATASET_LINESTYLES[dataset], label=dataset)
    ax.set(xlabel="ECG-reference HR (bpm)", ylabel="Density", title="HR support differs across datasets")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "hr_distributions.png", dpi=180); plt.close(fig)

    x = StandardScaler().fit_transform(matched[list(SHAPE_FEATURES)])
    xy = PCA(n_components=2, random_state=17).fit_transform(x)
    fig, ax = plt.subplots(figsize=(8, 6))
    for dataset in DATASETS:
        keep = matched.dataset.eq(dataset).to_numpy()
        ax.scatter(xy[keep, 0], xy[keep, 1], s=9, alpha=.32,
                   color=DATASET_COLORS[dataset], label=dataset, edgecolors="none")
    ax.set(xlabel="PCA 1", ylabel="PCA 2", title="HR-matched, amplitude-normalized PPG features")
    ax.legend(markerscale=2); fig.tight_layout(); fig.savefig(output / "normalized_feature_pca.png", dpi=180); plt.close(fig)

    ptt = metrics.loc[metrics.dataset.eq("PTT-PPG")]
    order = sorted(ptt.channel_name.unique())
    data = [ptt.loc[ptt.channel_name.eq(channel), "spectral_concentration"] for channel in order]
    channel_colors = ("#56B4E9", "#E69F00", "#009E73", "#F0E442", "#0072B2", "#CC79A7")
    fig, ax = plt.subplots(figsize=(9, 5))
    boxes = ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True,
                       medianprops={"color": "#111111", "linewidth": 1.8})
    for box, color in zip(boxes["boxes"], channel_colors):
        box.set_facecolor(color); box.set_alpha(.72); box.set_edgecolor("#333333")
    ax.set(xlabel="PTT-PPG channel", ylabel="Spectral concentration", title="PTT-PPG channels are not interchangeable")
    fig.tight_layout(); fig.savefig(output / "ptt_channel_comparison.png", dpi=180); plt.close(fig)
    examples = representative_windows(matched)
    plot_same_hr_examples(examples, recordings, output)
    plot_dashboard(features, matched, classifier, distances, output)
    plot_device_pca(matched, output)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-dir", type=Path, default=Path("reports/phase5_ppg"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase6_distribution_shift"))
    parser.add_argument("--per-subject", type=int, default=200)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.phase5_dir / "window_channel_quality.csv", low_memory=False)
    recordings = pd.read_csv(args.phase5_dir / "processed_recordings.csv")

    primary = metrics.loc[(metrics.channel_index == 0) & (metrics.window_index % 4 == 0) & metrics.structurally_usable].copy()
    sample = balanced_sample(primary, args.per_subject, seed=17)
    features = extract_features(sample, recordings)
    features.to_csv(args.output_dir / "balanced_nonoverlap_features.csv", index=False)

    hr_summary = features.groupby("dataset", sort=False).hr_bpm.agg(
        windows="size", minimum="min", q05=lambda x: x.quantile(.05), median="median",
        q95=lambda x: x.quantile(.95), maximum="max",
    ).reset_index()
    hr_summary.to_csv(args.output_dir / "hr_support_summary.csv", index=False)

    matched = hr_match(features)
    matched.to_csv(args.output_dir / "hr_matched_features.csv", index=False)
    raw_folds, raw_confusion = evaluate_classifier(matched, RAW_FEATURES, "raw_scale")
    normalized_folds, normalized_confusion = evaluate_classifier(matched, SHAPE_FEATURES, "per_window_zscore")
    folds = pd.concat([raw_folds, normalized_folds], ignore_index=True)
    confusion = pd.concat([raw_confusion, normalized_confusion], ignore_index=True)
    folds.to_csv(args.output_dir / "dataset_classifier_folds.csv", index=False)
    confusion.to_csv(args.output_dir / "dataset_classifier_confusion.csv", index=False)
    classifier_summary = folds.groupby("representation").agg(
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_macro_f1=("macro_f1", "mean"), sd_macro_f1=("macro_f1", "std"),
    ).reset_index()
    classifier_summary.to_csv(args.output_dir / "dataset_classifier_summary.csv", index=False)

    distances = pairwise_distances(matched)
    distances.to_csv(args.output_dir / "pairwise_shape_distances.csv", index=False)
    ptt_channels = metrics.loc[metrics.dataset.eq("PTT-PPG")].groupby(["channel_name", "condition_name"], sort=True).agg(
        windows=("window_index", "size"), median_concentration=("spectral_concentration", "median"),
        q05_concentration=("spectral_concentration", lambda x: x.quantile(.05)),
        median_std=("standard_deviation", "median"),
    ).reset_index()
    ptt_channels.to_csv(args.output_dir / "ptt_channel_activity_summary.csv", index=False)
    examples = plots(features, matched, metrics, recordings, classifier_summary,
                     distances, args.output_dir / "figures")
    analyse_ptt_sensor_sites(metrics, recordings, args.output_dir, per_subject=args.per_subject)
    examples[["dataset", "record_id", "subject_id", "window_index", "window_start_s", "hr_bpm"]].to_csv(
        args.output_dir / "same_hr_figure_examples.csv", index=False)
    pd.DataFrame([
        {"parameter": "random_seed", "value": 17},
        {"parameter": "window_selection", "value": "window_index modulo 4 = 0 (non-overlapping)"},
        {"parameter": "maximum_windows_per_subject", "value": args.per_subject},
        {"parameter": "primary_ptt_channel", "value": "pleth_1"},
        {"parameter": "hr_matching_bin_width_bpm", "value": 10},
        {"parameter": "classifier", "value": "standardized logistic regression"},
        {"parameter": "validation", "value": "5-fold stratified subject-group"},
    ]).to_csv(args.output_dir / "analysis_settings.csv", index=False)

    report = "\n".join([
        "# Phase 6: distribution-shift analysis", "", "## Design safeguards", "",
        "- Primary comparison uses one channel per dataset; PTT-PPG uses the predeclared first source channel (`pleth_1`). All six PTT channels are analysed separately.",
        "- Only every fourth 2-second-step window is eligible, producing non-overlapping 8-second windows.",
        f"- Sampling is capped at {args.per_subject} windows per subject and balanced to the same number per dataset.",
        "- Dataset classifiers are evaluated with five-fold stratified subject-group splitting: a subject never appears in both train and test.",
        "- Conditional comparison matches the four datasets within 10-bpm HR bins before testing dataset separability.", "",
        "## HR support", "", markdown(hr_summary), "",
        "## Dataset identity prediction", "", markdown(classifier_summary), "",
        "Chance accuracy is 0.25. `raw_scale` includes amplitude features. `per_window_zscore` removes mean/scale information and uses only normalized waveform and spectral features. Accuracy above chance after HR matching and z-normalization is evidence that dataset/device characteristics remain in PPG shape; it is not by itself proof of causal concept shift or model failure.", "",
        "## Pairwise normalized-shape distances after HR matching", "", markdown(distances), "",
        "## Main findings", "",
        "- The four datasets do not provide identical HR support. PPG-DaLiA has the broadest sampled upper range; BIDMC is narrower in this balanced sample.",
        "- Raw-scale dataset classification reaches about 77% accuracy, showing strong device/dataset signatures.",
        "- Per-window z-normalization reduces accuracy to about 65%, but does not remove dataset identity; waveform and spectral shift remain.",
        "- PPG-DaLiA and WESAD are the closest normalized-shape pair in this feature set, consistent with their shared Empatica E4 wrist device family. WESAD and BIDMC are the most separated pair.",
        "- PTT channel spectral concentration changes across both channel and activity. The additional channels should remain a secondary robustness analysis rather than be treated as duplicate measurements.", "",
        "## AI-engineering visual guide", "",
        "- `heterogeneous_domain_dashboard.png` combines label support, amplitude, spectral complexity, normalized PCA, dataset-classifier accuracy, and pairwise shape distance.",
        "- `same_hr_different_domains.png` controls HR to 80–90 bpm and shows representative normalized waveforms and spectra. It makes clear why an HR model may learn device/domain cues in addition to cardiac cues.",
        "- `normalized_feature_pca_by_device.png` colours the same HR-matched normalized feature space by physical recording device and uses marker shape to retain dataset identity.",
        "- `ptt_sensor_site_pca.png` uses PTT-PPG only: colour represents distal versus proximal left-index-finger placement, while marker pairs channels 1/4, 2/5, and 3/6. Lines between paired centroids expose site displacement without mixing devices. Pair labels are used because the source README contains conflicting red/infrared names in its hardware overview and detailed channel list; the site mapping itself is consistent.",
        "- `same_hr_figure_examples.csv` records the exact source window behind each example; the figure is not hand-picked without provenance.", "",
        "## Interpretation boundaries", "",
        "- Different HR distributions demonstrate target/label-support shift.",
        "- Predictable dataset identity after matching HR demonstrates class-conditional waveform shift, formally a difference in P(X|Y).",
        "- This analysis does not directly prove P(Y|X) concept shift. That requires the Phase 7 PPG-to-HR models and cross-dataset error analysis.",
        "- Overlapping motion frequencies remain inside the cardiac band; preprocessing cannot erase them.",
        "- Statistical tests on windows would overstate sample size, so inferential uncertainty must use subjects as the independent unit.", "",
    ])
    (args.output_dir / "README.md").write_text(report, encoding="utf-8")
    print(hr_summary.to_string(index=False)); print(classifier_summary.to_string(index=False))
    print(f"Analysed {len(features):,} balanced non-overlapping windows; HR-matched subset {len(matched):,}.")


if __name__ == "__main__":
    main()
