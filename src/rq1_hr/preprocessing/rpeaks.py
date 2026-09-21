"""A deterministic ECG R-peak detector and annotation matching metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, find_peaks, sosfiltfilt


@dataclass(frozen=True, slots=True)
class PeakMatch:
    reference_count: int
    detected_count: int
    matched_count: int
    precision: float
    recall: float
    f1: float
    median_absolute_timing_error_ms: float


def detect_rpeaks(
    ecg: NDArray[np.floating],
    fs_hz: float,
    *,
    bandpass_hz: tuple[float, float] = (5.0, 25.0),
    integration_s: float = 0.12,
    minimum_distance_s: float = 0.25,
    threshold_mad: float = 12.0,
    refinement_s: float = 0.10,
) -> NDArray[np.int64]:
    """Detect R-peaks using bandpass, derivative energy, and local refinement.

    The detector is polarity-independent because candidates are refined to the
    largest absolute filtered ECG deflection. Parameters are fixed for every
    PPG-DaLiA subject. This is a development detector; its parameters and
    performance must be validated independently before transfer to a new ECG
    device domain.
    """
    values = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if values.size < int(2 * fs_hz):
        raise ValueError("ECG must contain at least two seconds")
    if not np.all(np.isfinite(values)):
        raise ValueError("ECG contains non-finite values")
    low, high = bandpass_hz
    if not 0 < low < high < fs_hz / 2:
        raise ValueError("invalid ECG bandpass for the sampling rate")

    filtered = sosfiltfilt(
        butter(3, [low, high], btype="bandpass", fs=fs_hz, output="sos"), values
    )
    derivative = np.diff(filtered, prepend=filtered[0])
    energy = derivative * derivative
    integration_samples = max(1, int(round(integration_s * fs_hz)))
    integrated = np.convolve(
        energy, np.ones(integration_samples) / integration_samples, mode="same"
    )
    median = float(np.median(integrated))
    mad = float(np.median(np.abs(integrated - median)))
    if mad == 0:
        raise ValueError("ECG detector energy has zero median absolute deviation")

    candidates, _ = find_peaks(
        integrated,
        height=median + threshold_mad * mad,
        prominence=mad,
        distance=max(1, int(round(minimum_distance_s * fs_hz))),
    )
    radius = max(1, int(round(refinement_s * fs_hz)))
    refined: list[int] = []
    for candidate in candidates:
        left = max(0, candidate - radius)
        right = min(values.size, candidate + radius + 1)
        refined.append(left + int(np.argmax(np.abs(filtered[left:right]))))
    return np.unique(np.asarray(refined, dtype=np.int64))


def detect_rpeaks_xqrs(
    ecg: NDArray[np.floating], fs_hz: float
) -> NDArray[np.int64]:
    """Detect QRS complexes with WFDB's adaptive XQRS implementation."""
    from wfdb import processing

    values = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if values.size < int(2 * fs_hz):
        raise ValueError("ECG must contain at least two seconds")
    if not np.all(np.isfinite(values)):
        raise ValueError("ECG contains non-finite values")
    return np.asarray(
        processing.xqrs_detect(sig=values, fs=fs_hz, learn=True, verbose=False),
        dtype=np.int64,
    )


def detect_rpeaks_emrich2023(
    ecg: NDArray[np.floating], fs_hz: float
) -> NDArray[np.int64]:
    """Detect R-peaks with NeuroKit2's FastNVG/Emrich 2023 method."""
    import neurokit2 as nk

    values = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("ECG contains non-finite values")
    cleaned = nk.ecg_clean(values, sampling_rate=fs_hz, method="emrich2023")
    _, information = nk.ecg_peaks(
        cleaned,
        sampling_rate=fs_hz,
        method="emrich2023",
        correct_artifacts=False,
    )
    return np.asarray(information["ECG_R_Peaks"], dtype=np.int64)


def detect_rpeaks_sleepecg(
    ecg: NDArray[np.floating], fs_hz: float
) -> NDArray[np.int64]:
    """Detect heartbeats with SleepECG's adaptive Pan–Tompkins method."""
    import sleepecg

    values = np.asarray(ecg, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("ECG contains non-finite values")
    return np.asarray(sleepecg.detect_heartbeats(values, fs_hz), dtype=np.int64)


def match_rpeaks(
    reference: NDArray[np.integer],
    detected: NDArray[np.integer],
    fs_hz: float,
    *,
    tolerance_s: float = 0.10,
) -> PeakMatch:
    """One-to-one chronological matching within a fixed timing tolerance."""
    ref = np.asarray(reference, dtype=np.int64)
    det = np.asarray(detected, dtype=np.int64)
    tolerance = int(round(tolerance_s * fs_hz))
    ref_pos = det_pos = 0
    errors: list[int] = []
    while ref_pos < ref.size and det_pos < det.size:
        difference = int(det[det_pos] - ref[ref_pos])
        if abs(difference) <= tolerance:
            errors.append(abs(difference))
            ref_pos += 1
            det_pos += 1
        elif difference < 0:
            det_pos += 1
        else:
            ref_pos += 1
    matched = len(errors)
    precision = matched / det.size if det.size else 0.0
    recall = matched / ref.size if ref.size else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    timing = float(np.median(errors) / fs_hz * 1000) if errors else np.nan
    return PeakMatch(ref.size, det.size, matched, precision, recall, f1, timing)
