"""ECG R-peak to window-level heart-rate conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class WindowHR:
    starts_s: NDArray[np.float64]
    values_bpm: NDArray[np.float64]
    candidate_rr_count: NDArray[np.int64]
    valid_rr_count: NDArray[np.int64]
    invalid_rr_count: NDArray[np.int64]


def window_hr_from_rpeaks(
    rpeak_indices: NDArray[np.integer],
    fs_hz: float,
    duration_s: float,
    *,
    window_s: float = 8.0,
    step_s: float = 2.0,
    min_hr_bpm: float = 35.0,
    max_hr_bpm: float = 220.0,
    min_valid_rr: int = 4,
) -> WindowHR:
    """Calculate mean beat-level HR in fixed windows.

    Each RR interval is placed at the midpoint between its two R-peaks. Values
    outside the declared physiological range are flagged and excluded. A window
    receives no HR unless it contains ``min_valid_rr`` valid intervals.
    """
    peaks = np.asarray(rpeak_indices)
    if peaks.ndim != 1 or not np.issubdtype(peaks.dtype, np.integer):
        raise ValueError("rpeak_indices must be a one-dimensional integer array")
    if peaks.size > 1 and np.any(np.diff(peaks) <= 0):
        raise ValueError("rpeak_indices must be strictly increasing")
    if fs_hz <= 0 or duration_s <= 0 or window_s <= 0 or step_s <= 0:
        raise ValueError("sampling rate, duration, window, and step must be positive")
    if duration_s < window_s:
        raise ValueError("recording is shorter than one complete window")
    if min_valid_rr < 1:
        raise ValueError("min_valid_rr must be at least 1")

    peak_times = peaks.astype(np.float64) / fs_hz
    rr_s = np.diff(peak_times)
    beat_hr = 60.0 / rr_s
    rr_midpoints = (peak_times[:-1] + peak_times[1:]) / 2.0
    plausible = np.isfinite(beat_hr) & (beat_hr >= min_hr_bpm) & (beat_hr <= max_hr_bpm)

    n_windows = int(np.floor((duration_s - window_s) / step_s + 1e-9)) + 1
    starts = np.arange(n_windows, dtype=np.float64) * step_s
    values = np.full(n_windows, np.nan, dtype=np.float64)
    candidates = np.zeros(n_windows, dtype=np.int64)
    valid_counts = np.zeros(n_windows, dtype=np.int64)

    for index, start in enumerate(starts):
        inside = (rr_midpoints >= start) & (rr_midpoints < start + window_s)
        valid = inside & plausible
        candidates[index] = np.count_nonzero(inside)
        valid_counts[index] = np.count_nonzero(valid)
        if valid_counts[index] >= min_valid_rr:
            values[index] = float(np.mean(beat_hr[valid]))

    return WindowHR(
        starts_s=starts,
        values_bpm=values,
        candidate_rr_count=candidates,
        valid_rr_count=valid_counts,
        invalid_rr_count=candidates - valid_counts,
    )
