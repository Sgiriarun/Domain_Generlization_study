"""Transparent frequency-domain heart-rate baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SpectralEstimate:
    hr_bpm: float
    peak_frequency_hz: float
    peak_power_fraction: float


def estimate_hr_spectral(
    values: NDArray[np.floating],
    fs_hz: float,
    *,
    min_hr_bpm: float = 35.0,
    max_hr_bpm: float = 220.0,
    n_fft: int = 4096,
) -> SpectralEstimate:
    """Estimate HR from the dominant Hann-windowed PPG spectral peak.

    Zero-padding provides a smooth, deterministic frequency grid but does not
    add physiological information beyond the original eight-second window.
    """
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("PPG window must contain at least two finite samples")
    if not np.isfinite(fs_hz) or fs_hz <= 0:
        raise ValueError("fs_hz must be finite and positive")
    if not 0 < min_hr_bpm < max_hr_bpm < fs_hz * 30:
        raise ValueError("HR limits must be ordered and below Nyquist")
    if n_fft < x.size:
        raise ValueError("n_fft must be at least the window length")

    centred = x - np.mean(x)
    if float(np.std(centred)) <= np.finfo(np.float32).eps:
        raise ValueError("constant PPG window has no usable spectrum")
    tapered = centred * np.hanning(x.size)
    power = np.abs(np.fft.rfft(tapered, n=n_fft)) ** 2
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / fs_hz)
    cardiac = (frequencies >= min_hr_bpm / 60.0) & (frequencies <= max_hr_bpm / 60.0)
    indices = np.flatnonzero(cardiac)
    if indices.size == 0 or float(np.sum(power[indices])) <= 0:
        raise ValueError("PPG window has no power inside the cardiac band")
    peak_index = int(indices[np.argmax(power[indices])])
    peak_frequency = float(frequencies[peak_index])
    concentration = float(power[peak_index] / np.sum(power[indices]))
    return SpectralEstimate(peak_frequency * 60.0, peak_frequency, concentration)

