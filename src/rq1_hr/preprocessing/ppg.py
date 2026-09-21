"""Common PPG preparation and transparent window-quality measurements."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, resample_poly, sosfiltfilt, welch


@dataclass(frozen=True, slots=True)
class PPGWindowQuality:
    finite_fraction: float
    mean: float
    standard_deviation: float
    peak_to_peak: float
    flat_difference_fraction: float
    edge_value_fraction: float
    dominant_frequency_hz: float
    spectral_concentration: float
    structurally_usable: bool
    structural_flag: str


def prepare_ppg(
    values: NDArray[np.floating],
    native_fs_hz: float,
    *,
    target_fs_hz: float = 64.0,
    low_hz: float = 0.5,
    high_hz: float = 4.0,
    filter_order: int = 4,
) -> NDArray[np.float32]:
    """Resample a complete recording and apply a zero-phase band-pass filter.

    Processing the continuous recording avoids artificial filter edges at every
    overlapping window. ``resample_poly`` includes anti-alias filtering.
    """
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2 or signal.shape[0] == 0:
        raise ValueError("PPG must be a non-empty samples-by-channels array")
    if not np.all(np.isfinite(signal)):
        raise ValueError("continuous PPG contains non-finite values")
    if not 0 < low_hz < high_hz < target_fs_hz / 2:
        raise ValueError("band-pass cutoffs must lie below the target Nyquist rate")

    ratio = Fraction(target_fs_hz / native_fs_hz).limit_denominator(10_000)
    if not np.isclose(native_fs_hz * ratio.numerator / ratio.denominator, target_fs_hz):
        raise ValueError("sampling-rate ratio cannot be represented accurately")
    if ratio.numerator != ratio.denominator:
        signal = resample_poly(signal, ratio.numerator, ratio.denominator, axis=0)

    sos = butter(filter_order, [low_hz, high_hz], btype="bandpass", fs=target_fs_hz, output="sos")
    signal = sosfiltfilt(sos, signal, axis=0)
    return np.asarray(signal, dtype=np.float32)


def measure_ppg_window(values: NDArray[np.floating], fs_hz: float) -> PPGWindowQuality:
    """Return descriptive quality evidence without motion-quality rejection."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size < 2:
        raise ValueError("a PPG window needs at least two samples")
    finite = np.isfinite(x)
    finite_fraction = float(np.mean(finite))
    if not np.all(finite):
        return PPGWindowQuality(finite_fraction, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, "nonfinite")

    mean = float(np.mean(x))
    std = float(np.std(x))
    peak_to_peak = float(np.ptp(x))
    tolerance = np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(x))))
    flat_fraction = float(np.mean(np.abs(np.diff(x)) <= tolerance))
    edge_fraction = float(np.mean((x == np.min(x)) | (x == np.max(x))))
    usable = bool(peak_to_peak > tolerance)

    frequency, power = welch(x, fs=fs_hz, nperseg=min(x.size, 256))
    pulse_band = (frequency >= 0.5) & (frequency <= 4.0)
    if usable and np.any(pulse_band) and float(np.sum(power[pulse_band])) > 0:
        band_frequency = frequency[pulse_band]
        band_power = power[pulse_band]
        peak_index = int(np.argmax(band_power))
        dominant = float(band_frequency[peak_index])
        neighborhood = np.abs(band_frequency - dominant) <= 0.25
        concentration = float(np.sum(band_power[neighborhood]) / np.sum(band_power))
    else:
        dominant, concentration = np.nan, np.nan

    return PPGWindowQuality(
        finite_fraction, mean, std, peak_to_peak, flat_fraction, edge_fraction,
        dominant, concentration, usable, "pass" if usable else "constant",
    )
