import numpy as np
import pytest

from rq1_hr.baselines import estimate_hr_spectral


def test_spectral_estimator_recovers_known_heart_rate():
    fs = 64.0
    time = np.arange(8 * int(fs)) / fs
    signal = np.sin(2 * np.pi * 1.5 * time)
    estimate = estimate_hr_spectral(signal, fs)
    assert estimate.hr_bpm == pytest.approx(90.0, abs=1.0)
    assert 0 < estimate.peak_power_fraction <= 1


def test_spectral_estimator_rejects_constant_window():
    with pytest.raises(ValueError):
        estimate_hr_spectral(np.ones(512), 64.0)

