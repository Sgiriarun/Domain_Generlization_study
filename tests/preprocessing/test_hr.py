from __future__ import annotations

import unittest

import numpy as np

from rq1_hr.preprocessing import (
    detect_rpeaks,
    detect_rpeaks_xqrs,
    match_rpeaks,
    window_hr_from_rpeaks,
)


class WindowHRTests(unittest.TestCase):
    def test_regular_60_bpm_produces_60_bpm_windows(self) -> None:
        peaks = np.arange(0, 21, dtype=np.int64) * 100
        result = window_hr_from_rpeaks(peaks, 100, 20, window_s=8, step_s=2)
        np.testing.assert_allclose(result.values_bpm, 60)
        self.assertTrue(np.all(result.valid_rr_count >= 4))

    def test_implausible_interval_is_flagged_and_excluded(self) -> None:
        peaks = np.array([0, 100, 200, 220, 320, 420, 520, 620, 720, 820])
        result = window_hr_from_rpeaks(peaks, 100, 9, window_s=8, step_s=2)
        self.assertEqual(result.invalid_rr_count[0], 1)
        self.assertAlmostEqual(result.values_bpm[0], 60)

    def test_too_few_valid_intervals_returns_nan(self) -> None:
        peaks = np.array([0, 100, 200, 300])
        result = window_hr_from_rpeaks(peaks, 100, 8, window_s=8, min_valid_rr=4)
        self.assertTrue(np.isnan(result.values_bpm[0]))


class PeakMatchingTests(unittest.TestCase):
    def test_matching_is_one_to_one(self) -> None:
        result = match_rpeaks(
            np.array([100, 200, 300]), np.array([98, 199, 250, 302]), 100,
            tolerance_s=0.05,
        )
        self.assertEqual(result.matched_count, 3)
        self.assertEqual(result.detected_count, 4)
        self.assertAlmostEqual(result.precision, 0.75)
        self.assertAlmostEqual(result.recall, 1.0)

    def test_detector_finds_clean_synthetic_beats(self) -> None:
        fs = 200
        time = np.arange(10 * fs) / fs
        ecg = 0.02 * np.sin(2 * np.pi * time)
        expected = np.arange(1, 10) * fs
        for peak in expected:
            offsets = np.arange(-8, 9)
            ecg[peak + offsets] += np.exp(-0.5 * (offsets / 2.0) ** 2)
        detected = detect_rpeaks(ecg, fs)
        result = match_rpeaks(expected, detected, fs, tolerance_s=0.05)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 1.0)

    def test_xqrs_returns_ordered_indices(self) -> None:
        fs = 200
        time = np.arange(20 * fs) / fs
        ecg = 0.01 * np.sin(2 * np.pi * time)
        for peak in np.arange(1, 20) * fs:
            offsets = np.arange(-8, 9)
            ecg[peak + offsets] += np.exp(-0.5 * (offsets / 2.0) ** 2)
        detected = detect_rpeaks_xqrs(ecg, fs)
        self.assertGreater(detected.size, 0)
        self.assertTrue(np.all(np.diff(detected) > 0))


if __name__ == "__main__":
    unittest.main()
