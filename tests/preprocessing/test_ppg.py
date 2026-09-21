from __future__ import annotations

import unittest

import numpy as np

from rq1_hr.preprocessing.ppg import measure_ppg_window, prepare_ppg


class PPGPreparationTests(unittest.TestCase):
    def test_resampling_has_expected_shape_and_preserves_pulse_frequency(self) -> None:
        native_fs = 125
        time = np.arange(16 * native_fs) / native_fs
        source = np.sin(2 * np.pi * 1.25 * time)
        result = prepare_ppg(source, native_fs)
        self.assertEqual(result.shape, (16 * 64, 1))
        quality = measure_ppg_window(result[: 8 * 64, 0], 64)
        self.assertAlmostEqual(quality.dominant_frequency_hz, 1.25, delta=0.26)
        self.assertTrue(quality.structurally_usable)

    def test_multichannel_input_retains_channels(self) -> None:
        time = np.arange(8 * 500) / 500
        source = np.column_stack([np.sin(2 * np.pi * f * time) for f in (1.0, 1.5, 2.0)])
        result = prepare_ppg(source, 500)
        self.assertEqual(result.shape, (8 * 64, 3))

    def test_constant_window_is_not_structurally_usable(self) -> None:
        quality = measure_ppg_window(np.zeros(512), 64)
        self.assertFalse(quality.structurally_usable)
        self.assertEqual(quality.structural_flag, "constant")


if __name__ == "__main__":
    unittest.main()
