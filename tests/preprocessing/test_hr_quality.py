from __future__ import annotations

import unittest

from rq1_hr.preprocessing.hr_quality import classify_hr_window


class HRWindowQualityTests(unittest.TestCase):
    def test_valid_window_passes(self) -> None:
        result = classify_hr_window(72.0, 8, detector_disagreement_bpm=0.2)
        self.assertTrue(result.include_primary)
        self.assertFalse(result.review_recommended)
        self.assertEqual(result.quality_flag, "pass")

    def test_fewer_than_four_rr_intervals_is_excluded(self) -> None:
        result = classify_hr_window(72.0, 3)
        self.assertFalse(result.include_primary)
        self.assertEqual(result.quality_flag, "insufficient_valid_rr")

    def test_large_disagreement_is_flagged_but_retained(self) -> None:
        result = classify_hr_window(72.0, 8, detector_disagreement_bpm=10.1)
        self.assertTrue(result.include_primary)
        self.assertTrue(result.review_recommended)
        self.assertEqual(result.quality_flag, "detector_disagreement")


if __name__ == "__main__":
    unittest.main()
