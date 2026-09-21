"""Frozen Phase 4 rules for accepting and flagging window-level HR labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class HRWindowQuality:
    """Decision for one HR window.

    ``include_primary`` is a minimum usability decision. ``review_recommended``
    is deliberately separate: detector disagreement is evidence of uncertainty,
    not proof that the selected detector is wrong.
    """

    include_primary: bool
    review_recommended: bool
    quality_flag: str


def classify_hr_window(
    hr_bpm: float,
    valid_rr_count: int,
    *,
    detector_disagreement_bpm: float | None = None,
    min_valid_rr: int = 4,
    disagreement_threshold_bpm: float = 10.0,
) -> HRWindowQuality:
    """Apply the dataset-independent Phase 4 HR quality policy."""
    if not np.isfinite(hr_bpm) or valid_rr_count < min_valid_rr:
        return HRWindowQuality(False, True, "insufficient_valid_rr")

    if (
        detector_disagreement_bpm is not None
        and np.isfinite(detector_disagreement_bpm)
        and detector_disagreement_bpm > disagreement_threshold_bpm
    ):
        return HRWindowQuality(True, True, "detector_disagreement")

    return HRWindowQuality(True, False, "pass")
