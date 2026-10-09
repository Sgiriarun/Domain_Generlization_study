"""Utilities for frozen PPG-window manifests and leakage-safe subject splits."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


PRIMARY_CHANNELS = {
    "BIDMC": "PLETH",
    "PPG-DaLiA": "BVP",
    "PTT-PPG": "pleth_1",
    "WESAD": "BVP",
}

# Site membership is consistent in the PTT-PPG documentation. Wavelength names
# are deliberately not encoded here because the v1.0 README contains conflicting
# mappings between its hardware overview and detailed channel list.
PTT_SITE_METADATA = {
    "pleth_1": ("distal", "pair_1_4"),
    "pleth_2": ("distal", "pair_2_5"),
    "pleth_3": ("distal", "pair_3_6"),
    "pleth_4": ("proximal", "pair_1_4"),
    "pleth_5": ("proximal", "pair_2_5"),
    "pleth_6": ("proximal", "pair_3_6"),
}


def zscore_window(values: NDArray[np.floating]) -> NDArray[np.float32]:
    """Normalize one model input window without using other windows or subjects."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("window must be non-empty and finite")
    standard_deviation = float(np.std(x))
    if standard_deviation <= np.finfo(np.float32).eps:
        raise ValueError("constant window cannot be standardized")
    return np.asarray((x - np.mean(x)) / standard_deviation, dtype=np.float32)


def deterministic_subject_folds(
    subject_ids: Iterable[str], *, n_folds: int = 5, seed: int = 17
) -> dict[str, int]:
    """Assign unique subjects to stable, approximately balanced folds.

    A cryptographic hash makes the assignment independent of input row order and
    Python's process-randomized hash implementation.
    """
    subjects = sorted({str(subject_id) for subject_id in subject_ids})
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(subjects) < n_folds:
        raise ValueError("number of unique subjects must be at least n_folds")

    def key(subject_id: str) -> bytes:
        return hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()

    ordered = sorted(subjects, key=key)
    return {subject_id: index % n_folds for index, subject_id in enumerate(ordered)}

