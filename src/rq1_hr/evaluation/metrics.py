"""Regression metrics used consistently across RQ1 baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def hr_metrics(reference: ArrayLike, prediction: ArrayLike) -> dict[str, float | int]:
    y = np.asarray(reference, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(p)):
        raise ValueError("reference and prediction must be equally sized, non-empty, and finite")
    error = p - y
    correlation = float(np.corrcoef(y, p)[0, 1]) if np.std(y) > 0 and np.std(p) > 0 else np.nan
    bias = float(np.mean(error))
    error_sd = float(np.std(error, ddof=1)) if error.size > 1 else 0.0
    return {
        "windows": int(y.size),
        "mae_bpm": float(np.mean(np.abs(error))),
        "rmse_bpm": float(np.sqrt(np.mean(error**2))),
        "pearson_r": correlation,
        "within_5_bpm_percent": float(100 * np.mean(np.abs(error) <= 5.0)),
        "bland_altman_bias_bpm": bias,
        "bland_altman_lower_loa_bpm": bias - 1.96 * error_sd,
        "bland_altman_upper_loa_bpm": bias + 1.96 * error_sd,
    }

