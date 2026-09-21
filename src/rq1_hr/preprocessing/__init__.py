"""Common signal preprocessing and window generation."""
"""Common preprocessing components for ECG references and PPG inputs."""

from rq1_hr.preprocessing.hr import WindowHR, window_hr_from_rpeaks
from rq1_hr.preprocessing.rpeaks import (
    PeakMatch,
    detect_rpeaks,
    detect_rpeaks_emrich2023,
    detect_rpeaks_sleepecg,
    detect_rpeaks_xqrs,
    match_rpeaks,
)

__all__ = [
    "PeakMatch",
    "WindowHR",
    "detect_rpeaks",
    "detect_rpeaks_emrich2023",
    "detect_rpeaks_sleepecg",
    "detect_rpeaks_xqrs",
    "match_rpeaks",
    "window_hr_from_rpeaks",
]
from .hr_quality import HRWindowQuality, classify_hr_window
from .ppg import PPGWindowQuality, measure_ppg_window, prepare_ppg
