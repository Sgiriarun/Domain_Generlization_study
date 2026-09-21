# WESAD ECG-derived HR and detector-quality report

## Method

NeuroKit2 Emrich 2023/FastNVG is applied with its frozen default implementation to 700 Hz WESAD chest ECG. HR uses 8-second windows, 2-second step, RR-midpoint assignment, 35–220 bpm interval filtering, at least four valid RR intervals, and arithmetic mean.

WESAD has no supplied R-peaks or HR. Therefore this report does not claim accuracy. XQRS and SleepECG are independent comparison detectors; disagreement flags identify windows needing review but do not prove which detector is correct.

## Detector agreement

| comparison | matched_windows | median_absolute_difference_bpm | mean_absolute_difference_bpm | p95_absolute_difference_bpm | windows_over_5_bpm | windows_over_10_bpm | correlation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Emrich 2023 vs XQRS | 43385 | 0.0151 | 0.1996 | 0.1850 | 362 | 270 | 0.9935 |
| Emrich 2023 vs SleepECG | 43385 | 0.0148 | 0.1783 | 0.1377 | 343 | 219 | 0.9940 |
| XQRS vs SleepECG | 43385 | 0.0033 | 0.1142 | 0.0971 | 261 | 167 | 0.9973 |

## Subject summary

| subject_id | windows | emrich_hr_coverage | emrich_hr_median_bpm | emrich_hr_min_bpm | emrich_hr_max_bpm | emrich_invalid_rr | disagreement_over_5_bpm | disagreement_over_10_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S2 | 3036 | 1.0000 | 71.3877 | 57.3576 | 119.6853 | 0 | 315 | 267 |
| S3 | 3243 | 1.0000 | 56.8602 | 44.1813 | 131.6523 | 8 | 8 | 8 |
| S4 | 3208 | 1.0000 | 63.0172 | 51.2143 | 99.3275 | 4 | 0 | 0 |
| S5 | 3126 | 1.0000 | 66.9497 | 53.7869 | 102.5913 | 0 | 19 | 6 |
| S6 | 3532 | 1.0000 | 69.2047 | 57.7695 | 102.4714 | 0 | 0 | 0 |
| S7 | 2616 | 1.0000 | 68.6047 | 51.6821 | 129.0717 | 0 | 4 | 0 |
| S8 | 2730 | 1.0000 | 73.4161 | 56.3086 | 112.9087 | 0 | 36 | 14 |
| S9 | 2608 | 1.0000 | 77.5429 | 61.1460 | 119.3819 | 0 | 7 | 0 |
| S10 | 2745 | 1.0000 | 92.5682 | 60.2230 | 138.6401 | 0 | 20 | 7 |
| S11 | 2613 | 1.0000 | 86.2948 | 60.4475 | 139.2874 | 0 | 0 | 0 |
| S13 | 2765 | 1.0000 | 85.4097 | 65.0699 | 132.1905 | 0 | 0 | 0 |
| S14 | 2771 | 1.0000 | 80.7832 | 63.5149 | 151.9880 | 0 | 4 | 2 |
| S15 | 2623 | 1.0000 | 79.7290 | 56.9761 | 109.6506 | 0 | 10 | 8 |
| S16 | 2812 | 1.0000 | 80.6125 | 55.7780 | 154.3015 | 4 | 4 | 4 |
| S17 | 2957 | 1.0000 | 70.3286 | 50.9773 | 140.8743 | 0 | 4 | 0 |

## Protocol-condition summary

| condition_id | condition_name | windows | subjects | emrich_hr_median_bpm | emrich_hr_q05_bpm | emrich_hr_q95_bpm | emrich_hr_coverage | disagreement_over_5_bpm | disagreement_over_10_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | not_defined | 19710 | 15 | 75.1966 | 56.5529 | 105.7357 | 1.0000 | 342 | 279 |
| 1 | baseline | 8804 | 15 | 71.5377 | 55.5598 | 96.3218 | 1.0000 | 17 | 8 |
| 2 | stress | 4981 | 15 | 93.0738 | 73.0951 | 133.7765 | 1.0000 | 65 | 24 |
| 3 | amusement | 2788 | 15 | 72.3318 | 54.8892 | 90.0252 | 1.0000 | 5 | 5 |
| 4 | meditation | 5905 | 15 | 69.4321 | 54.6826 | 85.0512 | 1.0000 | 2 | 0 |
| 5 | not_defined | 393 | 14 | 80.1099 | 54.8713 | 94.3637 | 1.0000 | 0 | 0 |
| 6 | not_defined | 393 | 15 | 77.6052 | 59.7278 | 108.4246 | 1.0000 | 0 | 0 |
| 7 | not_defined | 411 | 15 | 75.1017 | 57.9500 | 90.6801 | 1.0000 | 0 | 0 |

## Interpretation rules

- Emrich HR is the selected candidate reference.
- Missing HR means fewer than four valid RR intervals.
- Pairwise disagreement above 5 or 10 bpm is retained as a quality flag, not automatically rejected.
- Protocol labels 0, 5, 6, and 7 remain `not_defined` according to the source label scheme and should not be interpreted as named experimental conditions.
- Final acceptance requires visual review of representative agreement and disagreement windows and later BIDMC monitor-HR validation.
