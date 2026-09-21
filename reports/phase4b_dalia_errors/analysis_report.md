# Phase 4B — PPG-DaLiA ECG detector error analysis

## Error distribution

| absolute_error_bpm | windows | fraction |
| --- | --- | --- |
| ≤2 | 57268 | 0.885 |
| 2–5 | 2363 | 0.037 |
| 5–10 | 1128 | 0.017 |
| 10–20 | 2533 | 0.039 |
| >20 | 1405 | 0.022 |

Because windows overlap by 6 seconds, neighbouring window errors are correlated and must not be interpreted as independent failures.

## Subjects with highest MAE

| subject_id | windows | mae_bpm | median_ae_bpm | p95_ae_bpm | max_ae_bpm | windows_over_5_bpm | windows_over_10_bpm | total_missed_peaks | total_extra_peaks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S10 | 5321 | 4.569 | 0.176 | 27.305 | 67.615 | 1093 | 923 | 1199 | 3354 |
| S8 | 4037 | 3.329 | 0.368 | 21.020 | 74.810 | 497 | 412 | 70 | 1286 |
| S14 | 4476 | 2.583 | 0.355 | 16.092 | 46.536 | 563 | 399 | 753 | 1457 |
| S7 | 4668 | 2.352 | 0.355 | 15.232 | 59.605 | 500 | 407 | 154 | 1126 |
| S4 | 4572 | 1.968 | 0.250 | 14.369 | 53.748 | 429 | 346 | 55 | 863 |

## Activity IDs with highest MAE

| activity_id | windows | mae_bpm | p95_ae_bpm | windows_over_10_bpm | missed_peaks | extra_peaks |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 2310 | 6.486 | 28.862 | 626 | 376 | 1667 |
| 2 | 3239 | 3.672 | 24.697 | 385 | 553 | 1709 |
| 7 | 4697 | 2.915 | 19.036 | 522 | 779 | 1548 |
| 0 | 17515 | 2.155 | 14.658 | 1344 | 1545 | 3788 |
| 4 | 3473 | 1.946 | 13.136 | 219 | 241 | 909 |

Activity is assigned as the modal source activity ID in each 8-second window. IDs are retained instead of attaching undocumented names.

## Observed failure pattern

Most windows are accurate: 88.5% have absolute error at or below 2 bpm. However, 6.1% exceed 10 bpm. The errors are episodic rather than uniformly poor; for example, S10 has median absolute error 0.18 bpm but 95th-percentile error 27.3 bpm.

In the inspected worst windows, extra detections dominate. The detector responds to large non-R deflections between supplied beats, which shortens RR intervals and can approximately double estimated HR. The S10 examples also show strong ECG clipping/distortion. These observations justify investigating morphology/RR-consistency rejection or a validated adaptive detector; they do not justify changing a threshold solely to improve this dataset.

Activity ID 3 has the largest aggregate MAE, followed by IDs 2 and 7. This is an association with protocol segments, not evidence that motion caused every error.

## How to read the diagnostic plots

Green solid lines are supplied R-peaks; red dashed lines are our detections. A red-only line is typically an extra peak and a green-only line a missed peak. Wrist ACC magnitude provides motion context but does not prove that motion caused an error.

## Scope

This phase diagnoses the frozen development detector. It does not change detector parameters or delete windows. Selected examples are listed in `selected_examples.csv`, and plots are under `diagnostic_plots/`.
