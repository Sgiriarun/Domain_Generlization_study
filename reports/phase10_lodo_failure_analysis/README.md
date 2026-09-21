# TimePPG full-LODO failure analysis — seed 17

## Scope and integrity

- Every LODO prediction was paired one-to-one with the within-dataset out-of-fold prediction for the same target window.
- Matching keys contain no missing values; physical start times/sample bounds agree; every subject occurs in exactly one within-test fold; and labels/predictions are finite.
- Analysis covers 136,625 windows and 98 dataset-specific subjects.
- Subject bootstrap keeps each person's overlapping windows together.
- Subgroup and adjusted results are post-hoc associations, not isolated causal effects.

## Overall paired result

| dataset | within_mae_bpm | lodo_mae_bpm | pooled_paired_gap_bpm | lodo_rmse_bpm | lodo_bias_bpm | lodo_p95_absolute_error_bpm | lodo_over_20_bpm_percent |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BIDMC | 1.916 | 1.859 | -0.057 | 4.055 | -0.774 | 7.002 | 0.860 |
| WESAD | 4.887 | 4.631 | -0.256 | 8.650 | -0.458 | 18.839 | 4.455 |
| PTT-PPG | 4.694 | 15.009 | 10.314 | 20.675 | 11.899 | 42.399 | 32.055 |
| PPG-DaLiA | 6.036 | 7.353 | 1.317 | 15.043 | -3.493 | 33.633 | 9.940 |

## How widespread is deterioration?

| dataset | subjects | subjects_worse | subjects_worse_percent | subjects_improved | median_subject_gap_bpm | mean_subject_gap_bpm | min_subject_gap_bpm | max_subject_gap_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BIDMC | 46 | 31 | 67.391 | 15 | 0.276 | 0.111 | -9.427 | 9.606 |
| WESAD | 15 | 5 | 33.333 | 10 | -0.329 | -0.241 | -0.932 | 0.557 |
| PTT-PPG | 22 | 21 | 95.455 | 1 | 11.332 | 10.331 | -40.449 | 28.104 |
| PPG-DaLiA | 15 | 11 | 73.333 | 4 | 0.529 | 1.474 | -1.581 | 8.864 |

PTT-PPG is the dominant failure: pooled MAE rises from **4.694** to
**15.009 bpm**, a paired window-level difference of
**+10.314 bpm**. At the subject level, the mean gap is
**+10.331 bpm** with a subject-bootstrap 95% interval of
**+4.361 to +14.937 bpm**.

## Mechanism-level findings

- **PTT is a low-HR overestimation failure:** LODO bias is **+11.899 bpm** overall; the `<60` and `60-80` bpm rows in `hr_bin_metrics.csv` show the largest errors. Those bins contain many source-training windows, so simple HR-bin count scarcity is not a sufficient explanation.
- **PTT tracks the target HR range poorly:** its prediction-versus-reference calibration slope is **0.404** (ideal 1.0), consistent with weak target tracking rather than a constant offset alone.
- **PTT failure persists across quality tertiles:** low spectral concentration is harder, but even the high-concentration tertile remains poor. Quality contributes without fully explaining transfer failure.
- **Motion alone is unsupported:** unadjusted sitting is hardest, while adjusted walking-versus-sitting and running-versus-sitting bootstrap intervals include zero.
- **Spectral disagreement remains informative:** larger FFT/reference mismatch is associated with larger TimePPG error after within-subject adjustment.
- **DaLiA has a different failure shape:** error rises sharply above 140 bpm, where source training support is sparse, and predictions are strongly biased downward.
- **Failures are sustained episodes:** `large_error_episodes.csv` identifies long consecutive sequences, so overlapping windows must not be counted as independent failure events.

## Worst PTT-PPG subjects

| subject_id | within_mae_bpm | lodo_mae_bpm | gap_bpm | lodo_bias_bpm | lodo_over_20_bpm_percent |
| --- | --- | --- | --- | --- | --- |
| s9 | 7.085 | 30.148 | 23.062 | 30.006 | 75.728 |
| s16 | 1.225 | 29.329 | 28.104 | 29.329 | 85.477 |
| s14 | 11.628 | 27.419 | 15.791 | 27.306 | 66.207 |
| s20 | 4.218 | 24.849 | 20.631 | 22.362 | 53.984 |
| s12 | 2.405 | 20.536 | 18.131 | 16.882 | 38.633 |

## PTT adjusted within-subject diagnostic

| predictor | coefficient_bpm | subject_bootstrap_ci_low_bpm | subject_bootstrap_ci_high_bpm |
| --- | --- | --- | --- |
| hr_per_10_bpm | -4.212 | -6.444 | -2.338 |
| spectral_concentration_per_0_1 | -1.088 | -1.919 | -0.359 |
| fft_mismatch_per_10_bpm | 1.196 | 0.515 | 1.695 |
| walking_vs_sitting | -1.586 | -5.588 | 2.804 |
| running_vs_sitting | -2.272 | -5.985 | 1.828 |

These coefficients compare changes within a subject while adjusting for the listed predictors. They
help reject simplistic explanations but do not prove causality. Spectral concentration is a proxy,
not an independently validated signal-quality label.

## Interpretation boundary and next decisions

1. Use `subject_bootstrap_ci.csv` to judge whether target gaps survive subject variation.
2. Use `ptt_activity_hr_metrics.csv` and the adjusted table to determine whether activity remains associated after HR and quality adjustment.
3. Use `source_hr_support.csv` to distinguish sparse source HR coverage from residual waveform mismatch.
4. Inspect `large_error_episodes.csv` and corresponding waveforms before treating adjacent windows as independent failures.
5. Repeat LODO with prespecified additional seeds before selecting an RQ2 mechanism.
6. Run paired PTT distal/proximal TimePPG transfer separately; primary LODO uses only `pleth_1` and cannot isolate site causally.

## Files

- `overall_paired_metrics.csv`: complete target metrics and paired gaps.
- `subject_metrics.csv`: localisation and subject-level variation.
- `condition_metrics.csv`, `hr_bin_metrics.csv`, `quality_tertile_metrics.csv`: one-factor diagnostics.
- `joint_stratum_metrics.csv`: jointly stratified diagnostics with sample counts.
- `ptt_adjusted_within_subject_associations.csv`: exploratory adjusted PTT analysis.
- `source_hr_support.csv`: source-training coverage versus target error.
- `large_error_episodes.csv`: contiguous >20-bpm LODO error episodes.
- `figures/`: presentation-ready diagnostic plots.
