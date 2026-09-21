# BIDMC ECG-derived HR validation

## Method

Frozen Emrich 2023, XQRS, and SleepECG detectors are applied to lead II. Their 8-second/2-second-step HR is compared with the mean of finite 1 Hz bedside-monitor HR values in the same interval.

The monitor series is an independent clinical comparison but not sample-level R-peak ground truth. Monitor smoothing, update timing, and proprietary processing can contribute to disagreement.

## Overall comparison

| comparison | windows | mae_bpm | rmse_bpm | bias_bpm | correlation | over_5_bpm | over_10_bpm | over_20_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| monitor vs emrich2023 | 12561 | 1.6235 | 3.6715 | 0.4410 | 0.9622 | 998 | 315 | 67 |
| monitor vs xqrs | 12560 | 3.4910 | 10.3230 | 2.6353 | 0.7496 | 1778 | 978 | 533 |
| monitor vs sleepecg | 12561 | 1.7294 | 4.0332 | 0.7021 | 0.9553 | 1089 | 402 | 94 |

## Records with highest Emrich-versus-monitor MAE

| record_id | subject_id | monitor_coverage | emrich2023_mae_bpm | emrich2023_rmse_bpm | emrich2023_correlation | detector_disagreement_over_10_bpm |
| --- | --- | --- | --- | --- | --- | --- |
| bidmc41 | s26868 | 1.0000 | 13.1306 | 18.5876 | -0.2418 | 6 |
| bidmc45 | s29622 | 1.0000 | 5.7796 | 6.8941 | 0.0443 | 4 |
| bidmc26 | s13314 | 1.0000 | 5.5895 | 6.4737 | -0.1573 | 181 |
| bidmc53 | s32628 | 1.0000 | 4.9940 | 6.3486 | 0.0340 | 102 |
| bidmc17 | s08936 | 1.0000 | 4.8256 | 6.1004 | 0.3910 | 0 |
| bidmc28 | s15646 | 1.0000 | 4.1844 | 5.6121 | 0.3391 | 0 |
| bidmc24 | s12531 | 1.0000 | 3.8268 | 4.6553 | 0.2212 | 0 |
| bidmc33 | s19981 | 1.0000 | 3.0780 | 3.9731 | 0.2337 | 94 |
| bidmc03 | s01795 | 1.0000 | 2.3538 | 4.0189 | 0.1896 | 18 |
| bidmc23 | s11342 | 1.0000 | 2.1567 | 4.4216 | 0.0687 | 24 |

All window values and quality counts are retained in `window_hr_and_quality.csv`. No window is removed by this validation report.
