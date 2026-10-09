# Phase 8: conventional frequency-domain HR baseline

## Frozen estimator

For each filtered 8-second primary-channel PPG window, subtract its mean, apply a Hann taper, calculate a 4096-point zero-padded FFT, select the strongest spectral component between 35 and 220 bpm, and convert frequency to HR using `bpm = Hz × 60`.

The estimator has no training data and no dataset-specific tuning. Zero-padding produces a finer numerical frequency grid but does not create new physiological information. Motion and harmonic peaks can still be mistaken for HR; that limitation is part of this baseline.

## Primary-channel results

| dataset | windows | mae_bpm | rmse_bpm | pearson_r | within_5_bpm_percent | bland_altman_bias_bpm | bland_altman_lower_loa_bpm | bland_altman_upper_loa_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPG-DaLiA | 64697 | 17.620 | 29.706 | 0.401 | 52.817 | -14.043 | -65.351 | 37.264 |
| PTT-PPG | 15982 | 10.262 | 20.817 | 0.530 | 75.579 | -7.655 | -45.599 | 30.289 |
| WESAD | 43385 | 12.595 | 22.079 | 0.329 | 57.218 | -9.237 | -48.543 | 30.068 |
| BIDMC | 12561 | 3.962 | 13.910 | 0.641 | 89.308 | -0.572 | -27.814 | 26.669 |

## PTT six-channel secondary results

| channel_name | sensor_site | site_pair | windows | mae_bpm | rmse_bpm | pearson_r | within_5_bpm_percent | bland_altman_bias_bpm | bland_altman_lower_loa_bpm | bland_altman_upper_loa_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pleth_1 | distal | pair_1_4 | 15982 | 10.262 | 20.817 | 0.530 | 75.579 | -7.655 | -45.599 | 30.289 |
| pleth_2 | distal | pair_2_5 | 15982 | 8.846 | 19.345 | 0.568 | 79.740 | -6.627 | -42.250 | 28.996 |
| pleth_3 | distal | pair_3_6 | 15982 | 5.644 | 14.528 | 0.700 | 87.085 | -4.166 | -31.446 | 23.114 |
| pleth_4 | proximal | pair_1_4 | 15982 | 23.722 | 33.816 | 0.081 | 42.779 | -18.264 | -74.046 | 37.519 |
| pleth_5 | proximal | pair_2_5 | 15982 | 22.454 | 33.174 | 0.100 | 46.308 | -16.399 | -72.923 | 40.125 |
| pleth_6 | proximal | pair_3_6 | 15982 | 14.301 | 25.671 | 0.357 | 66.569 | -11.980 | -56.482 | 32.522 |

Metrics are descriptive across overlapping windows. `subject_metrics.csv` is the correct basis for later subject-level uncertainty and comparisons. The spectral baseline does not itself have a leave-one-dataset-out training stage because it learns no parameters; it is applied unchanged to every dataset. The learned CNN will use the frozen Phase 7 LODO splits.
