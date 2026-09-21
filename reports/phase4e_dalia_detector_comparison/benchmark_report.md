# PPG-DaLiA ECG detector comparison

## Overall comparison

| detector | peak_precision_macro | peak_recall_macro | peak_f1_macro | peak_timing_error_ms_macro | runtime_s_total | matched_windows | coverage | hr_mae_bpm | hr_rmse_bpm | hr_bias_bpm | hr_correlation | windows_over_5_bpm | windows_over_10_bpm | windows_over_20_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | 0.9855 | 0.9950 | 0.9902 | 1.1429 | 3.8416 | 64697 | 1.0000 | 1.7751 | 5.8517 | 1.2604 | 0.9695 | 5066 | 3938 | 1405 |
| xqrs | 0.9918 | 0.9929 | 0.9923 | 0.0952 | 225.2126 | 64697 | 1.0000 | 0.9049 | 3.0703 | 0.2900 | 0.9911 | 2397 | 1422 | 336 |
| emrich2023 | 0.9959 | 0.9935 | 0.9947 | 1.4286 | 44.1756 | 64696 | 1.0000 | 0.6725 | 2.1586 | -0.0055 | 0.9955 | 1518 | 767 | 134 |
| sleepecg | 0.9895 | 0.9904 | 0.9899 | 0.5714 | 1.3516 | 64695 | 1.0000 | 1.0401 | 3.4075 | 0.2719 | 0.9890 | 3089 | 1617 | 393 |

All detectors use one-to-one annotation matching with ±100 ms tolerance and the same frozen RR-to-window-HR calculation. XQRS uses learning with default WFDB configuration; NeuroKit2 uses Emrich 2023/FastNVG without artifact correction; SleepECG uses its default compiled adaptive Pan–Tompkins implementation. No detector-specific threshold was tuned in this comparison.

## Decision rule

Choose using clinically relevant HR MAE/RMSE and large-error counts together with peak F1, coverage, runtime, reproducibility, and external PTT-PPG evidence. This is development comparison on PPG-DaLiA and cannot replace external validation.
