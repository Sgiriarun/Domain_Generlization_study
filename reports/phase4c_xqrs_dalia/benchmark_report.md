# WFDB XQRS benchmark on PPG-DaLiA

## Overall comparison

| detector | peak_precision_macro | peak_recall_macro | peak_f1_macro | peak_timing_error_ms_macro | matched_windows | coverage | hr_mae_bpm | hr_rmse_bpm | hr_bias_bpm | hr_correlation | windows_over_5_bpm | windows_over_10_bpm | windows_over_20_bpm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development | 0.9855 | 0.9950 | 0.9902 | 1.1429 | 64697 | 1.0000 | 1.7751 | 5.8517 | 1.2604 | 0.9695 | 5066 | 3938 | 1405 |
| xqrs | 0.9918 | 0.9929 | 0.9923 | 0.0952 | 64697 | 1.0000 | 0.9049 | 3.0703 | 0.2900 | 0.9911 | 2397 | 1422 | 336 |

Both detectors use one-to-one annotation matching with ±100 ms tolerance and the same frozen RR-to-window-HR calculation. XQRS is run with learning enabled and otherwise default WFDB configuration.

## Decision rule

Prefer XQRS only if it improves the clinically relevant HR errors—not merely peak timing—and does not hide reduced window coverage. This remains development validation on PPG-DaLiA; the selected detector must next run unchanged on PTT-PPG for external validation.
