# Phase 4A — PPG-DaLiA ECG-to-HR validation

## Design

Two paths are evaluated independently. Path 1 uses dataset-supplied R-peaks and therefore tests RR conversion, window placement, physiological filtering, and aggregation. Path 2 detects peaks from raw ECG and tests the complete deployable reference pipeline.

Published labels are aligned using the documented 8-second windows starting every 2 seconds. RR intervals are assigned by their midpoint; intervals implying HR outside 35–220 bpm are excluded; at least four valid RR intervals are required; remaining beat-level HR values are averaged.

## Overall HR agreement with published labels

| route | matched_windows | mae_bpm | rmse_bpm | bias_bpm | correlation | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| provided R-peaks | 64697 | 0.3314 | 0.6358 | -0.0252 | 0.9996 | 1.0000 |
| detected R-peaks | 64697 | 1.7751 | 5.8517 | 1.2604 | 0.9695 | 1.0000 |

## ECG detector annotation agreement

| metric | macro_mean |
| --- | --- |
| peak_precision | 0.9855 |
| peak_recall | 0.9950 |
| peak_f1 | 0.9902 |
| peak_median_timing_error_ms | 1.1429 |

Peak matching is one-to-one with a fixed ±100 ms tolerance. The development detector uses a fixed 5–25 Hz ECG bandpass, derivative-energy integration over 120 ms, a median-plus-12-MAD threshold, a 250 ms candidate separation, and ±100 ms polarity-independent refinement. The same parameters are used for every subject.

These are development-set results, not an unbiased external estimate: PPG-DaLiA annotations were available while the detector design was being checked. The supplied-peak HR calculation is considered validated; the raw-ECG detector remains provisional because occasional false peaks produce a much larger RMSE than MAE, especially for S10.

## Interpretation boundary

Strong agreement on PPG-DaLiA validates this implementation for RespiBAN ECG in this dataset. It does not by itself prove validity for BIDMC or WESAD. WESAD uses the same chest-device family and should be checked next by waveform review and detector quality diagnostics; BIDMC requires separate validation against monitor HR because supplied R-peaks are unavailable.

Subject-level results are in `subject_metrics.csv`; every window and its quality counts are in `window_comparison.csv`.
