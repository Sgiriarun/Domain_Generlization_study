# Phase 3 — Complete raw-record validation

## Result

Loaded **149/149** discovered recordings. The audit found **0 errors** and **11 warnings**.

An error means the WP1 loading gate is not satisfied. A warning identifies a real source-data characteristic that must be explained or handled later; Phase 3 does not alter it.

## Dataset summary

| dataset | records | loaded_records | subjects | total_ppg_hours | ppg_nonfinite_values | ecg_nonfinite_values | records_with_rpeaks | records_with_provided_hr | pass_records | warning_records | error_records | warnings | errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BIDMC | 53 | 53 | 46 | 7.067 | 0 | 0 | 0 | 53 | 52 | 1 | 0 | 1 | 0 |
| PPG-DaLiA | 15 | 15 | 15 | 35.969 | 0 | 0 | 15 | 15 | 7 | 8 | 0 | 10 | 0 |
| PTT-PPG | 66 | 66 | 22 | 9.010 | 0 | 0 | 66 | 0 | 66 | 0 | 0 | 0 | 0 |
| WESAD | 15 | 15 | 15 | 24.130 | 0 | 0 | 0 | 0 | 15 | 0 | 0 | 0 | 0 |

## Availability of reference information

`provided_hr` means a dataset-supplied HR series. It is retained for provenance and comparison, but it is not automatically the common RQ1 ECG-derived target.

| Dataset | PPG fs | ECG fs | PPG channels | R-peaks supplied | HR series supplied |
|---|---:|---:|---:|:---:|:---:|
| BIDMC | 125 Hz | 125 Hz | 1 | No | Yes |
| PPG-DaLiA | 64 Hz | 700 Hz | 1 | Yes | Yes |
| PTT-PPG | 500 Hz | 500 Hz | 6 | Yes | No |
| WESAD | 64 Hz | 700 Hz | 1 | No | No |

## Validation rules

- Dataset-specific PPG/ECG sampling rates and PPG channel names must match the documented source format.
- PPG and ECG durations may differ by no more than one sample period.
- Non-finite values are counted and reported, never repaired here.
- Supplied R-peaks must satisfy the schema; RR intervals implying HR outside 35–220 bpm are flagged for review, not deleted.
- BIDMC uses the source MIMIC patient as the leakage-control identity.

## Issues

| severity | check | count |
| --- | --- | --- |
| warning | provided_hr_nonfinite | 1 |
| warning | rpeak_rr_plausibility | 8 |
| warning | source_rpeak_duplicates | 2 |

See `validation_issues.csv` for record-level details.

## Scope boundary

This report validates loading and raw structure only. It does not filter, resample, derive ECG HR, create windows, or reject low-quality samples.
