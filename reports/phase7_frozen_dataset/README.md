# Phase 7: frozen modelling dataset and splits

## Decision aligned with the RQ1 documents

- Primary task: single-channel PPG to ECG-derived HR.
- Every input is an 8-second, 512-sample window at 64 Hz with a 2-second step.
- PTT-PPG uses `pleth_1` only in the primary four-dataset benchmark.
- `ptt_site_window_manifest.csv` preserves all six aligned PTT channels for the secondary site-shift analysis.
- Model input normalization is per-window z-scoring and must happen after slicing.
- The primary analysis retains detector-disagreement review windows. The manifest marks them so a documented sensitivity analysis can exclude them.
- No motion-quality threshold was invented after viewing target data; only structural failures are rejected.

## Primary dataset

| dataset | subjects | records | accepted_windows | detector_warning_windows |
| --- | --- | --- | --- | --- |
| PPG-DaLiA | 15 | 15 | 64697 | 0 |
| PTT-PPG | 22 | 66 | 15982 | 0 |
| WESAD | 15 | 15 | 43385 | 316 |
| BIDMC | 46 | 53 | 12561 | 890 |

## Leakage control

- `subject_splits.csv` assigns complete subjects, never windows, to five deterministic folds using seed 17.
- Within-dataset evaluation rotates the test fold; training-time validation must be selected only from the remaining subjects.
- For each leave-one-dataset-out experiment, every held-out-dataset subject is `target_test`.
- Source fold 0 is `source_validation`; all other source folds are `source_train`.
- Target subjects are not used for training, normalization fitting, early stopping, or hyperparameter selection.

## Files

- `main_window_manifest.csv`: frozen single-channel input references for RQ1.
- `ptt_site_window_manifest.csv`: six-channel paired-site secondary analysis.
- `subject_splits.csv`: reusable subject-level split assignments for RQ1–RQ3.
- `dataset_summary.csv` and `lodo_split_summary.csv`: audit tables.

The manifests reference immutable Phase 5 `.npy` recordings using `[start_sample_64hz, end_sample_64hz)` and `channel_index`; they do not duplicate overlapping signal windows.
