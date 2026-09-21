# RQ1 implementation plan

## Objective

Measure how PPG-based HR estimators degrade on unseen datasets and characterise
the degradation associated with population, device/body site, sampling rate,
recording environment, and motion/condition differences.

## Fixed experimental principles

1. No subject may occur in more than one split. BIDMC must be grouped by its 46
   MIMIC source patient IDs, not its 53 recording names.
2. All learned models receive the same primary PPG representation. Motion data
   is retained for analysis but is not a required model input.
3. Reference HR is derived from ECG through one common method wherever possible.
   Supplied HR is retained for validation, not silently mixed with derived HR.
4. Preprocessing parameters are fitted or selected without target-test data.
5. Every output is linked to dataset version, configuration, seed, and code.

## Work packages and gates

### WP1 — Dataset preparation and characterisation

- [x] Download and structurally audit all four datasets.
- [x] Create the recording manifest and leakage-safe subject identifiers.
- [x] Compare dataset domains, sensors, sampling rates, and labels.
- [x] Implement one loader per dataset behind a common interface.
- [x] Validate signal channels, lengths, timestamps, finite values, and alignment.
- [ ] Produce subject/recording duration and HR-distribution tables.
- [ ] Produce representative PPG/ECG plots for every dataset.

**Gate:** every manifest row loads into the common record schema, and validation
reports contain no unexplained failures.

### WP2 — Common preprocessing

- PPG filtering, polarity handling, normalization, and common resampling.
- ECG R-peak detection plus use of verified/provided R-peaks where available.
- ECG-derived HR on a documented time grid.
- Fixed window generation with explicit boundary and label rules.
- Signal-quality, missing-data, and physiologic-range rejection.
- Processed-data cache containing provenance and quality flags.

**Gate:** manually inspect samples from every dataset and quantify retained versus
rejected windows by dataset, subject, and condition.

### WP3 — Leakage-safe split definitions

- Within-dataset subject-independent train/validation/test splits.
- Four leave-one-dataset-out folds: train on three, test on the fourth.
- Optional pairwise train-one/test-one splits kept separate from the primary study.
- Fixed seeds and serialized subject lists.

**Gate:** automated tests prove zero subject overlap and zero held-out target use.

### WP4 — Conventional baseline

- FFT/Welch dominant-frequency HR estimator.
- Physiologic frequency bounds and deterministic peak-selection rules.
- Within-dataset and held-out-dataset evaluation.

**Gate:** predictions exist for every accepted test window and pass sanity checks.

### WP5 — Supervised baseline

- Published Deep PPG/PPG-only 1D-CNN-style supervised model.
- No new architecture is designed in RQ1.
- Identical input, target, training budget, and early-stopping policy across domains.
- Dataset-balanced sampling so large datasets do not dominate silently.

**Gate:** reproducible training runs and complete within-domain results for fixed seeds.

### WP6 — Evaluation

- MAE, RMSE, Pearson correlation, and percentage within ±5 bpm.
- Bland–Altman bias and 95% limits of agreement.
- Generalisation gap: unseen-domain MAE minus the matched within-domain reference.
- Bootstrap confidence intervals grouped by subject, not independent windows.

**Gate:** one tidy prediction table can reproduce every reported metric and plot.

### WP7 — Failure characterisation and reporting

- Per-subject and per-condition error distributions.
- Low-motion versus high-motion comparisons where labels support them.
- Clinical versus healthy/wearable comparisons without inventing BIDMC activities.
- Four primary leave-one-dataset-out result rows; pairwise 4×4 results only as an
  explicitly secondary experiment.
- Tables, figures, limitations, and presentation generated from saved results.

**Gate:** all claims trace to a table/figure and all tables/figures trace to predictions.

## Work deferred until RQ1 is complete

RQ2 will reuse the frozen RQ1 data, splits, 1D encoder, and leave-one-dataset-out
protocol. Self-supervised pretraining, CORAL/MMD, Transformer alternatives,
foundation representations, and the proposed device-agnostic representation are
not part of the initial RQ1 implementation. The method chosen for RQ2 must be
motivated by the failure pattern measured in RQ1.

RQ3 will use the best RQ2 model for ensembles, error detection, selective
prediction, conformal intervals, and comparison with signal-quality measures.

## Immediate next sprint

1. Define the common `PPGRecord` schema and loader contract.
2. Implement the BIDMC WFDB loader first and test all 53 recordings.
3. Implement PTT-PPG next because it also uses WFDB and supplies verified R-peaks.
4. Implement PPG-DaLiA and WESAD pickle loaders.
5. Run a complete loader validation report before filtering or windowing.

Do not begin model training until the WP1 gate is satisfied.

Validated results, interpretations, and candidate presentation figures are
tracked in `docs/presentation_results_log.md`. This log will be converted into
presentation slides after the relevant pipeline decisions are frozen.
