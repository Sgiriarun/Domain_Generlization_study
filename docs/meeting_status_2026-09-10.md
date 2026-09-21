# Supervisor meeting status — 10 September 2026

## Research goal

RQ1 measures how PPG heart-rate estimation changes across datasets, devices,
activities, environments, populations, and PTT sensor sites. ECG-derived HR is
the common reference. The central future experiment is train-three-datasets and
test the completely unseen fourth dataset.

## Completed and defensible

1. Four datasets audited and loaded through one canonical schema.
2. ECG/R-peak reference pipeline validated and frozen.
3. PPG resampled to 64 Hz, filtered at 0.5–4 Hz, and divided into 8-second
   windows with a 2-second step.
4. Subject-independent manifests and split roles frozen; no window-level subject
   leakage.
5. Distribution-shift analysis completed across dataset/device, environment,
   activity, stress, signal quality, HR support, and PTT sensor site.
6. Fixed FFT baseline completed on all four datasets.
7. Official TimePPG-Big repository audited and implemented as a clearly labelled
   PPG-only adaptation.
8. Five-fold PPG-DaLiA sensitivity experiment completed: subject-macro MAE
   5.97 +/- 2.71 bpm; pooled MAE 5.89 versus FFT 17.62 bpm.
9. Primary three-fold BIDMC completed: pooled MAE 1.92 bpm versus FFT 3.96.
10. Primary three-fold PTT-PPG completed: pooled MAE 4.69 bpm versus FFT 10.26.

## Important early findings

- Dataset identity remains predictable after HR matching and per-window
  normalization, showing persistent waveform/spectral domain signatures.
- The FFT estimator is much stronger on BIDMC than on wearable motion datasets.
- TimePPG substantially improves over FFT within completed familiar domains.
- BIDMC TimePPG performance is stable across folds (1.53–2.12 bpm MAE).
- PTT performance is less stable (2.25–8.10 bpm fold MAE) because subject `s2`
  has 44.27 bpm MAE.
- On PTT, walking is harder than sitting for both conventional and learned HR
  estimation.
- PPG-DaLiA stairs remains a major learned-model failure condition.

## Work to complete before the meeting if runtime permits

1. Allow WESAD folds 0–2 and PPG-DaLiA folds 0–2 to finish.
2. Aggregate all three-fold unseen-subject predictions for four datasets.
3. Verify every subject/window occurs exactly once in the combined test results.
4. Produce the within-dataset TimePPG-versus-FFT table and figures.
5. Inspect the PTT `s2` failure without automatically excluding it.
6. Prepare a short presentation/status narrative from generated reports.

## Work explicitly not complete

- Four leave-one-dataset-out TimePPG experiments.
- Generalisation-gap table.
- Cross-site TimePPG training/testing with all PTT channel pairs.
- Seed-sensitivity experiments beyond seed 17.
- BeliefPPG and PaPaGei comparisons.

These should be presented as planned work. Rushing LODO before the meeting risks
target leakage, incomplete balancing, or changing the frozen model after seeing
target performance.

## Recommended meeting message

> The common preprocessing and label pipeline is complete and audited. We have
> demonstrated measurable PPG distribution differences and established FFT and
> TimePPG familiar-domain baselines. TimePPG substantially improves over FFT,
> but performance remains activity- and subject-dependent. The current notable
> failure is PTT subject s2. The next central experiment is the frozen
> train-three/test-one LODO benchmark, which will quantify the actual unseen-
> dataset generalisation gap.

