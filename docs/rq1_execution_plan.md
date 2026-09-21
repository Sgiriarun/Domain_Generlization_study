# RQ1 execution plan aligned with the supervisor documents

## Research question

How large are the performance losses caused by dataset-, device-, activity-,
environment-, and sensor-site-related shifts in PPG-based heart-rate estimation?

The main contribution is a reproducible benchmark and failure characterisation,
not a new model. ECG supplies the reference HR and is never a model input.

## Completed foundation

1. Dataset audit, canonical schema, and four real-data loaders.
2. Structural validation of 149 recordings.
3. ECG/R-peak investigation and frozen ECG-derived HR policy.
4. Common continuous PPG preparation at 64 Hz and 0.5–4 Hz.
5. Descriptive domain analysis, including HR support, PCA, device identity,
   activity, stress, signal quality, environment, and PTT sensor site.
6. Frozen single-channel modelling manifest, secondary PTT site manifest, and
   deterministic subject-level splits.

## Frozen experimental decisions

- Input: PPG only; one channel for the primary benchmark.
- PTT primary channel: `pleth_1`, chosen as a fixed comparable channel, not
  claimed to be the best channel.
- Window: 8 seconds; step: 2 seconds; 512 samples at 64 Hz.
- Model-time input scaling: independent z-score within each window.
- Reference: ECG-derived mean beat-level HR using the frozen Phase 4 rules.
- Primary windows include detector-disagreement warnings; a sensitivity analysis
  will repeat evaluation without those flagged WESAD/BIDMC windows.
- Split unit: subject. No target subject or target statistic is available during
  source training or tuning.
- PTT sensor-site shift is a secondary controlled experiment using paired
  distal/proximal channels at identical times and with identical HR labels.

## Upcoming work

### Phase 8 — Frequency-domain baseline — completed

Implement `PPG -> Hann-windowed FFT -> dominant cardiac frequency -> HR`, using
the frozen 35–220 bpm search range. Freeze all estimator choices before testing.
Run subject-independent within-dataset evaluation first. Store predictions,
subject-level metrics, dataset metrics, Bland–Altman inputs, and figures.

### Phase 9 — TimePPG-Big PPG-only supervised baseline

Implement the published TimePPG-Big dilated TCN topology from the official
Q-PPG repository, with only the documented adaptations required for one-channel,
512-sample inputs and the common RQ1 protocol. Tune only using source
training/validation subjects. Record architecture, seed, checkpoint, training
history, and exact manifest/split identifiers.

Current status: pipeline verification and all five PPG-DaLiA subject folds for
seed 17 are complete (subject-macro MAE 5.97 bpm). To control computation, the
primary common within-dataset protocol is now three outer subject folds for all
four datasets; the PPG-DaLiA five-fold result is retained as sensitivity work.

### Phase 10 — Main leave-one-dataset-out benchmark

For each target dataset, train the CNN on the other three, tune on source
validation subjects, and evaluate once on every untouched target subject.
Report MAE, RMSE, Pearson correlation, percentage within ±5 bpm, Bland–Altman
bias/limits, and the generalisation gap relative to within-dataset performance.

### Phase 11 — Failure and domain analysis

Stratify predictions by dataset condition/activity, subject, HR range, and signal
quality. For PTT, compare within-site and cross-site performance for channel
pairs 1/4, 2/5, and 3/6. Treat subjects—not overlapping windows—as independent
units for confidence intervals and statistical inference.

### Phase 12 — Published comparators and reporting

Compare against BeliefPPG and, after a documented pretraining-data audit, one
available modern pretrained representation such as PaPaGei. These are additions
to, not replacements for, the two required baselines. Build the final transfer
table, reproducibility package, paper figures, and presentation from generated
logs.

## Immediate next action

Complete the frozen three-fold within-dataset runs for BIDMC, PTT-PPG, WESAD,
and PPG-DaLiA, aggregate unseen-subject results, and then begin the four main
leave-one-dataset-out experiments.
