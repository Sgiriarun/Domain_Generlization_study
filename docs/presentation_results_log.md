# Presentation results log

This file records results that should later be converted into presentation
slides. Values must be copied from the linked generated reports, not retyped
from memory. Development results must remain labelled as development results.

The complete, presentation-ready interpretation of every Phase 6 figure is
preserved in `docs/phase6_presentation_interpretation.md`. Use that document for
slide narration, research claims, limitations, and the transition to Phase 7.

## Dataset preparation and validation

### Slide: Common dataset inventory

- 149 recordings across four datasets.
- BIDMC: 53 recordings from 46 patients.
- PPG-DaLiA: 15 subjects.
- PTT-PPG: 66 recordings from 22 subjects and three activities.
- WESAD: 15 subjects.
- PPG and ECG timelines aligned for every loaded record.
- No non-finite values in the primary PPG or ECG signals.
- Phase 3 outcome: 149/149 loaded, zero structural errors, 11 documented
  warnings across nine records.

Source: `reports/phase3/validation_report.md`.

## PPG-DaLiA ECG-derived HR validation

### Slide: Why supplied R-peaks were tested first

Show the two paths separately:

1. Supplied R-peaks → RR intervals → window HR. This isolates and validates the
   HR calculation.
2. Raw ECG → detected R-peaks → RR intervals → window HR. This evaluates the
   complete reference-generation pipeline.

Configuration:

- ECG: 700 Hz.
- Window: 8 seconds.
- Step: 2 seconds.
- RR interval assigned by its midpoint.
- Accept HR from 35 to 220 bpm.
- Require at least four valid RR intervals.
- Use arithmetic mean HR in each window.

### Slide: HR calculation validation

Using supplied PPG-DaLiA R-peaks over 64,697 windows:

- MAE: 0.33 bpm.
- RMSE: 0.64 bpm.
- Correlation: 0.9996.
- Window coverage: 100%.

Interpretation: the R-peak-to-window-HR calculation and documented window
alignment are validated on PPG-DaLiA.

Source: `reports/phase4a_ppg_dalia/validation_report.md`.

### Slide: Failure analysis of the development detector

- 88.5% of windows had error at or below 2 bpm.
- 6.1% of overlapping windows exceeded 10 bpm error.
- Main inspected failure: extra detections on non-R deflections, producing
  shortened RR intervals and sometimes approximately doubled HR.
- S10 contained episodic severe ECG clipping/distortion.
- Important caveat: overlapping windows are correlated; adjacent bad windows
  may represent one short failure episode.

Use selected ECG figures from:
`reports/phase4b_dalia_errors/diagnostic_plots/`.

Source: `reports/phase4b_dalia_errors/analysis_report.md`.

### Slide: XQRS comparison

| Metric | Development detector | WFDB XQRS |
|---|---:|---:|
| Peak F1 | 99.02% | 99.23% |
| HR MAE | 1.78 bpm | 0.90 bpm |
| HR RMSE | 5.85 bpm | 3.07 bpm |
| HR correlation | 0.9695 | 0.9911 |
| Windows with error >10 bpm | 3,938 | 1,422 |
| Windows with error >20 bpm | 1,405 | 336 |
| Coverage | 100% | 100% |

Interim interpretation: XQRS became the leading candidate because it approximately
halves MAE and substantially reduces large HR errors without reducing coverage.
This decision was later superseded by the three-detector comparison below.

Status: **development validation only**. PPG-DaLiA annotations were available
during method development. Do not describe this as external validation.

Source: `reports/phase4c_xqrs_dalia/benchmark_report.md`.

## Decisions currently supported

- Use supplied R-peaks for final PPG-DaLiA reference construction.
- Use manually verified supplied R-peaks for final PTT-PPG reference
  construction.
- Use NeuroKit2 Emrich 2023/FastNVG as the selected detector for WESAD and
  BIDMC, subject to their dataset-specific quality checks.
- Keep XQRS as the established comparator and SleepECG as a sensitivity method.
- Foundation ECG models are deferred because
  they add model-specific preprocessing and are not needed for sample-accurate
  reference peak generation at this stage.

## Future presentation work

- Convert this log into a concise sequence of academic slides after the current
  pipeline decisions are frozen.
- Rebuild all tables directly from generated CSV files.
- Include representative clean and failed ECG windows.
- Clearly distinguish supplied-peak results, development-detector results, and
  external validation results.
- Present the complete three-detector comparison rather than the earlier interim
  XQRS-only selection.

## Phase 7 frozen modelling dataset

### Slide/document note: benchmark inputs and leakage control

- The primary single-channel manifest contains 136,625 accepted windows:
  64,697 PPG-DaLiA, 15,982 PTT-PPG, 43,385 WESAD, and 12,561 BIDMC.
- Every window references exactly 512 samples at 64 Hz and uses an ECG-derived
  HR label from the same 8-second interval.
- PTT-PPG uses `pleth_1` in the primary benchmark. This is a fixed comparable
  channel choice, not a claim that it is the best PTT channel.
- A separate PTT manifest contains 95,892 channel-windows: six aligned channels
  for each of the 15,982 time windows. This supports controlled distal/proximal
  site-shift analysis without changing the primary experiment.
- Complete subjects are assigned to deterministic five-fold splits. In every
  leave-one-dataset-out experiment, all subjects from the target dataset are
  test-only; source fold 0 is validation and the remaining source folds are
  training.
- WESAD's 316 and BIDMC's 890 detector-disagreement windows remain in the
  primary analysis and are explicitly marked for a sensitivity analysis.

Source: `reports/phase7_frozen_dataset/README.md`.

## Phase 8 frequency-domain baseline

### Slide/document note: non-AI reference performance

- One frozen estimator was applied unchanged to all 136,625 primary windows:
  mean removal, Hann taper, zero-padded FFT, strongest peak in the 35–220 bpm
  band, and `HR = frequency × 60`.
- BIDMC was easiest: MAE 3.96 bpm and 89.31% of windows within ±5 bpm.
- PTT-PPG: MAE 10.26 bpm and 75.58% within ±5 bpm.
- WESAD: MAE 12.59 bpm and 57.22% within ±5 bpm.
- PPG-DaLiA was hardest: MAE 17.62 bpm and 52.82% within ±5 bpm.
- The result is consistent with low-motion clinical PPG being easier for a
  dominant-frequency estimator than wearable PPG containing daily-life motion.
  It is an observed association, not proof that motion alone caused the gap.
- On PTT-PPG, distal channels outperformed their corresponding proximal channel
  in all three documented pairs. MAE was 10.26 versus 23.72 bpm for pair 1/4,
  8.85 versus 22.45 bpm for pair 2/5, and 5.64 versus 14.30 bpm for pair 3/6.
- Do not attach red/infrared names to pairs 1/4 and 2/5 until the conflicting
  v1.0 channel documentation is resolved. Site membership is consistent.
- Since this baseline learns no parameters, it has no LODO training operation.
  It provides a fixed non-AI comparison for the learned model.
- Clear explanation for narration: the method sees one PPG window, selects its
  strongest allowed frequency, and outputs HR. ECG is revealed only afterwards
  to measure the error; it never guides peak selection. Therefore “evaluation
  on the same dataset” does not mean training and testing on the same data.
- PTT condition results support a motion-related failure pattern: sitting MAE
  was 2.68 bpm, walking 18.12 bpm, and running 10.02 bpm.
- WESAD condition results were also heterogeneous: baseline 7.78 bpm, stress
  28.99 bpm, meditation 4.98 bpm, and amusement 6.14 bpm.
- All four datasets showed negative mean bias. This means the dominant-frequency
  rule more often underestimated than overestimated ECG-derived HR, although
  the mechanism cannot be identified from aggregate metrics alone.
- Use careful wording: the condition associations are evidence of estimator
  fragility under heterogeneous recordings, not causal proof that activity or
  stress alone produced every error.

Source: `reports/phase8_spectral_baseline/README.md`.

## Phase 9 model provenance decision

- The primary supervised model will use the officially released TimePPG-Big
  temporal convolutional architecture rather than an invented residual TCN.
- Reference: Burrello et al., DOI `10.1145/3487910`; official repository
  `eml-eda/q-ppg`; audited commit
  `ddf3866da6d5f9dda4da7d7884b4f1f3b809a6ba` (Apache-2.0).
- The official reported setup is multimodal: one PPG plus three accelerometer
  axes, 32 Hz, and PPG-DaLiA leave-one-subject-out evaluation.
- Our model must be described as a reference-faithful PPG-only adaptation. It
  retains the official block design and training standard but changes input to
  one PPG channel, uses the frozen 64-Hz windows, and follows four-dataset LODO.
- Do not claim an exact TimePPG reproduction or directly compare our adapted
  score with the paper's multimodal score as if the protocols were identical.

Source: `docs/model_references/timeppg_reference_audit.md`.

## Phase 9A TimePPG implementation check

- The reference-faithful TimePPG-Big PPG-only implementation passed its initial
  real-data pipeline check on PPG-DaLiA.
- Input batch shape was `(126, 1, 512)` and output shape was `(126,)`, meaning
  one HR prediction was produced for every single-channel 8-second window.
- Training subjects were S2, S3, S4, S5, S6, S7, S9, S10, S11, S12, S14, and
  S15. Validation subjects were S1, S8, and S13. Subject overlap was zero.
- The adapted model contains 254,784 trainable parameters.
- Manifest slicing, per-window z-normalisation, forward pass, log-cosh loss,
  backward pass, Adam update, validation inference, finite-output check, and
  checkpoint writing all passed.
- Five automated model/data tests passed.
- The smoke run used only 252 training and 126 validation windows for three
  epochs. Its loss and MAE are plumbing diagnostics and must never be reported
  as research performance or compared with Phase 8.

Source: `reports/phase9_timeppg/pipeline_check/README.md`.

## Phase 9B preliminary within-dataset result

### PPG-DaLiA fold 0, seed 17 — one-fold result only

- Proper subject separation: 9 training subjects, 3 validation subjects, and 3
  untouched test subjects (S1, S8, S13); no subject overlap.
- Early stopping selected epoch 25 using validation log-cosh loss. Training
  stopped at epoch 45 after 20 epochs without improvement, preventing the later
  reduction in training error from being mistaken for better generalisation.
- On 13,205 test windows, TimePPG achieved MAE 5.33 bpm, RMSE 10.23 bpm,
  Pearson r 0.8639, 72.35% within +/-5 bpm, and bias +2.24 bpm.
- On exactly the same test subjects/windows, FFT achieved MAE 12.21 bpm, RMSE
  20.36 bpm, Pearson r 0.5901, 54.43% within +/-5 bpm, and bias -6.63 bpm.
- TimePPG reduced same-window MAE by 6.88 bpm or 56.3% relative to FFT.
- Subject variation remains substantial: TimePPG MAE was 4.92 bpm for S1, 9.08
  bpm for S8, and 2.44 bpm for S13. This supports reporting subject-level
  results rather than only pooled windows.
- Activity variation also remains: working 2.00, baseline 2.92, driving 3.39,
  cycling 3.77, lunch 4.61, table soccer 7.79, walking 7.91, and stairs 14.35
  bpm. `no_activity`/transition windows had 6.81 bpm MAE.
- Careful claim: on this fold, learned temporal features strongly outperform a
  dominant-frequency rule, but stairs and some subjects remain difficult.
- This is not final within-dataset performance, a confidence interval, or
  evidence of LODO generalisation. All folds and the four held-out-dataset tests
  are still required.

Source: `reports/phase9_timeppg/within_dataset/ppg_dalia/test_fold_0/seed_17/evaluation/README.md`.

## Phase 9B complete PPG-DaLiA five-fold result — seed 17

- All five subject-wise folds completed successfully. Every one of the 15
  subjects appears in exactly one untouched test fold, covering all 64,697
  frozen PPG-DaLiA windows exactly once.
- Fold MAEs were 5.33, 7.34, 4.83, 7.21, and 5.08 bpm; fold mean was 5.96 +/-
  1.22 bpm.
- The primary subject-level result is macro MAE **5.97 +/- 2.71 bpm** across 15
  subjects. The subject-bootstrap 95% CI is **4.67–7.33 bpm**.
- Pooled-window performance was MAE 5.89 bpm, RMSE 11.80 bpm, Pearson r 0.8581,
  72.67% within +/-5 bpm, and bias -0.74 bpm.
- On the same 64,697 windows, FFT MAE was 17.62 bpm. TimePPG reduced pooled MAE
  by 11.73 bpm or 66.6%.
- Subject variation remains important: S5 was hardest at 11.09 bpm MAE,
  followed by S9 at 10.20 and S8 at 9.08 bpm. Aggregate window metrics hide
  these difficult individuals.
- Stairs was the hardest activity at 17.92 bpm MAE, followed by walking at 9.52
  bpm. Baseline was 2.71 and working was 2.86 bpm.
- Correct claim: the adapted TimePPG model generalises to unseen subjects within
  PPG-DaLiA and substantially outperforms FFT. This does not yet demonstrate
  transfer to an unseen dataset or device.
- Limitation: this complete five-fold result uses one random seed. Seed
  sensitivity and cross-dataset LODO experiments remain outstanding.

Source: `reports/phase9_timeppg/within_dataset/ppg_dalia/aggregate_seed_17/README.md`.

## Within-dataset protocol amendment: three folds

- To control total computation, the primary within-dataset experiment was
  reduced to three outer subject folds for all four datasets, seed 17.
- The roles are frozen before training in
  `reports/phase9_timeppg/three_fold_subject_roles.csv`.
- A naive three-way fold rotation would leave only one-third of subjects for
  training. Instead, approximately one-third are outer-test subjects, about 20%
  are source-only validation subjects, and the remaining subjects train the
  model. Roles never overlap.
- Every subject becomes test data exactly once across the three outer folds.
- The already completed five-fold PPG-DaLiA result is retained as a stronger
  sensitivity analysis, not discarded or mixed silently with the new primary
  three-fold aggregate.
- This protocol reduction must be disclosed in methods and limitations; it is a
  computational-budget decision rather than an accuracy-driven choice.

Source: `reports/phase9_timeppg/three_fold_role_summary.csv`.

### Runtime provenance note

- During the three-fold run on macOS, active BIDMC/PTT epochs generally took
  about 10–16 seconds, but the histories contained repeated approximately
  923-second gaps. These gaps indicate system sleep/process suspension rather
  than model computation and explain the overnight wall-clock duration.
- Future runs use macOS `caffeinate -i` when available and retain memory maps
  for up to 128 recording files instead of four, reducing sleep interruptions
  and repeated file opening during shuffled training.
- These runtime changes do not alter input samples, model parameters, loss,
  optimiser, split roles, checkpoint selection, or predictions.

## Interim three-fold results while training continues

- BIDMC and PTT-PPG completed all three within-dataset subject folds for seed
  17. WESAD and the primary three-fold PPG-DaLiA runs were still running when
  this interim note was created.
- BIDMC: pooled MAE 1.92 bpm, subject-macro MAE 1.83 bpm, RMSE 4.34 bpm,
  Pearson r 0.9457, and 90.57% within +/-5 bpm. Same-window FFT MAE was 3.96
  bpm, giving a 51.6% TimePPG reduction.
- PTT-PPG: pooled and subject-macro MAE both approximately 4.69 bpm, RMSE 11.51
  bpm, Pearson r 0.6241, and 81.28% within +/-5 bpm. Same-window FFT MAE was
  10.26 bpm, giving a 54.3% TimePPG reduction.
- PTT folds are unstable (2.25–8.10 bpm MAE) because subject `s2` has 44.27 bpm
  MAE. This is a research finding requiring investigation, not a subject to
  silently exclude.
- PTT activity MAE: sitting 3.21, running 4.71, walking 6.17 bpm.
- Correct interim conclusion: learned temporal features improve over FFT in
  both completed familiar domains, but subject-specific failures remain severe
  in PTT. No LODO conclusion is yet permitted.

Source: `reports/phase9_timeppg/interim_completed_datasets.md`.

## Generated RQ1 progress presentation

- A 20-slide evidence-linked progress deck was generated at
  `presentation/output/rq1_progress_review.pptx`.
- It includes background, research gap, datasets, common pipeline, HR-label
  validation, frozen manifests, distribution-shift evidence, FFT method and
  results, PTT site evidence, TimePPG provenance, evaluation protocol, interim
  learned-model results, failures, current conclusion and remaining work.
- Results are explicitly marked as completed, interim/sensitivity, or pending.
  No unseen-dataset LODO result is claimed before those experiments exist.
- Speaker notes are embedded in the PPTX and exported separately to
  `presentation/output/rq1_progress_speaker_script.md`.
- The deck passed ZIP integrity, 20-slide count, embedded-notes and slide-bound
  checks.

## Phase 9A TimePPG reference audit

- The supervised baseline is anchored to the official TimePPG/Q-PPG repository,
  not an invented generic CNN.
- Official TimePPG uses an 8-second, 32 Hz input with one PPG and three
  accelerometer axes. Its code explicitly supports a PPG-only mode, but the
  released float model hard-codes four input channels.
- RQ1 necessarily uses a documented PPG-only adaptation because accelerometer
  data are not common to all four datasets.
- The published convolutional topology, TimePPG-Big channel widths, dilation
  pattern, ReLU6, batch normalization, LogCosh loss, Adam learning rate 0.001,
  batch size 128, and early-stopping rule are retained.
- Required differences are 1-channel input, 64 Hz/512 samples, canonical ECG HR,
  frozen subject/LODO splits, and no target fine-tuning.
- The adapted model has 254,784 parameters and passed forward, backward, shape,
  finite-gradient, loss, window, metric, and spectral tests.
- Presentation wording must be “TimePPG-Big PPG-only adaptation”, not “exact
  reproduction” and not the paper's headline TimePPG result.

Source: `reports/phase9_timeppg/reference_audit.md`.

### Viva/presentation explanation: how FFT gives predicted HR

Use this explanation:

> A PPG pulse repeats approximately once per heartbeat. In the time domain we
> see repeating pulse waves. The FFT represents the same 8-second signal using
> frequency components and their strengths. We search only within the accepted
> 35–220 bpm range, which is approximately 0.583–3.667 Hz. The frequency with
> the largest spectral power in this range is selected as the pulse frequency.
> Predicted heart rate is that frequency in hertz multiplied by 60.

Concrete example:

- If the strongest spectral peak is at 1.5 Hz, the PPG pattern repeats 1.5
  times each second.
- `predicted HR = 1.5 × 60 = 90 bpm`.
- If ECG-derived HR for the same eight seconds is 94 bpm, the absolute error is
  `|90 − 94| = 4 bpm`.

Exact algorithm used:

1. Take the 512 samples belonging to one 8-second PPG window at 64 Hz.
2. Subtract the window mean to remove its remaining constant/DC level.
3. Multiply by a Hann taper to reduce spectral leakage caused by cutting a
   continuous recording at the window boundaries.
4. Calculate a 4096-point real FFT. The window still contains only 512 measured
   samples; zero-padding makes the frequency grid smoother but does not create
   new physiological information.
5. Calculate spectral power as the squared FFT magnitude.
6. Ignore frequencies outside 35–220 bpm (0.583–3.667 Hz).
7. Choose the in-range frequency with maximum power and multiply it by 60.

Important distinction:

- PPG alone determines the predicted frequency.
- ECG is not supplied to the estimator. ECG-derived HR is used only after the
  prediction to calculate error and agreement metrics.
- No parameters are learned, so this baseline has no training or validation
  dataset and no LODO training operation.

Limitation to state:

> The strongest in-band frequency is not guaranteed to be heart rate. Motion,
> a PPG harmonic, or noise may have more spectral power than the fundamental
> cardiac component. This baseline shows how far a fixed signal-processing
> method can work before using a learned model.

## Frozen Phase 4 HR-label construction

### Slide/document note: final reference policy

- PPG-DaLiA and PTT-PPG use supplied R-peaks.
- WESAD and BIDMC use NeuroKit2 Emrich 2023/FastNVG ECG detections.
- All datasets use 8-second windows, 2-second steps, RR-midpoint assignment,
  35–220 bpm interval limits, at least four valid RR intervals, and arithmetic
  mean beat-level HR.
- Detector disagreement above 10 bpm is a review flag, not an automatic
  exclusion. This distinction is required because the BIDMC investigation
  showed that detector/monitor disagreement does not by itself identify which
  source is wrong.
- The canonical output contains 136,625 windows: 64,697 PPG-DaLiA, 15,982
  PTT-PPG, 43,385 WESAD, and 12,561 BIDMC.
- All 136,625 windows meet the primary minimum label rule. WESAD has 316 and
  BIDMC has 890 detector-disagreement review flags. These windows remain
  available for transparent sensitivity analysis.

Source: `reports/phase4k_canonical_hr/README.md`.

## Phase 5 common PPG preparation

### Slide/document note: making PPG inputs comparable

- Every continuous PPG recording is resampled to 64 Hz with polyphase
  anti-alias filtering and then filtered at 0.5–4 Hz using a fourth-order,
  zero-phase Butterworth filter.
- Each frozen HR label maps to exactly 512 PPG samples: 8 seconds × 64 Hz.
- Filtering is performed before window extraction to avoid creating a filter
  boundary at every overlapping window.
- All six PTT-PPG channels are preserved. Channel selection remains a documented
  experiment rather than a hidden preprocessing decision.
- Amplitude is not normalized during Phase 5 because device/amplitude variation
  is relevant to the distribution-shift analysis. Any model-time normalization
  will be compared explicitly later.
- Descriptive PPG quality metrics are stored, but motion-based rejection is not
  imposed without empirical threshold analysis.

Source: `reports/phase5_ppg/README.md`.

### Slide/document note: why the PPG band is 0.5–4 Hz

- Convert frequency to HR using `bpm = Hz × 60`: 0.5–4 Hz is approximately
  30–240 bpm, covering the accepted 35–220 bpm label range with margin.
- The lower edge reduces DC and slow baseline drift; the upper edge reduces
  faster noise outside the main pulse-rate band.
- Band-pass filtering is common in PPG HR estimation, although the exact band is
  task-dependent rather than universal. Published work includes 0.4–5,
  0.5–2.5, 0.5–4, and 0.5–5 Hz choices.
- The supervisor specified 0.5–4 Hz for the common pipeline. Applying it
  unchanged to every dataset prevents target-specific filter tuning.
- Limitation to state clearly: an ordinary band-pass cannot remove motion whose
  frequency overlaps HR, and this band is intended for HR estimation rather
  than detailed PPG morphology analysis.

Literature links and full reasoning are stored in
`reports/phase5_ppg/README.md`.

## XQRS external validation on PTT-PPG

### Slide: Cross-device detector validation

The XQRS configuration selected during PPG-DaLiA development was applied
unchanged to all 66 PTT-PPG recordings (22 subjects × sit/walk/run). PTT-PPG
uses a different ECG device and 500 Hz sampling, with manually verified peaks.

- 46,405 reference peaks and 46,539 detected peaks.
- Micro peak precision: 99.46%.
- Micro peak recall: 99.75%.
- Micro peak F1: 99.60%.
- HR MAE: 0.35 bpm.
- HR RMSE: 2.51 bpm.
- HR correlation: 0.9855.
- 15,982/15,982 reference windows received detected-peak HR.

Activity HR MAE:

- Sitting: 0.17 bpm.
- Running: 0.43 bpm.
- Walking: 0.44 bpm.

Important limitation: errors are concentrated in a few records. `s12_run`
has 6.95 bpm MAE, `s12_walk` 5.39 bpm, `s13_sit` 3.16 bpm, and `s1_walk`
2.35 bpm. XQRS was therefore supported as an interim candidate, but automatic
ECG/beat quality flags are still required for datasets without annotations.

Source: `reports/phase4d_xqrs_ptt/validation_report.md`.

### Historical note: why XQRS was the interim candidate

XQRS is an adaptive QRS-complex detector provided by the open-source WFDB
toolkit. It bandpass-filters ECG, enhances QRS-shaped energy, learns signal and
noise levels, applies an adaptive threshold, rejects likely T waves, enforces a
refractory period, and searches backwards when a beat may have been missed. Its
output is the ECG sample index of each detected QRS/R-peak.

XQRS was the best method tested at that stage, not a claim that it was
universally the best ECG detector. It was provisionally selected because:

1. It substantially outperformed our development detector on PPG-DaLiA HR
   errors while retaining all windows.
2. It was then applied unchanged to the independent PTT-PPG device/domain and
   achieved 99.60% peak F1 and 0.35 bpm HR MAE.
3. It works at all project ECG sampling rates (125, 500, and 700 Hz).
4. It requires no target-dataset labels or model training.
5. It is available through the already-versioned WFDB dependency and is easier
   to reproduce and audit than an externally trained AI model.

Limitations that must remain visible:

- A few PTT-PPG records have concentrated errors despite strong aggregate
  performance.
- PPG-DaLiA and PTT-PPG supplied annotations remain preferable to detected
  peaks for final reference construction.
- WESAD has no supplied peaks, so XQRS quality must be assessed indirectly.
- BIDMC must be checked against its monitor HR and through ECG-quality review.
- This interim decision was superseded after testing Emrich 2023 and SleepECG.

## Final three-detector comparison

### Development comparison on PPG-DaLiA

| Detector | Peak F1 | HR MAE | HR RMSE | Error >20 bpm | Coverage |
|---|---:|---:|---:|---:|---:|
| Development | 99.02% | 1.78 bpm | 5.85 bpm | 1,405 | 100% |
| XQRS | 99.23% | 0.90 bpm | 3.07 bpm | 336 | 100% |
| Emrich 2023 | 99.47% | 0.67 bpm | 2.16 bpm | 134 | 99.9985% |
| SleepECG | 98.99% | 1.04 bpm | 3.41 bpm | 393 | 99.9969% |

### External comparison on PTT-PPG

| Detector | Peak F1 | HR MAE | HR RMSE | Error >20 bpm | Coverage |
|---|---:|---:|---:|---:|---:|
| XQRS | 99.60% | 0.35 bpm | 2.51 bpm | 54 | 100% |
| Emrich 2023 | 99.89% | 0.08 bpm | 0.85 bpm | 7 | 100% |
| SleepECG | 99.63% | 0.35 bpm | 2.68 bpm | 61 | 100% |

### Detector decision

NeuroKit2 Emrich 2023/FastNVG is selected because it is best on both the
PPG-DaLiA development comparison and the unchanged PTT-PPG external test. It
has the highest peak F1, lowest HR MAE/RMSE, lowest bias, and fewest large-error
windows. This is evidence across two devices and sampling rates, not a claim of
universal superiority.

Sources:

- `reports/phase4e_dalia_detector_comparison/benchmark_report.md`
- `reports/phase4f_ptt_detector_comparison/validation_report.md`

## WESAD ECG-derived HR generation

### Slide/document note: applying the selected detector without annotations

Frozen Emrich 2023/FastNVG was applied to all 15 WESAD subjects:

- 43,385 complete 8-second windows at a 2-second step.
- HR available for 43,385/43,385 windows.
- Median Emrich–XQRS HR difference: 0.015 bpm.
- Median Emrich–SleepECG HR difference: 0.015 bpm.
- 431 windows (0.99%) disagree with at least one comparator by more than 5 bpm.
- 316 windows (0.73%) disagree by more than 10 bpm.
- S2 accounts for 267 of the >10 bpm disagreement windows.

Critical wording: WESAD has no supplied R-peaks or HR, so detector agreement is
a quality diagnostic and **not an accuracy measurement**. Emrich HR remains the
selected candidate. Disagreement flags are preserved for later visual review
and sensitivity analysis; they are not automatically rejected.

The stress condition has a higher median candidate HR than baseline in this
dataset, but this is a descriptive observation and not a causal or inferential
result.

Source: `reports/phase4g_wesad_hr/validation_report.md`.

### Slide/document note: WESAD S2 disagreement review

- S2 accounts for 267/316 WESAD windows with detector disagreement above 10 bpm.
- 250 of those S2 windows have source condition ID 0 (`not_defined`).
- Ten separated episodes were plotted to avoid presenting overlapping windows
  as independent failures.
- Inspected ECG contains distorted, multi-deflection morphology; detectors
  select different deflections and consequently produce different HR.
- Without manual annotations, no detector can be declared correct in these
  episodes. Preserve them as quality flags and include a sensitivity analysis.

Source: `reports/phase4h_wesad_s2_review/review.md`.

## BIDMC ECG-derived HR validation

Frozen detectors were applied to all 53 BIDMC lead-II ECG records and compared
with the mean finite bedside-monitor HR over the same 8-second windows.

| Detector | MAE | RMSE | Bias | Correlation | Error >20 bpm |
|---|---:|---:|---:|---:|---:|
| Emrich 2023 | 1.62 bpm | 3.67 bpm | +0.44 bpm | 0.9622 | 67 |
| XQRS | 3.49 bpm | 10.32 bpm | +2.64 bpm | 0.7496 | 533 |
| SleepECG | 1.73 bpm | 4.03 bpm | +0.70 bpm | 0.9553 | 94 |

Emrich 2023 is again the strongest tested detector. The record-level median
Emrich MAE is 0.85 bpm, but `bidmc41` is a major outlier at 13.13 bpm MAE.
Monitor HR is an independent clinical comparison, not R-peak ground truth;
monitor smoothing, latency, and proprietary processing may contribute.

Source: `reports/phase4i_bidmc_hr/validation_report.md`.

### Slide/document note: difficult BIDMC records

- `bidmc41`: all detectors agree near 109 bpm on visible ECG complexes while
  monitor HR is near 54 bpm, approximately a 2:1 discrepancy. Treat as
  monitor-versus-ECG disagreement, not an isolated Emrich failure.
- `bidmc45`: all detectors agree on a slow rhythm; monitor HR changes sharply
  around the window, so smoothing/latency may contribute.
- `bidmc26`: detectors disagree on irregular morphology and the affected windows
  require an ECG-reference quality flag.
- Testing descriptive monitor shifts from -10 to +10 seconds did not resolve
  `bidmc41`; no global monitor offset is adopted.
- Wording must remain non-diagnostic: ectopy or pulse deficit may be plausible,
  but this signal review does not establish a clinical diagnosis.

Source: `reports/phase4j_bidmc_review/review.md`.

### Future document and presentation production rule

- Every numerical table must be generated from its report CSV.
- Every figure must retain the script and input file that produced it.
- Slides should label results as source-annotation validation, development
  validation, external validation, or application without peak annotations.
- The final written methods section must include detector version, parameters,
  peak-matching tolerance, window rule, HR range, minimum-RR rule, and failure
  handling.
- Maintain both a detailed research document and a shorter presentation; do not
  make slides the only record of methodological decisions.

## Phase 6 distribution-shift analysis

### Slide: analysis safeguards

- Primary comparison: one channel per dataset; PTT-PPG uses the predeclared
  first source channel (`pleth_1`). All six PTT channels remain in a secondary
  channel/activity analysis.
- Use every fourth 2-second-step window, giving non-overlapping 8-second
  observations.
- Cap sampling at 200 windows per subject and balance to 3,000 windows per
  dataset (12,000 total).
- Match all four datasets within 10-bpm HR bins before conditional waveform
  comparison; matched subset: 6,384 windows.
- Dataset-classifier validation is five-fold and subject-grouped, so no subject
  occurs in both training and test partitions.

### Slide: measured dataset shift

| Representation | Dataset classification accuracy | Macro F1 |
|---|---:|---:|
| Raw-scale features | 76.95% | 76.99% |
| Per-window z-normalized features | 64.64% | 64.52% |
| Chance level | 25.00% | — |

Interpretation: amplitude normalization removes some dataset/device information,
but substantial shape and spectral differences remain even after HR matching.
This is evidence of class-conditional waveform shift, `P(X|Y)`, not direct proof
that `P(Y|X)` has changed or that an HR model will fail.

### Slide: HR support and pairwise findings

- Balanced-sample HR medians: PPG-DaLiA 85.16, PTT-PPG 85.48, WESAD 75.52,
  and BIDMC 88.95 bpm.
- PPG-DaLiA has the broadest sampled high-HR tail (95th percentile 134.88 bpm;
  maximum 183.15 bpm). BIDMC is narrower (95th percentile 114.84 bpm;
  maximum 129.11 bpm).
- Based on mean standardized Wasserstein distance across normalized features,
  PPG-DaLiA/WESAD is the closest pair (0.194), consistent with their shared
  Empatica E4 wrist-device family.
- WESAD/BIDMC is the most separated pair (1.014), showing a strong
  wearable-laboratory versus clinical-sensor domain difference.
- PTT-PPG channel spectral distributions differ, and sitting generally has
  stronger spectral concentration than walking/running. Treat this as a
  descriptive motion/channel observation, not a universal signal-quality law.

Sources:

- `reports/phase6_distribution_shift/README.md`
- `reports/phase6_distribution_shift/dataset_classifier_folds.csv`
- `reports/phase6_distribution_shift/pairwise_shape_distances.csv`
- `reports/phase6_distribution_shift/ptt_channel_activity_summary.csv`

### Recommended Phase 6 presentation figures

- `figures/heterogeneous_domain_dashboard.png`: one-slide AI-engineering view
  combining HR-support shift, device/amplitude shift, spectral-complexity shift,
  HR-matched normalized PCA, dataset-classification accuracy, and pairwise
  normalized-shape distance.
- `figures/same_hr_different_domains.png`: controlled example showing one
  representative 80–90 bpm window from every dataset after z-normalization,
  together with its normalized spectrum. Use this to explain that equal HR does
  not imply equal PPG appearance across heterogeneous devices and settings.
- Exact provenance for the four plotted examples is stored in
  `reports/phase6_distribution_shift/same_hr_figure_examples.csv`.
- `figures/normalized_feature_pca_by_device.png`: device-focused PCA. Colour
  represents Empatica E4, MAX30101, or clinical pulse oximeter; marker shape
  distinguishes datasets. PPG-DaLiA and WESAD intentionally share the Empatica
  colour so within-device environmental variation remains visible.

Presentation wording: these figures demonstrate domain-identifiable PPG
structure after controlling approximately for HR. They motivate cross-dataset
testing; they do not themselves prove cross-dataset HR prediction failure.

### Slide: sensor-placement shift within PTT-PPG

- Use PTT-PPG only; do not mix datasets or devices for this figure.
- Distal site: `pleth_1–3`, measured at the distal phalanx of the left index
  finger on the palmar side.
- Proximal site: `pleth_4–6`, measured at the proximal phalanx of the same finger
  on the palmar side.
- In `figures/ptt_sensor_site_pca.png`, colour represents placement and marker
  pairs channels 1/4, 2/5, and 3/6 across sites. The centroid panel connects each
  same-sensor-channel pair, reducing channel/site interpretation confusion.
- Plain-language figure labels: blue is the sensor nearer the fingertip
  (`pleth_1–3`); orange is the sensor nearer the base of the same finger
  (`pleth_4–6`). Pair A compares `pleth_1 ↔ pleth_4`, Pair B compares
  `pleth_2 ↔ pleth_5`, and Pair C compares `pleth_3 ↔ pleth_6`.
- Source mapping: `datasets/raw/ptt_ppg/README.txt`, detailed channel descriptions
  at lines 116–127.
- Provenance caveat: the README hardware overview and detailed channel list swap
  the red/infrared names for pairs 1/4 and 2/5. The distal/proximal mapping is
  consistent, so the site analysis uses neutral paired-channel labels until the
  wavelength metadata is resolved from an authoritative source.

### Slides: separate within-domain PCA views

- PPG-DaLiA daily-life activity PCA: baseline, stairs, table soccer, cycling,
  driving, lunch, walking, working, and no-activity/transition periods.
- PTT-PPG activity PCA: sitting, walking, and running, using `pleth_1` only.
- WESAD condition PCA: baseline, stress, amusement, and meditation; undefined
  transition labels excluded.
- Environment PCA: daily life, controlled exercise, laboratory protocol, and
  clinical ICU. State clearly that each environment corresponds to a different
  dataset/device/population, so the factors are confounded.
- Signal-quality PCA: within-dataset low/medium/high spectral-concentration
  tertiles. The features defining those tertiles are excluded from the PCA to
  reduce circularity.
- Activity and stress PCA samples are balanced within 10-bpm HR bins before
  plotting, so visible differences are less dominated by different HR ranges.
- Main visual result: activity and stress groups overlap strongly after HR
  control, while environment/device and spectral-quality domains show stronger
  structure. PCA remains descriptive, not proof of causality or model failure.
- Never call the spectral-concentration groups verified motion artifacts. They
  are signal-regularity/quality proxies unless accelerometer or manual evidence
  is added.

Figures and source rows:

- `reports/phase6_distribution_shift/figures/dalia_activity_pca.png`
- `reports/phase6_distribution_shift/figures/ptt_activity_pca.png`
- `reports/phase6_distribution_shift/figures/wesad_stress_condition_pca.png`
- `reports/phase6_distribution_shift/figures/environment_domain_pca.png`
- `reports/phase6_distribution_shift/figures/signal_quality_domain_pca.png`
- `reports/phase6_distribution_shift/domain_pca_sample_counts.csv`

## RQ1 progress deck and detailed narration

- Generated deck: `presentation/output/rq1_progress_review.pptx`
- Standalone narration: `presentation/output/rq1_progress_speaker_script.md`
- Reproducible builder: `presentation/script/build_rq1_progress_deck.py`
- The deck contains 27 slides and embeds speaker notes on every slide.
- Notes are approximately 72–99 words per slide, targeting about 30–40 seconds
  per slide and 11–13 minutes for the full presentation.
- Figure narration states the dataset, comparison domain, numerical evidence,
  interpretation, and limitation. In particular, PCA is described as evidence
  of feature-distribution structure—not causal proof or direct proof of model
  generalisation failure.
- Interim TimePPG findings are labelled as within-dataset results. The deck does
  not claim a final RQ1 cross-dataset conclusion until the four LODO experiments
  are complete.

### Literature strengthening added on 10 September 2026

Sources reviewed:

- `docs/PPG_HR_Generalization_Literature_Evidence_Map.xlsx`
- `docs/PPG_HR_Literature_Evidence_Deck_with_RQ1_Progress_Notes.pptx`

Three literature slides were inserted before the dataset/method section:

1. Evidence-map scope and verification boundary: 65 screened papers, 20
   direct/core studies, 30 strong adjacent studies, and 28 records marked for
   further full-text checking.
2. Research trajectory: signal processing, supervised deep HR estimation,
   cross-dataset/uncertainty work, and self-supervised/foundation models.
3. Defendable RQ1 gap: quantify and diagnose residual failure under target-free
   LODO evaluation, while auditing foundation-model pretraining overlap.

Presentation claim boundary:

- Cross-dataset PPG-HR, LODO evaluation, self-supervision and domain adaptation
  already exist and must not be presented as untouched novelty.
- Methods using unlabeled or labelled target data are adaptation, not target-free
  domain generalisation.
- Adjacent PPG/ECG/rPPG/foundation literature can motivate methods, but precise
  PPG-HR claims must be grounded in directly verified PPG-HR studies.
- The workbook is an initial evidence map, not yet a completed systematic review.

Four background-teaching slides were subsequently inserted before the RQ1 slide:

1. **Domain definition:** `D = P(X,Y) + sensing context`, where the context
   includes device, body site, activity, population and environment.
2. **Model mapping:** `x → z = fθ(x) → ŷ = gφ(z)` for a 512-sample PPG window,
   latent representation and predicted HR.
3. **Generalisation problem:** training minimises source risk while deployment
   measures target risk under `P_source(X,Y) ≠ P_target(X,Y)`; the slide separates
   input, HR-support and conditional shifts.
4. **Broader DG history:** PACS, MLDG, RSC and DomainBed, including what each
   addressed and the remaining limitation relevant to physiological time series.

The resulting narrative is: clinical motivation → domain definition → model
mapping → mathematical failure mechanism → historical DG lesson → working RQ1
→ PPG-HR evidence history → defendable gap → our data and experiments.

### Detailed historical timelines

The two historical slides were redesigned as alternating horizontal timelines:

- **General domain generalisation:** PACS (2017), MLDG (2018), RSC (2020),
  DomainBed (2021), and large-scale pretraining (2022+). Each milestone states
  what it addressed and what limitation remained for our physiological setting.
- **PPG-HR development:** TROIKA (2015), Deep PPG (2019), TimePPG (2022),
  BeliefPPG (2023), SPEAR (2024), and PPG foundation models (2025–26).
  Each node distinguishes the contribution from the remaining generalisation
  limitation.

The timelines end in two explicit project implications: first establish a
frozen leakage-safe ERM/pretrained baseline; then measure residual target-free
failure and associate it with device, site, activity, signal quality and
population without claiming unsupported causality.

### Phase-8 FFT domain evidence added to the deck

- The four-dataset slide is now explicitly labelled **dataset-level domains**;
  it does not claim an isolated device effect.
- A separate slide reads `condition_metrics.csv` directly and plots logged
  within-dataset comparisons:
  - PTT-PPG: sitting 2.68, running 10.02, walking 18.12 bpm MAE.
  - WESAD: meditation 4.98, amusement 6.14, baseline 7.78, stress 28.99 bpm.
  - PPG-DaLiA: baseline 2.85 and stairs 36.77 bpm, alongside all other labelled
    activities in the chart.
- These are condition associations. HR range, motion, subject composition and
  signal quality may still co-vary, so they are not presented as causal effects.
- The existing PTT six-channel slide remains the narrower sensor-site analysis.
- Generated chart: `presentation/output/rq1_progress_assets/fft_within_dataset_domains.png`.

### ECG detector-selection evidence added to the deck

A dedicated slide now follows HR-reference validation and records why
NeuroKit2 Emrich 2023/FastNVG was frozen for WESAD and BIDMC.

- Candidates examined: the project development detector, WFDB XQRS,
  NeuroKit2 Emrich 2023/FastNVG, and SleepECG.
- PPG-DaLiA versus supplied peaks, window-HR MAE: development 1.78, XQRS 0.90,
  Emrich 0.67, SleepECG 1.04 bpm.
- External PTT-PPG versus supplied peaks: Emrich peak F1 0.9989, HR MAE 0.076
  bpm and 100% window coverage; XQRS and SleepECG were approximately 0.35 bpm.
- BIDMC versus bedside-monitor HR: Emrich 1.62, SleepECG 1.73 and XQRS 3.49
  bpm MAE. Monitor HR is an independent comparator, not R-peak ground truth.
- WESAD has no supplied peaks or HR. Pairwise agreement and waveform review were
  used as safeguards; 316 disagreement windows remain flagged, not discarded.
- Selection considered peak precision/recall/F1, HR error, coverage, large-error
  counts, cross-dataset consistency, runtime/reproducibility and visual review.
- Generated chart: `presentation/output/rq1_progress_assets/ecg_detector_selection.png`.

### Detailed teaching and defence script

- File: `presentation/rq1_progress_detailed_teaching_script.md`.
- Covers all 29 slides in the current generated deck.
- Contains approximately 4,000 words, separate from the shorter embedded notes.
- Explains visible statements, abbreviations, evidence values, equations and
  every mathematical symbol used in the background, FFT and evaluation slides.
- Includes claim boundaries for dataset/device confounding, PCA interpretation,
  detector validation, within-dataset versus LODO results and causal language.
- Concept-heavy slides include a short delivery paragraph for presentation use;
  the longer material is intended for learning and supervisor questions.

## Phase 9 primary three-fold within-dataset completion

Audit date: 10 September 2026. Source directories:
`reports/phase9_timeppg/within_dataset_3fold/`.

- All 12 planned runs are complete: four datasets times three outer test folds.
- Every `metrics.json` reports `status=complete`, `subject_overlap=false`, and
  `test_evaluations=1`.
- Combined outer-fold predictions cover each frozen dataset manifest once:
  BIDMC 12,561; PTT-PPG 15,982; WESAD 43,385; PPG-DaLiA 64,697 windows.

Pooled unseen-subject results across the three test folds:

| Dataset | Fold MAE (bpm) | Pooled MAE | RMSE | Pearson r | Within 5 bpm | Bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BIDMC | 2.055, 1.535, 2.120 | 1.916 | 4.342 | 0.946 | 90.57% | +0.304 |
| PTT-PPG | 3.861, 2.249, 8.103 | 4.694 | 11.507 | 0.624 | 81.28% | -1.784 |
| WESAD | 5.883, 4.302, 4.442 | 4.887 | 9.090 | 0.831 | 73.50% | +0.100 |
| PPG-DaLiA | 5.057, 6.652, 6.474 | 6.036 | 11.904 | 0.855 | 71.55% | -0.234 |

Interpretation:

- The within-dataset TimePPG task is complete for seed 17 under the frozen
  three-fold subject protocol.
- BIDMC is the strongest and most stable familiar-domain result.
- PPG-DaLiA has the highest pooled MAE, consistent with difficult daily-life
  wrist recordings.
- PTT-PPG has the largest fold variation (fold-MAE SD 3.02 bpm), driven by the
  previously identified difficult subject/fold rather than uniform failure.
- WESAD is reasonably stable across folds, with pooled MAE 4.89 bpm.

## Presentation refresh after within-dataset completion

- Updated the main deck to report all four completed primary three-fold
  TimePPG results rather than the earlier interim subset.
- Added a dedicated fold-stability and full-metric slide covering MAE, RMSE,
  Pearson correlation and percentage within ±5 bpm.
- Updated the conclusion to distinguish the completed familiar-domain result
  from the still-pending unseen-dataset LODO result.
- Updated the next-work slide so the first remaining experiment is four-way
  LODO, not additional within-dataset training.
- The generated deck now contains 30 slides, with synchronized embedded notes,
  speaker script and detailed teaching script.
- These values establish familiar-domain references only; they are not LODO
  results and do not complete the cross-dataset answer to RQ1.

Execution note: the last PPG-DaLiA console log ends with a shell message
`t: command not found` after the run outputs were written. Its `metrics.json` is
complete and its test predictions contain the full fold, so this did not
invalidate training or evaluation. The current runner no longer contains that
stray command.

## Phase 9 pilot LODO: PPG-DaLiA held out

A time-bounded, genuine train-three/test-one pilot was completed with seed 17.
This is a **pilot LODO**, because it uses balanced source subsets rather than all
available source-training windows.

- Held-out target: PPG-DaLiA.
- Joint training sources: BIDMC, PTT-PPG and WESAD.
- Source training: 5,000 windows per dataset, 15,000 total.
- Source validation: 1,500 windows per dataset, 4,500 total.
- Target test: all 64,697 PPG-DaLiA windows.
- PPG-DaLiA was not used for training, validation, early stopping or checkpoint
  selection.
- Training stopped after 13 epochs with patience 3; epoch 10 was selected using
  source-validation loss.
- The untouched target was evaluated exactly once.
- Total runtime: 419.4 seconds, approximately 7.0 minutes.

Pilot target results:

| MAE | RMSE | Pearson r | Within ±5 bpm | Bias |
| ---: | ---: | ---: | ---: | ---: |
| 9.537 bpm | 16.877 bpm | 0.718 | 54.98% | -5.566 bpm |

Compared with the completed PPG-DaLiA familiar-domain MAE of 6.036 bpm, the
pilot generalisation gap is **+3.501 bpm**, or approximately **58.0% higher
MAE**. This is preliminary evidence of cross-dataset degradation, not the final
full-data LODO estimate.

Source: `reports/phase9_timeppg/lodo_pilot/target_ppg_dalia/seed_17/`.

### Pilot LODO presentation update

- Added a dedicated pilot LODO slide after the completed within-dataset profile.
- The slide shows the joint three-source training design, source-only validation,
  complete untouched DaLiA test, target metrics and +3.501-bpm gap.
- The result is explicitly labelled balanced-subset, one-target and one-seed
  preliminary evidence; it is not presented as the final full-data benchmark.
- Updated the conclusion and next-work slides to distinguish the available pilot
  from the required four-target full-source LODO evaluation.
- The generated deck now contains 31 slides, with synchronized speaker notes and
  detailed teaching script.

## Phase 9 full-data four-target LODO completion

All four full frozen-data LODO runs completed for seed 17 using Apple MPS. Each
run trained one new model jointly on three source datasets, used source-only
validation, excluded the complete target from checkpoint selection, and
evaluated the target exactly once.

| Target | Familiar MAE | LODO MAE | Gap |
| --- | ---: | ---: | ---: |
| BIDMC | 1.916 | 1.859 | -0.057 bpm |
| WESAD | 4.887 | 4.631 | -0.256 bpm |
| PTT-PPG | 4.694 | 15.009 | +10.315 bpm |
| PPG-DaLiA | 6.036 | 7.353 | +1.317 bpm |

Interpretation: transfer is strongly target-dependent. BIDMC and WESAD transfer
well, DaLiA degrades moderately, and PTT-PPG is the dominant unseen-domain
failure. Multiple target factors change together, so this result does not yet
isolate device, site, activity, population or HR-support causally.

The presentation now replaces the pilot slide with the complete four-target
comparison and updates its conclusion and next-work sequence. Additional seeds
and PTT-focused failure analysis remain.

### Full-LODO domain-stratified presentation update

- Replaced the older within-dataset outlier slide with evidence calculated from
  the four full-LODO `target_test_predictions.csv` files.
- Added logged activity/condition examples: WESAD stress 9.22 bpm and DaLiA
  stairs 24.25 bpm.
- Added HR-range failures: DaLiA >140 bpm is 48.18 bpm; PTT 60–80 bpm is 22.49
  bpm and <60 bpm is 31.42 bpm.
- Added the spectral-concentration proxy contrast and worst PTT subjects.
- Added the counterexample that PTT sitting is harder than walking and running,
  showing that motion alone does not explain the target failure.
- The conclusion labels these as associations and routes RQ2 through seed
  replication, subject bootstrap, adjusted mixed-effects analysis and paired
  sensor-site testing before choosing a mechanism.

## Phase 10 approved failure-analysis interpretation

Use the following interpretation in future presentations, reports and spoken
explanations.

### Supported findings

- PTT transfer degradation is widespread: **21 of 22 subjects** have higher
  LODO MAE than within-dataset MAE, although the severity varies greatly.
- PTT's strongest failure zone is below 80 bpm. The transferred model is
  systematically biased upward in this range.
- PTT error decreases toward approximately 100–120 bpm. However, 80–100 bpm
  remains materially inaccurate at **11.98 bpm LODO MAE**.
- Sitting has the highest unadjusted PTT error mainly because it contains many
  low-HR windows. The current evidence does not establish sitting itself as the
  cause.
- BIDMC performs well across its common HR ranges. WESAD deteriorates above
  100 bpm. PPG-DaLiA performs reasonably below 100 bpm but increasingly
  underpredicts above 100 bpm.
- PTT results above 120 bpm come from only one subject and must not be
  generalised.
- PTT low HR is overpredicted, whereas DaLiA high HR is underpredicted. This is
  consistent with target-dependent prediction-range compression toward a
  middle HR region.
- Activity is not proven as the cause: after accounting for HR, signal proxy
  and subject, walking and running do not have clear independent associations.
- Signal quality is not a complete explanation because PTT remains inaccurate
  in the high spectral-concentration tertile.
- PTT low-HR failure is not explained by simple source-window scarcity: the
  source data contain 6,848 windows below 60 bpm and 37,174 windows from 60–80
  bpm. By contrast, DaLiA's >=140-bpm target region has only 226 source-training
  windows, so limited coverage is a plausible contributor there.
- Temporal traces and episode analysis show sustained failures rather than only
  isolated bad windows.
- Subject s2 is an important exception: its within-dataset model fails much
  more severely than its LODO model and requires separate trace, label and fold
  inspection while remaining in the main analysis.

### Wording boundary

Do not say that low-HR labels were not learned or that sitting caused the
failure. The ECG-derived labels are valid, and low-HR source examples are
available. Use this wording instead:

> The model did not transfer or calibrate properly for low-HR PTT signals and
> systematically predicted them too high.

Do not claim that 80–120 bpm is the best range for every target. The failure
profile is target-specific, and sparsely supported bins must be interpreted
using both window and subject counts.

### Presentation-ready conclusion

> Phase 10 shows that TimePPG transfer failure is target- and HR-range-dependent:
> PTT experiences widespread low-HR overprediction across subjects, while
> DaLiA experiences high-HR underprediction; activity and signal quality
> contribute but do not independently explain the failures.

These are post-hoc diagnostic associations from seed 17. They identify strong
failure patterns but do not prove isolated causal mechanisms. Confirmatory work
requires additional training seeds, subject-aware uncertainty, targeted trace
inspection and controlled sensor-site experiments.

Source: `reports/phase10_lodo_failure_analysis/`.

## Agreed RQ1-to-RQ2 execution sequence

The frozen next-work order is:

1. Replicate the full LODO result with seeds 29 and 43, prioritising PTT-PPG
   because it is the dominant seed-17 failure.
2. Audit severe PTT temporal episodes and the exceptional s2 result against
   the raw ECG, R-peaks, reference HR, PPG waveform and spectrum.
3. Run a controlled paired PTT distal-versus-proximal TimePPG experiment using
   synchronised channels from the same subjects, recordings and windows.
4. Run a WESAD-to-DaLiA and DaLiA-to-WESAD comparison to reduce device
   confounding because both use the Empatica E4 wrist device family.
5. Perform HR-matched TimePPG-embedding domain/site classification to test
   whether the learned representation retains nuisance-domain information.
6. Freeze the RQ1 mechanism-level conclusion, then evaluate one strong modern
   pretrained or self-supervised PPG baseline under the identical four-target
   LODO protocol.
7. Select the RQ2 intervention from the residual replicated failure rather than
   from aggregate MAE alone.

The full-LODO launcher accepts `TIMEPPG_SEED`; the prespecified replication
wrapper runs seeds 29 and 43 without overwriting seed 17.

## PTT full-LODO replication: seeds 17, 29 and 43

The two prespecified PTT replication runs completed using the same 15,982
untouched target windows and frozen source/validation protocol as seed 17.
Window identities, timestamps and reference labels match across all runs.

| Seed | MAE | Bias | Pearson r | Within ±5 bpm |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 15.009 | +11.899 | 0.375 | 32.75% |
| 29 | 14.324 | +12.155 | 0.400 | 35.33% |
| 43 | 14.489 | +11.181 | 0.296 | 40.80% |

- Mean LODO MAE is **14.607 ± 0.357 bpm**, where ± is the descriptive standard
  deviation across the three training seeds.
- Mean signed bias is **+11.745 ± 0.505 bpm** and mean correlation is
  **0.357 ± 0.054**.
- **20 of 22 subjects** are worse than the fixed within-dataset reference in
  every LODO seed. Seed 17 alone had 21 of 22 subjects worse.
- The seed-averaged subject-macro gap is **+9.927 bpm**, with a subject-bootstrap
  95% interval of **+3.804 to +14.788 bpm** conditional on these fitted models.
- Strong positive bias below 80 bpm repeats in all three seeds. The principal
  PTT transfer failure and its direction are therefore not a seed-17 accident.
- PTT bins above 120 bpm still represent only one subject and must not be used
  for a general high-HR conclusion.

Source: `reports/phase10_lodo_failure_analysis/ptt_multiseed/`.

## Phase 11 initial controlled PTT sensor-site transfer

The completed seed-17 PTT three-fold models, originally trained on distal
`pleth_1`, were evaluated on both distal `pleth_1` and its synchronized paired
proximal `pleth_4` channel. The comparison uses the same model checkpoints,
held-out subjects, activities, recording times, 8-second windows and ECG-derived
HR labels. All 15,982 PTT windows are covered exactly once across the folds.

- Familiar distal `pleth_1` MAE: **4.694 bpm**.
- Unseen proximal `pleth_4` MAE: **7.805 bpm**.
- Pooled site-transfer increase: **+3.111 bpm**.
- Subject-macro site gap: **+3.109 bpm**.
- Subject-bootstrap 95% interval: **+2.009 to +4.204 bpm**.
- Proximal evaluation is worse for **20 of 22 subjects**.

Interpretation: changing only the paired input placement/channel while holding
the model, subject, time and HR constant produces a reproducible loss of
usability. This is stronger evidence for a placement/channel contribution than
the four-dataset comparison. It remains a one-direction diagnostic, not yet a
symmetric or isolated causal site result. The next confirmation is to train on
proximal `pleth_4`, test familiar proximal and transfer back to distal
`pleth_1`. Wavelength names remain neutral because local documentation versions
conflict, while the 1/4 pairing and distal/proximal relationship are documented.

Source: `reports/phase11_ptt_site_transfer/`.

## Phase 11 completed bidirectional PTT site result

All three proximal `pleth_4` training folds completed on MPS. Their familiar-site
fold MAEs were 7.931, 6.113 and 10.677 bpm, giving a pooled proximal familiar-site
MAE of **8.223 bpm**. The same proximal-trained checkpoints were then evaluated
on synchronized distal `pleth_1` windows.

| Training channel | Test channel | Pooled MAE | Meaning |
| --- | --- | ---: | --- |
| Distal `pleth_1` | Distal `pleth_1` | 4.694 bpm | Familiar distal site |
| Distal `pleth_1` | Proximal `pleth_4` | 7.805 bpm | Distal-to-proximal transfer |
| Proximal `pleth_4` | Proximal `pleth_4` | 8.223 bpm | Familiar proximal site |
| Proximal `pleth_4` | Distal `pleth_1` | 5.575 bpm | Proximal-to-distal transfer |

For distal-to-proximal evaluation, MAE increases by **+3.111 bpm** and proximal
is worse for 20 of 22 subjects. For the reverse direction, moving from proximal
to distal improves MAE by **2.648 bpm**; the subject-macro reverse difference is
-2.647 bpm with a subject-bootstrap 95% interval of -3.665 to -1.744 bpm, and
distal is worse for only 2 of 22 subjects.

### Approved interpretation

> The result is not a generic penalty for any unseen site. Distal `pleth_1`
> produces lower error than proximal `pleth_4` under both training directions.
> This supports greater distal signal usability for TimePPG and a substantial
> placement/channel contribution to performance.

Do not describe this as a pure causal body-site result. Although subject,
activity, timing, HR labels and paired channel relationship are controlled,
placement remains tied to the physical channel and potentially its optical or
hardware characteristics. Wavelength names remain neutral because the local
documentation versions conflict.

The combined presentation figure is
`reports/phase11_ptt_site_transfer/figures/bidirectional_site_summary.png`.

## Phase 11 frozen PTT pair-1/4 outcome

The bidirectional exact-window analysis is now frozen for PTT pair `pleth_1` /
`pleth_4`, TimePPG-Big PPG-only, seed 17, frozen preprocessing and frozen
three-fold subject roles. Input prediction tables are protected by recorded
SHA-256 hashes in `reports/phase11_ptt_site_transfer/frozen_outcome.json`.

- Distal-trained subject-macro proximal penalty: **+3.109 bpm**; subject-
  bootstrap 95% interval **+2.009 to +4.204 bpm**.
- Proximal-trained subject-macro proximal penalty: **+2.647 bpm**; interval
  **+1.744 to +3.665 bpm**.
- Bidirectional subject-macro proximal penalty: **+2.878 bpm**; interval
  **+1.926 to +3.881 bpm**.
- The bidirectional proximal penalty is positive for **20 of 22 subjects**.
- By activity, the descriptive bidirectional proximal penalties are +4.329 bpm
  for running, +3.818 for walking and +0.493 for sitting. These are effect-
  modification descriptions, not isolated causal activity effects.
- By HR, the penalty is +2.172 bpm at 60–80, +2.782 at 80–100 and +8.169 at
  100–120 bpm. The <60 estimate is +7.362 bpm but has only four subjects;
  estimates above 120 bpm have only one subject and are not generalisable.

Approved frozen statement:

> For PTT pair 1/4, proximal input is consistently less usable for TimePPG than
> synchronized distal input. The disadvantage remains across both training
> directions while subject, activity, time and HR label are held constant.

This supports a placement/channel contribution, not a universally causal claim
about anatomical site. Physical channel, optical and hardware characteristics
remain tied to placement. Pairs 2/5 and 3/6 are optional sensitivity extensions
and are outside this frozen pair-1/4 outcome.

## Approved RQ1 contribution and presentation framing

Use the following as the conclusion-framework wording:

> We do not claim that causes of domain shift have never been studied. We
> provide a controlled failure-attribution framework that measures where
> transfer fails and quantifies specific domain-related contributions before
> developing an RQ2 solution.

Use the following as the literature-gap wording:

> Existing PPG-HR studies often demonstrate cross-dataset failure or propose a
> solution, but relatively few isolate and quantify how much individual
> factors—such as sensor site, HR-support mismatch, activity and signal
> quality—contribute under one target-free, controlled protocol.

In this wording, **target-free** means that no samples, labels, summary
statistics or validation results from the held-out target dataset are used to
train, tune, calibrate, select or early-stop the model. The target is opened
only for the final evaluation. This is domain generalisation. If unlabeled or
labeled target data are used to change the model, the experiment becomes
domain adaptation rather than target-free domain generalisation.

`Target-free` does not mean there is only one target dataset. In each LODO run,
one complete dataset is selected as the target and kept untouched; the process
is repeated so that each of the four datasets serves as the target once.

### Approved concise methodological gap

> Relatively few PPG-HR studies use a single consistent, target-free and
> controlled evaluation protocol to isolate individual domain factors and
> quantify how much each factor is associated with transfer failure.

This is the current defendable RQ1 gap. It is not a claim that domain-shift
factors have never been studied. The novelty is the consistent failure-
attribution framework: hold other variables constant where the data permit,
avoid target-based tuning, calculate the additional transfer error, and compare
multiple domain factors under the same protocol.

Use **associated with**, **contributes to**, or **supports a factor effect**.
Use **causes** only when the factor has been independently manipulated or
isolated strongly enough for causal identification.

## Approved presentation framing: controlled failure attribution

Use the following as the presentation gap statement:

> Existing PPG-HR studies often demonstrate cross-dataset failure or propose a
> solution, but relatively few isolate and quantify how much individual
> factors—such as sensor site, HR-support mismatch, activity and signal
> quality—contribute under one target-free, controlled protocol.

Here, **target-free** means that the held-out target dataset contributes no data
to model training, validation, early stopping, checkpoint selection,
hyperparameter tuning or calibration. It is revealed only for the final test.
This is the LODO domain-generalisation setting. It differs from domain
adaptation, where labelled or unlabelled target-domain data are available while
the model is being adapted.

Use the following as the presentation conclusion/framework statement:

> We do not claim that causes of domain shift have never been studied. We
> provide a controlled failure-attribution framework that measures where
> transfer fails and quantifies specific domain-related contributions before
> developing an RQ2 solution.

This is the approved novelty boundary. Do not claim that this is the first study
until the focused literature search for controlled, paired PPG-HR domain-factor
experiments is complete.

## Phase 12 frozen Pulse-PPG linear-probe failure localisation

The full paired post-hoc analysis covers all **136,625** primary windows. Each
window is matched across Pulse-PPG within-dataset, Pulse-PPG LODO, TimePPG
within-dataset and seed-17 TimePPG LODO predictions using dataset, recording,
subject and window index.

Frozen headline findings:

- Pulse-PPG uses the released frozen encoder with an 8-second 50-Hz adaptation
  and a source-trained Ridge head. This condition is not equivalent to the
  authors' 4-minute pretraining input or to end-to-end fine-tuning.
- PTT-PPG remains the dominant transfer failure: **14.452 bpm LODO MAE** versus
  **5.991 bpm within-dataset**, a same-window paired gap of **+8.461 bpm**.
  The gap is positive for **21/22 subjects**.
- PTT 60–80 bpm has **19.301 bpm LODO MAE**, **+17.018 bpm signed bias** and a
  **+14.901 bpm paired gap** over 5,169 windows from 21 subjects. The source
  training data contain 37,174 windows in this HR bin, so absent low-HR labels
  alone are not an adequate explanation.
- The PTT low-HR positive bias occurs in sitting, walking and running and the
  overall failure persists in the highest spectral-concentration tertile
  (**14.870 bpm MAE; +8.968 bpm paired gap**). Activity or simple spectral
  quality alone therefore cannot explain it.
- DaLiA shows the opposite direction at high HR: at 120–140 bpm, LODO MAE is
  **29.371 bpm** with **−29.047 bpm bias**; at ≥140 bpm, it is **51.169 bpm**
  with **−51.164 bpm bias**. Source support is sparse in these bins (1,831 and
  226 training windows), making tail coverage a plausible contributor.
- Prediction-range compression remains: LODO prediction-versus-reference
  slopes are **0.427 for PTT**, **0.462 for DaLiA** and **0.619 for WESAD**.
  These descriptive slopes do not establish a causal mechanism.

Approved conclusion:

> Large-scale field pretraining with a frozen linear head does not remove the
> target- and HR-dependent failure. PTT shows widespread low-HR overprediction
> despite substantial source low-HR support, while DaLiA shows high-HR
> underprediction together with sparse source-tail coverage. These observations
> support HR-conditional acquisition shift and source-support mismatch as
> different target-specific mechanisms, but do not yet isolate causality.

The next controlled ablations are input context (8 seconds versus longer), head
capacity (linear versus nonlinear), and encoder adaptation (frozen versus
source-only fine-tuning). Change one factor at a time and retain target-free
source validation. Do not claim that Pulse-PPG generally fails until these
adaptation conditions are tested.

### Approved explanation: four-line HR-range comparison

The HR-range figure deliberately contains all four frozen conditions:

1. Pulse-PPG within-dataset,
2. Pulse-PPG LODO,
3. TimePPG within-dataset, and
4. TimePPG LODO.

Colour identifies the model: green is Pulse-PPG and orange is TimePPG. Line
style identifies the protocol: dashed is the familiar, within-dataset test and
solid is the unseen-dataset LODO test. Every point in one target/HR bin is
calculated from the same ECG-labelled test windows; `n` is the number of those
windows.

Use this simple presentation wording:

> We compare both models in familiar and unseen datasets. For one model, the
> distance between its dashed and solid lines shows the transfer penalty. For
> one protocol, the distance between the green and orange lines shows which
> model has lower error in that heart-rate range.

Use this interpretation rule:

- High within-dataset and high LODO error indicates a broader model or input
  limitation; it should not be described as transfer failure alone.
- Low within-dataset error followed by high LODO error is evidence of a
  transfer-specific failure for that target and HR range.
- Similar within-dataset and LODO error means little additional transfer
  penalty, but it does not necessarily mean the absolute accuracy is good.

Approved one-line conclusion:

> The full comparison separates model weakness from transfer weakness: PTT's
> low-HR deterioration is mainly transfer-specific, whereas some extreme-HR
> errors are already present during familiar-domain testing.

Important boundary: the plot compares frozen evaluations on identical target
windows, but the two model families use different representations and training
procedures. It supports a fair performance comparison, not a causal claim that
one architectural component produced the difference.

Phase-12 image-level evidence is stored in
`reports/phase12_pulseppg/04_failure_analysis/`:

- `overall_model_protocol_comparison.png` gives the pooled four-condition result;
- `hr_range_failure.png` localises absolute error by HR range;
- `lodo_signed_bias_by_hr.png` shows overprediction versus underprediction;
- `subject_transfer_gap.png` shows whether deterioration is widespread across people;
- `ptt_failure_checks.png` checks whether PTT failure persists across activity,
  HR group and spectral-concentration strata.

## Phase 12D frozen nonlinear-head ablation

The Pulse-PPG encoder and all 136,625 cached embeddings remained frozen. Two
predeclared MLP heads were compared using source-validation subjects only.
Selection averaged source-validation dataset-macro MAE over seeds 17, 29 and
43; the `512→256→64→1` head was selected for every within fold and LODO run.
Every window-level prediction count was verified.

Mean ± SD MAE across the three seeds:

| Target | Nonlinear within | Nonlinear LODO | LODO − within |
|---|---:|---:|---:|
| BIDMC | 3.54 ± 0.06 | 3.64 ± 0.22 | +0.10 ± 0.21 |
| WESAD | 6.34 ± 0.03 | 6.48 ± 0.08 | +0.14 ± 0.10 |
| PTT-PPG | 5.47 ± 0.11 | 12.15 ± 1.26 | +6.68 ± 1.15 |
| PPG-DaLiA | 8.52 ± 0.05 | 9.59 ± 0.02 | +1.08 ± 0.06 |

Relative to the frozen Ridge head, nonlinear LODO MAE improves by 1.80 bpm on
BIDMC, 2.19 on WESAD, 2.31 on PTT and 0.88 on DaLiA. BIDMC within-dataset MAE
worsens by about 0.40 bpm. The PTT transfer gap remains large and its LODO MAE
is the most seed-sensitive (11.18–13.56 bpm).

Approved presentation wording:

> A small nonlinear head accesses more useful HR information from the frozen
> Pulse-PPG representation and improves unseen-dataset MAE on all four targets.
> However, it does not eliminate target-dependent transfer failure: PTT retains
> a large and seed-sensitive generalisation gap, and TimePPG remains stronger
> on BIDMC, WESAD and DaLiA. Head linearity is therefore part of the limitation,
> but not the complete explanation.

Presentation figure:
`reports/phase12_pulseppg/03_head_and_tuning_ablation/nonlinear_head/nonlinear_head_comparison.png`.
