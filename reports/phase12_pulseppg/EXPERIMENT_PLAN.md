# Phase 12 experimental plan

## Scientific gap being tested

Existing PPG-HR studies include supervised cross-dataset evaluation and several
domain-adaptation methods that use unlabelled target signals. Far less evidence
exists for target-free domain generalisation and controlled attribution of the
remaining error to individual acquisition and physiological factors.

Phase 12 asks whether large-scale self-supervised PPG pretraining closes that
gap or merely changes the aggregate error while retaining the same failures.

## Goal hierarchy

The project is not trying to accumulate model comparisons. Each stage resolves
one decision:

1. **Existence:** does target-free PPG-HR failure remain after strong pretraining?
2. **Localisation:** is the residual failure concentrated by target, HR range,
   subject or sensor placement?
3. **Mechanism:** do embeddings retain acquisition identity after controlling HR?
4. **Existing solutions:** which established DG mechanisms address the failure,
   using one controlled backbone and protocol?
5. **Intervention:** only if a residual gap remains, can a mechanism-specific
   source-only objective remove acquisition information without removing HR?
6. **Reliability:** does the intervention improve unseen targets and subjects,
   rather than only pooled average MAE?

## Task 12A — provenance and overlap audit

### Question

Are the downloaded weights genuinely the base self-supervised Pulse-PPG encoder,
and were any RQ1 datasets used to construct those weights?

### Record

- official repository URL and exact commit;
- licence;
- weight URL, filename and SHA-256 checksum;
- architecture and parameter count;
- pretraining dataset, participants, input duration and sampling rate;
- pretraining objective and confirmation that it used no HR labels;
- downstream datasets used by the original paper;
- explicit status for BIDMC, WESAD, PTT-PPG and PPG-DaLiA: pretraining,
  downstream-only, absent or unresolved.

### Gate

Do not train an HR head until the checkpoint can be identified as a base encoder.

## Task 12B — input compatibility

### Problem

Our canonical input is 8 seconds at 64 Hz, while Pulse-PPG was pretrained with
longer field segments. Variable-length execution does not prove that an
8-second embedding contains enough stable HR information.

### Tests

1. Determine the released encoder's expected sampling rate and filtering.
2. Define one documented resampling path from the frozen 64-Hz signals.
3. Confirm deterministic embeddings with no NaN or infinite values.
4. Measure embedding variance and duplicate/collapse rates by dataset.
5. Compare 8-second input with one longer-context sensitivity condition, such
   as 30 or 32 seconds, without crossing subject or recording boundaries.

### Gate

Eight-second input is primary only if embeddings are numerically stable and a
source-only validation test shows useful HR signal. Otherwise, longer context
must be declared as a protocol change and evaluated separately.

## Task 12C — frozen representation benchmark

### Primary model

Freeze every Pulse-PPG encoder parameter. Train a linear regression head using
ECG-derived HR labels from source-training subjects only.

### Evaluations

- Four within-dataset three-fold tests using the frozen subject roles.
- Four LODO tests: train on three complete source datasets and test the untouched
  fourth dataset once.
- Report MAE, RMSE, signed bias, Pearson correlation and percentage within
  ±5 bpm using the same aggregation rules as TimePPG.
- Calculate `LODO MAE − within-dataset MAE` for each target.

### Required controls

- TimePPG and FFT values from their frozen reports.
- A randomly initialized encoder with the same linear-head protocol, or another
  architecture-matched control if technically feasible.
- Source-only validation for all model selection.

### Interpretation

Improvement over TimePPG supports transferable pretrained features. It does not
by itself prove domain invariance.

## Task 12D — adaptation ablation

Run only after the frozen linear probe is complete.

1. Frozen encoder plus small nonlinear head.
2. Unfreeze the final encoder block plus HR head.
3. Full fine-tuning as a secondary condition if compute permits.

The purpose is to distinguish three possibilities:

- HR information is already linearly accessible;
- HR information exists but needs a nonlinear mapping;
- source-labelled adaptation must modify the encoder.

Select hyperparameters from source validation only. Never select a condition
because it performs best on a held-out LODO target.

## Task 12E — failure localisation

Re-run the Phase-10 analyses on Pulse-PPG predictions:

- per-subject within-versus-LODO error;
- HR-bin MAE and signed bias;
- prediction-range compression;
- activity and HR joint analysis;
- signal-quality strata;
- temporal traces for representative persistent failures;
- source HR support versus target HR support.

### Questions

- Is PTT below 80 bpm still systematically overpredicted?
- Is DaLiA above 100 bpm still underpredicted?
- Does Pulse-PPG reduce error tails or only pooled MAE?
- Are improvements widespread across subjects or driven by a few people?

## Task 12F — controlled PTT site transfer

Apply the selected source-only Pulse-PPG condition to all synchronized pairs
1/4, 2/5 and 3/6 using the Phase-11 exact-window design.

Hold person, activity, time, HR label and split fixed. Report the bidirectional
subject-macro proximal penalty and subject-bootstrap 95% interval for every pair.

### Question

Does field pretraining reduce the repeatable proximal-input disadvantage found
with TimePPG, or is sensor placement still encoded as a major nuisance factor?

## Task 12G — representation audit

### Tests

- Predict dataset identity from frozen embeddings with subject-grouped splits.
- Repeat after HR matching.
- Visualize embeddings only as descriptive support, not causal evidence.
- Where sample size allows, test site and activity identity within PTT.

### Question

Does the representation retain device/dataset/site information even when HR
performance improves?

## Task 12H — frozen conclusion

Freeze only after predictions, metrics, checksums and protocol audits exist.
The final conclusion must answer:

1. Does pretraining improve familiar-domain accuracy?
2. Does it reduce the target-specific LODO gap?
3. Which TimePPG failure zones remain?
4. Does it reduce the controlled sensor-site penalty?
5. Are embeddings more domain-invariant?
6. What residual failure is sufficiently strong and reproducible to motivate
   the RQ2 method?

## RQ2 decision rule

- If frozen pretraining fixes most failures, RQ2 should study efficient and
  leakage-safe pretrained transfer.
- If fine-tuning is necessary, RQ2 should study source-only adaptation without
  catastrophic loss of invariance.
- If HR-range compression remains, RQ2 should target conditional/HR-aware
  calibration or representation learning.
- If site penalties remain across pairs, RQ2 should target acquisition-factor
  invariance or controlled domain-factor disentanglement.
- If uncertainty rises reliably on failures, an uncertainty-aware reliability
  method such as a BeliefPPG-style head becomes well motivated.

After Phase 12, the representative existing-method benchmark in
`reports/phase13_dg_benchmark/` is mandatory. The preferred method direction is
defined in `RESEARCH_DIRECTION.md`, but remains a candidate until the literature
audit, Phase-12 evidence and Phase-13 benchmark are complete.

