# Goal-motivated path from RQ1 to RQ2

## 1. The key problem

The main problem is not simply that one model has high MAE. Our evidence shows
that PPG-HR prediction changes systematically with the acquisition domain:

- TimePPG LODO performance depends strongly on the held-out target.
- PTT low HR is systematically overpredicted.
- PPG-DaLiA high HR is systematically underpredicted.
- Predictions are compressed toward a middle HR range.
- PTT proximal input produces more error than synchronized distal input across
  all three sensor pairs, with person, activity, time and ECG HR held constant.
- Activity and signal quality alone do not fully explain these failures.

Together, these results support a working mechanism:

> **HR-conditional acquisition shift:** the same target HR has different PPG
> morphology and spectral structure across devices, sites and environments.
> Models may encode these acquisition characteristics instead of a stable
> cardiac representation.

This is consistent with a shift in `P(X | Y)`: the distribution of PPG input
`X` changes even when ECG-derived HR `Y` is held approximately constant.

## 2. What is missing in existing work

The careful gap statement is:

> PPG-HR domain adaptation increasingly uses unlabelled target data, and a small
> number of studies perform cross-dataset or LODO evaluation. However,
> target-free controlled attribution of transfer failure to specific acquisition
> factors remains limited, and generic alignment does not explicitly ensure that
> signals with the same HR align across acquisition domains without collapsing
> distinct HR values.

We do not claim that domain shift, sensor placement or DG has never been studied.
Our contribution is the chain from controlled diagnosis to a mechanism-motivated,
source-only intervention.

## 3. Why Pulse-PPG is needed before proposing a method

Pulse-PPG tests whether large-scale field pretraining already learns a stable
cardiac representation.

- If it removes TimePPG's failures, generic self-supervised pretraining may be
  sufficient and inventing a complex DG method is not justified.
- If aggregate MAE improves but HR-range bias, site penalties or domain identity
  remain, the unresolved mechanism is clearer.
- If it does not improve transfer, scale alone is insufficient for this task.

Therefore, Phase 12 is a falsification step for the need for our method.

## 4. Existing-DG benchmark required before our method

We must not move directly from ERM and Pulse-PPG to a new objective. Phase 13
will test representative existing DG mechanisms under the same source-only LODO
protocol:

- inter-domain Mixup;
- CORAL and MMD source-feature alignment;
- source-domain adversarial learning (DANN);
- GroupDRO;
- VREx and IRM;
- MLDG.

Methods will first share the TimePPG backbone so that the learning strategy—not
the encoder architecture—is the main changed variable. Only methods that improve
more than one target and pass failure-level checks will be transferred to the
Pulse-PPG backbone. See `reports/phase13_dg_benchmark/`.

The candidate idea below is permitted to proceed only if strong existing methods
do not adequately resolve the reproducible failure mechanism.

## 5. Candidate own idea for RQ2

### Working name

**HR-Conditioned Acquisition-Invariant Representation Learning**

This is a provisional research direction, not yet a novelty claim.

### Core principle

Do not align all source PPG features globally. Global alignment can mix signals
with different HR values and remove useful cardiac information. Instead:

1. preserve HR-discriminative information;
2. bring representations from different source domains closer only when their
   ECG-derived HR values are similar;
3. separate representations when HR values are physiologically different;
4. balance source domains and HR ranges so the regression head does not collapse
   predictions toward the most common HR region.

### Candidate training objective

For source-only training data:

`Total loss = HR regression loss + conditional cross-domain consistency loss + HR-support balancing term`

- **HR regression loss:** retains accurate cardiac prediction.
- **Conditional cross-domain consistency:** treats examples with similar HR but
  different source domains as positive representation pairs.
- **HR-support balancing:** prevents common middle-HR windows from dominating
  learning and reduces prediction-range compression.

An optional domain-adversarial term may be tested only if it is conditioned on
HR. An unconditional domain adversary risks deleting useful HR information when
domain and HR distributions are correlated.

### PTT paired-site extension

When PTT is a source dataset, synchronized distal/proximal windows can provide a
strong consistency signal because their person, activity, time and ECG HR are
identical. This is a secondary ablation, not part of the universal primary
method: it cannot be used when PTT is the held-out target because that would
leak target data.

## 6. Target-free safety rule

For every LODO target:

- use only the other three datasets for training and validation;
- construct HR-conditioned pairs only from source datasets;
- calculate balancing weights from source training data only;
- never use target signals for alignment, normalization or model selection;
- reveal target labels only for the frozen final evaluation and failure audit.

This keeps the proposed method as domain generalisation rather than unsupervised
domain adaptation.

## 7. Hypotheses to test

### H1 — residual failure after pretraining

Pulse-PPG will improve some aggregate errors but will retain target-dependent
HR-range bias or measurable acquisition identity in its embeddings.

### H2 — conditional alignment

HR-conditioned cross-domain consistency will reduce LODO MAE and signed bias
more reliably than unconditioned feature alignment.

### H3 — range compression

Source HR-support balancing will reduce low-HR overprediction and high-HR
underprediction without materially damaging common-range accuracy.

### H4 — acquisition sensitivity

The proposed representation will reduce synchronized PTT proximal penalties
when PTT is available as a source, while maintaining HR accuracy.

## 8. Required baselines and ablations

To attribute any improvement to our idea, compare:

1. TimePPG ERM;
2. Pulse-PPG frozen linear probe;
3. the strongest existing DG methods selected by Phase 13;
4. Pulse-PPG source-only fine-tuning;
5. fine-tuning plus HR-support balancing only;
6. fine-tuning plus conditional consistency only;
7. the complete proposed objective;
8. optional unconditional domain alignment as a mechanism control.

Use identical source splits, validation rules, LODO targets and metrics.

## 9. Success criteria

The proposed method is supported only if it:

- improves source-only LODO performance across more than one target;
- reduces the worst positive generalisation gap, not only average MAE;
- reduces systematic HR-bin bias and prediction-range compression;
- improves most subjects rather than a small subgroup;
- does not obtain improvement through target access;
- preserves familiar-domain performance within a declared acceptable margin;
- shows reduced domain/site information or reduced controlled site sensitivity.

## 10. What would falsify the idea

- Pulse-PPG already removes the residual failures.
- Conditional consistency improves sources but worsens unseen targets.
- Improvements disappear across seeds or are driven by one subject.
- Domain predictability falls while HR performance also collapses.
- The method works only when target-like PTT pairs are included as sources.
- A standard existing DG method resolves the same failures equally well with
  less complexity.

These outcomes are informative and must be reported rather than hidden.

## 11. Proposed RQ2 wording

> Can source-only HR-conditioned acquisition-invariant representation learning
> reduce target-dependent and sensor-related generalisation failures in
> PPG-based heart-rate estimation without access to the deployment domain?
