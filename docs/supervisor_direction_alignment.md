# Alignment with supervisor direction

Governing document: `docs/Arun-PhD Direction 1.docx`.

## Thesis dependency

RQ1 is not merely preliminary model training. It creates the frozen benchmark on
which RQ2 and RQ3 depend:

```text
RQ1: measure and characterise failure under domain shift
  -> identifies which shifts and target datasets cause failure
RQ2: develop one representation method motivated by that failure pattern
  -> produces the best zero-target-label model
RQ3: quantify uncertainty and abstain when that model is unreliable
```

RQ2 should therefore not be finalised before RQ1 results exist. Its proposed
method must address an observed RQ1 failure rather than a presumed failure.

## Fixed common pipeline

- Primary input: PPG/BVP only.
- Reference: ECG-derived HR.
- PPG filter: 0.5–4 Hz.
- Common PPG rate: 64 Hz.
- Window length: 8 s.
- Window step: 2 s.
- Window normalization: zero mean and unit standard deviation.
- Reject missing/saturated signals and reference HR outside 35–220 bpm.
- Visually inspect at least 10 randomly selected recordings per dataset before
  batch processing.
- Freeze accepted-window manifest and subject splits before model development.

## RQ1 scope

Only two baselines are required:

1. Dominant spectral-peak HR estimator.
2. Published Deep PPG/1D-CNN-style supervised model.

RQ1 evaluates both within datasets and under four leave-one-dataset-out folds.
It reports MAE, RMSE, Pearson correlation, predictions within ±5 bpm, and
Bland–Altman bias/95% limits of agreement. Activity-stratified analysis applies
to PPG-DaLiA, WESAD, and PTT-PPG; BIDMC receives patient-level analysis.

## One point requiring clarification

The document specifies train-three/test-one leave-one-dataset-out evaluation,
which produces four primary transfer experiments. It also requests a “4×4
transfer table,” which normally means 16 pairwise train-one/test-one cells.

Until clarified, implementation should store results in a tidy table capable of
rendering either format. The primary protocol remains the four explicitly stated
train-three/test-one folds; pairwise experiments must not be silently substituted
for them.
