# Task 12B — input compatibility and small source-only HR gate

## Reproducible checks

Run from the repository root with the project's virtual environment:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/models/pulseppg/input_compatibility_smoke.py
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/models/pulseppg/source_validation_probe.py
```

The first script uses `torch.load(..., weights_only=True)` and loads the released
`net` weights into the official ResNet1D with strict key matching. It reads
up to 24 label-blind windows per dataset from the frozen Phase-7 manifest.
Both 8-second and 32-second inputs remain within one recording. The second
script trains a *small diagnostic* fixed-alpha Ridge head on 720 source
windows and checks 192 source-validation windows in a PTT-held-out split.
**No PTT signal or label is read by this HR probe.** Neither script selects
hyperparameters from a held-out target.

## Input path and mismatch from author pretraining

- The [Pulse-PPG paper](https://arxiv.org/html/2502.01108) reports 4-minute,
  50-Hz wearable-field PPG, with person-specific global z-normalisation and
  no signal-specific filter for encoder pretraining.
- Our canonical frozen inputs are 8-second, 64-Hz, 0.5–4-Hz filtered PPG.
  We resample each extracted 64-Hz slice to 50 Hz with
  `scipy.signal.resample_poly(..., 25, 32)`, yielding 400 points for 8 seconds
  or 1,600 for 32 seconds, then z-score **that input slice only**. No test
  subject or target-dataset statistics are fitted. This is a documented
  adaptation, **not an exact recreation** of author pretraining preprocessing.
- The 32-second condition gives the model more future context than the
  original 8-second target window, so it is a numerical sensitivity condition
  only, not an interchangeable HR benchmark input.

## Observed numerical results

See `smoke_metrics.json` for all numbers. The released checkpoint loaded with
strict key matching on CPU and returned 512-dimensional embeddings for both
durations. For each of BIDMC, WESAD, PTT-PPG and PPG-DaLiA, all 24 sampled
8-second embeddings were finite; repeated inference differed by exactly zero;
and no sampled pair had cosine similarity above 0.9999. The same checks passed
for 32 seconds. Thus **8-second input is numerically usable**, but this does
not establish robust HR information.

PTT-PPG's median pairwise cosine similarity was high (0.995 at 8 seconds),
although no pair met the near-duplicate threshold. This deserves monitoring
in the representation audit; it is not, by itself, proof of feature collapse.
Embedding norms differ substantially between 8 and 32 seconds, consistent
with the network's max temporal pooling and changed context. Raw norms across
durations should not be interpreted as HR accuracy.

## Source-only HR-signal gate

See `source_validation_probe.json`. A fixed Ridge probe on frozen 8-second
embeddings achieved **6.56 bpm MAE** on source-validation windows versus
**9.80 bpm** for predicting the source-training mean HR; Pearson `r=0.617`.
This confirms useful aggregate HR signal in the 8-second representation.
However, WESAD validation was **12.99 bpm**, worse than that simple mean
baseline (**10.86 bpm**), while BIDMC and DaLiA improved. The tiny,
subject-sampled pilot must not be cited as a final performance result or
compared numerically with full TimePPG/LODO metrics.

## Gate decision

**Proceed to Task 12C with 8-second input as the primary documented
adaptation**, retaining 32-second input only as a predeclared sensitivity
analysis. The gate is about technical viability, not a claim that Pulse-PPG
generalises well. Task 12C must use all frozen source windows and proper
subject-wise folds, and must explicitly check the WESAD weakness and PTT
embedding similarity. No target-conditioned preprocessing or tuning is allowed.

The official network emitted a PyTorch warning because it constructs an
`InstanceNorm1d` with a stored `num_features` value that differs from the
single input channel; with `affine=False`, PyTorch does not use that value.
The warning did not prevent finite inference and should be documented rather
than silently edited in the external source.
