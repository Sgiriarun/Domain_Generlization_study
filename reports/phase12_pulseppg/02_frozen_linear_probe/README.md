# Task 12C — full-data frozen-encoder linear probe

## Status

The full-data benchmark is implemented but **not yet complete**. A local CPU
extraction was stopped after 2,432 of 136,625 windows because throughput
indicated hours of runtime. The preallocated `.npy` matrix in
`artifacts/pulseppg/phase12_embeddings/` is **incomplete**; it has no completion
metadata and must not be used for evaluation. No Phase-12 within-dataset or
LODO result should be reported yet.

## Frozen protocol

The script `scripts/models/pulseppg/run_full_linear_probe.py`:

1. reads **all 136,625** primary Phase-7 windows (BIDMC 12,561; WESAD
   43,385; PTT-PPG 15,982; PPG-DaLiA 64,697), with no subject/window sampling;
2. converts each frozen 64-Hz, 0.5–4-Hz signal window to 50 Hz, then applies
   per-window z-scoring, without fitting target statistics;
3. loads the official encoder checkpoint with `weights_only=True` and strict
   key matching, freezes it, and caches one 512-dimensional embedding per
   window; the cache is only valid when `metadata.json` says `complete` and
   the manifest/checkpoint hashes match;
4. fits a `StandardScaler` and Ridge head on **training subjects only**;
   selects `alpha` from `[0.1, 1, 10, 100, 1000]` using validation subjects
   only, averaging validation MAE equally over source datasets; then evaluates
   a held-out set once;
5. runs the same three subject-disjoint outer folds as Phase 9 for every
   dataset, so every window is tested exactly once within its dataset;
6. runs four full-data train-three/test-one LODO evaluations using the frozen
   Phase-7 source-train/source-validation/target-test roles;
7. reports pooled MAE, RMSE, Pearson `r`, percentage within ±5 bpm, signed
   bias, subject-macro MAE with subject-bootstrap 95% interval, and LODO minus
   within-dataset MAE. Same-window seed-17 TimePPG comparisons are checked by
   exact dataset/record/subject/window keys.

All eligible source windows are used. Window counts differ naturally between
datasets; results are presented **separately per target**, and the validation
choice averages dataset MAE equally. The primary Ridge fit itself is
unweighted, matching the full-data TimePPG training convention; equal-source
weighting would be a separately labelled sensitivity analysis, not a silent
change to the primary benchmark.

## Run commands

On a machine with this repository, datasets, weights and CUDA available:

```bash
python scripts/models/pulseppg/run_full_linear_probe.py --stage all --device cuda --batch-size 128
```

If extraction and evaluation need separate jobs:

```bash
python scripts/models/pulseppg/run_full_linear_probe.py --stage extract --device cuda --batch-size 128
python scripts/models/pulseppg/run_full_linear_probe.py --stage evaluate
```

On this local workspace only, the interrupted cache can be resumed with
`--stage extract --resume --device cpu --batch-size 128`; the script verifies
that it contains one contiguous prefix of completed batches. This would take
hours locally, so a fresh CUDA extraction on the GPU server is preferred.
Do not use `--resume` on a cache of unknown origin.

After completion, `comparison_summary.csv` and `run_provenance.json` are the
entry points; per-fold, per-subject and window-level files are kept under
`within_dataset/` and `lodo/`.
