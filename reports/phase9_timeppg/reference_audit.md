# Phase 9A: TimePPG reference-code audit

## Sources

- Paper: Burrello et al., *Embedding Temporal Convolutional Networks for
  Energy-Efficient PPG-Based Heart Rate Monitoring*, ACM HEALTH 2022,
  DOI `10.1145/3487910`.
- Official repository: `https://github.com/eml-eda/q-ppg`, preserved locally at
  `external/q-ppg/` under Apache-2.0.
- Primary inspected implementation:
  `external/q-ppg/precision_search/model/TimePPG_float.py`.
- Training and preprocessing were checked in `architecture_search/config.py`,
  `architecture_search/preprocessing/preprocessing_Dalia.py`, and
  `architecture_search/trainer/train_TEMPONet.py`.

## What the official implementation establishes

- Original input: 8 seconds at 32 Hz, giving 256 time samples.
- Default input: one PPG channel plus three accelerometer axes.
- The architecture-search code also explicitly supports `ppg_only_1`.
- TimePPG-Big widths: `32, 32, 63, 64, 64, 121, 122, 104, 76, 82, 61`.
- Dilations: `2, 2, 1, 4, 4, 8, 8`.
- Blocks use Conv1d, average pooling, ReLU6, and batch normalization, followed
  by two dense regressors and one scalar output.
- Training uses LogCosh loss, Adam at 0.001, batch size 128, maximum 500 epochs,
  and early stopping on validation MAE with minimum improvement 0.01 bpm and
  patience 35 epochs.
- The published best results can include smoothing and subject fine-tuning.

## RQ1 adaptation boundary

An unchanged reproduction is impossible for the primary RQ1 benchmark because
BIDMC has no accelerometer and an unseen target cannot be fine-tuned. The common
pipeline is also frozen at 64 Hz by the supervisor specification.

We therefore preserve the complete convolutional topology and published
TimePPG-Big widths, but change:

1. four input channels to one PPG channel;
2. 256 input samples to the frozen 512 samples;
3. the final dense input dimension to match the longer feature map;
4. the original labels/splits to canonical ECG HR and frozen subject/LODO splits;
5. remove target fine-tuning and target-specific calibration.

The experiment name is consequently **TimePPG-Big PPG-only adaptation**. It
must not be described as reproducing the paper's headline MAE.

## Verification completed

- PyTorch 2.14 installed in `.venv`.
- Input `(batch, 1, 512)` produces one finite HR value per item.
- Forward and backward passes produce finite gradients.
- Adapted 512-sample model parameter count: 254,784.
- The same one-channel topology at the paper's 256 samples has 229,856
  parameters; the difference comes from the doubled feature length entering the
  first dense regressor.
- Twelve focused model, window, metric, and spectral tests pass.

## Next implementation checkpoint

Build the manifest-backed training loader and verify a short source-only training
run. Do not evaluate any LODO target until sampling, validation, checkpointing,
and training logs have been inspected and frozen.

