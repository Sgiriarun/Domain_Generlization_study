# Task 12A — official files and checkpoint-role audit

## Step 1: identify and obtain official code and weights (complete)

- Official repository: <https://github.com/maxxu05/pulseppg>
- Local code: `external/pulseppg/`, commit `716eaf9cf966e8f76436f2263872ef38b1f90166`
- Code licence: MIT, per `external/pulseppg/LICENSE`.
- Official archive: <https://zenodo.org/records/17345536/files/pulseppg_model_weights.zip?download=1>
- The repository README references DOI `10.5281/zenodo.17270930`, while its
  `download_model.sh` references Zenodo record `17345536`. We selected the
  latter because the official script points to it and Zenodo identifies it as
  the Pulse-PPG model resource. This discrepancy must remain documented.
- Archive location: `artifacts/pulseppg/pulseppg_model_weights.zip` (315,300,817 bytes).
- Published archive MD5: `2d0ebda9afeb9648674464098a698c37`; local MD5
  matches. ZIP integrity check passed.
- Archive SHA-256: `5685093114036e9ceb720b98342d68188c8786ed4e48466aa608a325cd890918`.
- Extracted main checkpoint:
  `artifacts/pulseppg/pulseppg/experiments/out/pulseppg/checkpoint_best.pkl`;
  SHA-256 `485ade5033b3baa9b82e252fc131042dda897d7bdfc0d1a030d9746a1e98857c`.
- Extracted MotifDist checkpoint:
  `artifacts/pulseppg/pulseppg/experiments/out/motifdist/checkpoint_best.pkl`;
  SHA-256 `cb3fb8f9e33cad409fed875be94e09c17b25eca43133d3331f1c1a8ab7218ac7`.

Both checkpoint files have `.pkl` extensions. They were extracted but **not
deserialized or executed** during Step 1. Downloaded binaries are ignored by
the root `.gitignore`; code provenance remains separate from model artifacts.

## Step 2: checkpoint role and dataset exposure (complete, bounded audit)

### Checkpoint identity and safe inspection

- Both `.pkl` files are PyTorch ZIP archives. We read only ZIP member names and
  pickle opcodes with the standard-library `zipfile` and `pickletools` modules;
  we did **not** call `torch.load` or otherwise deserialize the downloaded files.
- The main `pulseppg/checkpoint_best.pkl` static pickle structure has top-level
  keys `net`, `optimizer`, `test_loss`, and `epoch` (recorded epoch 5). Its `net`
  includes ResNet-style names such as `first_block_conv.conv.weight` and
  `basicblock_list.0.conv1.conv.weight`. There is no `trained_net` or HR-head
  top-level entry. This agrees with `RelCon_Model.create_state_dict`, which
  saves the pretraining encoder and optimizer, not a downstream predictor.
- The separate `motifdist/checkpoint_best.pkl` has the same top-level key types
  (recorded epoch 15), with cross-attention/dilated-convolution names such as
  `q_func.layer0.dilated_conv.weight`. `RelCon_Model` loads it as a learned
  similarity guide during self-supervised encoder training.
- The official downstream HR linear-probe implementation freezes the encoder,
  extracts embeddings, and trains a separate standardised Ridge regression
  head. That head is saved in separate evaluation outputs (including joblib
  files), not in the released main checkpoint. The official `RelCon_Model`
  constructor still loads MotifDist even for evaluation, so using their full
  wrapper may require both files; direct encoder-only inference should not
  require MotifDist, subject to the Step-01 compatibility test.
- Static inspection establishes checkpoint *role*, not numerical loading
  compatibility or that the archive's weights came from precisely the paper's
  stated training run. Those require a controlled, trusted-model load and smoke
  test. The paper states a pretraining batch size of 64; the released
  `PulsePPG_expconfigs.py` currently sets 16, another reason not to claim exact
  training-run reconstruction from configuration alone.

### Reported pretraining data versus our RQ1 targets

- The [authors' paper](https://arxiv.org/html/2502.01108) identifies **MOODS**
  (Mobile Open Observation of Daily Stressors) as pretraining data: 50-Hz,
  4-minute PPG segments from a 100-day Fossil Sport wearable field study.
  The source study had 122 participants; PPG from 120 was used. The paper
  states 84/18/18 subject-wise train/validation/test groups and says the
  pretraining test group was not used for encoder pretraining. It also states
  that no signal-specific filtering was used, with person-specific global
  z-normalisation instead. The local pretraining loader additionally applies
  an upper clipping threshold when configured; the exact author-side
  normaliser pickle and raw-data preparation are not distributed here.
- The paper explicitly lists **PPG-DaLiA** and **WESAD** as downstream
  evaluation datasets, not MOODS pretraining data. **BIDMC** and **PTT-PPG**
  are not listed among that paper's pretraining/evaluation datasets. Thus our
  four RQ1 datasets are **not reported** as inputs to encoder pretraining;
  PPG-DaLiA and WESAD have nevertheless been used by the authors for
  downstream model evaluation and development. This is *evaluation exposure*,
  not evidence that their HR labels are in the released encoder weights.
- The released checkpoint contains no source-dataset or participant manifest.
  Therefore we cannot prove record- or participant-level non-overlap, exclude
  all publication-stage model-selection influence, or certify that an unseen
  RQ1 target was never considered by the original authors. Our own LODO runs
  must use only source-dataset HR labels to fit/select our probe, with no
  target-domain adaptation or tuning.

**Admissible claim:** “The released Pulse-PPG checkpoint is a self-supervised
encoder checkpoint reportedly pretrained on MOODS, a field study separate
from the four named RQ1 datasets; PPG-DaLiA and WESAD were author-side
downstream evaluation datasets. Exact participant non-overlap and checkpoint
training provenance cannot be independently verified from the release.”

Step 2 is complete at this evidence level. A controlled, restricted-loader
encoder smoke test and source-only input-compatibility gate have since passed;
see `../01_input_compatibility/README.md`. This does not upgrade the
participant-overlap claim or establish full-benchmark HR performance.
