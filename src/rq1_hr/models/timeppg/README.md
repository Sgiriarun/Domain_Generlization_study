# TimePPG supervised baseline

This model folder is reserved for the reference-faithful PPG-only adaptation of
the published TimePPG architecture.

Primary sources:

- Burrello et al., *Embedding Temporal Convolutional Networks for
  Energy-Efficient PPG-Based Heart Rate Monitoring*, ACM Transactions on
  Computing for Healthcare, DOI: 10.1145/3487910.
- Official authors' repository: https://github.com/eml-eda/q-ppg
- Audited repository commit: `ddf3866da6d5f9dda4da7d7884b4f1f3b809a6ba`
- Repository license: Apache-2.0.

This folder must not describe our experiment as an exact paper reproduction.
The paper/repository uses PPG plus three accelerometer axes, 32-Hz inputs, and
PPG-DaLiA leave-one-subject-out evaluation. RQ1 instead freezes PPG-only input,
64 Hz, and four-dataset leave-one-dataset-out evaluation.

The architecture and training audit is stored in
`docs/model_references/timeppg_reference_audit.md`. Implementation must retain
source attribution at file level and document every deliberate deviation.

## Local files

- `model.py`: attributed TimePPG-Big float topology adapted to `(batch, 1, 512)`.
- `dataset.py`: lazy reader for frozen Phase 7 signal slices.
- `configs/models/timeppg.toml`: audited architecture and training settings.
- `scripts/models/timeppg/check_pipeline.py`: non-reportable Phase 9A smoke test.
- `reports/phase9_timeppg/pipeline_check/`: generated check, history, and smoke
  checkpoint.
