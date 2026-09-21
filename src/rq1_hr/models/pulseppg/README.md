# Pulse-PPG model integration

This package will contain the minimal adapter between the released Pulse-PPG
encoder and the frozen RQ1 PPG manifests. It must not contain experimental
outputs or downloaded third-party weights.

## Integration boundary

- Input comes from the frozen Phase-7 manifests.
- The initial experiment uses the released self-supervised encoder weights.
- The primary downstream model freezes the encoder and learns only a linear HR
  regression head from ECG-derived source labels.
- A held-out LODO target must never enter head training, early stopping,
  hyperparameter selection or normalization fitting.
- External code commit, weight checksum, licence and preprocessing are recorded
  under `reports/phase12_pulseppg/00_provenance_audit/`.

Implementation starts only after Task 12A is frozen.

