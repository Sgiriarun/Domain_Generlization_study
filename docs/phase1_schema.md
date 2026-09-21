# Phase 1 — Common data schema

## Purpose

Phase 1 defines the in-memory contract that every dataset loader must satisfy.
It prevents dataset-specific storage details from leaking into preprocessing and
experiments. It does not read a dataset; reading begins only after this phase is
reviewed and accepted.

## Scientific decisions

### Native data are preserved

The schema stores raw values and native sampling rates. Filtering, resampling,
normalization, HR derivation, windowing, and quality rejection are excluded.
This separation lets us audit the original signal and test every transformation.

### Signals use `(samples, channels)`

Every `SampledSignal` has an explicit two-dimensional shape. A single-channel
PPG is `(N, 1)`, not an ambiguous `(N,)`; PTT-PPG can retain several channels as
`(N, 6)`. Channel names and units must match the second dimension.

### Every sequence retains its own clock

PPG, ECG, ACC, HR, and condition labels can have different sampling rates. Each
object therefore stores `fs_hz` and `t0_s`. Array index equality across signals
must never be mistaken for time alignment.

### R-peaks remain ECG sample indices

`EventSeries` stores strictly increasing integer indices and the ECG sampling
rate. This retains the precision of supplied annotations. Times in seconds are
derived when needed rather than stored as a second potentially inconsistent copy.

### Supplied HR is not common reference HR

`provided_hr` preserves a dataset's published/monitor HR for validation. The
common ECG-derived HR will be created later using one documented method, so the
two cannot be silently confused.

### Subject identity differs from recording identity

`subject_id` controls data splitting; `record_id` identifies a file/session.
This is essential for BIDMC, whose 53 records represent 46 source patients.

### Auxiliary signals are retained but optional

`extra_signals` can contain ACC, respiration, temperature, or other modalities.
They support domain-shift and signal-quality analysis, while `ppg` remains the
only required primary model input.

### Structural and quality validation are separated

The schema rejects impossible structures such as negative sampling rates,
incorrect channel counts, unordered R-peaks, and annotations outside ECG bounds.
It intentionally allows NaN/Inf values because Phase 2 validation must count and
report real quality problems instead of hiding them during loading.

## Public objects

- `SampledSignal`: regularly sampled one- or multi-channel signal.
- `EventSeries`: annotations such as ECG R-peak indices.
- `SampledValues`: regularly sampled HR targets or condition labels.
- `PPGRecord`: one recording with identity, provenance, signals, and annotations.
- `SchemaError`: explicit structural-contract failure.

## Phase boundary

Phase 1 is complete when its unit tests pass and the schema is accepted. Dataset
loaders, duration/alignment tolerances, signal-quality rules, and transformations
belong to later phases and must not be added here.
