# Phase 2 — Real-dataset loaders

## Goal

Phase 2 translates each downloaded source format into the Phase 1 `PPGRecord`
contract. It does not make the signals comparable yet. Filtering, resampling,
ECG peak detection, HR derivation, windowing, and quality rejection remain later
phases.

This boundary matters scientifically: a loader should preserve what the dataset
actually contains. Otherwise a later result cannot distinguish source data from
our analytical decisions.

## Dataset mappings

| Dataset | Source unit | PPG | ECG/reference | Conditions and auxiliary data |
|---|---|---|---|---|
| BIDMC | one WFDB ICU record | `PLETH`, 125 Hz | lead `II`, 125 Hz; monitor HR, 1 Hz | respiration, other ECG leads; source patient retained as `subject_id` |
| WESAD | one subject pickle | wrist `BVP`, 64 Hz | chest `ECG`, 700 Hz | protocol labels, wrist/chest ACC and other signals |
| PTT-PPG | one subject–activity WFDB record | `pleth_1`–`pleth_6`, 500 Hz | ECG, 500 Hz; verified R-peak annotations | sit/walk/run, ACC/gyroscope and sensor channels |
| PPG-DaLiA | one subject pickle | wrist `BVP`, 64 Hz | chest ECG, 700 Hz; R-peaks and provided HR | activity labels, wrist/chest ACC and other signals |

## Important choices

1. **Native clocks are retained.** PPG, ECG, HR, labels, and ACC are not forced
   into one array because they have different sampling frequencies.
2. **BIDMC is split by patient, not record.** Several recordings can originate
   from one MIMIC patient; using the source patient as `subject_id` prevents
   leakage in later subject-level splits.
3. **All six PTT-PPG channels are retained.** Channel selection is an experiment
   decision, not a loading decision.
4. **Dataset HR is provenance, not automatically the common target.** BIDMC
   monitor HR and PPG-DaLiA supplied HR are stored as `provided_hr`. Phase 4 will
   define the common ECG-derived reference consistently.
5. **No missing values are silently repaired.** Their amount and location must
   be reported during validation and signal-quality analysis.
6. **Exact duplicate PPG-DaLiA R-peak indices are provenance-tracked.** An event
   series cannot contain two distinct beats at the same ECG sample. The loader
   collapses only exact duplicates, stores their source count in metadata, and
   Phase 3 reports each affected record as a warning.

## Verification levels

- Phase 1 schema unit tests use tiny synthetic arrays to isolate contract rules.
- Phase 2 smoke tests load one real record per dataset and verify known channels,
  rates, identities, and annotations.
- Phase 3 will audit every record and produce a machine-readable validation
  report before any preprocessing begins.

Run the current checks from the project root:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```
