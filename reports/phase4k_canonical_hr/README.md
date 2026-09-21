# Frozen Phase 4 canonical HR labels

## Decision

- Supplied R-peaks are the primary source for PPG-DaLiA and PTT-PPG.
- Emrich 2023 ECG detections are the primary source for WESAD and BIDMC.
- HR uses 8-second windows, 2-second steps, RR-midpoint assignment, 35–220 bpm limits, and at least four valid RR intervals.
- Detector disagreement above 10 bpm is retained as a review flag, not automatically excluded.
- BIDMC monitor HR and published PPG-DaLiA HR are comparison fields, not replacements for the selected ECG reference.

## Dataset summary

| dataset | records | subjects | windows | included_primary | review_recommended | primary_coverage_percent |
| --- | --- | --- | --- | --- | --- | --- |
| PPG-DaLiA | 15 | 15 | 64697 | 64697 | 0 | 100.000 |
| PTT-PPG | 66 | 22 | 15982 | 15982 | 0 | 100.000 |
| WESAD | 15 | 15 | 43385 | 43385 | 316 | 100.000 |
| BIDMC | 53 | 46 | 12561 | 12561 | 890 | 100.000 |

## Quality flags

| dataset | quality_flag | windows |
| --- | --- | --- |
| PPG-DaLiA | pass | 64697 |
| PTT-PPG | pass | 15982 |
| WESAD | pass | 43069 |
| WESAD | detector_disagreement | 316 |
| BIDMC | pass | 11671 |
| BIDMC | detector_disagreement | 890 |

The table preserves `subject_id` for subject-wise splitting and `record_id` for provenance. No PPG filtering, resampling, or model input construction occurs in Phase 4.
