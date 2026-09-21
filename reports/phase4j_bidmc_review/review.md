# BIDMC difficult-record review

Four high-error records were reviewed using separated 8-second episodes. Each
plot shows raw and filtered lead-II ECG, peaks from three detectors, PLETH, and
the 1 Hz monitor HR around the window.

`monitor_lag_sensitivity.csv` tests shifts from -10 to +10 seconds only as a
descriptive check for monitor delay. The best shift is not adopted as a fitted
parameter because monitor processing and latency are undocumented and differ by
record.

Interpretation must separate: detector disagreement, agreement of all detectors
on ECG beats, ECG/PPG signal distortion, and mismatch with monitor HR. Monitor
HR is an independent clinical comparator, not R-peak ground truth. No record or
window is removed by this review.

## Visual findings

- `bidmc41`: all three detectors agree at approximately 109 bpm and their peaks
  align with visible ECG complexes, while monitor HR is approximately 54 bpm.
  The near 2:1 relationship suggests monitor/beat-counting disagreement rather
  than an isolated Emrich failure. PLETH has a different dominant pulse pattern,
  so ectopy or pulse deficit is plausible but cannot be diagnosed here.
- `bidmc45`: all detectors agree near 47 bpm and the slow ECG/PLETH rhythm is
  visually consistent. Monitor HR changes sharply around the inspected windows,
  making monitor smoothing or latency a plausible contributor.
- `bidmc26`: detectors select different events in irregular ECG morphology.
  These windows are genuine detector-disagreement cases and should receive an
  ECG-reference quality flag.
- Descriptive monitor shifts do not resolve `bidmc41`. No global monitor lag is
  adopted.

These are signal-level interpretations, not clinical diagnoses.
