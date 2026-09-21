# Phase 12E: frozen Pulse-PPG failure localisation

The analysis pairs **all 136,625 primary windows** by dataset, recording, person and window index. Each window has the same ECG-derived HR under four frozen comparisons: (1) Pulse-PPG three-fold within-dataset, (2) Pulse-PPG train-three/test-one LODO, (3) TimePPG three-fold within-dataset and (4) seed-17 TimePPG LODO. It makes no training or model-selection changes. Run:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python scripts/models/pulseppg/analyse_full_linear_probe_failures.py
```

The [HR-range figure](hr_range_failure.png) and CSVs in this folder come from frozen predictions. `paired_gap_bpm` is Pulse-PPG LODO absolute error minus its within-dataset absolute error on **the same window**; positive means transfer added error. Signed bias is prediction minus ECG HR: positive is too high.

## Strong observations

1. **PTT-PPG transfer failure is widespread and concentrated at low HR.** Pulse-PPG LODO MAE is **14.45 bpm** versus **5.99 bpm** within-dataset (paired gap **+8.46**); **21/22 people** have a positive subject-level gap. At **60–80 bpm**, LODO MAE is **19.30** and bias **+17.02 bpm**, versus within MAE **4.40** (5,169 windows, 21 people). The `<60` result is more extreme (63.63 MAE), but has only 279 windows from four people. The frozen source-training sets contain **37,174 windows from 28 source people at 60–80 bpm** and 6,848 below 60; missing source low-HR labels alone cannot explain the 60–80 failure. Source counts do not prove that source *waveforms* match PTT at a given HR.

2. **This is not simply an activity or noisy-spectrum effect.** PTT low-HR overprediction occurs in sitting, walking and running: their `<80` LODO biases are **+17.40**, **+27.08** and **+19.53 bpm**, respectively. Their people and HR distributions differ, so this is not a causal activity comparison. In PTT's highest spectral-concentration tertile, LODO MAE is still **14.87 bpm** with a **+8.97 bpm** paired gap (5,328 windows, 22 people). Spectral concentration is a proxy, not a complete signal-quality measure.

3. **DaLiA has the opposite signed failure: high HR is predicted too low.** At 120–140 bpm, Pulse-PPG LODO MAE is **29.37** and bias **−29.05 bpm** (4,881 windows, 14 people); at ≥140 bpm, MAE is **51.17** and bias **−51.16 bpm** (2,229 windows, 11 people). The train-three source sets contain only **1,831** windows at 120–140 and **226** at ≥140; sparse source HR support is a plausible contributor. Its pooled LODO-minus-within gap of **+0.36 bpm** hides a **+10.18 bpm** gap at 120–140 and **+16.68 bpm** at ≥140, offset by better LODO results below 100 bpm. The within-dataset Ridge head also performs poorly at high HR, so this is not solely a transfer problem.

4. **Pulse-PPG does not uniformly beat TimePPG.** On the same windows Pulse-PPG LODO MAE is higher by **3.58 bpm** for BIDMC, **4.03** for WESAD and **3.12** for DaLiA. PTT is slightly lower overall (**14.45 vs 15.01 bpm**), but Pulse-PPG is much worse below 60 bpm and better at 60–100 bpm. This is a frozen-encoder **linear-head** benchmark, not a verdict on all ways of adapting Pulse-PPG.

5. **Prediction-range compression is measurable.** For DaLiA LODO, the descriptive slope of prediction versus reference is **0.462** (1 would track the HR range without compression); WESAD is **0.619**, PTT **0.427**. These are post-hoc summaries, not calibrations applied to predictions. A single slope cannot distinguish waveform shift, HR-support imbalance, encoder context mismatch or head limitation.

## Interpretation and boundary

The best-supported diagnosis is **target- and HR-dependent miscalibration**. For PTT, plentiful source low-HR examples, positive bias across activities and persistent error in high-spectral-concentration windows make *label scarcity or simple spectral noise alone* inadequate explanations. HR-conditional waveform/site differences (`P(X|Y)` shift) are a **plausible mechanism, not a proven cause**. For DaLiA, sparse high-HR source coverage and strong high-HR underprediction support a **tail-coverage problem**, alongside substantial familiar-domain model error.

## Presentation-ready explanation of the four curves

> We compare both models in familiar and unseen datasets. For one model, the distance between its dashed and solid lines shows the transfer penalty. For one protocol, the distance between the green and orange lines shows which model has lower error in that heart-rate range.

- Green circles are Pulse-PPG; orange squares are TimePPG.
- Dashed lines are within-dataset tests; solid lines are LODO tests.
- High error in both protocols suggests a broader model/input limitation.
- Low within-dataset error followed by high LODO error shows transfer-specific deterioration.
- Similar within and LODO errors indicate little additional transfer penalty, but the common error may still be unacceptable.
- `n` is the shared number of ECG-labelled target windows in that HR bin.

**Approved conclusion:** The full comparison separates model weakness from transfer weakness: PTT's low-HR deterioration is mainly transfer-specific, whereas some extreme-HR errors are already present during familiar-domain testing.

The four conditions use exactly matched target windows, but the model families have different representations and training procedures. Therefore, this is a fair performance comparison, not causal proof that a particular architectural component created the difference.

The released Pulse-PPG encoder was pretrained on much longer 50-Hz segments, whereas this comparison feeds 8-second segments resampled from 64 Hz, and only a Ridge head is learned. **No experiment here isolates 8-second context, resampling, the frozen encoder or the linear head as the cause**. To identify the lever, keep the same frozen targets and source-only tuning, then compare (a) 8 versus longer context, (b) linear versus nonlinear head, and (c) frozen versus carefully fine-tuned encoder. Repeat with several seeds and subject-level intervals before claiming a stable ranking.

## Files

- `target_metrics.csv`: pooled target-level paired comparisons.
- `hr_bins_metrics.csv`: HR-range MAE, signed bias, window and person counts.
- `subjects_metrics.csv`: one row per person for prevalence checks.
- `conditions_metrics.csv`, `ptt_activity_hr_metrics.csv`: condition and PTT activity × HR summaries.
- `spectral_tertiles_metrics.csv`: descriptive signal-proxy strata.
- `source_hr_support.csv`: source-training windows and people by target/HR bin.
- `prediction_range.csv`: prediction-versus-reference slopes and spread.
- `overall_model_protocol_comparison.png`: pooled MAE for both models under within-dataset and LODO evaluation.
- `hr_range_failure.png`: four target panels comparing Pulse-PPG and TimePPG under both within-dataset and LODO evaluation. Colour denotes the model; dashed lines denote within-dataset tests and solid lines denote LODO tests.
- `lodo_signed_bias_by_hr.png`: shows whether each LODO model predicts too high or too low in every HR range.
- `subject_transfer_gap.png`: shows whether transfer becomes better or worse for each person rather than only in the pooled average.
- `ptt_failure_checks.png`: tests whether the PTT failure remains across activities, HR strata and spectral-concentration groups.
