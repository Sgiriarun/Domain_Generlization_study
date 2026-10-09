# Phase 12 frozen outcome — Pulse-PPG representation and final-block adaptation

Status: **primary complete and frozen for analysis**

## Protocol

The released self-supervised Pulse-PPG encoder was evaluated on all 136,625 primary 8-second windows. The final experiment warm-started the source-selected nonlinear HR head, froze the stem and residual blocks 0–10, and updated only residual block 11 plus the head. Learning rate and early stopping used source-validation people only. No held-out LODO target signal, label or statistic entered training or selection. Four within-dataset targets (three subject folds) and four train-three/test-one LODO targets were completed for seeds 17, 29 and 43 on CUDA.

## Primary results

| Target | Within MAE | LODO MAE | Gap (LODO − within) |
|---|---:|---:|---:|
| BIDMC | 3.15 ± 0.07 | 3.18 ± 0.51 | +0.03 ± 0.43 |
| WESAD | 5.82 ± 0.04 | 5.68 ± 0.15 | -0.14 ± 0.16 |
| PTT-PPG | 5.34 ± 0.04 | 12.18 ± 0.45 | +6.84 ± 0.43 |
| PPG-DaLiA | 7.51 ± 0.05 | 8.85 ± 0.30 | +1.34 ± 0.35 |

Values are pooled-window MAE, mean ± sample SD across three seeds. Within-dataset predictions pool the three subject-disjoint folds before calculating each seed's metric.

## What final-block fine-tuning changed

- **BIDMC:** LODO MAE changed from 3.64 to 3.18 bpm (-0.45); within-dataset MAE changed by -0.39 bpm.
- **WESAD:** LODO MAE changed from 6.48 to 5.68 bpm (-0.80); within-dataset MAE changed by -0.52 bpm.
- **PTT-PPG:** LODO MAE changed from 12.15 to 12.18 bpm (+0.04); within-dataset MAE changed by -0.12 bpm.
- **PPG-DaLiA:** LODO MAE changed from 9.59 to 8.85 bpm (-0.75); within-dataset MAE changed by -1.01 bpm.

Final-block adaptation therefore improves BIDMC, WESAD and DaLiA LODO MAE, but leaves PTT essentially unchanged. Better fitting of the source domains is not sufficient to remove the PTT acquisition/site-related failure.

## Failure localisation

- PTT retains the largest gap (+6.84 bpm). Across seed–person evaluations, the transfer penalty is positive in about 95.5% of cases.
- PTT at 60–80 bpm has mean LODO MAE 20.01 bpm, signed bias +19.87 bpm and paired transfer gap +16.39 bpm (5169 windows per seed). The transferred model systematically predicts this range too high.
- DaLiA at 120–140 bpm has mean LODO MAE 24.58 bpm and signed bias -23.68 bpm (4881 windows per seed), retaining the high-HR underprediction pattern.
- These are post-hoc associations. They localise failure but do not prove that one device, site, activity or physiological factor is the sole cause.

## Selection and integrity

All 48 expected runs, prediction files, subject-metric files, checkpoints and selection records are present. Every selection record declares `target_data_used_for_selection: false`. Fine-tuning was selected over the no-adaptation control in 47/48 runs; one within-PTT run retained the frozen source-selected control.

The frozen provenance records CUDA, batch size 64, seeds 17/29/43, the two predeclared learning rates, and matching hashes for the encoder, manifest and subject roles.

## Approved conclusion

> Self-supervised Pulse-PPG pretraining and source-only final-block adaptation improve several targets, but do not eliminate target-dependent generalisation failure. PTT retains a large, widespread low-HR overprediction gap, while DaLiA retains high-HR underprediction. Representation quality and modest source-only adaptation are therefore helpful but insufficient; RQ2 still requires controlled tests of domain-generalisation mechanisms that preserve HR structure while reducing acquisition-specific dependence.

This conclusion applies to the released Pulse-PPG checkpoint, 8-second inputs resampled to 50 Hz, the declared nonlinear head and final-block adaptation protocol. It is not a claim that every self-supervised or fully fine-tuned PPG foundation model will behave identically.

## Files

- `partial_seed_metrics.csv`: complete per-target/per-seed metrics.
- `partial_generalisation_gaps.csv`: paired within/LODO values and gaps.
- `model_ladder.csv`: FFT, TimePPG, Ridge, nonlinear-head and final-block conditions.
- `hr_bin_metrics.csv`, `subject_metrics.csv`, `condition_metrics.csv`: post-hoc failure localisation.
- `selection_summary.csv`: source-validation selection behaviour.
- `figures/model_ladder.png`: complete model/protocol comparison.
- `figures/fine_tuning_effect.png`: matched change from frozen nonlinear to final-block adaptation.
- `figures/hr_range_error_and_bias.png`: target-specific HR-range error direction.
- `figures/subject_transfer_prevalence.png`: prevalence of transfer deterioration.
