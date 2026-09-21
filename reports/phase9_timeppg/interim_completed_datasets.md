# Interim three-fold TimePPG findings

Status date: 2026-09-09. BIDMC and PTT-PPG are complete; WESAD and PPG-DaLiA
three-fold runs are still in progress. These are within-dataset unseen-subject
results for seed 17, not leave-one-dataset-out results.

| Dataset | Subjects | Windows | TimePPG pooled MAE | Subject-macro MAE | FFT same-window MAE | TimePPG reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BIDMC | 46 | 12,561 | 1.916 bpm | 1.830 bpm | 3.962 bpm | 51.6% |
| PTT-PPG | 22 | 15,982 | 4.694 bpm | 4.695 bpm | 10.262 bpm | 54.3% |

Additional pooled metrics:

- BIDMC: RMSE 4.342 bpm, Pearson r 0.9457, 90.57% within +/-5 bpm, bias
  +0.304 bpm.
- PTT-PPG: RMSE 11.507 bpm, Pearson r 0.6241, 81.28% within +/-5 bpm, bias
  -1.784 bpm.

## Important observations

- TimePPG improves substantially over FFT on both completed datasets.
- BIDMC is consistently easier: fold MAEs are 2.055, 1.535, and 2.120 bpm.
- PTT fold MAEs are 3.861, 2.249, and 8.103 bpm, showing much greater fold
  instability.
- PTT subject `s2` has 44.268 bpm MAE and heavily influences fold 2 and the
  overall RMSE. The next-worst PTT subjects are `s14` at 11.628 bpm and `s9` at
  7.085 bpm. This must be investigated rather than hidden by an average.
- PTT activity MAE is 3.205 bpm for sitting, 4.711 for running, and 6.173 for
  walking. Walking remains the hardest activity on average.

## Interpretation boundary

These results establish familiar-domain performance on unseen subjects. They do
not test transfer to an unseen dataset/device. PTT `s2` requires signal, label,
and prediction inspection before making a causal claim about its failure.

