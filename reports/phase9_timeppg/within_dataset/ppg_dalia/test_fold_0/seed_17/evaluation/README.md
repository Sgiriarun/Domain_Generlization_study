# Evaluation: PPG-DaLiA TimePPG fold 0, seed 17

## Validity

- Training stopped at epoch 45 after 20 epochs without validation-loss improvement.
- Epoch 25 was selected using validation data only.
- Training, validation, and test subjects are disjoint.
- The saved test set was evaluated once after checkpoint selection.
- This is one fold and one seed; it is not the final within-dataset estimate.

## Same-window method comparison

| Method | MAE (bpm) | RMSE (bpm) | Pearson r | Within +/-5 bpm | Bias (bpm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TimePPG | 5.333 | 10.232 | 0.8639 | 72.35% | 2.237 |
| FFT | 12.214 | 20.358 | 0.5901 | 54.43% | -6.633 |

TimePPG reduces MAE by 6.881 bpm (56.3%) on exactly the same three test subjects and windows.

## Interpretation boundary

The improvement demonstrates that the learned temporal representation is much
stronger than selecting one dominant PPG frequency on this fold. It does not yet
establish final within-dataset performance, statistical significance, or
cross-dataset generalisation. Those claims require all subject folds and the
four LODO experiments.
