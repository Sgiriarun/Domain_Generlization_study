# Phase 5: common PPG preparation

## Frozen processing

1. Load the complete native PPG recording.
2. Resample to 64 Hz using polyphase anti-alias filtering.
3. Apply a fourth-order 0.5–4 Hz Butterworth band-pass in zero phase.
4. Reference each 8-second window by exact 64 Hz sample indices (512 samples).
5. Preserve every source PPG channel; PTT-PPG therefore remains six-channel.
6. Record descriptive quality evidence without arbitrary motion rejection.

The 0.5–4 Hz band corresponds approximately to 30–240 beats/min and surrounds the frozen 35–220 bpm HR range. Filtering the complete recording, rather than each window separately, reduces artificial boundary effects. No amplitude normalization is applied here because amplitude/device differences are part of the distribution-shift question; model-time normalization can be evaluated later as an explicit experiment.

## Structural results

| dataset | records | subjects | window_channels | structurally_usable | median_std | median_flat_fraction | median_spectral_concentration |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPG-DaLiA | 15 | 15 | 64697 | 64697 | 54.2894 | 0.0000 | 0.6281 |
| PTT-PPG | 66 | 22 | 95892 | 95892 | 53.4402 | 0.0000 | 0.7208 |
| WESAD | 15 | 15 | 43385 | 43385 | 27.4520 | 0.0000 | 0.7091 |
| BIDMC | 53 | 46 | 12561 | 12561 | 0.1347 | 0.0000 | 0.8055 |

## Descriptive distributions

| dataset | std_q05 | std_median | std_q95 | concentration_q05 | concentration_median | concentration_q95 |
| --- | --- | --- | --- | --- | --- | --- |
| PPG-DaLiA | 15.4366 | 54.2894 | 140.9124 | 0.3962 | 0.6281 | 0.8516 |
| PTT-PPG | 10.8771 | 53.4402 | 329.2792 | 0.4047 | 0.7208 | 0.8885 |
| WESAD | 5.9266 | 27.4520 | 125.8310 | 0.4294 | 0.7091 | 0.8638 |
| BIDMC | 0.0570 | 0.1347 | 0.6037 | 0.5382 | 0.8055 | 0.9294 |

All window-channels are structurally usable. This does **not** mean that all are clean: spectral concentration, variability, flat differences, and edge-value fractions are retained for distribution analysis and later justified sensitivity rules. The amplitude statistics must not be compared as physical units across devices without normalization.

`signals/` stores each processed continuous recording once. `window_channel_quality.csv` provides its channel and `[start_sample_64hz, end_sample_64hz)` slice, HR label, condition, and quality measures.
