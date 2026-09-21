# Phase 5: common PPG preparation

## Frozen processing

1. Load the complete native PPG recording.
2. Resample to 64 Hz using polyphase anti-alias filtering.
3. Apply a fourth-order 0.5–4 Hz Butterworth band-pass in zero phase.
4. Reference each 8-second window by exact 64 Hz sample indices (512 samples).
5. Preserve every source PPG channel; PTT-PPG therefore remains six-channel.
6. Record descriptive quality evidence without arbitrary motion rejection.

The 0.5–4 Hz band corresponds approximately to 30–240 beats/min and surrounds the frozen 35–220 bpm HR range. Filtering the complete recording, rather than each window separately, reduces artificial boundary effects. No amplitude normalization is applied here because amplitude/device differences are part of the distribution-shift question; model-time normalization can be evaluated later as an explicit experiment.

## Why 0.5–4 Hz?

Frequency in hertz converts to heart rate by `bpm = Hz × 60`; therefore,
0.5–4 Hz represents approximately 30–240 bpm. This covers the project's accepted
35–220 bpm reference range with a small margin. The high-pass edge reduces the
DC component and slow baseline drift, while the low-pass edge reduces faster
noise outside the main pulse-rate band.

Band-pass filtering is common in PPG heart-rate work, but **0.5–4 Hz is not a
universal standard**. Published examples use nearby bands such as 0.4–5 Hz,
0.5–2.5 Hz, and 0.5–5 Hz depending on the population, sensor, task, and whether
waveform morphology must be preserved. A dual-wavelength PPG HR study also
searched for HR-related spectral components specifically within 0.5–4 Hz.

Our exact band was specified in the supervisor's common pipeline and is
physiologically consistent with our declared HR range. It is applied unchanged
to all datasets, preventing dataset-specific filter tuning from giving one
domain an unfair advantage.

Limitations: filtering cannot remove motion artifacts whose frequencies overlap
the cardiac band, and the 4 Hz upper edge suppresses higher-frequency waveform
morphology. It is suitable for the primary HR-estimation task, but should not be
assumed suitable for morphology, vascular-age, or detailed fiducial analysis.

Supporting literature:

- Charlton et al. describe band-pass filtering as common for baseline removal
  and report a 0.4–5 Hz example in wearable PPG HR processing:
  https://www.mdpi.com/2076-3417/14/17/7451
- Chen et al. use a 0.5–5 Hz passband for PPG peak/HR estimation:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8869811/
- Casson et al. use 0.5–4 Hz as the HR spectral search range in a PPG HR method:
  https://www.mdpi.com/1424-8220/22/24/9955
- Zieliński et al. use a fourth-order 0.5–2.5 Hz band-pass for wearable PPG HR:
  https://www.mdpi.com/1424-8220/20/6/1783

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
