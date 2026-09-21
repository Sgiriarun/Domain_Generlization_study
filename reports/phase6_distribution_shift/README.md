# Phase 6: distribution-shift analysis

## Design safeguards

- Primary comparison uses one channel per dataset; PTT-PPG uses the predeclared first source channel (`pleth_1`). All six PTT channels are analysed separately.
- Only every fourth 2-second-step window is eligible, producing non-overlapping 8-second windows.
- Sampling is capped at 200 windows per subject and balanced to the same number per dataset.
- Dataset classifiers are evaluated with five-fold stratified subject-group splitting: a subject never appears in both train and test.
- Conditional comparison matches the four datasets within 10-bpm HR bins before testing dataset separability.

## HR support

| dataset | windows | minimum | q05 | median | q95 | maximum |
| --- | --- | --- | --- | --- | --- | --- |
| PPG-DaLiA | 3000 | 42.3078 | 59.3683 | 85.1645 | 134.8808 | 183.1537 |
| PTT-PPG | 3000 | 48.6418 | 65.5915 | 85.4798 | 107.8644 | 145.7832 |
| WESAD | 3000 | 46.3475 | 57.0439 | 75.5169 | 109.1829 | 151.3638 |
| BIDMC | 3000 | 46.9368 | 67.0936 | 88.9460 | 114.8423 | 129.1067 |

## Dataset identity prediction

| representation | mean_accuracy | sd_accuracy | mean_macro_f1 | sd_macro_f1 |
| --- | --- | --- | --- | --- |
| per_window_zscore | 0.6464 | 0.0115 | 0.6452 | 0.0142 |
| raw_scale | 0.7695 | 0.0440 | 0.7699 | 0.0378 |

Chance accuracy is 0.25. `raw_scale` includes amplitude features. `per_window_zscore` removes mean/scale information and uses only normalized waveform and spectral features. Accuracy above chance after HR matching and z-normalization is evidence that dataset/device characteristics remain in PPG shape; it is not by itself proof of causal concept shift or model failure.

## Pairwise normalized-shape distances after HR matching

| dataset_a | dataset_b | mean_standardized_wasserstein | max_standardized_wasserstein |
| --- | --- | --- | --- |
| PPG-DaLiA | PTT-PPG | 0.5403 | 0.9605 |
| PPG-DaLiA | WESAD | 0.1940 | 0.2888 |
| PPG-DaLiA | BIDMC | 0.9182 | 1.5000 |
| PTT-PPG | WESAD | 0.6349 | 0.9369 |
| PTT-PPG | BIDMC | 0.4935 | 1.0360 |
| WESAD | BIDMC | 1.0141 | 1.5334 |

## Main findings

- The four datasets do not provide identical HR support. PPG-DaLiA has the broadest sampled upper range; BIDMC is narrower in this balanced sample.
- Raw-scale dataset classification reaches about 77% accuracy, showing strong device/dataset signatures.
- Per-window z-normalization reduces accuracy to about 65%, but does not remove dataset identity; waveform and spectral shift remain.
- PPG-DaLiA and WESAD are the closest normalized-shape pair in this feature set, consistent with their shared Empatica E4 wrist device family. WESAD and BIDMC are the most separated pair.
- PTT channel spectral concentration changes across both channel and activity. The additional channels should remain a secondary robustness analysis rather than be treated as duplicate measurements.

## AI-engineering visual guide

- `heterogeneous_domain_dashboard.png` combines label support, amplitude, spectral complexity, normalized PCA, dataset-classifier accuracy, and pairwise shape distance.
- `same_hr_different_domains.png` controls HR to 80–90 bpm and shows representative normalized waveforms and spectra. It makes clear why an HR model may learn device/domain cues in addition to cardiac cues.
- `normalized_feature_pca_by_device.png` colours the same HR-matched normalized feature space by physical recording device and uses marker shape to retain dataset identity.
- `ptt_sensor_site_pca.png` uses PTT-PPG only: colour represents distal versus proximal left-index-finger placement, while marker pairs channels 1/4, 2/5, and 3/6. Lines between paired centroids expose site displacement without mixing devices. Pair labels are used because the source README contains conflicting red/infrared names in its hardware overview and detailed channel list; the site mapping itself is consistent.
- `same_hr_figure_examples.csv` records the exact source window behind each example; the figure is not hand-picked without provenance.

## Interpretation boundaries

- Different HR distributions demonstrate target/label-support shift.
- Predictable dataset identity after matching HR demonstrates class-conditional waveform shift, formally a difference in P(X|Y).
- This analysis does not directly prove P(Y|X) concept shift. That requires the Phase 7 PPG-to-HR models and cross-dataset error analysis.
- Overlapping motion frequencies remain inside the cardiac band; preprocessing cannot erase them.
- Statistical tests on windows would overstate sample size, so inferential uncertainty must use subjects as the independent unit.
