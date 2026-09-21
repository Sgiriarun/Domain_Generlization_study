# PPG-DaLiA five-fold TimePPG result — seed 17

## Integrity

- All 5 runs completed with disjoint training, validation, and test subjects.
- All 15 subjects appear in exactly one test fold.
- The combined test predictions contain 64,697 unique windows, matching the frozen PPG-DaLiA manifest.
- Checkpoints were selected by validation loss; each test fold was evaluated once.

## Results

- Subject-macro MAE: **5.970 +/- 2.714 bpm** across 15 subjects.
- Subject-bootstrap 95% CI for macro MAE: **4.672 to 7.330 bpm**.
- Pooled-window MAE: **5.890 bpm**; RMSE: **11.799 bpm**.
- Pooled Pearson r: **0.8581**; within +/-5 bpm: **72.67%**.
- Same-window FFT pooled MAE: **17.620 bpm**.
- TimePPG pooled MAE reduction relative to FFT: **11.730 bpm (66.6%)**.
- Hardest pooled activity: **stairs (17.924 bpm MAE)**.

## Correct interpretation

This is the complete five-fold within-dataset PPG-DaLiA result for one random
seed. It supports unseen-subject generalisation within PPG-DaLiA and establishes
the familiar-domain reference for its later LODO generalisation gap. It does not
demonstrate transfer to a different dataset/device. Seed sensitivity and the
four leave-one-dataset-out experiments remain outstanding.
