# Phase 9A TimePPG pipeline check

Status: **PASS**

This is a short real-data plumbing check, not a model comparison and not a
reportable HR-performance result. It used only PPG-DaLiA, with frozen fold
0 subjects for validation and different subjects for training.

- Official upstream commit: `ddf3866da6d5f9dda4da7d7884b4f1f3b809a6ba`
- Training windows: 252
- Validation windows: 126
- Subject overlap: none
- Batch input shape: `(126, 1, 512)`
- Output shape: `(126,)`
- Trainable parameters: 254,784
- Forward pass, backward pass, Adam update, checkpoint writing: passed
- Predictions finite: yes

Loss values in `training_history.csv` only demonstrate executable optimization
on a tiny subset. They must never be placed in the final results table.
