# Phase 11: controlled PTT sensor-site transfer — seed 17

## Design

- The completed three-fold TimePPG models were trained on distal `pleth_1`.
- Each model was evaluated on familiar distal `pleth_1` and synchronized paired
  proximal `pleth_4` for the exact same held-out subjects and 8-second windows.
- Subject, activity, recording time, reference HR, split and model checkpoint are
  held fixed. The changed input is sensor placement/channel within pair 1/4.
- All 15,982 primary PTT windows are covered once across the three test folds.

## Result

- Distal familiar-site MAE: **4.694 bpm**.
- Proximal unseen-site MAE: **7.805 bpm**.
- Pooled window-level site gap: **+3.111 bpm**.
- Subject-macro site gap: **+3.109 bpm**,
  subject-bootstrap 95% interval **+2.009 to +4.204 bpm**.
- Proximal is worse for **20/22 subjects**.

## Interpretation boundary

This is a controlled one-direction transfer diagnostic: a `pleth_1`-trained model
is challenged with paired `pleth_4`. It provides stronger evidence that placement/
channel shift changes model usability, but it is not yet a symmetric site experiment.
Training on `pleth_4` and reversing the test direction is required next. Wavelength
names remain neutral because the local documentation versions conflict, although the
1/4 pairing and distal/proximal placement are documented.
