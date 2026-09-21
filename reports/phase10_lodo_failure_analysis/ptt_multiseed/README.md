# PTT-PPG full-LODO replication across seeds 17, 29 and 43

All three runs use the same 15,982 untouched PTT target windows and the same frozen
source/validation protocol. Matching keys, timestamps and reference labels agree.

- Mean LODO MAE: **14.607 ± 0.357 bpm** (training-seed SD).
- Mean signed bias: **+11.745 ± 0.505 bpm**.
- Mean correlation: **0.357 ± 0.054**.
- **20/22 subjects** are worse than the fixed within-dataset reference in all three LODO seeds.
- Seed-averaged subject-macro gap: **+9.927 bpm**; subject-bootstrap
  95% interval **+3.804 to +14.788 bpm**.

The low-HR positive-bias pattern repeats across seeds. Therefore, the main PTT transfer
failure and its direction are not peculiar to seed 17. The SD across only three seeds is
descriptive and should not be presented as a precise population confidence interval.
