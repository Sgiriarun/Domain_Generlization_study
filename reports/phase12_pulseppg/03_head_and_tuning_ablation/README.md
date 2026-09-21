# Task 12D — head and encoder-adaptation ablation

## Current experiment: frozen encoder with a nonlinear head

This stage asks whether Pulse-PPG's frozen 512-dimensional representation
contains HR information that the linear Ridge head cannot extract. It does not
yet fine-tune the Pulse-PPG encoder.

The script `scripts/models/pulseppg/run_nonlinear_head_ablation.py`:

- reuses the complete 136,625-window embedding cache and frozen subject roles;
- compares two predeclared heads: `512→128→1` and `512→256→64→1`, with ReLU
  and 0.10 dropout;
- fits feature scaling and HR standardisation on training subjects only;
- trains with Huber loss and AdamW;
- early-stops and selects capacity using dataset-macro MAE on source-validation
  subjects only;
- opens within-dataset test subjects or the untouched LODO target only after
  selection;
- saves exact window predictions, subject metrics, checkpoints, training
  histories, hashes and provenance.

The candidate grid is intentionally small: this is a controlled test of head
nonlinearity, not an unrestricted neural-architecture search.

Primary local run:

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python \
  scripts/models/pulseppg/run_nonlinear_head_ablation.py \
  --protocol all --seeds 17 --device mps --batch-size 2048
```

Replication run on CUDA after the primary result passes integrity checks:

```bash
MPLCONFIGDIR=/tmp/matplotlib python \
  scripts/models/pulseppg/run_nonlinear_head_ablation.py \
  --protocol all --seeds 17,29,43 --device cuda --batch-size 4096
```

Do not interpret `/tmp/pulseppg_nonlinear_smoke`; that two-epoch run only
verified software execution. Scientific outputs belong under
`nonlinear_head/` and require the declared 60-epoch maximum and patience 8.

## Decision rule

- If the nonlinear head consistently improves source validation and frozen
  tests, HR information was present but not linearly accessible.
- If it does not improve, additional head capacity is not the main missing
  component.
- The next separate condition will unfreeze only the final encoder block. It
  must not be mixed into this result.

## Frozen nonlinear-head outcome — seeds 17, 29 and 43

The complete run contains 24 protocol/target/seed results: four within-dataset
tests and four LODO tests for each of three seeds. Exact prediction-count checks
passed for all 136,625 windows per seed. Source validation selected
`512→256→64→1` for every fold and LODO run when the validation MAE was averaged
over the three seeds.

| Target | Linear within | Nonlinear within | Linear LODO | Nonlinear LODO | Nonlinear gap |
|---|---:|---:|---:|---:|---:|
| BIDMC | 3.14 | 3.54 ± 0.06 | 5.44 | 3.64 ± 0.22 | +0.10 ± 0.21 |
| WESAD | 7.17 | 6.34 ± 0.03 | 8.66 | 6.48 ± 0.08 | +0.14 ± 0.10 |
| PTT-PPG | 5.99 | 5.47 ± 0.11 | 14.45 | 12.15 ± 1.26 | +6.68 ± 1.15 |
| PPG-DaLiA | 10.11 | 8.52 ± 0.05 | 10.47 | 9.59 ± 0.02 | +1.08 ± 0.06 |

Values are pooled MAE in bpm; nonlinear results are mean ± sample SD across
seeds. The nonlinear LODO MAE is lower than the Ridge LODO MAE for every target:
−1.80 BIDMC, −2.19 WESAD, −2.31 PTT and −0.88 DaLiA. It also reduces the
linear-probe generalisation gap for BIDMC, WESAD and PTT. However, it increases
the DaLiA gap from +0.36 to +1.08 bpm because its familiar-domain improvement
is larger than its LODO improvement. BIDMC familiar-domain MAE worsens by about
0.40 bpm.

PTT remains the unresolved transfer failure. Its mean nonlinear gap is +6.68
bpm and its LODO MAE varies from 11.18 to 13.56 bpm across seeds. Thus added
head capacity helps substantially but does not remove the target-specific
failure, and PTT is more seed-sensitive than the other targets.

### Approved conclusion

> A small nonlinear head accesses more useful HR information from the frozen
> Pulse-PPG representation and improves unseen-dataset MAE on all four targets.
> However, it does not eliminate target-dependent transfer failure: PTT retains
> a large and seed-sensitive generalisation gap, and TimePPG remains stronger
> on BIDMC, WESAD and DaLiA. Head linearity is therefore part of the limitation,
> but not the complete explanation.

This conclusion concerns the released frozen encoder with 8-second resampled
inputs. It does not establish that end-to-end Pulse-PPG fine-tuning fails or
that the nonlinear architecture itself causes the observed target differences.

Files:

- `nonlinear_head/run_summary.csv`: every protocol/target/seed result;
- `nonlinear_head/frozen_comparison.csv`: frozen comparison with Ridge and TimePPG;
- `nonlinear_head/nonlinear_head_comparison.png`: presentation-ready summary;
- split-level `source_validation_selection.json`: complete source-only training histories;
- seed folders: checkpoints, exact predictions and subject metrics;
- `nonlinear_head/run_provenance.json`: hashes and declared training settings.
