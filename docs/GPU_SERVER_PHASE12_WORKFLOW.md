# GPU server workflow — Phase 12 Pulse-PPG

## Goal

Run the already designed **partial final-block fine-tuning** experiment on the
GPU server. The model starts from the frozen Pulse-PPG encoder and the
source-selected nonlinear HR head. It updates only residual block 12 and the
HR head, selecting learning rate and early stopping from source-validation
subjects. No LODO target data are used for training or selection.

## What to transfer

Do **not** copy `datasets/raw/` (about 42 GB) for this stage. The run needs
only the canonical PPG files, manifests, frozen roles, Pulse-PPG checkpoint,
cached 50-Hz inputs, and Phase-12D head checkpoints. The supplied transfer
script copies roughly 0.7 GB plus code.

Before transfer, configure SSH access from the machine that can reach the GPU
server. The server setup email says access may require ECMSRDP as a jumpbox;
if so, run the transfer command from that authorised environment or configure
an SSH `ProxyJump` first.

```bash
chmod +x scripts/hpc/*.sh
./scripts/hpc/sync_phase12_bundle.sh USER@GPU_HOST /absolute/remote/project/path
```

The local workspace initially has no `.git` directory. To use a private
GitHub/GitLab repository, first initialize Git locally. The project's
`.gitignore` tracks code, configuration, documentation and report narratives,
while excluding datasets, checkpoints, prediction tables and generated figures.
Before every commit, run `scripts/hpc/check_git_payload.sh` to reject staged
data paths or files larger than 5 MiB. Retain the large data/checkpoint bundle
under `rsync`.

## On the GPU server

```bash
chmod +x scripts/hpc/*.sh
./scripts/hpc/verify_phase12_gpu.sh /absolute/remote/project/path
```

This prints the GPU, CUDA/PyTorch availability and runs a tiny **source-only**
training check. It does not produce a scientific result.

## First scientific run: PTT target, seed 17

```bash
./scripts/hpc/run_phase12_partial_gpu.sh \
  /absolute/remote/project/path lodo PTT-PPG 17
```

This trains on BIDMC, WESAD and PPG-DaLiA only, selects using their validation
subjects only, and evaluates untouched PTT-PPG once. Inspect
`reports/phase12_pulseppg/03_head_and_tuning_ablation/partial_final_block/run_summary.csv`
before launching the complete run.

## Full declared run after the PTT check passes

```bash
./scripts/hpc/run_phase12_partial_gpu.sh \
  /absolute/remote/project/path all all 17,29,43
```

This executes four within-dataset tests and four LODO targets across three
seeds. Keep the generated `run_provenance.json`, checkpoints and exact
window-level predictions. Results must be copied back with `rsync` before
analysis and presentation updates.
