#!/usr/bin/env bash
set -euo pipefail

# Complete PPG-DaLiA five-fold within-dataset evaluation for seed 17.
# Fold 0 is complete. For every test fold, the next fold is validation and the
# other three folds are training subjects.

for test_fold in 1 2 3 4; do
  validation_fold=$(( (test_fold + 1) % 5 ))
  run_dir="reports/phase9_timeppg/within_dataset/ppg_dalia/test_fold_${test_fold}/seed_17"
  mkdir -p "${run_dir}"

  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "Skipping completed test fold ${test_fold}: metrics.json exists"
    continue
  fi

  echo "Starting test fold ${test_fold}, validation fold ${validation_fold}"
  PYTHONPATH=src .venv/bin/python scripts/models/timeppg/train_within_dataset.py \
    --dataset PPG-DaLiA \
    --test-fold "${test_fold}" \
    --validation-fold "${validation_fold}" \
    --seed 17 \
    2>&1 | tee "${run_dir}/console.log"
done

echo "All remaining PPG-DaLiA folds completed."

