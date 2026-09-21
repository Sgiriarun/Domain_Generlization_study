#!/usr/bin/env bash
set -euo pipefail

# Main three-fold within-dataset protocol. Every dataset uses the same frozen
# outer-fold design. Completed runs are skipped safely.

if command -v caffeinate >/dev/null 2>&1; then
  keep_awake=(caffeinate -i)
else
  keep_awake=()
fi

split_file="reports/phase9_timeppg/three_fold_subject_roles.csv"
if [[ ! -f "${split_file}" ]]; then
  PYTHONPATH=src .venv/bin/python scripts/models/timeppg/build_three_fold_splits.py
fi

for dataset in BIDMC PTT-PPG WESAD PPG-DaLiA; do
  slug=$(printf '%s' "${dataset}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  for test_fold in 0 1 2; do
    run_dir="reports/phase9_timeppg/within_dataset_3fold/${slug}/test_fold_${test_fold}/seed_17"
    mkdir -p "${run_dir}"
    if [[ -f "${run_dir}/metrics.json" ]]; then
      echo "Skipping completed ${dataset} test fold ${test_fold}"
      continue
    fi
    echo "Starting ${dataset}: test fold ${test_fold}"
    PYTHONPATH=src "${keep_awake[@]}" .venv/bin/python scripts/models/timeppg/train_within_dataset.py \
      --dataset "${dataset}" \
      --test-fold "${test_fold}" \
      --validation-fold -1 \
      --splits-file "${split_file}" \
      --role-column "within3_test_fold_${test_fold}_role" \
      --output-root reports/phase9_timeppg/within_dataset_3fold \
      --seed 17 \
      2>&1 | tee "${run_dir}/console.log"
  done
done

echo "All three-fold within-dataset runs completed."
