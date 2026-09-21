#!/usr/bin/env bash
set -euo pipefail

# Phase 11 reverse-site models: train on proximal pleth_4 using the frozen PTT
# three-fold subject roles. Completed folds are skipped.

device="${TIMEPPG_DEVICE:-auto}"
seed="${TIMEPPG_SEED:-17}"
split_file="reports/phase9_timeppg/three_fold_subject_roles.csv"
output_root="reports/phase11_ptt_site_transfer/proximal_training"

if command -v caffeinate >/dev/null 2>&1; then
  keep_awake=(caffeinate -i)
else
  keep_awake=()
fi

for test_fold in 0 1 2; do
  run_dir="${output_root}/ptt_ppg/test_fold_${test_fold}/seed_${seed}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "Skipping completed proximal pleth_4 fold ${test_fold}, seed ${seed}"
    continue
  fi
  echo "Training proximal pleth_4: fold=${test_fold}, seed=${seed}, device=${device}"
  PYTHONPATH=src "${keep_awake[@]}" .venv/bin/python scripts/models/timeppg/train_within_dataset.py \
    --dataset PTT-PPG \
    --manifest-file reports/phase7_frozen_dataset/ptt_site_window_manifest.csv \
    --channel-name pleth_4 \
    --test-fold "${test_fold}" \
    --validation-fold -1 \
    --splits-file "${split_file}" \
    --role-column "within3_test_fold_${test_fold}_role" \
    --output-root "${output_root}" \
    --seed "${seed}" \
    --device "${device}" \
    --epochs 200 \
    --patience 20 \
    2>&1 | tee "${run_dir}/console.log"
done

echo "All requested proximal pleth_4 folds completed."
PYTHONPATH=src .venv/bin/python scripts/models/timeppg/evaluate_ptt_reverse_site.py
