#!/usr/bin/env bash
set -euo pipefail

# Phase 11 sensitivity extension: train both sides of PTT pairs 2/5 and 3/6
# with the already-frozen three-fold subject roles. Completed folds are skipped.

device="${TIMEPPG_DEVICE:-auto}"
seed="${TIMEPPG_SEED:-17}"
epochs="${TIMEPPG_EPOCHS:-200}"
patience="${TIMEPPG_PATIENCE:-20}"
split_file="reports/phase9_timeppg/three_fold_subject_roles.csv"
manifest="reports/phase7_frozen_dataset/ptt_site_window_manifest.csv"
output_base="reports/phase11_ptt_site_transfer/pair_sensitivity/training"

if command -v caffeinate >/dev/null 2>&1; then
  keep_awake=(caffeinate -i)
else
  keep_awake=()
fi

for channel in pleth_2 pleth_5 pleth_3 pleth_6; do
  output_root="${output_base}/${channel}"
  for test_fold in 0 1 2; do
    run_dir="${output_root}/ptt_ppg/test_fold_${test_fold}/seed_${seed}"
    mkdir -p "${run_dir}"
    if [[ -f "${run_dir}/metrics.json" && -f "${run_dir}/best_checkpoint.pt" ]]; then
      echo "Skipping completed ${channel}, fold ${test_fold}, seed ${seed}"
      continue
    fi
    echo "Training ${channel}: fold=${test_fold}, seed=${seed}, device=${device}"
    PYTHONPATH=src "${keep_awake[@]}" .venv/bin/python scripts/models/timeppg/train_within_dataset.py \
      --dataset PTT-PPG \
      --manifest-file "${manifest}" \
      --channel-name "${channel}" \
      --test-fold "${test_fold}" \
      --validation-fold -1 \
      --splits-file "${split_file}" \
      --role-column "within3_test_fold_${test_fold}_role" \
      --output-root "${output_root}" \
      --seed "${seed}" \
      --device "${device}" \
      --epochs "${epochs}" \
      --patience "${patience}" \
      2>&1 | tee "${run_dir}/console.log"
  done
done

echo "All remaining channel models completed; freezing the three-pair sensitivity result."
PYTHONPATH=src .venv/bin/python scripts/models/timeppg/freeze_ptt_all_site_pairs.py --seed "${seed}"

