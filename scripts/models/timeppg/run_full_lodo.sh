#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/models/timeppg/run_full_lodo.sh PPG-DaLiA  # one target
#   ./scripts/models/timeppg/run_full_lodo.sh all         # all four targets
# Optional environment settings:
#   TIMEPPG_DEVICE=cpu or TIMEPPG_DEVICE=mps
#   TIMEPPG_SEED=17 (default; use 29 and 43 for planned replication)

requested_target="${1:-PPG-DaLiA}"
device="${TIMEPPG_DEVICE:-auto}"
seed="${TIMEPPG_SEED:-17}"

if command -v caffeinate >/dev/null 2>&1; then
  keep_awake=(caffeinate -i)
else
  keep_awake=()
fi

if [[ "${requested_target}" == "all" ]]; then
  targets=(BIDMC WESAD PTT-PPG PPG-DaLiA)
else
  targets=("${requested_target}")
fi

for target in "${targets[@]}"; do
  slug=$(printf '%s' "${target}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  run_dir="reports/phase9_timeppg/lodo_full/target_${slug}/seed_${seed}"
  mkdir -p "${run_dir}"
  if [[ -f "${run_dir}/metrics.json" ]]; then
    echo "Skipping completed full LODO target ${target}"
    continue
  fi
  echo "Starting full LODO with held-out target ${target} on ${device}"
  PYTHONPATH=src "${keep_awake[@]}" .venv/bin/python \
    scripts/models/timeppg/train_lodo_pilot.py \
    --target "${target}" \
    --full-data \
    --device "${device}" \
    --seed "${seed}" \
    --epochs 200 \
    --patience 20 \
    --output-root reports/phase9_timeppg/lodo_full \
    2>&1 | tee "${run_dir}/console.log"
done

echo "Requested full-data LODO run(s) completed."
