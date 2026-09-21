#!/usr/bin/env bash
set -euo pipefail

# Replicate the prespecified full-data LODO experiment without overwriting seed 17.
# Usage:
#   TIMEPPG_DEVICE=mps ./scripts/models/timeppg/run_lodo_replication.sh PTT-PPG
#   TIMEPPG_DEVICE=mps ./scripts/models/timeppg/run_lodo_replication.sh all

target="${1:-PTT-PPG}"
for seed in 29 43; do
  echo "Starting LODO replication: target=${target}, seed=${seed}"
  TIMEPPG_SEED="${seed}" ./scripts/models/timeppg/run_full_lodo.sh "${target}"
done

echo "Requested seed-29 and seed-43 LODO replications completed."
