#!/usr/bin/env bash
# Official submission runs: full run per seed, locked tail-EMA recipe baked
# into the trainer defaults (no env overrides). Claimed step: 2720.
# Usage: SEEDS="2 0 1 ..." GPUS=1 run.sh  (launch from a dir with data/fineweb10B)
set -euo pipefail

SEEDS="${SEEDS:-2 0 1 3 4 5 6 7}"
GPUS="${GPUS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="${SCRIPT_DIR}/train_gpt_tailema_2720.py"

for seed in ${SEEDS}; do
  echo "running seed ${seed}"
  torchrun --standalone --nproc_per_node="${GPUS}" "${TRAINER}" --seed "${seed}"
done
