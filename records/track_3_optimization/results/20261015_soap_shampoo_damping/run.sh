#!/usr/bin/env bash
set -euo pipefail
# Run from any directory; data and output logs are relative to the repository root.
submission_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$submission_dir/../../../.."
nproc="${NPROC_PER_NODE:-2}"
seed="${1:-49}"
exec torchrun --standalone --nproc_per_node="$nproc" "$submission_dir/train_gpt_soap_shampoo.py" --seed "$seed"
