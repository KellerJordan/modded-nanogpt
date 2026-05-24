#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$RESULT_DIR/../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_RUN:-4}}"
DATA_TOKENS="${DATA_TOKENS:-40}"
INSTALL_REQS="${INSTALL_REQS:-0}"
CHECK_DATA="${CHECK_DATA:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
OUT_DIR="${OUT_DIR:-$RESULT_DIR/repro_runs/$(date -u +%Y%m%d_%H%M%S)}"

cd "$REPO_ROOT"

if [[ "$INSTALL_REQS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r "$RESULT_DIR/requirements.txt"
fi

if [[ "$CHECK_DATA" == "1" ]]; then
  shopt -s nullglob
  train_shards=(data/fineweb10B/fineweb_train_*.bin)
  val_shards=(data/fineweb10B/fineweb_val_*.bin)
  shopt -u nullglob
  if (( ${#train_shards[@]} < DATA_TOKENS || ${#val_shards[@]} < 1 )); then
    if [[ "$DOWNLOAD_DATA" == "1" ]]; then
      "$PYTHON_BIN" data/cached_fineweb10B.py "$DATA_TOKENS"
    else
      echo "FineWeb data missing: found ${#train_shards[@]} train shards and ${#val_shards[@]} val shards." >&2
      echo "Run with DOWNLOAD_DATA=1 to fetch data/cached_fineweb10B.py $DATA_TOKENS." >&2
      exit 1
    fi
  fi
fi

mkdir -p "$OUT_DIR"
{
  echo "result_dir=$RESULT_DIR"
  echo "repo_root=$REPO_ROOT"
  echo "seeds=$SEEDS"
  echo "nproc_per_node=$NPROC_PER_NODE"
  echo "data_tokens=$DATA_TOKENS"
  echo "install_reqs=$INSTALL_REQS"
  echo "check_data=$CHECK_DATA"
  echo "download_data=$DOWNLOAD_DATA"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/metadata.txt"

for seed in $SEEDS; do
  seed_dir="$OUT_DIR/seed_${seed}"
  log_dir="$seed_dir/logs"
  mkdir -p "$log_dir"
  echo "starting seed=$seed out=$seed_dir"
  SEED="$seed" \
  LOG_DIR="$log_dir" \
  EARLY_STOP=0 \
  TARGET_VAL_LOSS=0 \
  TRAIN_PROGRESS_INTERVAL=0 \
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
    "$RESULT_DIR/train_gpt_simple_trustlight.py" \
    2> >(tee "$seed_dir/stderr.txt" >&2) | tee "$seed_dir/stdout.txt"
  grep 'step:3175/3175 val_loss:' "$seed_dir/stdout.txt" > "$seed_dir/final_val_loss.txt" || true
done

echo "all done: $OUT_DIR"
