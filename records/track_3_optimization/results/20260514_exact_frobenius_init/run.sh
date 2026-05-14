#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
RESULT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$RESULT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/data/cached_fineweb10B.py" ]]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -f "$REPO_ROOT/data/cached_fineweb10B.py" && -f "/workspace/modded-nanogpt/data/cached_fineweb10B.py" ]]; then
  REPO_ROOT="/workspace/modded-nanogpt"
fi
if [[ ! -f "$REPO_ROOT/data/cached_fineweb10B.py" ]]; then
  echo "Could not find repo root from $RESULT_DIR; expected data/cached_fineweb10B.py under modded-nanogpt." >&2
  echo "Run from the full repo copy, e.g. /workspace/modded-nanogpt/records/track_3_optimization/results/20260514_hyperball_radial_experiments/b_late_hyperball_radial/run.sh" >&2
  exit 1
fi
EXPERIMENT_SLUG="$(basename "$RESULT_DIR")"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$RESULT_DIR/train_gpt_${EXPERIMENT_SLUG}.py}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
RUN_MODE="${RUN_MODE:-${MODE:-auto}}"
SEEDS="${SEEDS:-0}"
DATA_TOKENS="${DATA_TOKENS:-20}"
INSTALL_REQS="${INSTALL_REQS:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"
CHECK_DATA="${CHECK_DATA:-1}"
SETUP_ONLY="${SETUP_ONLY:-0}"
PARALLEL_SEEDS="${PARALLEL_SEEDS:-0}"
PIP_PACKAGES="${PIP_PACKAGES:-torch==2.11 huggingface_hub numpy tqdm datasets tiktoken setuptools typing-extensions}"
OUT_DIR="${OUT_DIR:-$RESULT_DIR/runs/$(date -u +%Y%m%d_%H%M%S)}"

resolve_run_mode() {
  case "$RUN_MODE" in
    auto|AUTO)
      if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_name
        gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || true)"
        if [[ "$gpu_name" == *"A100"* ]]; then
          RUN_MODE="a100"
        elif [[ "$gpu_name" == *"H100"* || "$gpu_name" == *"H200"* ]]; then
          RUN_MODE="h100"
        else
          RUN_MODE="generic"
        fi
      else
        RUN_MODE="generic"
      fi
      ;;
    a100|A100) RUN_MODE="a100" ;;
    h100|H100|h200|H200) RUN_MODE="h100" ;;
    generic|GENERIC) RUN_MODE="generic" ;;
    *)
      echo "Unknown RUN_MODE=$RUN_MODE. Use auto, a100, h100, or generic." >&2
      exit 1
      ;;
  esac
}

apply_run_mode_defaults() {
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  case "$RUN_MODE" in
    a100)
      # A100 is bf16-capable, but compile/NCCL behavior can be more brittle than
      # H100 pods. Keep math unchanged; just use stable runtime defaults.
      export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
      export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
      ;;
    h100)
      export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
      ;;
    generic)
      ;;
  esac
}

split_list() {
  local raw="${1:-}"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  SPLIT_LIST_RESULT=($raw)
}

detect_gpus() {
  if [[ -n "${GPU_IDS:-}" ]]; then
    split_list "$GPU_IDS"
    GPU_LIST=("${SPLIT_LIST_RESULT[@]}")
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    split_list "$CUDA_VISIBLE_DEVICES"
    GPU_LIST=("${SPLIT_LIST_RESULT[@]}")
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_LIST < <(nvidia-smi --query-gpu=index --format=csv,noheader | sed 's/[[:space:]]//g')
  else
    echo "Could not detect GPUs. Set GPU_IDS, e.g. GPU_IDS=0,1,2,3." >&2
    exit 1
  fi
  if (( ${#GPU_LIST[@]} == 0 )); then
    echo "No GPUs found. Set GPU_IDS or CUDA_VISIBLE_DEVICES." >&2
    exit 1
  fi
}

join_by_comma() {
  local IFS=,
  echo "$*"
}

make_gpu_chunks() {
  GPU_CHUNKS=()
  local total=${#GPU_LIST[@]}
  if (( GPUS_PER_RUN < 1 || GPUS_PER_RUN > total )); then
    echo "GPUS_PER_RUN=$GPUS_PER_RUN is invalid for ${total} visible GPUs: ${GPU_LIST[*]}" >&2
    exit 1
  fi
  local start=0
  while (( start + GPUS_PER_RUN <= total )); do
    local chunk=("${GPU_LIST[@]:start:GPUS_PER_RUN}")
    GPU_CHUNKS+=("$(join_by_comma "${chunk[@]}")")
    start=$((start + GPUS_PER_RUN))
  done
}

check_or_download_data() {
  if [[ "$CHECK_DATA" != "1" ]]; then
    return
  fi
  shopt -s nullglob
  local train_shards=(data/fineweb10B/fineweb_train_*.bin)
  local val_shards=(data/fineweb10B/fineweb_val_*.bin)
  shopt -u nullglob
  if (( ${#train_shards[@]} >= DATA_TOKENS && ${#val_shards[@]} >= 1 )); then
    return
  fi
  if [[ "$DOWNLOAD_DATA" != "1" ]]; then
    echo "FineWeb data missing: found ${#train_shards[@]} train shards and ${#val_shards[@]} val shards." >&2
    echo "Run with DOWNLOAD_DATA=1 DATA_TOKENS=$DATA_TOKENS to fetch it." >&2
    exit 1
  fi
  "$PYTHON_BIN" data/cached_fineweb10B.py "$DATA_TOKENS"
}

run_seed() {
  local seed="$1"
  local gpu_csv="$2"
  shift 2
  local seed_dir="$OUT_DIR/seed_${seed}"
  local log_dir="$seed_dir/experiment_logs"
  mkdir -p "$log_dir"

  {
    echo "seed=$seed"
    echo "cuda_visible_devices=$gpu_csv"
    echo "nproc_per_node=$GPUS_PER_RUN"
    echo "train_script=$TRAIN_SCRIPT"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$seed_dir/metadata.txt"

  echo "starting seed=$seed gpus=$gpu_csv out=$seed_dir"
  CUDA_VISIBLE_DEVICES="$gpu_csv" \
  LOG_DIR="$log_dir" \
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$GPUS_PER_RUN" "$TRAIN_SCRIPT" --seed "$seed" "$@" \
    2> >(tee "$seed_dir/stderr.txt" >&2) | tee "$seed_dir/stdout.txt"

  grep 'step:3010/3020 val_loss:' "$seed_dir/stdout.txt" > "$seed_dir/val_3010.txt" || true
  grep 'step:3020/3020 val_loss:' "$seed_dir/stdout.txt" > "$seed_dir/final_val_loss.txt" || true
  echo "finished seed=$seed gpus=$gpu_csv"
}

cd "$REPO_ROOT"
resolve_run_mode
apply_run_mode_defaults
detect_gpus

if [[ -z "${GPUS_PER_RUN:-}" ]]; then
  if [[ -n "${NPROC_PER_NODE:-}" ]]; then
    GPUS_PER_RUN="$NPROC_PER_NODE"
  elif [[ "$PARALLEL_SEEDS" == "1" ]]; then
    GPUS_PER_RUN=1
  else
    GPUS_PER_RUN=${#GPU_LIST[@]}
  fi
fi
make_gpu_chunks

split_list "$SEEDS"
SEED_LIST=("${SPLIT_LIST_RESULT[@]}")
if (( ${#SEED_LIST[@]} == 0 )); then
  echo "No seeds specified. Set SEEDS='0 1 2' or SEEDS=0,1,2." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
{
  echo "result_dir=$RESULT_DIR"
  echo "repo_root=$REPO_ROOT"
  echo "train_script=$TRAIN_SCRIPT"
  echo "run_mode=$RUN_MODE"
  echo "seeds=${SEED_LIST[*]}"
  echo "gpu_ids=${GPU_LIST[*]}"
  echo "gpu_chunks=${GPU_CHUNKS[*]}"
  echo "gpus_per_run=$GPUS_PER_RUN"
  echo "parallel_seeds=$PARALLEL_SEEDS"
  echo "data_tokens=$DATA_TOKENS"
  echo "install_reqs=$INSTALL_REQS"
  echo "download_data=$DOWNLOAD_DATA"
  echo "check_data=$CHECK_DATA"
  echo "setup_only=$SETUP_ONLY"
  echo "nccl_debug=${NCCL_DEBUG:-}"
  echo "torch_nccl_async_error_handling=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}"
  echo "pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-}"
  echo "cuda_device_max_connections=${CUDA_DEVICE_MAX_CONNECTIONS:-}"
  echo "torchinductor_compile_threads=${TORCHINDUCTOR_COMPILE_THREADS:-}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT_DIR/metadata.txt"

if [[ "$INSTALL_REQS" == "1" ]]; then
  "$PYTHON_BIN" -m pip install $PIP_PACKAGES
fi

check_or_download_data

if [[ "$SETUP_ONLY" == "1" ]]; then
  echo "setup complete: $OUT_DIR"
  exit 0
fi

if [[ ! -f "$TRAIN_SCRIPT" && "$(basename "$TORCHRUN_BIN")" != "echo" ]]; then
  echo "Missing trainer script: $TRAIN_SCRIPT" >&2
  echo "Add or copy the experiment trainer to this path before running $EXPERIMENT_SLUG." >&2
  exit 1
fi

if [[ "$PARALLEL_SEEDS" == "1" ]]; then
  status=0
  index=0
  while (( index < ${#SEED_LIST[@]} )); do
    pids=()
    for gpu_csv in "${GPU_CHUNKS[@]}"; do
      if (( index >= ${#SEED_LIST[@]} )); then
        break
      fi
      run_seed "${SEED_LIST[$index]}" "$gpu_csv" "$@" &
      pids+=("$!")
      index=$((index + 1))
    done
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
  done
  if (( status != 0 )); then
    echo "one or more seed runs failed; see $OUT_DIR" >&2
    exit "$status"
  fi
else
  gpu_csv="${GPU_CHUNKS[0]}"
  for seed in "${SEED_LIST[@]}"; do
    run_seed "$seed" "$gpu_csv" "$@"
  done
fi

echo "all done: $OUT_DIR"
