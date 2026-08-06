#!/bin/bash
# NanoGPT speedrun record attempt — exact launch line.
# 8xH100-80GB single node; torch cu128 + triton + tiktoken + numpy.
# DATA_PATH must contain data/fineweb10B/fineweb_{train,val}_*.bin (tmpfs recommended).
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8
export DATA_PATH=${DATA_PATH:-/dev/shm}
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/ind.XXXX)
export TRITON_CACHE_DIR=$(mktemp -d /tmp/trt.XXXX)
trap 'rm -rf "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"' EXIT
# KX_STEPS=1178 is the certified record shape (default); KX_SEED=<n> optional for reproduction runs.
KX_STEPS=${KX_STEPS:-1178} timeout 2700 torchrun --standalone --nproc_per_node=8 train_gpt.py
