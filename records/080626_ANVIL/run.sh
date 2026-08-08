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
# KX_STEPS=1178 is the record shape (default). Every mechanism knob's DEFAULT is
# the record configuration — this script sets nothing besides KX_STEPS.
# NOTE on the step count: the trainer's baked KX_GROW=40 default inserts 40 extra
# scheduled steps into stage 2 (boundaries after stage 2 shift by +40; the LR
# cooldown re-anchors to the grown length). The effective schedule is therefore
# KX_STEPS + 40 = 1218 scheduled + 23 extension = 1241 trained steps. KX_STEPS
# remains the PRE-growth count so its meaning matches the schedule tables in
# train_gpt.py; set KX_GROW=0 to reproduce an ungrown KX_STEPS-step schedule.
# KX_SEED=<n> optional for reproduction runs (records run unseeded).
KX_STEPS=${KX_STEPS:-1178} timeout 2700 torchrun --standalone --nproc_per_node=8 train_gpt.py
