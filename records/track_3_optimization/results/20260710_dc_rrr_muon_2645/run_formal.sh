#!/usr/bin/env bash
set -euo pipefail

trainer=records/track_3_optimization/results/20260710_dc_rrr_muon_2645/train_gpt_dc_projection_2645.py

for seed in $(seq "${START_SEED:-0}" "${END_SEED:-18}"); do
  MODDED_NANOGPT_SEED="$seed" \
  MODDED_NANOGPT_STEP_CAP=2645 \
  MODDED_NANOGPT_FINE_VAL_START_STEP=2620 \
  MODDED_NANOGPT_FINE_VAL_STEP_FREQ=5 \
  MODDED_NANOGPT_STOP_AT_TARGET=0 \
  MODDED_NANOGPT_SAVE_CHECKPOINTS=0 \
  torchrun --standalone --nproc_per_node=1 "$trainer"
done
