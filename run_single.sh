#!/bin/bash

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
    train_gpt2_refactored.py \
    --data_path "data/tinystories_char" \
    --num_iterations 100 \
    --head_mode "euc" \
    > last_logs.txt 2>&1
    
# --curvature "${curvature}" \
# --k_lr "${k_lr}" \
# --seed "${seed}" \