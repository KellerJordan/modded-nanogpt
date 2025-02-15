#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
    train_gpt2_refactored.py \
    --data_path "data/tinystories_char" \
    --device_batch_size 2 \
    --batch_size 4 \
    --num_iterations 300 \
    --generate_every 100 \
    --n_heads 4 \
    --n_layers 6 \
    --head_dim 32 \
    --attn_mode "hyp" \
    --curvature 1. \
    --k_lr 1. \
    > last_logs_hyp_attn.txt 2>&1
    
# --curvature "${curvature}" \
# --k_lr "${k_lr}" \
# --seed "${seed}" \