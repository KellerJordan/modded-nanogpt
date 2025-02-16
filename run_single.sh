#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
    train_gpt2_main.py \
    --data_path "data/tinystories_char" \
    --device_batch_size 32 \
    --batch_size 64 \
    --num_iterations 300 \
    --generate_every 100 \
    --n_heads 4 \
    --n_layers 6 \
    --head_dim 32 \
    --attn_mode "hyp" \
    --curvature 1. \
    --k_lr 10. \
    > last_logs_hyp_attn.txt 2>&1
    
# --curvature "${curvature}" \
# --k_lr "${k_lr}" \
# --seed "${seed}" \