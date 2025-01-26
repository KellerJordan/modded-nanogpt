#!/bin/bash

seeds=(0)
num_iterations=1000

for seed in "${seeds[@]}"; do
    echo "Running script with euclidean head and seed=${seed}"
    CUDA_VISIBLE_DEVICES=3,7 torchrun --standalone --nproc_per_node=2 \
        train_gpt2_hyp.py \
        --data_path '/home/jovyan/fokin/modded-nanogpt/data/tinystories' \
        --num_iterations "${num_iterations}" \
        --seed "${seed}" \
        > logs/fineweb_euc_seed_${seed}.txt 2>&1
    echo "Finished run with euclidean head and seed=${seed}, logs saved to logs/fineweb_euc_seed_${seed}.txt"
done

echo "All experiments completed!"
