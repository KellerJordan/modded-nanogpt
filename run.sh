#!/bin/bash

# curvatures=(300. 1000. 3000.)
# rates=(0.1 1.)
seeds=(1)

for seed in "${seeds[@]}"; do
    echo "Running script with euclidean head and seed=${seed}"
    CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 \
        train_gpt2_hyp.py \
        --lm_head 'euc' \
        --seed "${seed}" \
        > logs/fineweb_euc_seed_${seed}.txt 2>&1
    echo "Finished run with euclidean head and seed=${seed}, logs saved to logs/fineweb_euc_seed_${seed}.txt"
done

echo "All experiments completed!"
