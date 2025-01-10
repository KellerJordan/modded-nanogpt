#!/bin/bash

curvatures=(10. 30. 100. 300. 1000.)
seeds=(1 2 3)

for seed in "${seeds[@]}"; do
    echo "Running script with euclidean head and seed=${seed}"
    CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
        train_gpt2_hyp.py \
        --lm_head 'euc' \
        --seed "${seed}" \
        > logs/fineweb_euc_seed_${seed}.txt 2>&1
    echo "Finished run with euclidean head and seed=${seed}, logs saved to logs/fineweb_euc_seed_${seed}.txt"
done

for curvature in "${curvatures[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "Running script with fixed curvature=${curvature} and seed=${seed}"
        CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
            train_gpt2_hyp.py \
            --lm_head 'hyp' \
            --curvature "${curvature}" \
            --seed "${seed}" \
            > logs/fineweb_curvature_${curvature}_seed_${seed}.txt 2>&1
        echo "Finished run with curvature=${curvature} and seed=${seed}, logs saved to logs/fineweb_curvature_${curvature}_seed_${seed}.txt"
    done
    # and now learnable
    for seed in "${seeds[@]}"; do
        echo "Running script with learnable curvature from ${curvature} and seed=${seed}"
        CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 \
            train_gpt2_hyp.py \
            --lm_head 'hyp' \
            --learnable True \
            --curvature "${curvature}" \
            --seed "${seed}" \
            > logs/fineweb_learnable_curvature_${curvature}_seed_${seed}.txt 2>&1
        echo "Finished run with learnable curvature=${curvature} and seed=${seed}, logs saved to logs/fineweb_learnable_curvature_${curvature}_seed_${seed}.txt"
    done
done

echo "All experiments completed!"
