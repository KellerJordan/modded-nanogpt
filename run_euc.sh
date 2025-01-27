#!/bin/bash
seeds=(2 3 4)
num_iterations=5000

head_modes=('euc')
paths=('/home/jovyan/fokin/modded-nanogpt/data/fineweb10B')

for seed in "${seeds[@]}"; do
    for path in "${paths[@]}"; do
        for head_mode in "${head_modes[@]}"; do
            if [ "$head_mode" == "hyp" ]; then
            k_lrs=(10.)
            curvatures=(1. 100.)
            else
            k_lrs=(0.) 
            curvatures=(0.)
            fi

            for curvature in "${curvatures[@]}"; do
                for k_lr in "${k_lrs[@]}"; do
                echo "Running ${head_mode} head run with seed=${seed}, k_lr=${k_lr}, k from ${curvature}"
                CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
                train_gpt2_refactored.py \
                --data_path "${path}" \
                --num_iterations "${num_iterations}" \
                --head_mode "${head_mode}" \
                --curvature "${curvature}" \
                --k_lr "${k_lr}" \
                --seed "${seed}" \
                > last_logs_${head_mode}_head_lr=${k_lr}_k_${curvature}_seed${seed}.txt 2>&1

                echo "Finished ${head_mode} head run with seed=${seed}, k_lr=${k_lr}, k from ${curvature}."
                done
            done
        done
    done
done

seeds=(0 1 2 3 4)
num_iterations=5000
head_modes=('hyp')

for seed in "${seeds[@]}"; do
    for path in "${paths[@]}"; do
        for head_mode in "${head_modes[@]}"; do
            if [ "$head_mode" == "hyp" ]; then
            k_lrs=(10.)
            curvatures=(500.)
            else
            k_lrs=(0.) 
            curvatures=(0.)
            fi

            for curvature in "${curvatures[@]}"; do
                for k_lr in "${k_lrs[@]}"; do
                echo "Running ${head_mode} head run with seed=${seed}, k_lr=${k_lr}, k from ${curvature}"
                CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
                train_gpt2_refactored.py \
                --data_path "${path}" \
                --num_iterations "${num_iterations}" \
                --head_mode "${head_mode}" \
                --curvature "${curvature}" \
                --k_lr "${k_lr}" \
                --seed "${seed}" \
                > last_logs_${head_mode}_head_lr=${k_lr}_k_${curvature}_seed${seed}.txt 2>&1

                echo "Finished ${head_mode} head run with seed=${seed}, k_lr=${k_lr}, k from ${curvature}."
                done
            done
        done
    done
done
echo "All experiments completed!"
