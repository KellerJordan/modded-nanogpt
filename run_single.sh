#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 train_gpt2_split.py > logs/test_run.txt 2>&1 
