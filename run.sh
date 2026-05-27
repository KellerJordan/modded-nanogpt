#!/bin/sh
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
torchrun --standalone --nproc_per_node=${NUM_GPUS} train_gpt.py