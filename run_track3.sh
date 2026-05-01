TORCH_LOGS="recompiles" torchrun --standalone --nproc_per_node=2 records/track_3_optimization/train_gpt_ddp.py 2>&1 | tee -a track3.log
