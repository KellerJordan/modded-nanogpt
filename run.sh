torchrun --standalone --nproc_per_node=8 train_gpt2.py
# torchrun --standalone --nproc_per_node=1 train_gpt2.py --train.batch_size 8 --train.sequence_length 32768