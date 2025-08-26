"""This script calculates the HellaSwag metric (https://huggingface.co/datasets/Rowan/hellaswag) of a model checkpoint."""

import sys, os, time
import argparse
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.distributed as dist
from datasets import load_dataset
import tiktoken

# add parent directory to sys.path to be able to import train_gpt.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, parent_dir)
from train_gpt import model, rank, world_size, master_process, args, ENDOFTEXT_ID
from train_gpt import distributed_data_generator, print0, get_window_size_blocks

tokenizer = tiktoken.get_encoding("gpt2")
PAD_ID = 0


@dataclass
class HellaswagSequence:
    """
    Packs multiple HellaSwag tasks into 1 sequence for efficient evaluation. For 
    val_seq_len = 260k, we pack ~800 tasks into 1 sequence which lets us evaluate
    all these tasks in 1 forward pass.

    Each task has 1 context, 4 possible endings, and 1 label indicating the index of 
    the correct ending.

    The sequence holds:
      - tokens: a list of token IDs with EOT tokens separating different endings/tasks
        (note: we don't add the last token of each ending, because it is not needed
        as we don't predict what comes after it)
      - targets: a shifted version of tokens to compute the loss
      - tasks: metadata for each task as (label, [(start_id, end_id), ...]), where 
        each pair contains the start and end ids of an ending

    Example with 2 tasks:
    - suppose task 0 has context [10,11] and 4 endings [20,21], [30,31], [40,41], [50,51])
    - suppose task 1 has context [110,111] and 4 endings [120,121], [130,131], [140,141], [150,151])

                                task 0                                                  task 1                                                          
                                ----------------------------------------------------    ---------------------------------------------------------------
    index                         0  1  2   3  4  5  6   7  8  9 10  11 12 13 14  15     16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31
    self.tokens              = [ 10 11 20 EOT 10 11 30 EOT 10 11 40 EOT 10 11 50 EOT    110 111 120 EOT 110 111 130 EOT 110 111 140 EOT 110 111 150 EOT ]
    self.targets             = [ 11 20 21 EOT 11 30 31 EOT 11 40 41 EOT 11 50 51 EOT    111 120 121 EOT 111 130 131 EOT 111 140 141 EOT 111 150 151 EOT ]
    self.tasks               = [(0,[(1, 2),      (5, 6),      (9,10),     (13,14)])     (1,[(17, 18),       (21, 22),       (25, 26),       (29, 30)] ) ]
    """

    tokens: Tensor                                  # shape (val_seq_len,)
    targets: Tensor                                 # shape (val_seq_len,)
    tasks: list[tuple[int, list[tuple[int, int]]]]  # one tuple for each task: (label, list of (start_id, end_id) for each of the 4 endings)


def get_hellaswag_task(raw_task: dict) -> tuple[list[list[int]], list[int], int]:
    ctx, label, endings = raw_task["ctx"], raw_task["label"], raw_task["endings"]
    assert len(endings) == 4

    ctx_tokens = tokenizer.encode_ordinary(ctx)
    options = [ctx_tokens + tokenizer.encode_ordinary(f" {ending}") for ending in endings]
    start_ids = [len(ctx_tokens) for _ in endings]  # holds index of first token of each ending
    return options, start_ids, int(label)


def pack_hellaswag_tasks_into_sequences(hellaswag_tasks: list[tuple[list[list[int]], list[int], int]]) -> list[HellaswagSequence]:
    sequences = []
    tokens, targets, tasks = [], [], []
    seq_len = args.val_seq_len

    def flush_sequence() -> None:
        nonlocal tokens, targets, tasks

        desired_length = seq_len
        padding_length = desired_length - len(tokens)
        tokens.extend([PAD_ID] * padding_length)
        targets.extend([PAD_ID] * padding_length)
        assert len(tokens) == len(targets) == seq_len

        sequence = HellaswagSequence(
            tokens=torch.tensor(tokens).to(dtype=torch.int32, device="cuda", non_blocking=True),
            targets=torch.tensor(targets).to(dtype=torch.int64, device="cuda", non_blocking=True),
            tasks=tasks,
        )
        sequences.append(sequence)
        tokens, targets, tasks = [], [], [] # reset

    for options, start_ids, label in hellaswag_tasks:
        # flush if adding this task would exceed maximum sequence length
        num_tokens = sum(len(option_tokens) for option_tokens in options)
        if len(tokens) + num_tokens >= seq_len:
            flush_sequence()

        # add task
        boundaries = []
        for option_tokens, start_id in zip(options, start_ids):            # iterate over 4 endings
            if len(tokens) > 0:
                tokens.append(ENDOFTEXT_ID)
                targets.append(ENDOFTEXT_ID)
            start_id_in_sequence = len(tokens) + start_id - 1              # -1 because we shift targets one position to the left
            end_id_in_sequence = len(tokens) + len(option_tokens) - 1 - 1  # -1 because we add one less token, and len(option_tokens) - 1 because this is the id of the last token
            boundaries.append((start_id_in_sequence, end_id_in_sequence))
            tokens.extend(option_tokens[:-1])
            targets.extend(option_tokens[1:])

        tasks.append((label, boundaries))

    if len(tokens) > 0:
        flush_sequence()

    return sequences


def evaluate_hellaswag_sequence(sequence: HellaswagSequence) -> tuple[int, int]:
    correct, count = 0, 0
    sliding_window_num_blocks = get_window_size_blocks(args.num_iterations)
    loss_per_token = model(sequence.tokens, sequence.targets, sliding_window_num_blocks)

    for label, boundaries in sequence.tasks:
        avg_loss_per_ending = [loss_per_token[start_id : end_id + 1].mean() for start_id, end_id in boundaries]

        # use the ending with the lowest loss as the predicted ending (this technique is often used when
        # evaluating HellaSwag, especially if the model is not very strong and would likely fail if we
        # asked it to output which of the 4 endings is the correct one)
        predicted = torch.tensor(avg_loss_per_ending).argmin().item()
        is_correct = (predicted == label)
        correct += is_correct
        count += 1

    return correct, count


def get_hellaswag_accuracy(sequences: list[HellaswagSequence]) -> float:
    correct, count = 0, 0

    with torch.inference_mode():
        for sequence in sequences:
            correct_, count_ = evaluate_hellaswag_sequence(sequence)
            correct += correct_
            count += count_

    correct_tensor = torch.tensor([correct], device="cuda")
    count_tensor = torch.tensor([count], device="cuda")
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
    correct = correct_tensor.item()
    count = count_tensor.item()
    acc = correct / count if count > 0 else 0.0

    return acc


def get_hellaswag_sequences_for_current_gpu() -> list[HellaswagSequence]:
    dataset = load_dataset(path="Rowan/hellaswag", split="validation", keep_in_memory=True)
    dataset = dataset.shuffle(seed=42)  # shuffle to have similar distribution of tasks across sequences and GPUs

    tasks = []
    for i in range(0, len(dataset), world_size):
        if i + rank <= len(dataset) - 1:
            task = get_hellaswag_task(dataset[i + rank])
            tasks.append(task)

    sequences = pack_hellaswag_tasks_into_sequences(tasks)

    # 10042 HellaSwag validation tasks are packed into ~16 sequences (2 forward passes per GPU if 8 GPUs)
    print(f" GPU {rank}: packed {len(tasks)} HellaSwag tasks into {len(sequences)} sequences")

    return sequences


if __name__ == "__main__":
    # load model
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    arguments = parser.parse_args()
    model_state_dict = torch.load(arguments.checkpoint)["model"]
    model.load_state_dict(model_state_dict)
    model.eval()
    dist.barrier()
    seconds = time.perf_counter() - t0
    print0(f"\nLoaded model from {arguments.checkpoint=} in {seconds=:.2f}", console=True)

    # warm up model
    t0 = time.perf_counter()
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, align_to_bos=True)
    for _ in range(10):
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(args.num_iterations))
    dist.barrier()
    seconds = time.perf_counter() - t0
    print0(f"Warmed up model in {seconds=:.2f}\n", console=True)

    # load HellaSwag
    t0 = time.perf_counter()
    sequences = get_hellaswag_sequences_for_current_gpu()
    dist.barrier()
    seconds = time.perf_counter() - t0
    print0(f"Loaded HellaSwag tasks in {seconds=:.2f}", console=True)

    # evaluate HellaSwag
    t0 = time.perf_counter()
    accuracy = get_hellaswag_accuracy(sequences)
    dist.barrier()
    seconds = time.perf_counter() - t0
    print0(f"\nCalculated HellaSwag {accuracy=:.6f} in {seconds=:.2f}\n", console=True)

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)

    dist.destroy_process_group()
