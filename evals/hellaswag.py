from typing import Callable
import time
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor
import torch.distributed as dist
from datasets import load_dataset
import tiktoken
from huggingface_hub import logging as hf_logging
hf_logging.set_verbosity_error()

tokenizer = tiktoken.get_encoding("gpt2")
rank = dist.get_rank()
world_size = dist.get_world_size()
EOT_ID = tokenizer.eot_token

@dataclass
class HellaswagTask:
    # store indices (start, end_excluding) of 4 endings with respect to the targets in PackedHellaswagSequence
    spans: tuple[tuple[int,int], tuple[int,int], tuple[int,int], tuple[int,int]]

    # label of correct ending of task
    label: int


@dataclass
class PackedHellaswagSequence:
    """Contains multiple HellaSwag tasks in 1 token sequence for efficient evaluation.

    If seq_len is ~260k we pack ~800 tasks into 1 sequence. For each task we store the
    tokens of 4 strings with each string being the context plus a possible ending.

    self.inputs does not store the last token of each ending because it's not used for 
    evaluation. Similarly, self.targets does not store the first token of each context.
    """
                               # EXAMPLE:
                               # The below sequence starts with Task0, which has context [10,11] 
                               # and 4 endings [20,21],[30,31],[40](correct ending),[50].
                               #            -Task0--------------------------------------- -Task1...
                               # indices     0  1  2   3  4  5  6   7  8  9  10 11 12  13 14 ...
    inputs: Tensor             # inputs  = [10 11 20 EOT 10 11 30 EOT 10 11 EOT 10 11 EOT 24 ...
    targets: Tensor            # targets = [11 20 21 EOT 11 30 31 EOT 11 40 EOT 11 50 EOT 37 ...
    tasks: list[HellaswagTask] # .spans        [1, 3),      [5, 7),      [9,10),  [12,13)    ...
                               # .label                                   2                  ...
    doc_end_positions: Tensor


def finalize_sequence(inputs: list[int], targets: list[int], tasks: list[HellaswagTask], seq_len: int) -> PackedHellaswagSequence:
    doc_len_min_average = 70 # use a smaller number than the median doc length of 94 (doc=context+one ending)
    max_num_docs = ((seq_len // doc_len_min_average) // 128 + 1) * 128  
    pad_id = 0  # can use any token id except EOT_ID which is reserved to delimit HellaSwag tasks and endings

    # pad to seq_len
    padding_length = seq_len - len(inputs)
    inputs.extend([pad_id] * padding_length)
    targets.extend([pad_id] * padding_length)
    assert len(inputs) == len(targets) == seq_len

    inputs_tensor = torch.tensor(inputs, dtype=torch.int32)
    targets_tensor = torch.tensor(targets, dtype=torch.int64)

    cum_lengths = torch.nonzero(inputs_tensor == EOT_ID)[:, 0] + 1  # add +1 to assign EOT_ID to preceding doc
    _cum_lengths = torch.full((max_num_docs,), seq_len)
    _cum_lengths[0] = 0
    _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths
    _cum_lengths = _cum_lengths.to(dtype=torch.int32)

    sequence = PackedHellaswagSequence(inputs=inputs_tensor,
                                       targets=targets_tensor,
                                       tasks=tasks,
                                       doc_end_positions=_cum_lengths)
    return sequence


def pack_tasks(raw_tasks: list[dict], seq_len: int) -> list[PackedHellaswagSequence]:
    """Pack `raw_tasks` into a list of sequences.

    The tasks are packed into a sequence like this:
    <ctx1><end11>EOT_ID<ctx1><end12>EOT_ID<ctx1><end13>EOT_ID<ctx1><end14>EOT_ID<ctx2><end21>...
    """

    sequences = []
    inputs, targets, packed_tasks = [], [], []

    def flush_sequence() -> None:
        nonlocal inputs, targets, packed_tasks
        sequence = finalize_sequence(inputs, targets, packed_tasks, seq_len=seq_len)
        sequences.append(sequence)
        inputs, targets, packed_tasks = [], [], [] # reset next sequence

    for raw_task in raw_tasks:
        context = tokenizer.encode_ordinary(raw_task["ctx"])
        # add space before ending because GPT2 tokenizer prefers predicting a word which starts with a space
        endings = [tokenizer.encode_ordinary(" " + ending) for ending in raw_task["endings"]]  
        assert len(endings) == 4, f"There must be 4 endings per task but got {len(endings)}"

        # flush if adding this task would exceed sequence length
        total_task_tokens = sum(len(context) + len(ending) for ending in endings)
        if len(inputs) + total_task_tokens >= seq_len:
            flush_sequence()

        # add task
        spans = []
        for ending in endings:
            if len(inputs) > 0:  # add separator EOT_ID between tasks, but not at the beginning of the sequence
                inputs.append(EOT_ID)
                targets.append(EOT_ID)
            context_plus_ending = context + ending
            start = len(inputs) + len(context) - 1  # -1 because targets are shifted one position to the left
            end_excluding = start + len(ending)
            spans.append((start, end_excluding))
            inputs.extend(context_plus_ending[:-1])
            targets.extend(context_plus_ending[1:])
        task = HellaswagTask(spans=tuple(spans), label=int(raw_task["label"]))
        packed_tasks.append(task)

    if len(inputs) > 0:
        flush_sequence()
    return sequences


def score_sequence(model, schedule_cfg, sequence: PackedHellaswagSequence, get_bigram_hash: Callable) -> tuple[int, int]:
    bigram_input_seq = get_bigram_hash(sequence.inputs).to(device="cuda")
    loss_per_token = model(input_seq=sequence.inputs.to(device="cuda"), 
                           target_seq=sequence.targets.to(device="cuda"), 
                           seqlens=sequence.doc_end_positions.to(device="cuda"),
                           schedule_cfg=schedule_cfg,
                           bigram_input_seq=bigram_input_seq,
                           )

    n_correct, n_count = 0, 0
    for task in sequence.tasks: # loop could be sped up by vectorizing it
        avg_loss_per_ending = [loss_per_token[start:end_excluding].mean() for start, end_excluding in task.spans]
        ending_id_with_lowest_loss = torch.stack(avg_loss_per_ending).argmin().item()
        is_correct = (ending_id_with_lowest_loss == task.label)
        n_correct += int(is_correct)
        n_count += 1
    return n_correct, n_count


@lru_cache(1)  # cache to speed up evaluation if this is run multiple times in same Python process
def get_sequences_for_current_rank(seq_len: int) -> list[PackedHellaswagSequence]:
    dataset = load_dataset(path="Rowan/hellaswag", split="validation")
    dataset = dataset.shuffle(seed=42)  # shuffle for similar distribution across sequences and GPUs
    tasks = [dataset[i+rank] for i in range(0, len(dataset) - rank, world_size)]
    sequences = pack_tasks(tasks, seq_len=seq_len)
    return sequences


def score_hellaswag(model, schedule_cfg, seq_len: int, get_bigram_hash: Callable) -> tuple[int, int]:
    sequences = get_sequences_for_current_rank(seq_len=seq_len)

    n_correct, n_count = 0, 0
    for sequence in sequences:
        _correct, _count = score_sequence(model, schedule_cfg, sequence, get_bigram_hash)
        n_correct += _correct
        n_count += _count

    n_correct_tensor = torch.tensor([n_correct], device="cuda")
    n_count_tensor = torch.tensor([n_count], device="cuda")
    dist.all_reduce(n_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_count_tensor, op=dist.ReduceOp.SUM)
    n_correct = n_correct_tensor.item()
    n_count = n_count_tensor.item()

    return n_correct, n_count
    

def evaluate(model, schedule_cfg, seq_len: int, get_bigram_hash: Callable, print0: Callable):
    """Calculates and prints accuracy of `model` on 10042 HellaSwag validation tasks.
    
    This function takes:
    - very first run on 8 x H100 machine:         ~15s (mainly: download dataset)
    - same machine, 1st run in a Python process:   ~5s
    - same machine, >=2nd run in a Python process: ~0.5s (= this is 
      how long 2 forward passes + some pre- and post-processing take)
    
    The 10042 tasks are split into 16 sequences. If world_size=8 then each
    GPU processes 2 sequences using 2 forward passes.
    """

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model.eval()
    with torch.inference_mode():
        n_correct, n_count = score_hellaswag(model, schedule_cfg, seq_len, get_bigram_hash)
        accuracy = n_correct / n_count

    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    print0(f"HellaSwag: {accuracy=:.3%} ({n_correct=} out of {n_count=} tasks in {seconds=:.2f})", console=True)
