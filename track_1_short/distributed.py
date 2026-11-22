"""Process-group setup. The speedrun always runs as 8 processes on one 8xH100 node."""
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

WORLD_SIZE = 8


@dataclass(frozen=True)
class DistEnv:
    rank: int
    world_size: int
    device: torch.device
    # A second NCCL communicator (own stream, high priority) for the n-gram table's row exchanges and the
    # big dense reduce-scatters queued behind them (optim/anvil.py), so they overlap the default group's
    # collectives instead of sharing its stream.
    ngram_group: dist.ProcessGroup

    @property
    def master_process(self) -> bool:
        # rank 0 does logging, checkpointing etc.
        return self.rank == 0


def setup_distributed() -> DistEnv:
    """Initialize NCCL from torchrun's RANK / WORLD_SIZE / LOCAL_RANK."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # The sampled softmax and the sparse row exchanges assume one microbatch per step on 8 ranks.
    assert world_size == WORLD_SIZE, f"track 1 runs on exactly {WORLD_SIZE} GPUs, got world_size={world_size}"
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
    dist.barrier()
    options = dist.ProcessGroupNCCL.Options()
    options.is_high_priority_stream = True
    ngram_group = dist.new_group(backend="nccl", pg_options=options)
    # NCCL connects a peer pair on its first message: do it for every pair now, off the clock.
    ones = torch.ones(world_size, dtype=torch.int64, device=device)
    dist.all_to_all_single(torch.empty_like(ones), ones, group=ngram_group)
    torch.cuda.synchronize()
    return DistEnv(rank=rank, world_size=world_size, device=device, ngram_group=ngram_group)
