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
    return DistEnv(rank=rank, world_size=world_size, device=device)
