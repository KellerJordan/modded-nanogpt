import os
import torch
import torch.distributed as dist

def setup_distributed():
    """Setup distributed training environment."""
    assert torch.cuda.is_available(), "CUDA is required for DDP but not available."
    
    try:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
    except KeyError as e:
        raise RuntimeError(f"Missing environment variable for DDP: {e}")
    
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    
    # Map local rank to CUDA device
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)
    
    if ddp_rank == 0:
        print(f"[Rank {ddp_rank}] Using device: {device}")
    
    return ddp_rank, ddp_world_size, device 