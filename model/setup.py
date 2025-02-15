import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from model.model import GPT
from data.loader import setup_tokenizer

def setup_model(config, device, ddp_rank):
    """Initialize and setup the model for distributed training."""
    # Setup tokenizer first to get vocab_size
    tokenizer, _ = setup_tokenizer(config)
    
    # Create model instance
    model = GPT(config, tokenizer)
    model = model.to(device)
    
    # Compile model if using PyTorch 2.0+
    model = torch.compile(model)
    
    # Wrap model in DDP
    model = DDP(model, device_ids=[device.index])
    raw_model = model.module
    
    if ddp_rank == 0:
        print(f"\n=== Minimal Report ===")
        print(f"Model Size: {raw_model.model_size()}")
        print("\n=== Relevant Hyperparameters ===")
        print(f"Data Path:            {config.data_path}")
        print(f"Sequence Length:      {config.sequence_length}")
        print(f"Batch Size (global):  {config.batch_size}")
        print(f"Batch Size (device):  {config.device_batch_size}")
        print(f"n_layers:             {config.n_layers}")
        print(f"n_heads:              {config.n_heads}")
        print(f"head_dim:            {config.head_dim}")
        print(f"n_embd:              {config.n_embd}")
        print(f"Seed:                {config.seed}")
        print("==============================\n")
    
    return model, raw_model