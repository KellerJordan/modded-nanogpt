import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.geoopt.optim import RiemannianSGD
from utils.muon import Muon

def get_parameter_groups(model):
    """Separate model parameters into different groups for optimization."""
    raw_model = model.module
    
    # Collect k parameters
    head_k_params, attn_k_params = [], []
    for name, param in raw_model.named_parameters():
        if "manifold.k" in name:
            head_k_params.append(param)
        elif "attn.k" in name:
            attn_k_params.append(param)
    
    k_params = head_k_params + attn_k_params
    
    # Collect other parameter groups
    lm_head_params = [p for name, p in raw_model.lm_head.named_parameters() 
                     if (p.requires_grad and ("manifold.k" not in name))]
    
    params = list(raw_model.transformer.h.parameters())
    matrix_params = [p for p in params if p.ndim == 2]
    wte_params = [raw_model.transformer.wte.weight]
    
    return {
        'head_k': head_k_params,
        'attn_k': attn_k_params,
        'k': k_params,
        'lm_head': lm_head_params,
        'matrix': matrix_params,
        'wte': wte_params
    }

def setup_lr_scheduler(optimizer, config):
    """Create learning rate scheduler."""
    init_lr = 1.0
    end_lr = 0.1
    
    def get_lr(it):
        t = max(0, min(1, 1 - it/config.num_iterations))
        w = min(t / config.cooldown_frac, 1.0)
        return w * init_lr + (1 - w) * end_lr
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

def setup_optimizers(model, config, master_process):
    """Configure optimizers and learning rate schedules."""
    param_groups = get_parameter_groups(model)
    
    # Set k parameter gradients based on config
    if config.k_lr:
        for p in param_groups['k']:
            p.requires_grad = True
    else:
        for p in param_groups['k']:
            p.requires_grad = False
    
    # Initialize optimizers
    optimizer_lm_head = RiemannianSGD(
        [{'params': param_groups['lm_head']}],
        lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True, stabilize=1
    )
    
    optimizer_muon = Muon(param_groups['matrix'], lr=0.05, momentum=0.95)
    
    optimizer_wte = torch.optim.Adam(
        param_groups['wte'], lr=0.6, betas=(0.8, 0.95), fused=True
    )
    
    optimizers = [optimizer_lm_head, optimizer_muon, optimizer_wte]
    
    # Add k optimizer if needed
    if param_groups['attn_k']:
        optimizer_k = torch.optim.SGD([
            {"params": param_groups['head_k'], "lr": config.k_lr},
            {"params": param_groups['attn_k'], "lr": config.k_lr}
        ], momentum=0.9, nesterov=True)
        optimizers.append(optimizer_k)
        if master_process:
            print("attn.k is learned")
    elif param_groups['head_k']:
        optimizer_k = torch.optim.SGD([
            {"params": param_groups['head_k'], "lr": config.k_lr}
        ], momentum=0.9, nesterov=True)
        optimizers.append(optimizer_k)
        if master_process:
            print(f"head.k is learned with {config.k_lr} lr")
    else:
        if master_process:
            print("k is not learned")
    
    # Create schedulers
    schedulers = [setup_lr_scheduler(opt, config) for opt in optimizers]
    
    return optimizers, schedulers 