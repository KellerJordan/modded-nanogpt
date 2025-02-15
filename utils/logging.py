import os
import json
import subprocess
import datetime
from torch.utils.tensorboard import SummaryWriter

def save_config(logdir: str, args, timestamp):
    """Save configuration to JSON file."""
    config_path = os.path.join(logdir, "config.json")
    config_dict = {
        **vars(args),
        "created_at": timestamp.isoformat(),
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        "git_branch": subprocess.getoutput("git rev-parse --abbrev-ref HEAD")
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    return config_dict

def initialize_logfile(logdir: str):
    """Initialize main log file with system information."""
    import torch
    
    logfile = os.path.join(logdir, 'run.log')
    with open(logfile, "w") as f:
        f.write("=== Environment Information ===\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA: {torch.version.cuda}\n")
        f.write(f"Device: {torch.cuda.get_device_name()}\n")
        f.write("\n=== GPU Information ===\n")
        try:
            nvidia_smi = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, text=True)
            f.write(nvidia_smi.stdout)
        except:
            f.write("nvidia-smi not available\n")
        f.write("\n=== Training Log ===\n")
    return logfile

def setup_logging(config, args, master_process):
    """Set up logging directories and writers."""
    if not master_process:
        return None, None, None
    
    now = datetime.datetime.now()
    timestamp = now.strftime('%m%d_%H%M')
    
    # Create run ID
    config_parts = []
    if config.head_mode == 'hyp':
        config_parts.append(f"k{config.curvature}")
        if config.k_lr:
            config_parts.append(f"lr{config.k_lr:.1e}")
    else:
        config_parts.append(config.head_mode)
    
    config_parts.extend([
        f"l{config.n_layers}",
        f"h{config.n_heads}",
        f"d{config.head_dim}",
        f"s{config.seed}"
    ])
    
    run_id = f"{timestamp}_{'_'.join(config_parts)}"
    
    # Setup directories
    logdir = os.path.join('runs', run_id)
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    checkpoints_dir = os.path.join(logdir, 'checkpoints')
    
    for directory in [logdir, tensorboard_dir, checkpoints_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"\n=== Logging Setup ===")
    print(f"Run ID: {run_id}")
    print(f"Log directory: {logdir}")
    
    writer = SummaryWriter(log_dir=tensorboard_dir)
    config_dict = save_config(logdir, args, now)
    logfile = initialize_logfile(logdir)
    
    return writer, logfile, run_id 