import torch
import random
import numpy as np
import time
import yaml


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config or {}


def set_seed(seed):
    """Set seed for reproducibility across all processes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_param_count(model):
    total_params = 0
    for _, param in model.named_parameters():
        total_params += param.numel()
    return total_params


class LerpTensor:
    def __init__(self, start_val, end_val, precision):
        self.start, self.end, self.prec = start_val, end_val, precision
        self.prev_val = None
        dtype = torch.int32 if isinstance(precision, int) else torch.float
        self.gpu_val = torch.tensor(0, dtype=dtype, device="cuda")

    def __call__(self, frac_done):
        val = (max((1 - frac_done), 0) * self.start + min(frac_done, 1) * self.end) // self.prec * self.prec
        if val != self.prev_val:
            self.gpu_val.copy_(val, non_blocking=True)
            self.prev_val = val
        return self.gpu_val
    

class LerpFloat:
    def __init__(self, start_val, end_val, precision):
        self.start, self.end, self.prec = start_val, end_val, precision
        self.prev_val = None
        
    def __call__(self, frac_done):
        val = (max((1 - frac_done), 0) * self.start + min(frac_done, 1) * self.end) // self.prec * self.prec
        if val != self.prev_val:
            self.prev_val = val
        return self.prev_val


class GlobalTimer:
    """Global timer that tracks elapsed time and can be paused/resumed."""
    def __init__(self):
        self.total_time = 0.0
        self.start_time = None
        self.is_running = False
    
    def start(self):
        """Start the timer."""
        if not self.is_running:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()
            self.is_running = True
    
    def pause(self):
        """Pause the timer and add elapsed time to total."""
        if self.is_running:
            torch.cuda.synchronize()
            self.total_time += time.perf_counter() - self.start_time
            self.is_running = False
    
    def resume(self):
        """Resume the timer."""
        self.start()
    
    def get_time(self):
        """Get total elapsed time including current session if running."""
        current_time = self.total_time
        if self.is_running:
            torch.cuda.synchronize()
            current_time += time.perf_counter() - self.start_time
        return current_time
    
    def reset(self):
        """Reset the timer to zero."""
        self.total_time = 0.0
        self.start_time = None
        self.is_running = False


def exclude_from_timer(timer):
    """Decorator that pauses the timer during function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer.pause()
            try:
                result = func(*args, **kwargs)
            finally:
                timer.resume()
            return result
        return wrapper
    return decorator
