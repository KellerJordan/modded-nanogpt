import torch


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
        val = ((1 - frac_done) * self.start + frac_done * self.end) // self.prec * self.prec
        if val != self.prev_val:
            self.gpu_val.copy_(val, non_blocking=True)
            self.prev_val = val
        return self.gpu_val