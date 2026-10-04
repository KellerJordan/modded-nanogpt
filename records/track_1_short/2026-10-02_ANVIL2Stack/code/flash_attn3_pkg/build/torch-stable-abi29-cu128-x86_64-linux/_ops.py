import torch
from . import _flash_attn3_cuda_8a730d9
ops = torch.ops._flash_attn3_cuda_8a730d9

def add_op_namespace_prefix(op_name: str):
    """
    Prefix op by namespace.
    """
    return f"_flash_attn3_cuda_8a730d9::{op_name}"
