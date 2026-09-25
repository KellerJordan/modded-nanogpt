"""The unchanged, hash-checked FA3 forward and reference backward from PR #360."""
import hashlib
from pathlib import Path
import torch
from kernels import get_kernel

module = get_kernel('devenpzak/flash-attn3-12864', revision='64c1e6d1f2780e7931839f41426ddcdb564a7cb9')
FA = module.flash_attn_interface
SO = next(Path(module.__file__).parent.glob('_flash_attn3_cuda_*.so'))
assert hashlib.sha256(SO.read_bytes()).hexdigest() == '9beee90dbef9c9c221e677dad9f932108b70c46aa35841fc0862281d5b84c09b'

def forward(q,k,v,cu,maxlen,scale,window):
    return FA._flash_attn_forward(q,k,v,cu_seqlens_q=cu,cu_seqlens_k=cu,
        max_seqlen_q=maxlen,max_seqlen_k=maxlen,softmax_scale=scale,causal=True,
        window_size_left=window,window_size_right=0)[:2]

def backward(g,q,k,v,o,lse,cu,maxlen,scale,window):
    dq,dk,dv=[torch.empty_like(x) for x in (q,k,v)]
    FA._flash_attn_backward(g,q,k,v,o,lse,cu_seqlens_q=cu,cu_seqlens_k=cu,
        max_seqlen_q=maxlen,max_seqlen_k=maxlen,dq=dq,dk=dk,dv=dv,
        softmax_scale=scale,is_causal=True,window_size_left=window,window_size_right=0)
    return dq,dk,dv
