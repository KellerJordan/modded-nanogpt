# This switches to the pre-installed flash attention if run on a GH200. 
#
# On lambda GH200, run:
#
# pip install huggingface-hub
# git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
# python data/cached_fineweb10B.py 9
#
# Update the `run_id` below for a better log file name.
#
# Launch with below:
#
# torchrun --standalone --nproc_per_node=1 train_gpt_attn_output_var_fix.py

import uuid
run_id = f"Merged Optims - {str(uuid.uuid4())[0:5]}"

import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from collections import defaultdict
from itertools import accumulate
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(
    1, device="cuda", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import triton
import triton.language as tl
#from kernels import get_kernel
from torch import Tensor, nn

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng


@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def XXT_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    XXT_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from XXT_kernel, but also loads and adds a block of A
    # Performance is slightly slower than XXT_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ba_plus_cAA_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out

# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def polar_express(G: torch.Tensor, split_baddbmm: bool = False):
    """
    Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
    by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
  
    # Select batched vs unbatched
    if split_baddbmm:
        BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
    else:
        aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm
    
    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        
        # Referencing X twice causes pytorch to make a defensive copy, 
        # resulting in a cudaMemcpyAsync in baddbmm.
        # For large matrices (i.e., the mlp weights), it's faster to split 
        # the operation into two kernels to avoid this.
        if split_baddbmm:
            BX_matmul(B, X, out=C)  # C = B @ X  
            C.add_(X, alpha=a)      # C = C + a*X  (in-place, X only read)
        else:
            aX_plus_BX(X, B, X, beta=a, out=C)  # C = a * X + B @ X
            
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# -----------------------------------------------------------------------------
# Compiled Helpers for NorMuon by @chrisjmccormick

@torch.compile(dynamic=False, fullgraph=True)
def cautious_wd_and_update_inplace(p, v, wd_tensor, lr_tensor):
    """Cautious weight decay + parameter update. wd_tensor and lr_tensor are 0-D CPU tensors."""
    mask = (v * p) >= 0
    wd_factor = wd_tensor.to(p.dtype)
    lr_factor = lr_tensor.to(p.dtype)
    p.copy_(p - (p * mask * wd_factor * lr_factor) - (v * lr_factor))
    

@torch.compile(dynamic=False, fullgraph=True)
def apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
    """NorMuon variance reduction. Algebraically fuses the normalization steps to minimize memory ops."""
    v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v_chunk.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v_chunk.mul_(final_scale.type_as(v_chunk))

@torch.compile(dynamic=False, fullgraph=True)
def _adam_update_inplace(p_slice, g_slice, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, wd, t, wd_mul):
    """
    Compiled Adam update math. 
    Args:
        lr, beta1, beta2, eps, wd, t, wd_mul: 0-dim CPU Tensors
    """
    # 1. Unconditional Weight Decay
    # We cannot use 'if wd != 0' because checking a tensor value causes a graph break.
    # If wd is 0, eff_weight_decay becomes 0, and p_slice.mul_(1) is a no-op.
    eff_weight_decay = lr * wd * wd_mul
    p_slice.mul_(1 - eff_weight_decay)

    # 2. Update running averages
    # beta1/beta2 are tensors, so operations here stay in the graph
    exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)

    # 3. Bias corrections
    # pow() with tensor exponents works fine on device
    bias1 = 1 - beta1.pow(t)
    bias2 = 1 - beta2.pow(t)

    # 4. Compute step
    denom = exp_avg_sq.sqrt().add_(eps)
    step_size = lr * (bias2.sqrt() / bias1)
    update = exp_avg.div(denom).mul_(step_size)
    p_slice.add_(other=update, alpha=-1.0)

import torch
from torch import Tensor
import torch.distributed as dist
from collections import defaultdict

# -----------------------------------------------------------------------------
# Combined Optimizer

class CombinedOptimizer(torch.optim.Optimizer):
    """Combines NorMuon and DistAdam for better communication-compute overlap."""
    def __init__(
        self, 
        params, 
        adam_lr: float = 1e-3, 
        adam_betas: tuple[float, float] = (0.9, 0.999), 
        adam_eps: float = 1e-8, 
        adam_weight_decay: float = 0.01,
        normuon_lr: float = 0.02,
        normuon_weight_decay: float = 0.01,
        normuon_momentum: float = 0.95,
        normuon_beta2: float = 0.95,
        normuon_custom_sizing: bool = True):
        
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        defaults = dict(
            adam_lr=adam_lr,
            adam_betas=adam_betas,
            adam_eps=adam_eps,
            adam_weight_decay=adam_weight_decay,
            normuon_lr=normuon_lr,
            normuon_weight_decay=normuon_weight_decay,
            normuon_momentum=normuon_momentum,
            normuon_beta2=normuon_beta2)
        
        params = list(params)
        
        # Separate params by optimizer type based on their labels
        adam_labels = ['lm_head', 'scalars', 'value_embed', 'embed']
        normuon_labels = ['smear_gate', 'attn_gate', 'attn', 'mlp']
        
        adam_params = [p for p in params if getattr(p, 'label', None) in adam_labels]
        normuon_params = [p for p in params if getattr(p, 'label', None) in normuon_labels]
        
        # Create param groups for Adam (processed in label order)
        adam_param_groups = []
        params_by_label = defaultdict(list)
        for p in adam_params:
            params_by_label[p.label].append(p)
        
        for label in adam_labels:
            if label in params_by_label:
                adam_param_groups.append(dict(
                    params=params_by_label[label],
                    optimizer_type='adam',
                    lr=adam_lr,
                    betas=adam_betas,
                    eps=adam_eps,
                    weight_decay=adam_weight_decay
                ))
        
        # Create param groups for NorMuon
        normuon_param_groups = []
        if normuon_custom_sizing and dist.get_world_size() == 8:
            normuon_param_groups = self.generate_custom_normuon_groups(normuon_params)
        else:
            normuon_param_groups = self.generate_standard_normuon_groups(normuon_params)
        
        # Mark all normuon groups with optimizer type and set hyperparameters
        for group in normuon_param_groups:
            group['optimizer_type'] = 'normuon'
            group['lr'] = normuon_lr
            group['weight_decay'] = normuon_weight_decay
            group['momentum'] = normuon_momentum
            group['beta2'] = normuon_beta2
        
        # Combine param groups (Adam first, then NorMuon)
        param_groups = adam_param_groups + normuon_param_groups
        
        super().__init__(param_groups, defaults)
        
        # Initialize Adam state for Adam parameters
        for p in adam_params:
            chunk_size = p.size(0) // self.world_size
            exp_avg = torch.zeros_like(p[:chunk_size], dtype=torch.bfloat16, device=p.device)
            exp_avg_sq = torch.zeros_like(exp_avg)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=exp_avg_sq)
        
        # Sync control flags
        self.should_sync_adam = False
        self.should_sync_normuon = True  # NorMuon syncs every step
        
        # Storage for async operations - unified list for interleaved processing
        self._reduce_scatter_hooks = []
        self._reduce_scatter_infos = []
        
        # Create a mapping from param to its group for efficient lookup in _sync_gradient
        self._param_to_group = {}
        for group in self.param_groups:
            for param in group['params']:
                self._param_to_group[param] = group
        
        self.register_backward_hooks()

    def reset(self):
        """Reset NorMuon momentum buffers."""
        for group in self.param_groups:
            if group.get('optimizer_type') == 'normuon':
                if "momentum_buffer" in group:
                    group["momentum_buffer"].zero_()
                if "second_momentum_buffer" in group:
                    group["second_momentum_buffer"].zero_()

    def generate_standard_normuon_groups(self, params):
        """
        Standard NorMuon grouping: one param group per label.
        Use this method if running on less than 8 GPU or experimenting with additional modules.
        """
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        return param_groups

    def generate_custom_normuon_groups(self, params):
        """
        Custom NorMuon grouping for 8 GPUs.
        Implementation requires that a single GPU does not receive both attn
        and mlp params when a param group is split across GPUs.
        """
        module_group_order = ['smear_gate', 'attn_gate', 'attn', 'mlp']
        params_list = list(params)
        params_list.sort(key=lambda x: module_group_order.index(x.label))

        idx = 0
        group_sizes = [1, 10, 16, 16]
        assert len(params_list) == sum(group_sizes), f"Expected {sum(group_sizes)} params, got {len(params_list)}"
        param_groups = []
        for size in group_sizes:
            chunk_size = (size + self.world_size - 1) // self.world_size
            group_params = params_list[idx: idx + size]
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
            idx += size

        return param_groups

    def register_backward_hooks(self):
        """Register hooks to sync gradients as they become available."""
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for param in params:
                hook = param.register_post_accumulate_grad_hook(self._sync_gradient)
                self._reduce_scatter_hooks.append(hook)

    @torch.no_grad()
    def _sync_gradient(self, param):
        """
        Callback triggered when a parameter's gradient is ready.
        Routes to appropriate sync method based on optimizer type.
        """
        group = self._param_to_group.get(param)
        if group is None:
            return
        
        optimizer_type = group.get('optimizer_type')
        
        if optimizer_type == 'adam':
            # Adam: sync individual parameter gradients via reduce_scatter
            if not self.should_sync_adam:
                return
            
            grad = param.grad
            rank_size = grad.shape[0] // self.world_size
            grad_slice = torch.empty_like(grad[:rank_size])
            reduce_future = dist.reduce_scatter_tensor(
                grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()
            
            # Store in unified list with metadata
            self._reduce_scatter_infos.append(dict(
                optimizer_type='adam',
                param=param,
                grad_slice=grad_slice,
                reduce_future=reduce_future
            ))
            
        elif optimizer_type == 'normuon':
            # NorMuon: sync entire groups at once
            # Check if this is the first param in the group to trigger group sync
            if not self.should_sync_normuon:
                return
            
            if param is group['params'][0]:
                self._sync_normuon_group(group)
    
    def _sync_normuon_group(self, group):
        """Sync gradients for an entire NorMuon parameter group."""
        params: list[Tensor] = group["params"]
        if not params:
            return
        
        chunk_size = group["chunk_size"]
        padded_num_params = chunk_size * self.world_size
        
        stacked_grads = torch.empty(
            (padded_num_params, *params[0].shape),
            dtype=params[0].dtype,
            device=params[0].device
        )
        
        for i, p in enumerate(params):
            if p.grad is not None:
                stacked_grads[i].copy_(p.grad, non_blocking=True)
            else:
                # TODO - This indicates skipped weights with no gradients--is that expected?
                stacked_grads[i].zero_()
        
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()
        
        grad_chunk = torch.empty_like(stacked_grads[:chunk_size])
        
        reduce_future = dist.reduce_scatter_tensor(
            grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
        ).get_future()
        
        # Store in unified list with metadata
        self._reduce_scatter_infos.append(dict(
            optimizer_type='normuon',
            group=group,
            grad_chunk=grad_chunk,
            reduce_future=reduce_future
        ))

    @torch.no_grad()
    def step(self, optimizer_types=None):
        """
        Single step to handle both optimizers.
        
        Args:
            optimizer_types: List of optimizer types to step. 
                           If None, steps all types.
                           E.g., ['normuon'] to only step NorMuon params,
                                 ['adam', 'normuon'] to step both.
        """
        if optimizer_types is None:
            optimizer_types = ['adam', 'normuon']
        
        rank = dist.get_rank()
        all_gather_futures = []
        
        # Process all reduce_scatter futures (interleaved Adam and NorMuon)
        for info in self._reduce_scatter_infos:
            info['reduce_future'].wait()
            
            if info['optimizer_type'] == 'adam' and 'adam' in optimizer_types:
                # Process Adam parameter update
                param = info['param']
                grad_slice = info['grad_slice']
                
                # Get group for this parameter to access hyperparameters
                group = self._param_to_group[param]
                beta1, beta2 = group['betas']
                eps = group['eps']
                wd = group['weight_decay']
                
                rank_size = param.shape[0] // self.world_size
                p_slice = param[rank * rank_size:(rank + 1) * rank_size]
            
                # Prepare Scalar Inputs as 0-dim CPU Tensors
                # This ensures one kernel is reused for all steps and all parameter groups
                
                # 1. Calculate scalar values (Python floats)
                lr_val = group['lr'] * getattr(param, "lr_mul", 1.0)
                wd_val = group['weight_decay']
                wd_mul_val = getattr(param, "wd_mul", 1.0)
                
                state = self.state[param]
                state["step"] += 1
                t_val = state["step"]

                # 2. Wrap as CPU Tensors
                # Using device='cpu' avoids eager CUDA API calls in the loop
                lr_t = torch.tensor(lr_val, device='cpu', dtype=torch.float32)
                wd_t = torch.tensor(wd_val, device='cpu', dtype=torch.float32)
                wd_mul_t = torch.tensor(wd_mul_val, device='cpu', dtype=torch.float32)
                t_t = torch.tensor(t_val, device='cpu', dtype=torch.float32)
                
                # 3. Handle Hyperparams (beta/eps are usually constant, but safe to wrap)
                beta1, beta2 = group['betas']
                eps = group['eps']
                beta1_t = torch.tensor(beta1, device='cpu', dtype=torch.float32)
                beta2_t = torch.tensor(beta2, device='cpu', dtype=torch.float32)
                eps_t = torch.tensor(eps, device='cpu', dtype=torch.float32)
                
                # 4. Call Compiled Helper
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                
                _adam_update_inplace(p_slice, grad_slice, exp_avg, exp_avg_sq, 
                                     lr_t, beta1_t, beta2_t, eps_t, wd_t, t_t, wd_mul_t)
                
                # Launch all_gather
                gather_future = dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future()
                all_gather_futures.append(gather_future)
                
            elif info['optimizer_type'] == 'normuon' and 'normuon' in optimizer_types:
                # Process NorMuon group update
                group = info['group']
                grad_chunk = info['grad_chunk']
                params = group["params"]
                chunk_size = group["chunk_size"]
                padded_num_params = chunk_size * self.world_size
                
                start_idx = rank * chunk_size
                module_idx = start_idx if start_idx < len(params) else 0
                num_params = min(chunk_size, max(0, len(params) - start_idx))
                
                if "momentum_buffer" not in group:
                    group["momentum_buffer"] = torch.zeros_like(grad_chunk[:num_params])
                momentum_buffer = group["momentum_buffer"]
                
                # Apply momentum
                momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
                updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])
                
                grad_shape = updated_grads.shape
                
                # Handle attn reshaping
                if num_params > 0 and params[module_idx].label == 'attn':
                    for p in params[module_idx:module_idx + num_params]:
                        assert p.label == 'attn'
                    updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1] // 4, grad_shape[2])
                
                ref_param = params[module_idx] if num_params > 0 else params[0]
                param_shape = ref_param.shape
                is_gate = ref_param.label in ['smear_gate', 'attn_gate']
                
                # Initialize second momentum buffer
                if "second_momentum_buffer" not in group:
                    if is_gate:
                        group["second_momentum_buffer"] = torch.zeros_like(updated_grads[..., :, :1])
                    else:
                        group["second_momentum_buffer"] = (
                            torch.zeros_like(updated_grads[..., :, :1])
                            if param_shape[-2] >= param_shape[-1] 
                            else torch.zeros_like(updated_grads[..., :1, :])
                        )
                second_momentum_buffer = group["second_momentum_buffer"]
                
                # Initialize lr/wd multipliers
                if "param_lr_cpu" not in group:
                    lr_mults = []
                    wd_mults = []
                    for p in params:
                        shape = p.shape
                        if len(shape) >= 2:
                            shape_mult = max(1.0, shape[-2] / shape[-1]) ** 0.5
                        else:
                            shape_mult = 1.0
                        lr_mults.append(shape_mult * getattr(p, "lr_mul", 1.0))
                        wd_mults.append(getattr(p, "wd_mul", 1.0))
                    group["param_lr_cpu"] = torch.tensor(lr_mults, dtype=torch.float32, device="cpu")
                    group["param_wd_cpu"] = torch.tensor(wd_mults, dtype=torch.float32, device="cpu")
                
                eff_lr_all = group["param_lr_cpu"] * group["lr"]
                eff_wd_all = group["param_wd_cpu"] * group["weight_decay"] * group["lr"]
                eff_lr_cpu = eff_lr_all[module_idx:module_idx + num_params]
                eff_wd_cpu = eff_wd_all[module_idx:module_idx + num_params]
                
                # Apply polar express orthogonalization
                if num_params == 0:
                    v_chunk = updated_grads
                else:
                    v_chunk = polar_express(updated_grads, split_baddbmm=(ref_param.label == 'mlp'))
                
                # Apply variance reduction
                red_dim = -1 if (is_gate or param_shape[-2] >= param_shape[-1]) else -2
                v_chunk = apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, group["beta2"], red_dim)
                v_chunk = v_chunk.view(grad_shape)
                
                # Apply cautious weight decay and update parameters
                updated_params = torch.empty_like(grad_chunk)
                if num_params > 0:
                    param_chunk = torch.stack(params[module_idx:module_idx + num_params])
                    for local_idx in range(num_params):
                        cautious_wd_and_update_inplace(
                            param_chunk[local_idx],
                            v_chunk[local_idx],
                            eff_wd_cpu[local_idx],
                            eff_lr_cpu[local_idx],
                        )
                else:
                    param_chunk = torch.zeros_like(v_chunk)
                
                updated_params[:num_params].copy_(param_chunk)
                if num_params < chunk_size:
                    updated_params[num_params:].zero_()
                
                # Launch all_gather
                stacked_params = torch.empty(
                    (padded_num_params, *param_shape),
                    dtype=updated_params.dtype,
                    device=updated_params.device,
                )
                gather_future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
                all_gather_futures.append((gather_future, stacked_params, params))
        
        # Wait for all_gather operations and copy results back
        for item in all_gather_futures:
            if isinstance(item, tuple):
                # NorMuon: (future, stacked_params, orig_params)
                gather_future, stacked_params, orig_params = item
                gather_future.wait()
                unstacked_params = torch.unbind(stacked_params)
                for i, p in enumerate(orig_params):
                    p.copy_(unstacked_params[i], non_blocking=True)
            else:
                # Adam: just the future
                item.wait()
        
        # Clear futures for next iteration
        self._reduce_scatter_infos.clear()



# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()  # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

# yarn implementation @classiclarryd
class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//4)])
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.cos = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.sin = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.cos.copy_(theta.cos())
        self.sin.copy_(theta.sin())
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

def rotary(x_BTHD: Tensor, cos: Tensor, sin: Tensor):
    assert cos.size(0) >= x_BTHD.size(-3)
    cos, sin = (
        cos[None, : x_BTHD.size(-3), None, :],
        sin[None, : x_BTHD.size(-3), None, :],
    )
    x1, x2 = x_BTHD.chunk(2, dim=-1)
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), 3)

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    cos: torch.Tensor
    sin: torch.Tensor
    attn_scale: float

# If we're on a GH200, just use the existing flash attention installed.
# Otherwise, if we're on an H100, we'll use the official flash attention for the speedrun.
gpu_name = torch.cuda.get_device_properties(0).name
if "H100" in gpu_name:  # H100
    from kernels import get_kernel
    flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface
else:  # GH200 or other
    from flash_attn import flash_attn_interface


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = 0.5 * (self.dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKVO weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        #
        # Stacked layout stores all QKVO heads horizontally, allowing us to 
        # correctly calculate variance for output heads in NorMuon without
        # splitting off O. @chrisjmccormick  
        self.qkvo_w = nn.Parameter(torch.empty(self.dim * 4, self.hdim))
        # label module to enable custom optimizer sizing
        self.qkvo_w.label = 'attn'

        with torch.no_grad():
            self.qkvo_w[:self.dim * 3].uniform_(-bound, bound)  # init QKV weights
            self.qkvo_w[self.dim * 3:].zero_()  # init O weights to zero

        # sparse gated attention to enable context based no-op by @classiclarryd
        self.attn_gate = CastedLinear(12, num_heads)
        # label module to enable custom optimizer sizing
        self.attn_gate.weight.label = 'attn_gate'

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        cos, sin = attn_args.cos, attn_args.sin
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, attn_scale, bm_size = attn_args.seqlens, attn_args.attn_scale, attn_args.bm_size

        q, k, v = F.linear(x, self.qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = rotary(q, cos, sin), rotary(k, cos, sin)
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v) # @ KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = sa_lambdas[0] * v

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                        max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                        causal=True, softmax_scale=attn_scale, window_size=(bm_size, 0))
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, self.qkvo_w[self.dim * 3:].type_as(y))
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim        
        # Transposed layout to match attention weights
        self.c_fc = nn.Parameter(torch.empty(hdim, dim))
        self.c_proj = nn.Parameter(torch.empty(hdim, dim))
        # label modules to enable custom optimizer sizing
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        # corrective factor to account for transpose
        self.c_proj.lr_mul = 2.

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc.type_as(x))
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = F.linear(x, self.c_proj.T.type_as(x))
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        # skip MLP blocks for first MLP layer by @EmelyanenkoK
        self.mlp = MLP(dim) if layer_idx != 0 else None

    def forward(self, x: Tensor, x0: Tensor, lambdas: Tensor, attn_args: AttnArgs):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        # label module to enable explicit optimizer grouping
        self.embed.weight.label = 'embed'
        
        self.smear_gate = CastedLinear(12, 1)
        # label modules to enable custom optimizer sizing
        self.smear_gate.weight.label = 'smear_gate'

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        # label modules to enable explicit optimizer grouping
        for ve in self.value_embeds:
            ve.weight.label = 'value_embed'
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=(model_dim**0.5)/448, w_s=2**-9, grad_s=1/448)
        # label module to enable explicit optimizer grouping
        self.lm_head.weight.label = 'lm_head'
        # Add learnable skip connection weights for decoder layers
        assert num_layers % 2 == 0
        pad = (-num_layers * 5 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    -1.5
                    * torch.ones(num_layers),  # skip_weights -> σ(-1.5) ≈ 0.18
                    *[
                        torch.tensor([1.1, 0.0]) for _ in range(num_layers)
                    ],  # block lambdas. 1.1 init such that layer i weight is i^(num_layers-i).
                        # ~3x higher weight to layer 1 compared to 12 at init.
                    *[
                        torch.tensor([0.5, 0.5]) for _ in range(num_layers)
                    ],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    0.5*torch.ones(1), # backout_lambda
                    torch.ones(pad),
                ]
            )
        )
        # label module to enable explicit optimizer grouping
        self.scalars.label = 'scalars'
        # set learning rates
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 1.0
        self.scalars.lr_mul = 5.0

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, ws_short: int, ws_long: int):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        # dropping first layer updates this to .12 ... 012
        ve = [None, ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [None, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == len(self.blocks)

        x = self.embed(input_seq)

        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)
        smear_lambda = self.scalars[5 * len(self.blocks)]
        backout_lambda = self.scalars[5 * len(self.blocks)+1]

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # create skip connection from layer 4 (long attn window) to layer 7 (no attn op)
        skip_connections = []
        n = len(self.blocks) // 2
        skip_in = [4]
        skip_out = [7]

        x_backout = None
        backout_layer = 8
        # skip layer zero
        for i in range(1,len(self.blocks)):
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                cos=self.yarn.cos,
                sin=self.yarn.sin,
                attn_scale=self.yarn.attn_scale
            )
            if i in skip_out:
                gate = torch.sigmoid(skip_weights[i - n])  # in (0, 1)
                x = x + gate * skip_connections.pop()
            x = self.blocks[i](x, x0, lambdas[i], attn_args)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # back out contributions from first 8 layers that are only required for downstream context and not direct prediction
        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits / 7.5)
        logits_for_loss = logits.float() if not self.training else logits
        loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            target_seq,
            reduction="sum" if self.training else "mean",
        )
        return loss

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256

class BOSFinder:
    # Helper for getting sequences that start at the beginning of documents by @varunneal based on work by @classiclarryd
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        # Precompute BOS positions once per shard
        self.tokens=tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            # only scan first 4 million tokens, then kickoff async thread to scan rest
            self.bos_idx = (tokens[:4_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        # if quickload was used, repoint to the full dataset after 5 batches
        if self.quickload and self.batch_iter==5:
            self.get()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter+=1
        return starts, ends

class DataPreloader:
    # Helper for asynchronously loading next shard and indexing bos tokens
    def __init__(self, file_iter, world_size: int = 1):
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    assert num_tokens % (world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)  # median doc length is ~400

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)

        new_params = yield (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True)
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len


# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files: str = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files: str = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens: int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # batch sizes
    train_bs_schedule: tuple = (8 * 2048 * 8, 16 * 2048 * 8, 24 * 2048 * 8)
    train_bs_extension: int = 24 * 2048 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 4 * 64 * 1024 * 8
    # optimization
    num_scheduled_iterations: int = 2115  # number of steps to complete lr and ws schedule
    num_extension_iterations: int = 40  # number of steps to continue training at final lr and ws
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.55  # fraction of num_scheduled_iterations spent cooling down the learning rate
    # evaluation and logging
    run_id: str = run_id #f"{uuid.uuid4()}"
    val_loss_every: int = 250  # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint: bool = False
    # attention masking
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_final: int = 13 # increase final validation ws, used for YaRN extension and short window size @classiclarryd
    ws_validate_post_yarn_ext: int = 20 # extend long windows out even further after applying YaRN

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"
grad_accum_steps = 8 // world_size
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = args.run_id
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running Triton version {triton.__version__}")

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

model: nn.Module = GPT(
    vocab_size=50257,
    num_layers=12,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=args.val_batch_size // (grad_accum_steps * world_size)
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n and "gate" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]
gate_params = [p for n, p in model.named_parameters() if "gate" in n]

# init the optimizer
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
optimizer = CombinedOptimizer(
    embed_params + scalar_params + head_params + hidden_matrix_params + gate_params,
    adam_lr=0.008,
    adam_betas=(0.65, 0.95),
    adam_eps=1e-8,
    adam_weight_decay=0.0,
    normuon_lr=0.023,
    normuon_momentum=0.95,
    normuon_beta2=0.95,
    normuon_weight_decay=1.2,
    normuon_custom_sizing=True
)
optimizers = [optimizer]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

# learning rate schedule: tied to batch size schedule, with cooldown at the end.
def get_lr(step: int):
    if step > args.num_scheduled_iterations:
        return 0.1
    lr_max = 1.0
    x = step / args.num_scheduled_iterations
    if x > 0.33:
       lr_max = 1.51  # (16/8)**0.6
    if x > 0.66:
        lr_max = 1.93  # (24/8)**0.6
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = lr_max * w + (1 - w) * 0.1
        return lr
    return lr_max

def get_ws(step: int):
    # set short window size to half of long window size
    # Higher ws on "extension" steps
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]


def get_bs(step: int):
    if step >= args.num_scheduled_iterations:
        return args.train_bs_extension
    x = step / args.num_scheduled_iterations
    bs_idx = int(len(args.train_bs_schedule) * x)
    return args.train_bs_schedule[bs_idx]


def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    # warmup phase: linearly increase momentum from min to max
    # cooldown phase: linearly decrease momentum from max to min
    momentum_cd_start = args.num_iterations - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum

def step_optimizers(step: int, optimizers, model):
    # update lr
    optimizer = optimizers[0]  # single combined optimizer
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * get_lr(step)

    # set muon momentum based on step
    momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        if group.get('optimizer_type') == 'normuon':
            group["momentum"] = momentum

    # on even steps, only step Muon params
    # on odd steps, step all params
    if step%2==0:
        # Even steps: only step NorMuon parameters
        optimizer.step(optimizer_types=['normuon'])
        # Zero grad only for NorMuon parameters
        for group in optimizer.param_groups:
            if group.get('optimizer_type') == 'normuon':
                for param in group['params']:
                    param.grad = None
    else:
        # Odd steps: step both Adam and NorMuon parameters
        optimizer.step(optimizer_types=['adam', 'normuon'])
        # Zero grad for all parameters
        model.zero_grad(set_to_none=True)
        # disable sync in the next training step for the adam optimizer
        optimizer.should_sync_adam = False

model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)

########################################
#            Warmup kernels            #
########################################

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) # save the initial state
train_loader = distributed_data_generator(args.train_files, args.train_bs_schedule[0], args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
ws_schedule = list(args.ws_schedule) + [args.ws_final]
bs_schedule = list(args.train_bs_schedule) + [args.train_bs_extension]
ws_long = ws_schedule[0]
model.yarn.reset()
assert len(ws_schedule) == len(bs_schedule), "This warmup assumes len(ws_schedule) == len(bs_schedule)"

for idx in range(len(ws_schedule)):
    send_args = None
    if idx != 0:
        new_ws_long = ws_schedule[idx]
        model.yarn.apply(ws_long, new_ws_long)
        ws_long = new_ws_long
        send_args = (bs_schedule[idx], args.train_max_seq_len, grad_accum_steps)
    for step in range(warmup_steps):
        inputs, targets, cum_seqlens = train_loader.send(send_args)
        model(inputs, targets, cum_seqlens, ws_long//2, ws_long).backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

model.yarn.reset() # rotary buffer is not stored in state_dict
optimizer.reset() # muon momentum buffers not in state dict
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

step_batch_size = args.train_bs_schedule[0]
train_loader = distributed_data_generator(args.train_files, step_batch_size, args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
ws_short, ws_long = get_ws(0)
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    ws_short, new_ws_long = get_ws(step)
    if new_ws_long != ws_long:
        model.yarn.apply(ws_long, new_ws_long)
        ws_long=new_ws_long

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            ws_long = args.ws_validate_post_yarn_ext
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        assert args.val_tokens % args.val_batch_size == 0
        val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
        val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets, cum_seqlens = next(val_loader)
                val_loss += model(inputs, targets, cum_seqlens, ws_short, ws_long)
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    new_step_batch_size = get_bs(step)
    send_args = (new_step_batch_size, args.train_max_seq_len, grad_accum_steps) if new_step_batch_size != step_batch_size else None
    step_batch_size = new_step_batch_size
    for idx in range(grad_accum_steps):
        # enable gradient sync for the DistAdam optimizer on the last iteration before we step it
        if idx == grad_accum_steps - 1 and step % 2 == 1:
            optimizers[0].should_sync_adam = True

        inputs, targets, cum_seqlens = train_loader.send(send_args)
        (model(inputs, targets, cum_seqlens, ws_short, ws_long) / grad_accum_steps).backward()
    step_optimizers(step, optimizers, model)

    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()