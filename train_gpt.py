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
import gc

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch

torch.empty(
    1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True
).backward()  # prevents a bug on some systems
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
from kernels import get_kernel
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
# Compiled helpers for NorMuon by @chrisjmccormick

@torch.compile(dynamic=False, fullgraph=True)
def cautious_wd_and_update_inplace(p, mantissa, grad, wd_tensor, lr_tensor):
    """
    Cautious weight decay + parameter update. wd_tensor and lr_tensor are 0-D CPU tensors.
    Mantissa is tracked to enable higher precision updates on bfloat16 parameters.
    bfloat16 format: 1 sign bit + 8 exponent bits + 7 mantissa bits = 16 bits total
    float32 format: 1 sign bit + 8 exponent bits + 23 mantissa bits = 32 bits total
    """
    assert p.dtype == mantissa.dtype == torch.uint16
    grad = grad.float()
    wd_factor = wd_tensor.to(torch.float32)
    lr_factor = lr_tensor.to(torch.float32)
    p_precise_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
    p_precise = p_precise_raw.view(torch.float32)
    mask = (grad * p_precise) >= 0
    p_precise.copy_(p_precise - (p_precise * mask * wd_factor * lr_factor) - (grad * lr_factor))
    p.copy_((p_precise_raw >> 16).to(torch.uint16))
    mantissa.copy_(p_precise_raw.to(torch.uint16))

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


# -----------------------------------------------------------------------------
# NorMuon optimizer

class NorMuon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).

    Differences from standard Muon:
    - Newton-Shulz is replaced with Polar Express for the orthogonalization step
    - NorMuon adds a low-rank variance estimator similar to Adafactor. https://arxiv.org/pdf/2510.05491
    - small 1D parameters handled here instead of in Adam
    - Cautious weight decay, a gated version of decoupled weight decay
    - Custom distributed sizing:
    The model stores all attn and mlp weights in the same shape, and then updates the view as
    needed on the forward pass. This enables attn and mlp weights to be contained within the same
    dist.reduce_scatter_tensor() call. The model architecture has been customized to enable
    (n_attn_layers+n_mlp_layers*2)%8==0 for batching across 8 GPUs with zero padding on mlp and attn.
    The scheduling is:
        1. reduce scatter attn/mlp round 1 (10 attn params 6 mlp params)
        2. reduce scatter attn/mlp round 2 (16 mlp params)
        3. wait on step 1, then compute update of 1 and schedule all gather
        4. wait on step 2, then compute update of 2 and schedule all gather
            GPUs receive [2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 ATTN, 2 MLP, 2 MLP, 2 MLP]
            GPUs that receive params of type attn reshape before computing update
        5. wait for each all gather to complete and update params
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95, custom_sizing=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # custom sizing requires 8 GPUs
        if custom_sizing and dist.get_world_size()==8:
            param_groups = self.generate_custom_param_groups(params)
        else:
            param_groups = self.generate_standard_param_groups(params)
        super().__init__(param_groups, defaults)

    def reset(self):
        # expose a reset for clearing buffers
        for group in self.param_groups:
            if "momentum_buffer" in group:
                group["momentum_buffer"].zero_()
                group["mantissa"].zero_()
                group["second_momentum_buffer"].zero_()

    def generate_standard_param_groups(self, params):
        """
        Use this method if running on less than 8 GPU or experimenting with additional attn or mlp modules.
        Creates one param group per module.
        """
        groups = defaultdict(list)
        for param in params:
            groups[param.label].append(param)

        param_groups = []
        for module_name, group_params in groups.items():
            chunk_size = (len(group_params) + self.world_size - 1) // self.world_size
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))

        return param_groups

    def generate_custom_param_groups(self, params):
        """
        Implementation requires that a single GPU does not receive both attn
        and mlp params when a param group is split across GPUs.
        """
        params_list = list(params)
        module_group_order = ['attn', 'mlp']
        group_sizes = [16, 16]  # 10 attn + 6 mlp, then 16 mlp
        params_list.sort(key=lambda x: module_group_order.index(x.label))

        idx = 0
        assert len(params_list) == sum(group_sizes)
        param_groups = []
        for size in group_sizes:
            chunk_size = (size + self.world_size - 1) // self.world_size
            group_params = params_list[idx: idx + size]
            param_groups.append(dict(params=group_params, chunk_size=chunk_size))
            idx += size

        return param_groups

    def step(self):
        self.step_p1()
        self.step_p2()
        self.step_p3()
        
    @torch.no_grad()
    def step_p1(self):
        """
        Part 1: Launch distributed reduce_scatter operations for parameter groups.
        """
        rank = dist.get_rank()
        self.group_infos = []
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue

            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            stacked_grads = torch.empty(
                (padded_num_params, *params[0].shape),
                dtype=params[0].dtype,
                device=params[0].device
            )
            for i, p in enumerate(params):
                stacked_grads[i].copy_(p.grad, non_blocking=True)
            if len(params) < padded_num_params:
                stacked_grads[len(params):].zero_()

            grad_chunk = torch.empty_like(stacked_grads[:chunk_size])

            reduce_future = dist.reduce_scatter_tensor(
                grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True
            ).get_future()

            self.group_infos.append(dict(grad_chunk=grad_chunk, reduce_future=reduce_future))
    
    @torch.no_grad()
    def step_p2(self):
        """
        Part 2: Compute gradient updates and launch all_gather operations
        Wait for all gather to complete for all parameter groups except final one
        """
        # Efficient distributed step by @YouJiacheng, @KonstantinWilleke, @alexrgilbert,
        # @adricarda, @tuttyfrutyee, @vdlad, @ryanyang0, @vagrawal, @varunneal, @chrisjmccormick
        rank = dist.get_rank()
        group_infos = self.group_infos
        
        self.all_gather_infos = []
        # Second pass: wait for gradients, compute updates for the local shard of parameters,
        # and launch all async all_gather operations.
        for group, info in zip(self.param_groups, group_infos):
            info["reduce_future"].wait()

            params = group["params"]
            grad_chunk = info["grad_chunk"].float()
            chunk_size = group["chunk_size"]
            padded_num_params = chunk_size * self.world_size

            start_idx = rank * chunk_size
            module_idx = start_idx if start_idx < len(params) else 0

            num_params = min(chunk_size, max(0, len(params) - start_idx))  # num params for this rank

            if "momentum_buffer" not in group:
                group["momentum_buffer"]  = torch.zeros_like(grad_chunk[:num_params], dtype=torch.float32)
                
            momentum_buffer = group["momentum_buffer"]
            # Apply momentum update to the persistent momentum buffer in-place
            momentum_buffer.lerp_(grad_chunk[:num_params], 1 - group["momentum"])
            updated_grads = grad_chunk[:num_params].lerp_(momentum_buffer, group["momentum"])

            grad_shape = updated_grads.shape
            if params[module_idx].label == 'attn':
                for p in params[module_idx:module_idx + num_params]:
                    assert p.label == 'attn'
                updated_grads = updated_grads.view(4 * grad_shape[0], grad_shape[1] // 4, grad_shape[2])

            ref_param = params[module_idx]
            param_shape = ref_param.shape

            if "second_momentum_buffer" not in group:
                group["second_momentum_buffer"] = (torch.zeros_like(updated_grads[..., :, :1], dtype=torch.float32)
                    if param_shape[-2] >= param_shape[-1] else torch.zeros_like(updated_grads[..., :1, :])
                )
            second_momentum_buffer = group["second_momentum_buffer"]

            if "param_lr_cpu" not in group:
                # Define multipliers for ALL params in this group (global, not per-shard)
                lr_mults = []
                wd_mults = []
                for p in params:
                    # Increase learning rate for modules with larger inputs than outputs.
                    # This shape check also assumes rows=input, columns=output, so take care
                    # when changing memory layouts. @chrisjmccormick
                    shape = p.shape
                    if len(shape) >= 2:
                        shape_mult = max(1.0, shape[-2] / shape[-1]) ** 0.5
                    else:
                        shape_mult = 1.0
                    lr_mults.append(shape_mult * getattr(p, "lr_mul", 1.0))
                    wd_mults.append(getattr(p, "wd_mul", 1.0))
                # Define as cpu tensors to enable Inductor constant folding
                group["param_lr_cpu"] = torch.tensor(lr_mults, dtype=torch.float32, device="cpu")
                group["param_wd_cpu"] = torch.tensor(wd_mults, dtype=torch.float32, device="cpu")

            eff_lr_all = group["param_lr_cpu"] * group["lr"]
            eff_wd_all = group["param_wd_cpu"] * group["weight_decay"] * group["lr"]

            # Slice the portion corresponding to this rank's shard
            eff_lr_cpu = eff_lr_all[module_idx:module_idx + num_params]
            eff_wd_cpu = eff_wd_all[module_idx:module_idx + num_params]

            # Compute zeropower for the entire chunk in a single, batched call.
            if num_params == 0:
                v_chunk = updated_grads
            else:
                v_chunk = polar_express(updated_grads, split_baddbmm=(ref_param.label == 'mlp'))

            # Note that the head orientation in O is transposed relative to QKV, so red_dim
            # is 'incorrect' for O. However, correcting this showed no improvement. @chrisjmccormick
            red_dim = -1 if param_shape[-2] >= param_shape[-1] else -2

            v_chunk = apply_normuon_variance_reduction(
                v_chunk, second_momentum_buffer, group["beta2"], red_dim
            )

            v_chunk = v_chunk.view(grad_shape)

            # # "Cautious" weight decay (https://arxiv.org/abs/2510.12402)
            updated_params = torch.empty_like(grad_chunk, dtype=torch.bfloat16)
            if num_params > 0:
                # Work on a stacked copy to avoid touching original params
                param_chunk = torch.stack(params[module_idx:module_idx + num_params])

                if "mantissa" not in group:
                    group["mantissa"] = torch.zeros_like(param_chunk, dtype=torch.uint16)
                mantissa = group["mantissa"]

                for local_idx in range(num_params):
                    cautious_wd_and_update_inplace(
                        param_chunk[local_idx].view(torch.uint16),
                        mantissa[local_idx],
                        v_chunk[local_idx],
                        eff_wd_cpu[local_idx],
                        eff_lr_cpu[local_idx]
                    )
            else:
                param_chunk = torch.zeros_like(v_chunk)

            updated_params[:num_params].copy_(param_chunk)
            if num_params < chunk_size:
                updated_params[num_params:].zero_()

            stacked_params = torch.empty(
                (padded_num_params, *param_shape),
                dtype=updated_params.dtype,
                device=updated_params.device,
            )

            gather_future = dist.all_gather_into_tensor(
                stacked_params, updated_params, async_op=True
            ).get_future()

            self.all_gather_infos.append(
                {
                    "gather_future": gather_future,
                    "stacked_params": stacked_params,
                    "orig_params": params,
                }
            )

        # Final pass: wait for all_gather to complete for all except final and copy results back into original parameter tensors.
        for info in self.all_gather_infos[:-1]:
            info["gather_future"].wait()
            stacked_params = info["stacked_params"]
            orig_params = info["orig_params"]

            unstacked_params = torch.unbind(stacked_params)
            for i, p in enumerate(orig_params):
                p.copy_(unstacked_params[i], non_blocking=True)

    @torch.no_grad()
    def step_p3(self):
        """
        Part 3: Wait for final all gather to complete and copy results back into original parameter tensors
        """
        info = self.all_gather_infos[-1]
        info["gather_future"].wait()
        stacked_params = info["stacked_params"]
        orig_params = info["orig_params"]

        unstacked_params = torch.unbind(stacked_params)
        for i, p in enumerate(orig_params):
            p.copy_(unstacked_params[i], non_blocking=True)


class DistAdam(torch.optim.Optimizer):
    def __init__(self, params, label_order: list[str], betas: list[list[float]], lr: float = 1e-3, eps: float = 1e-8, weight_decay: float = 0.01):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        params = list(params)
        # Group by label, with explicit ordering for execution control.
        params_by_label = defaultdict(list)
        for p in params:
            params_by_label[getattr(p, 'label', None)].append(p)
        param_groups = []
        for idx, label in enumerate(label_order):
            if label in params_by_label:
                param_groups.append(dict(params=params_by_label[label], betas=betas[idx]))
        # include any unlabeled params at the end (processed last)
        if None in params_by_label:
            param_groups.append(dict(params=params_by_label[None]))
        super().__init__(param_groups, defaults)
        # init state: small params (numel < 1024) use full-sized state, others use sharded
        for p in params:
            chunk = p if p.numel() < 1024 else p[:p.size(0) // self.world_size]
            exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=p.device)
            self.state[p] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))

        # tag the final param for optimizer pipelining, run all gather after muon copy
        param_groups[-1]['params'][-1].is_final_param = True
        
        # DistributedAdam implementation by @vagrawal, @akash5474
        self.should_sync = False
        self._reduce_scatter_hooks = []
        self._reduce_scatter_futures = {}
        # 0-D CPU tensors to avoid recompilation in _update_step
        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self.register_backward_hooks()

    def register_backward_hooks(self):
        for group in self.param_groups:
            for param in group["params"]:
                self._reduce_scatter_hooks.append(param.register_post_accumulate_grad_hook(self._sync_gradient))

    def load_state_dict(self, state_dict):
        """Override to preserve optimizer state dtypes (avoid BFloat16->Float32 cast that causes recompilation)."""
        # Save original state dtypes before loading
        original_dtypes = {}
        for p, s in self.state.items():
            original_dtypes[p] = {k: v.dtype for k, v in s.items() if isinstance(v, torch.Tensor)}
        
        # Call parent load_state_dict (which may cast dtypes to match param dtype)
        super().load_state_dict(state_dict)
        
        # Restore original dtypes
        for p, s in self.state.items():
            if p in original_dtypes:
                for k, v in s.items():
                    if isinstance(v, torch.Tensor) and k in original_dtypes[p]:
                        if v.dtype != original_dtypes[p][k]:
                            s[k] = v.to(original_dtypes[p][k])

    @torch.no_grad()
    def _sync_gradient(self, param):
        if not self.should_sync:
            return

        grad = param.grad
        if param.numel() < 1024:
            # Small params: use all_reduce (no scatter/gather needed)
            self._reduce_scatter_futures[param] = (
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future(),
                grad
            )
        else:
            rank_size = grad.shape[0] // self.world_size
            if grad is not None:
                grad_slice = torch.empty_like(grad[:rank_size])
                self._reduce_scatter_futures[param] = (
                    dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future(),
                    grad_slice
                )

    def copy_lm_to_embed(self):
        # run at 2/3 of training
        lm_head = self.param_groups[0]['params'][0]
        embed = self.param_groups[-2]['params'][0]
        lm_head_state = self.state[lm_head]
        embed_state = self.state[embed]
        embed_state['step'] = lm_head_state['step']
        embed_state['exp_avg'] = lm_head_state['exp_avg'].clone()
        embed_state['exp_avg_sq'] = lm_head_state['exp_avg_sq'].clone()
        embed.data.copy_(lm_head.data)

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _update_step(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        """Compiled Adam update step. step_size_t and eff_wd_t are 0-D CPU tensors to avoid recompilation."""
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)  # exp_avg = beta1 * exp_avg + (1 - beta1) * g_slice
        exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)  # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g_slice^2
        # compute step
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)  # update = (exp_avg / (sqrt(exp_avg_sq) + eps)) * step_size
        # cautious weight decay
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)  # update += eff_wd_t * p_slice * mask
        p_slice.add_(other=update, alpha=-1.0)  # p_slice -= update

    @torch.no_grad()
    def step(self, muon_opt):
        muon_opt.step_p1()
        rank = dist.get_rank()
        all_gather_futures: list[torch.Future] = []

        last_param = None
        last_p_slice = None
        for group in self.param_groups:      
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for param in group['params']:
                if param not in self._reduce_scatter_futures:
                    continue

                fut, g_slice = self._reduce_scatter_futures[param]
                fut.wait()

                is_small = param.numel() < 1024
                if is_small:
                    # Small params: g_slice is actually full grad, p_slice is full param
                    p_slice = param
                else:
                    rank_size = param.shape[0] // self.world_size
                    p_slice = param[rank * rank_size:(rank + 1) * rank_size]

                lr = group['lr'] * getattr(param, "lr_mul", 1.0)
                state = self.state[param]
                state["step"] += 1
                t = state["step"]
                
                # Pre-compute changing values as 0-D CPU tensors to avoid recompilation.
                # `.fill_(value)` is the same as "= value", but doesn't change the tensor object.
                bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
                self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
                self._eff_wd_t.fill_(lr * lr * wd * getattr(param, "wd_mul", 1.0)) # `lr` included twice to serve as weight decay schedule.

                DistAdam._update_step(p_slice, g_slice, state["exp_avg"], state["exp_avg_sq"],
                                      beta1, beta2, eps, self._step_size_t, self._eff_wd_t)

                if not is_small:
                    if getattr(param, "is_final_param", False):
                        last_param = param
                        last_p_slice = p_slice
                    else:
                        all_gather_futures.append(dist.all_gather_into_tensor(param, p_slice, async_op=True).get_future())
        self._reduce_scatter_futures.clear()

        muon_opt.step_p2()
        torch.futures.collect_all(all_gather_futures).wait()

        if last_param is not None:
            last_all_gather_future = dist.all_gather_into_tensor(last_param, last_p_slice, async_op=True).get_future()
        muon_opt.step_p3()
        torch.futures.collect_all([last_all_gather_future]).wait()

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


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.factor1 = nn.Buffer(
            theta.cos().to(torch.bfloat16), persistent=False
        )
        self.factor2 = nn.Buffer(
            theta.sin().to(torch.bfloat16), persistent=False
        )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.factor1.copy_(theta.cos())
        self.factor2.copy_(theta.sin())
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

class YarnPairedHead(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        t_even = 2 * t
        t_odd = 2 * t + 1
        theta1 = torch.outer(t_even, angular_freq)
        theta2 = torch.outer(t_odd, angular_freq)
        self.factor1 = nn.Buffer(
            torch.cat((theta1.cos(),theta2.cos()), dim=-1).to(torch.bfloat16), 
            persistent=False
        )
        self.factor2 = nn.Buffer(
            torch.cat((theta1.sin(),theta2.sin()), dim=-1).to(torch.bfloat16), 
            persistent=False
        )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        t_even = 2 * t
        t_odd = 2 * t + 1
        theta1 = torch.outer(t_even, self.angular_freq)
        theta2 = torch.outer(t_odd, self.angular_freq)
        self.factor1.copy_(torch.cat((theta1.cos(),theta2.cos()), dim=-1))
        self.factor2.copy_( torch.cat((theta1.sin(),theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor
    ve_gate_w: torch.Tensor

flash_attn_interface = get_kernel('varunneal/flash-attention-3').flash_attn_interface

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = self.dim ** -0.5
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKVO weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        # Simplified layout by @chrisjmccormick
        self.qkvo_w = nn.Parameter(torch.empty(self.dim * 4, self.hdim, dtype=torch.bfloat16))
        # label all modules for explicit optimizer grouping
        self.qkvo_w.label = 'attn'

        with torch.no_grad():
            self.qkvo_w[:self.dim * 3].uniform_(-bound, bound)  # init QKV weights
            self.qkvo_w[self.dim * 3:].zero_()  # init O weights to zero

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        yarn = attn_args.yarn
        ve, sa_lambdas, key_offset = attn_args.ve, attn_args.sa_lambdas, attn_args.key_offset
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        # sparse gated attention to enable context based no-op by @classiclarryd
        # only include gates on layers with value embeds used on forward pass
        attn_gate_w, ve_gate_w = attn_args.attn_gate_w, attn_args.ve_gate_w

        q, k, v = F.linear(x, sa_lambdas[0] * self.qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = yarn.rotary(q), yarn.rotary(k)
        if key_offset:
            # shift keys forward for the stationary head dims. Enables 1-layer induction.
            k[:, 1:, :, self.head_dim // 4:self.head_dim // 2] = k[:, :-1, :, self.head_dim // 4:self.head_dim // 2]
            k[:, 1:, :, 3 * self.head_dim // 4:] = k[:, :-1, :, 3 * self.head_dim // 4:]
        if ve is not None:
            ve_gate_out = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w)).view(B, T, self.num_heads, 1)
            v = v + ve_gate_out * ve.view_as(v) # @ KoszarskyB & @Grad62304977

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                        max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                        causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0))
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y

class PairedHeadCausalSelfAttention(nn.Module):
    """
    Pairs up attention heads such that queries from head 1 can attend to keys in head 2, and vice-versa.
    Implemented by interleaving the k, q, and v for pairs of heads to form twice as long sequences
    EG [k1_h1, k2_h1, k3_h1], [k1_h2, k2_h2, k3_h2] -> [k1_h1, k1_h2, k2_h1, k2_h2, k3_h1, k3_h2], repeat for q and v
    """
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim

        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        std = self.dim ** -0.5
        bound = (3 ** 0.5) * std
        self.qkvo_w = nn.Parameter(torch.empty(self.dim * 4, self.hdim, dtype=torch.bfloat16))
        self.qkvo_w.label = 'attn'

        with torch.no_grad():
            self.qkvo_w[:self.dim * 3].uniform_(-bound, bound)
            self.qkvo_w[self.dim * 3:].zero_()

    def forward(self, x: Tensor, attn_args: AttnArgs):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        yarn = attn_args.yarn
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        attn_gate_w, ve_gate_w = attn_args.attn_gate_w, attn_args.ve_gate_w

        q, k, v = F.linear(x, sa_lambdas[0] * self.qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)

        # delay q,k reshape until rotary makes data contiguous, to enable view (non-copy)
        q = q.view(B, T, self.num_heads // 2, self.head_dim * 2)
        k = k.view(B, T, self.num_heads // 2, self.head_dim * 2)
        v = v.reshape(B, T*2, self.num_heads//2, self.head_dim)

        q, k = yarn.rotary(q), yarn.rotary(k)

        q = q.view(B, T*2, self.num_heads//2, self.head_dim)
        k = k.view(B, T*2, self.num_heads//2, self.head_dim)

        if ve is not None:
            ve_gate_out = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w)).view(B, T*2, self.num_heads//2, 1)
            v = v + ve_gate_out * ve.view_as(v)

        max_len = args.train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        # paired head correction
        seqlens = 2 * seqlens
        max_len = 2 * max_len

        y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                        max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                        causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0))
        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * self.qkvo_w[self.dim * 3:].type_as(y))
        return y

@triton.jit
def linear_relu_square_kernel(a_desc, b_desc, c_desc, aux_desc,
                                 M, N, K,
                                 BLOCK_SIZE_M: tl.constexpr,
                                 BLOCK_SIZE_N: tl.constexpr,
                                 BLOCK_SIZE_K: tl.constexpr,
                                 GROUP_SIZE_M: tl.constexpr,
                                 NUM_SMS: tl.constexpr,
                                 FORWARD: tl.constexpr,
                                 ):
    dtype = tl.bfloat16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_SMS
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_am_c = pid_m * BLOCK_SIZE_M
        offs_bn_c = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)

        c0 = acc0.to(dtype)
        if not FORWARD:
            c0_pre = aux_desc.load([offs_am_c, offs_bn_c])
            c0 = 2 * c0 * tl.where(c0_pre > 0, c0_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c], c0)

        if FORWARD:
            c0_post = tl.maximum(c0, 0)
            c0_post = c0_post * c0_post
            aux_desc.store([offs_am_c, offs_bn_c], c0_post)

        c1 = acc1.to(dtype)
        if not FORWARD:
            c1_pre = aux_desc.load([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2])
            c1 = 2 * c1 * tl.where(c1_pre > 0, c1_pre, 0)

        c_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)

        if FORWARD:
            c1_post = tl.maximum(c1, 0)
            c1_post = c1_post * c1_post
            aux_desc.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1_post)


def linear_relu_square(a, b, aux=None):
    M, K = a.shape
    N, K = b.shape
    dtype = a.dtype

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    FORWARD = False
    if aux is None:
        FORWARD = True
        aux = torch.empty((M, N), device=a.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 64
    num_stages = 4 if FORWARD else 3
    num_warps = 8

    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_SIZE_M, BLOCK_SIZE_K])
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = TensorDescriptor.from_tensor(c, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])
    aux_desc = TensorDescriptor.from_tensor(aux, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2])

    def grid(META):
        return (min(
            NUM_SMS,
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        ), )

    linear_relu_square_kernel[grid](
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=1,
        NUM_SMS=NUM_SMS,
        FORWARD=FORWARD,
        num_stages=num_stages,
        num_warps=num_warps
    )

    if FORWARD:
        return c, aux
    else:
        return c

class FusedLinearReLUSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2):
        pre, post = linear_relu_square(x.view((-1, x.shape[-1])), W1)
        x3 = post @ W2
        ctx.save_for_backward(x, W1, W2, pre, post)
        return x3.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        dW2 = post.T @ grad_output
        dpre = linear_relu_square(grad_output.view((-1, grad_output.shape[-1])), W2, aux=pre)
        dW1 = dpre.T @ x
        dx = dpre @ W1
        return dx.view(x.shape), dW1, dW2

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        # Transposed layout to match attention weights
        self.c_fc = nn.Parameter(torch.empty(hdim, dim, dtype=torch.bfloat16))
        self.c_proj = nn.Parameter(torch.empty(hdim, dim, dtype=torch.bfloat16))
        # label all modules for explicit optimizer grouping
        self.c_fc.label = 'mlp'
        self.c_proj.label = 'mlp'
        self.c_proj.lr_mul = 2.

        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        with torch.no_grad():
            self.c_fc.uniform_(-bound, bound)
            self.c_proj.zero_() # zero init suggested by @Grad62304977

    def forward(self, x: Tensor):
        # relu(x)^2:
        # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977

        # This call computes relu(x @ W1.T)^2 @ W2.T
        return FusedLinearReLUSquareFunction.apply(x, self.c_fc, self.c_proj)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int, use_paired_head: bool):
        super().__init__()
        # skip attention of blocks.6 (the 7th layer) by @YouJiacheng
        if use_paired_head:
            self.attn = PairedHeadCausalSelfAttention(dim, head_dim, num_heads, layer_idx)
        else:
            self.attn = CausalSelfAttention(dim, head_dim, num_heads, layer_idx) if layer_idx != 6 else None
        # skip MLP blocks for first MLP layer by @EmelyanenkoK
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, attn_args: AttnArgs):
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args)
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

@dataclass
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    ws_short: int
    ws_long: int

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        vocab_size = next_multiple_of_n(vocab_size, n=128)

        self.smear_gate = CastedLinear(12, 1)
        self.smear_gate.weight.label = 'smear_gate'
        self.smear_gate.weight.lr_mul = 0.01
        self.smear_gate.weight.wd_mul = 0.0

        self.skip_gate = CastedLinear(12, 1)
        self.skip_gate.weight.label = 'skip_gate'
        self.skip_gate.weight.lr_mul = 0.05
        self.skip_gate.weight.wd_mul = 0.0

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        for embed in self.value_embeds:
            nn.init.zeros_(embed.weight)
        for ve in self.value_embeds:
            ve.weight.label = 'value_embed'
        
        # parameter banks for attention and value embedding gate weights
        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12)) # 10 layers
        self.attn_gate_bank.label = 'attn_gate_bank'
        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12)) # 5 layers
        self.ve_gate_bank.label = 've_gate_bank'

        self.paired_head_layers = [0, 2, 5, 9]
        self.blocks = nn.ModuleList([Block(model_dim, head_dim, num_heads, i, i in self.paired_head_layers) for i in range(num_layers)])
        self.yarn = Yarn(head_dim, max_seq_len)
        self.yarn_paired_head = YarnPairedHead(head_dim, max_seq_len)
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        use_fp8 = not os.environ.get("DISABLE_FP8", False)

        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=use_fp8, x_s=100/448, w_s=1.6/448, grad_s=0.75/448)
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)
        self.lm_head.weight.label = 'lm_head'

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.embed.weight.label = 'embed'

        # x0_lambdas separated out for different optimizer treatment (no beta smoothing)
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.x0_lambdas.label = 'x0_lambdas'
        self.x0_lambdas.lr_mul = 5.0
        self.x0_lambdas.wd_mul = 0.0

        pad = (-num_layers * 3 - 3) % dist.get_world_size()  # updated: 3*num_layers instead of 4*
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    1.1 * torch.ones(num_layers),  # resid lambdas. 1.1 init such that layer i weight is i^(num_layers-i).
                    *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    0.5*torch.ones(1), # backout_lambda
                    -1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18
                    torch.ones(pad),
                ]
            )
        )

        self.scalars.label = 'scalars'
        # set learning rates
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
            param.wd_mul = 5.
        for param in self.embed.parameters():
            param.wd_mul = 150.
        for param in self.lm_head.parameters():
            param.wd_mul = 150.
        self.scalars.lr_mul = 5.0
        self.scalars.wd_mul = 0.0

        self.split_embed = False

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, schedule_cfg: ForwardScheduleConfig):
        assert input_seq.ndim == 1

        # unpack schedule_cfg
        mtp_weights, ws_short, ws_long = schedule_cfg.mtp_weights, schedule_cfg.ws_short, schedule_cfg.ws_long

        # set configs
        skip_connections = []
        skip_in = [3] # long attention window on layer 3
        skip_out = [6] # no attn op on layer 6
        x_backout = None
        backout_layer = 7

        # set lambdas
        resid_lambdas = self.scalars[: 1 * self.num_layers]
        x0_lambdas = self.x0_lambdas
        sa_lambdas = self.scalars[1 * self.num_layers: 3 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[3 * self.num_layers]
        backout_lambda = self.scalars[3 * self.num_layers+1]
        skip_lambda = self.scalars[3 * self.num_layers+2]

        # set block masks and key shift
        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == self.num_layers
        key_offset = [b==long_bm for b in bm_sizes] # apply partial key offset to long windows

        # weight-tied: use lm_head.weight for embedding lookup (or separate embed after split)
        if self.split_embed:
            x = self.embed(input_seq)
        else:
            x = F.embedding(input_seq, self.lm_head.weight)
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        # dropping first layer updates this to .12 ... 012
        ve = [ve[1], ve[2]] + [None] * (self.num_layers - 5) + [ve[0], ve[1], ve[2]]
        assert len(ve) == self.num_layers

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # unbind gate banks to avoid select_backwards kernel
        ag = [w.bfloat16() for w in self.attn_gate_bank.unbind(0)] 
        veg = [w.bfloat16() for w in self.ve_gate_bank.unbind(0)]
        attn_gates = ag[:6] + [None] + ag[6:]
        ve_gates = [veg[0], veg[1]] + [None] * (self.num_layers - 5) + [veg[2], veg[3], veg[4]]
        assert len(attn_gates) == self.num_layers
        assert len(ve_gates) == self.num_layers

        for i in range(self.num_layers):
            yarn = self.yarn_paired_head if i in self.paired_head_layers else self.yarn
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                yarn=yarn,
                key_offset=key_offset[i],
                attn_gate_w=attn_gates[i],
                ve_gate_w=ve_gates[i]
            )
            if i in skip_out:
                skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)]))
                x = x + skip_gate_out * skip_connections.pop()
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0
            x = self.blocks[i](x, attn_args)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # back out contributions from first 7 layers that are only required for downstream context and not direct prediction
        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1). @classiclarryd updated to 23*sigmoid((logits+5)/7.5)
        logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        logits_for_loss = logits.float() if not self.training else logits

        n_predict = mtp_weights.size(0) if mtp_weights is not None else 1
        if self.training and n_predict > 1:
            # Multi-token prediction: take loss of the weighted average of next n_predict tokens
            logits_flat = logits_for_loss.view(-1, logits_for_loss.size(-1))
            idx = F.pad(target_seq, (0, n_predict - 1)).unfold(0, n_predict, 1)  # [T, n_predict] of shifted targets
            target_logits = logits_flat.gather(1, idx)
            cross_entropy = torch.logsumexp(logits_flat, dim=-1).unsqueeze(1) - target_logits
            for k in range(1, n_predict):  # zero out preds past end of sequence
                cross_entropy[-k:, k] = 0
            loss = (cross_entropy * mtp_weights).sum()
        elif self.training:
            loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="sum")
        else:
            loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="mean")
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
# Training Management
 
def get_bs(step: int):
    if step >= args.num_scheduled_iterations:
        return args.train_bs_extension
    x = step / args.num_scheduled_iterations
    bs_idx = int(len(args.train_bs_schedule) * x)
    return args.train_bs_schedule[bs_idx]

def get_ws(step: int):
    # set short window size to half of long window size
    # Higher ws on "extension" steps
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

# learning rate schedule: tied to batch size schedule, with cooldown at the end.
def get_lr(step: int):
    if step > args.num_scheduled_iterations:
        return 0.1
    lr_max = 1.0
    x = step / args.num_scheduled_iterations
    if x > 1/3:
       lr_max = 1.52  # (16/8)**0.6
    if x > 2/3:
        lr_max = 1.73  # (24/8)**0.5
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = lr_max * w + (1 - w) * 0.1
        return lr
    return lr_max

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

class TrainingManager():
    """
    Manages three optimizers for Adam embed/lm_head, Adam scalars, and Muon weight matrices.
    Notable Features:
        1. Scalars are given higher momentum terms to smooth learning @ChrisJMcCormick
        2. Scalar weights are temporarily frozen during batch size or window size updates @ChrisJMcCormick
        3. Adam optimizers are only stepped on odd steps @classiclarryd
        4. Adam optimizers have hooks to start gradient communication during backwards pass @akash5474
        5. Muon has a linear momentum warmup and cooldown schedule
        6. Learning rates follow a linear decay schedule
        7. Embed/lm_head weights and optimizer state splits at 2/3 of training @classiclarryd

    Manages model architecture, data, and target that changes during training
    Notable Features:
        1. Multi Token Prediction schedule of [1, 0.5, 0.25->0] -> [1, 0.5->0] -> [1] @varunneal
        2. Sliding Attention window schedule of [1,3] -> [3,7] -> [5,11] -> [6,13]
        3. YaRN updates to RoPE on window changes
        4. Split embed and lm head at 2/3 of training
        5. Batch size schedule of 8 -> 16 -> 24
        6. Post training extension of long windows from 13 to 20
    """
    def __init__(self, model):
        self.mtp_weights_schedule = self._build_mtp_schedule()
        self.model = model
        adam_betas = {
            'lm_head': [0.5, 0.95],
            'smear_gate': [0.9, 0.99],
            'attn_gate_bank': [0.9, 0.99],
            've_gate_bank': [0.9, 0.99],
            'skip_gate': [0.9, 0.99],
            'x0_lambdas': [0.65, 0.95],
            'scalars': [0.9, 0.99],
            'embed': [0.5, 0.95],
            'value_embed': [0.75, 0.95]
        }
        adam_labels = list(adam_betas.keys())
        adam_beta_values = list(adam_betas.values())
        muon_labels = ['attn', 'mlp']
        adam_params = [p for p in model.parameters() if getattr(p, 'label', None) in adam_labels]
        muon_params = [p for p in model.parameters() if getattr(p, 'label', None) in muon_labels]
        assert set(getattr(p, 'label', None) for p in model.parameters()) == set(adam_labels + muon_labels), "All params must have label"

        self.adam_opt = DistAdam(adam_params, adam_labels, adam_beta_values, lr=0.008, eps=1e-10, weight_decay=0.005)
        self.muon_opt = NorMuon(muon_params, lr=0.023, momentum=0.95, beta2=0.95, weight_decay=1.2)
        self.optimizers = [self.adam_opt, self.muon_opt]

        # split after odd number step
        self.split_step = math.ceil(args.split_embed_frac * args.num_scheduled_iterations) | 1

        # set defaults
        for opt in self.optimizers:
            opt.odd_step_only = False 
            opt.should_sync = True
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]

        # on even steps, only step Muon params
        self.adam_opt.odd_step_only = True

        self.reset()

    def _build_mtp_schedule(self):
        # Precompute MTP weights for all steps to avoid tensor allocation during training
        # Schedule: [1, 0.5, 0.25->0] -> [1, 0.5->0] -> [1]
        mtp_weights_schedule = []
        for s in range(args.num_iterations + 1):
            x = s / args.num_scheduled_iterations
            if x < 1/3:
                w = [1.0, 0.5, 0.25 * (1 - 3*x)]
            elif x < 2/3:
                w = [1.0, 0.5 * (1 - (3*x - 1))]
            else:
                w = [1.0]
            mtp_weights_schedule.append(torch.tensor(w, device=device))
        return mtp_weights_schedule

    def apply_final_ws_ext(self):
        self.ws_long = args.ws_validate_post_yarn_ext

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights = self.mtp_weights,
            ws_short = self.ws_short,
            ws_long = self.ws_long
        )
    
    def _is_active_step(self, opt, step: int):
        return (opt.odd_step_only and step%2==1) or not opt.odd_step_only

    def get_transition_steps(self):
        transition_steps = []
        ws_short, ws_long = get_ws(0)
        for step in range(1, args.num_iterations):
            ws_short, new_ws_long = get_ws(step)
            if new_ws_long != ws_long:
                transition_steps.append(step)
                ws_long = new_ws_long
        return transition_steps

    def advance_schedule(self, step: int):
        self.ws_short, new_ws_long = get_ws(step)
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long, new_ws_long)
            self.model.yarn_paired_head.apply(self.ws_long, new_ws_long)

        new_batch_size = get_bs(step)
        if new_batch_size != self.batch_size:
            self.train_loader_send_args = (new_batch_size, args.train_max_seq_len, grad_accum_steps)
        else:
            self.train_loader_send_args = None

        self.ws_long = new_ws_long
        self.mtp_weights = self.mtp_weights_schedule[step]
    
    def step_optimizers(self, step: int):                
        step_lr = get_lr(step)
        muon_momentum = get_muon_momentum(step)
        for group in self.muon_opt.param_groups:
            group["momentum"] = muon_momentum

        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * step_lr
                
        if self._is_active_step(self.adam_opt, step):
            # adam will interleave calls to muon step
            self.adam_opt.step(self.muon_opt)
            self.model.zero_grad(set_to_none=True)
            self.adam_opt.should_sync = False
        else:
            self.muon_opt.step()
            self.muon_opt.zero_grad(set_to_none=True)
            
        if step == self.split_step:
            self.adam_opt.copy_lm_to_embed()
            self.model.split_embed = True
    
    def activate_hooks(self, step: int):
        for opt in self.optimizers:
            if self._is_active_step(opt, step):
                opt.should_sync = True

    def reset(self, state=None):
        if state is not None:
            for opt, opt_state in zip(self.optimizers, state):
                opt.should_sync = False
                opt.load_state_dict(opt_state)

        # muon momentum buffers not in state dict
        self.muon_opt.reset()
        self.model.split_embed = False

        self.ws_short, self.ws_long = get_ws(0)
        self.batch_size = get_bs(0)
        self.model.yarn.reset()
        self.model.yarn_paired_head.reset()

    def get_state(self):
        return [copy.deepcopy(opt.state_dict()) for opt in self.optimizers]

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
    num_scheduled_iterations: int = 1735  # number of steps to complete lr and ws schedule
    num_extension_iterations: int = 40  # number of steps to continue training at final lr and ws
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.50  # fraction of num_scheduled_iterations spent cooling down the learning rate
    split_embed_frac: float = 2/3  # fraction of training when embeddings split from lm_head
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
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
    num_layers=11,
    num_heads=6,
    head_dim=128,
    model_dim=768,
    max_seq_len=args.val_batch_size // (grad_accum_steps * world_size)
).cuda()
for m in model.modules():
    if isinstance(m, (nn.Embedding, nn.Linear)):
        m.weight.data = m.weight.data.bfloat16()
model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)
training_manager = TrainingManager(model)

########################################
#            Warmup kernels            #
########################################
print0("Compiling model and warming up kernels (~7 minutes on first execution)", console=True)
# Warmup the training kernels, then re-initialize the state so we aren't cheating
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=training_manager.get_state()) # save the initial state
train_loader = distributed_data_generator(args.train_files, args.train_bs_schedule[0], args.train_max_seq_len, grad_accum_steps=grad_accum_steps)
val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)

transition_steps = training_manager.get_transition_steps()
# first few steps plus transitions
warmup_steps = sorted({0, 1, 2} | set(s + offset for s in transition_steps for offset in [-1, 0, 1] if s + offset >= 0)) 
print0(f"Sampling steps {warmup_steps} for warmup", console=True)
for step in warmup_steps:
    training_manager.advance_schedule(step)
    model.eval()
    with torch.no_grad():
        inputs, targets, cum_seqlens = next(val_loader)
        model(inputs, targets, cum_seqlens, training_manager.get_forward_args())
    model.train()
    for idx in range(grad_accum_steps):
        # enable gradient sync for the DistAdam optimizers on the last iteration before we step them
        if idx == grad_accum_steps - 1:
            training_manager.activate_hooks(step)
        send_args = training_manager.train_loader_send_args
        inputs, targets, cum_seqlens = train_loader.send(send_args)
        (model(inputs, targets, cum_seqlens, training_manager.get_forward_args()) / grad_accum_steps).backward()
    training_manager.step_optimizers(step)
print0("Resetting Model", console=True)
model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
training_manager.reset(initial_state["optimizers"])
del val_loader, train_loader, initial_state
model.train()

########################################
#        Training and validation       #
########################################
train_loader = distributed_data_generator(args.train_files, args.train_bs_schedule[0], args.train_max_seq_len, grad_accum_steps=grad_accum_steps)

gc.collect()

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    training_manager.advance_schedule(step)
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        if last_step:
            training_manager.apply_final_ws_ext()
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
                val_loss += model(inputs, targets, cum_seqlens, training_manager.get_forward_args())
        val_loss /= val_steps
        del val_loader
        dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
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
    for idx in range(grad_accum_steps):
        # enable gradient sync for the DistAdam optimizers on the last iteration before we step them
        if idx == grad_accum_steps - 1:
            training_manager.activate_hooks(step)
        send_args = training_manager.train_loader_send_args
        inputs, targets, cum_seqlens = train_loader.send(send_args)
        (model(inputs, targets, cum_seqlens, training_manager.get_forward_args()) / grad_accum_steps).backward()
    training_manager.step_optimizers(step)

    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()
