"""FP8 weight caches for the packed attention QKV projection (see qkv_rope.PackedFP8QKV).

What it replaces: per attention layer, cat([Wq; Wk; Wv]) -> amax -> quantize -> transpose in aten.
Why it is faster: one launch per batch of same-shaped layers reads the QK and V rows straight from
their banks (no cat) and writes both fp8 layouts the forward and backward GEMMs read, so neither
direction pays a transpose.
Invariants: the scale is exact (this step's own amax / 448), refreshed after every optimizer step;
the +-448 clamp is explicit, so the bytes match the aten chain.

Provenance: record #344 (packed weight cache); record #360 (one cache per attention width class).
"""
import torch
import triton
import triton.language as tl
from torch import nn

# Partial amaxes per layer: the amax pass splits each layer's packed rows over this many programs.
AMAX_PARTS = 32


@triton.jit
def _packed_batched_matrix_amax_partial_kernel(
    first, second, partial_amax,
    FIRST_ELEMENTS: tl.constexpr,
    SECOND_ELEMENTS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr,
    SECOND_STRIDE_B: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Amax over a logical row-pack without materializing the concatenation."""
    batch = tl.program_id(0)
    part = tl.program_id(1)
    total_elements = FIRST_ELEMENTS + SECOND_ELEMENTS
    offsets = tl.arange(0, BLOCK_SIZE)
    local_amax = tl.zeros((BLOCK_SIZE,), tl.float32)
    for start in tl.range(part * BLOCK_SIZE, total_elements, NUM_PARTS * BLOCK_SIZE):
        indices = start + offsets
        valid = indices < total_elements
        in_first = indices < FIRST_ELEMENTS
        first_values = tl.load(first + batch * FIRST_STRIDE_B + indices, mask=valid & in_first, other=0.0)
        second_indices = tl.maximum(indices - FIRST_ELEMENTS, 0)
        second_values = tl.load(second + batch * SECOND_STRIDE_B + second_indices, mask=valid & ~in_first, other=0.0)
        values = first_values.to(tl.float32) + second_values.to(tl.float32)
        local_amax = tl.maximum(local_amax, tl.abs(values))
    tl.store(partial_amax + batch * NUM_PARTS + part, tl.max(local_amax, axis=0))


@triton.jit
def _quantize_dual_layout_packed_batched_kernel(
    first, second, row, transposed, scale_ptr,
    M, N,
    FIRST_ROWS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr,
    SECOND_STRIDE_B: tl.constexpr,
    ROW_STRIDE_B: tl.constexpr,
    TRANSPOSED_STRIDE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Quantize a logical row-pack directly from its two source allocations: one read of each source,
    two fp8 writes (row-major, and transposed via a register-tile tl.trans)."""
    batch = tl.program_id(2)
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    in_first = offs_m[:, None] < FIRST_ROWS
    first_offsets = offs_m[:, None] * N + offs_n[None, :]
    second_rows = tl.maximum(offs_m - FIRST_ROWS, 0)
    second_offsets = second_rows[:, None] * N + offs_n[None, :]
    source_ptrs = tl.where(in_first, first + batch * FIRST_STRIDE_B + first_offsets,
                           second + batch * SECOND_STRIDE_B + second_offsets)
    values = tl.load(source_ptrs, mask=valid, other=0.0).to(tl.float32)
    scaled = values / tl.load(scale_ptr + batch)
    quantized = tl.maximum(tl.minimum(scaled, 448.0), -448.0).to(tl.float8e4nv)
    tl.store(row + batch * ROW_STRIDE_B + offs_m[:, None] * N + offs_n[None, :], quantized, mask=valid)
    transposed_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(transposed + batch * TRANSPOSED_STRIDE_B + offs_n[:, None] * M + offs_m[None, :],
             tl.trans(quantized), mask=transposed_mask)


def quantize_dual_layout_packed_batched(first, second, scales, partial_amax, row, transposed):
    """Exact-scale dual-layout quantization of the row concatenation [first; second], per batch entry."""
    B, first_rows, N = first.shape
    B2, second_rows, N2 = second.shape
    total_rows = first_rows + second_rows
    num_parts = partial_amax.shape[-1]
    assert (B2, N2) == (B, N) and first.device == second.device and first.dtype == second.dtype
    # Row-major sources: a strided dim-0 view of a bank is fine, a row-strided view is not.
    assert first.stride(1) == second.stride(1) == N and first.stride(2) == second.stride(2) == 1
    assert scales.shape == (B,) and partial_amax.shape == (B, num_parts)
    assert row.shape == (B, total_rows, N) and transposed.shape == (B, N, total_rows)
    assert row.dtype == transposed.dtype == torch.float8_e4m3fn
    assert row.is_contiguous() and transposed.is_contiguous()
    _packed_batched_matrix_amax_partial_kernel[(B, num_parts)](
        first, second, partial_amax,
        FIRST_ELEMENTS=first_rows * N, SECOND_ELEMENTS=second_rows * N,
        FIRST_STRIDE_B=first.stride(0), SECOND_STRIDE_B=second.stride(0),
        NUM_PARTS=num_parts, BLOCK_SIZE=2048, num_stages=1, num_warps=8,
    )
    # Exact-current scale: amax / 448, no headroom.
    torch.mul(partial_amax.amax(dim=1).clamp_(min=1.0e-12), 1.0 / 448.0, out=scales)
    block_m, block_n = 64, 128
    grid = (triton.cdiv(total_rows, block_m), triton.cdiv(N, block_n), B)
    _quantize_dual_layout_packed_batched_kernel[grid](
        first, second, row, transposed, scales,
        total_rows, N,
        FIRST_ROWS=first_rows,
        FIRST_STRIDE_B=first.stride(0), SECOND_STRIDE_B=second.stride(0),
        ROW_STRIDE_B=row.stride(0), TRANSPOSED_STRIDE_B=transposed.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, num_stages=2, num_warps=8,
    )
    return row, transposed


class PackedQKVFP8Cache(nn.Module):
    """The fp8 [Q; K; V] weights of one batch of same-shaped attention layers, in both layouts.

    Buffers (none persistent: every refresh rewrites them from the banks):
      row    [layers, rows, cols]  e4m3, the forward GEMM's operand
      col    [layers, cols, rows]  e4m3, the input-gradient GEMM's operand
      scales [layers]              each layer's exact dequant scale
    """
    def __init__(self, num_layers: int, rows: int, cols: int, device: torch.device):
        super().__init__()
        e4m3 = torch.float8_e4m3fn
        self.register_buffer("row", torch.empty(num_layers, rows, cols, dtype=e4m3, device=device), persistent=False)
        self.register_buffer("col", torch.empty(num_layers, cols, rows, dtype=e4m3, device=device), persistent=False)
        self.register_buffer("scales", torch.ones(num_layers, device=device), persistent=False)
        self.register_buffer("partial_amax", torch.zeros(num_layers, AMAX_PARTS, device=device), persistent=False)

    def refresh(self, qk: torch.Tensor, v: torch.Tensor):
        """Requantize from qk [layers, qk_rows, cols] and v [layers, v_rows, cols]."""
        quantize_dual_layout_packed_batched(qk, v, self.scales, self.partial_amax, self.row, self.col)
