"""Row scatter kernels for the row-sparse tables (the n-gram table, value_embeds): one program per source row.

What they replace:
    dst.index_add_(0, idx, src.to(dst.dtype))      scatter_add_rows
    dst.index_copy_(0, idx, src)                   scatter_copy_rows

Why they are faster: aten's index_add_ materialises the dtype cast as a temporary and then runs a
per-element indexing kernel (index math per element, one scalar 16-bit atomic per element). Here one
program moves one whole row: the loads vectorise, the cast happens in registers, and the 16-bit atomics
pack two lanes. index_copy_ likewise becomes one vectorised row copy per program.

Invariants: src and dst are 2-D, contiguous and of one row width; the tile is the largest power of two
<= MAX_BLOCK dividing the width, so no row needs a mask. Destination rows are not bounds-checked (aten's
release path does not either). scatter_add_rows' idx may repeat: every contribution lands through an
atomic, in unspecified order, as with index_add_. The cast is load -> fp32 (exact for fp16/bf16) -> the
atomic's round-to-nearest-even into dst's dtype. With `row_map`, source row i lands in destination row
row_map[idx[i]] (the n-gram owner merge, perf/kernels/ngram_adam.py). scatter_copy_rows moves raw bits,
so it is a bitwise index_copy_ and needs one dtype on both sides.

Provenance: record #360 (ANVIL2): bigram_kernels.py `bgfuse_scatter_add` / `bgfuse_scatter_copy`.
"""
import torch
import triton
import triton.language as tl

# Largest row tile; a 768-wide row is 3 tiles of 256.
MAX_BLOCK = 1024


@triton.jit
def _scatter_add_rows_kernel(SRC, IDX, DST, ROW_MAP, D: tl.constexpr, BLOCK: tl.constexpr, NCHUNK: tl.constexpr,
                             HAS_MAP: tl.constexpr):
    pid = tl.program_id(0)
    o64 = tl.arange(0, BLOCK).to(tl.int64)
    drow = tl.load(IDX + pid)
    if HAS_MAP:
        drow = tl.load(ROW_MAP + drow)
    # int64 offsets: a table shard exceeds 2**31 elements.
    sbase = pid.to(tl.int64) * D
    dbase = drow.to(tl.int64) * D
    for k in tl.static_range(NCHUNK):
        val = tl.load(SRC + (sbase + k * BLOCK + o64)).to(tl.float32)
        tl.atomic_add(DST + (dbase + k * BLOCK + o64), val, sem="relaxed")


@triton.jit
def _scatter_copy_rows_kernel(SRC, IDX, DST, D: tl.constexpr, BLOCK: tl.constexpr, NCHUNK: tl.constexpr):
    pid = tl.program_id(0)
    o64 = tl.arange(0, BLOCK).to(tl.int64)
    drow = tl.load(IDX + pid)
    sbase = pid.to(tl.int64) * D
    dbase = drow.to(tl.int64) * D
    for k in tl.static_range(NCHUNK):
        tl.store(DST + (dbase + k * BLOCK + o64), tl.load(SRC + (sbase + k * BLOCK + o64)))


def _row_tile(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor) -> tuple[int, int]:
    """(width, tile) after checking the shared contract of both kernels."""
    assert src.ndim == 2 and dst.ndim == 2 and src.shape[1] == dst.shape[1], f"{tuple(src.shape)} -> {tuple(dst.shape)}"
    assert src.is_contiguous() and dst.is_contiguous() and src.numel() < 2 ** 31
    assert idx.ndim == 1 and idx.is_contiguous() and idx.numel() == src.shape[0]
    assert idx.dtype in (torch.int32, torch.int64)
    d = src.shape[1]
    block = MAX_BLOCK
    while block > 1 and d % block:
        block //= 2
    assert block >= 32, f"row width {d} has no usable tile"
    return d, block


def scatter_add_rows(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, row_map: torch.Tensor | None = None):
    """dst[row_map[idx[i]] if row_map is given else idx[i]] += src[i], in dst's dtype; idx may repeat."""
    d, block = _row_tile(src, idx, dst)
    assert row_map is None or (row_map.dtype == torch.int32 and row_map.is_contiguous())
    if src.shape[0]:
        _scatter_add_rows_kernel[(src.shape[0],)](
            src, idx, dst, idx if row_map is None else row_map, D=d, BLOCK=block, NCHUNK=d // block,
            HAS_MAP=row_map is not None, num_warps=max(1, min(8, block // 64)), num_stages=1,
        )
    return dst


def scatter_copy_rows(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor):
    """dst[idx[i]] = src[i], bitwise (same dtype); idx must not repeat."""
    assert src.dtype == dst.dtype, f"a bitwise copy: {src.dtype} != {dst.dtype}"
    d, block = _row_tile(src, idx, dst)
    if src.shape[0]:
        _scatter_copy_rows_kernel[(src.shape[0],)](
            src, idx, dst, D=d, BLOCK=block, NCHUNK=d // block, num_warps=max(1, min(8, block // 64)), num_stages=1,
        )
    return dst
