"""Standalone numerical tests for Triton tiled transpose_copy and transpose_add."""

import torch
from triton_kernels import transpose_copy, transpose_add


def test_exact_shape():
    """Test with the exact lm_head/embed shapes: (768, 50304) -> (50304, 768)."""
    M, N = 768, 50304
    src = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty(N, M, device="cuda", dtype=torch.bfloat16)

    transpose_copy(src, dst)

    ref = src.T.contiguous()
    assert torch.equal(dst, ref), f"FAIL exact_shape: max diff = {(dst - ref).abs().max().item()}"
    print(f"PASS exact_shape ({M}, {N}) -> ({N}, {M})")


def test_small():
    """Quick sanity check with a small matrix."""
    M, N = 4, 8
    src = torch.arange(M * N, device="cuda", dtype=torch.bfloat16).reshape(M, N)
    dst = torch.empty(N, M, device="cuda", dtype=torch.bfloat16)

    transpose_copy(src, dst)

    ref = src.T.contiguous()
    assert torch.equal(dst, ref), f"FAIL small: max diff = {(dst - ref).abs().max().item()}"
    print(f"PASS small ({M}, {N}) -> ({N}, {M})")


def test_non_divisible():
    """Test shapes not evenly divisible by BLOCK_M=32, BLOCK_N=32."""
    for M, N in [(100, 200), (33, 65), (1, 50304), (768, 1), (31, 31), (63, 97)]:
        src = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        dst = torch.empty(N, M, device="cuda", dtype=torch.bfloat16)

        transpose_copy(src, dst)

        ref = src.T.contiguous()
        assert torch.equal(dst, ref), f"FAIL non_divisible ({M},{N}): max diff = {(dst - ref).abs().max().item()}"
        print(f"PASS non_divisible ({M}, {N}) -> ({N}, {M})")


def test_dtypes():
    """Test float32 and float16 in addition to bfloat16."""
    M, N = 128, 256
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        src = torch.randn(M, N, device="cuda", dtype=dtype)
        dst = torch.empty(N, M, device="cuda", dtype=dtype)

        transpose_copy(src, dst)

        ref = src.T.contiguous()
        assert torch.equal(dst, ref), f"FAIL dtype {dtype}: max diff = {(dst - ref).abs().max().item()}"
        print(f"PASS dtype {dtype} ({M}, {N})")


def test_dst_not_zeroed():
    """Ensure the kernel fully overwrites dst (pre-fill with garbage)."""
    M, N = 768, 50304
    src = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dst = torch.full((N, M), float("nan"), device="cuda", dtype=torch.bfloat16)

    transpose_copy(src, dst)

    ref = src.T.contiguous()
    assert torch.equal(dst, ref), f"FAIL dst_not_zeroed: max diff = {(dst - ref).abs().max().item()}"
    print(f"PASS dst_not_zeroed ({M}, {N})")


def bench():
    """Benchmark against PyTorch copy_ for the production shape."""
    M, N = 768, 50304
    src = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty(N, M, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        transpose_copy(src, dst)
    torch.cuda.synchronize()

    # Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        transpose_copy(src, dst)
    end.record()
    torch.cuda.synchronize()
    triton_us = start.elapsed_time(end) * 1000 / 100  # ms -> us

    # PyTorch copy_
    for _ in range(10):
        dst.copy_(src.T)
    torch.cuda.synchronize()

    start.record()
    for _ in range(100):
        dst.copy_(src.T)
    end.record()
    torch.cuda.synchronize()
    pytorch_us = start.elapsed_time(end) * 1000 / 100

    data_mb = M * N * 2 / 1e6  # BF16 = 2 bytes
    print(f"\nBenchmark ({M}, {N}) BF16 ({data_mb:.1f} MB):")
    print(f"  Triton transpose_copy : {triton_us:7.1f} us")
    print(f"  PyTorch .copy_(src.T) : {pytorch_us:7.1f} us")
    print(f"  Speedup               : {pytorch_us / triton_us:5.2f}x")


# =====================================================================
# transpose_add tests: dst += src.T
# =====================================================================

def test_add_exact_shape():
    """Test transpose_add with embed.grad (50304,768) added into lm_head.grad (768,50304)."""
    M, N = 50304, 768  # embed.grad shape
    src = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dst = torch.randn(N, M, device="cuda", dtype=torch.bfloat16)
    dst_orig = dst.clone()

    transpose_add(src, dst)

    ref = dst_orig + src.T
    assert torch.equal(dst, ref), f"FAIL add_exact_shape: max diff = {(dst - ref).abs().max().item()}"
    print(f"PASS add_exact_shape src=({M}, {N}) dst=({N}, {M})")


def test_add_small():
    """Quick sanity with known values."""
    M, N = 4, 8
    src = torch.arange(M * N, device="cuda", dtype=torch.bfloat16).reshape(M, N)
    dst = torch.ones(N, M, device="cuda", dtype=torch.bfloat16)

    transpose_add(src, dst)

    ref = torch.ones(N, M, device="cuda", dtype=torch.bfloat16) + src.T
    assert torch.equal(dst, ref), f"FAIL add_small: max diff = {(dst - ref).abs().max().item()}"
    print(f"PASS add_small src=({M}, {N}) dst=({N}, {M})")


def test_add_non_divisible():
    """Shapes not divisible by 32."""
    for M, N in [(100, 200), (33, 65), (1, 50304), (768, 1), (31, 31), (63, 97)]:
        src = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        dst = torch.randn(N, M, device="cuda", dtype=torch.bfloat16)
        dst_orig = dst.clone()

        transpose_add(src, dst)

        ref = dst_orig + src.T
        assert torch.equal(dst, ref), f"FAIL add_non_div ({M},{N}): max diff = {(dst - ref).abs().max().item()}"
        print(f"PASS add_non_divisible src=({M}, {N}) dst=({N}, {M})")


def test_add_zeros():
    """Adding transposed zeros should leave dst unchanged."""
    M, N = 768, 50304
    src = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    dst = torch.randn(N, M, device="cuda", dtype=torch.bfloat16)
    dst_orig = dst.clone()

    transpose_add(src, dst)

    assert torch.equal(dst, dst_orig), f"FAIL add_zeros: max diff = {(dst - dst_orig).abs().max().item()}"
    print(f"PASS add_zeros src=({M}, {N}) dst=({N}, {M})")


def bench_add():
    """Benchmark transpose_add against PyTorch .add_(src.T) for the production shape."""
    # embed.grad is (50304, 768), lm_head.grad is (768, 50304)
    M, N = 50304, 768
    src = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    dst = torch.randn(N, M, device="cuda", dtype=torch.bfloat16)

    # Warmup
    for _ in range(10):
        transpose_add(src, dst)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        transpose_add(src, dst)
    end.record()
    torch.cuda.synchronize()
    triton_us = start.elapsed_time(end) * 1000 / 100

    # PyTorch .add_
    for _ in range(10):
        dst.add_(src.T)
    torch.cuda.synchronize()

    start.record()
    for _ in range(100):
        dst.add_(src.T)
    end.record()
    torch.cuda.synchronize()
    pytorch_us = start.elapsed_time(end) * 1000 / 100

    data_mb = M * N * 2 / 1e6
    print(f"\nBenchmark transpose_add src=({M}, {N}) BF16 ({data_mb:.1f} MB):")
    print(f"  Triton transpose_add  : {triton_us:7.1f} us")
    print(f"  PyTorch .add_(src.T)  : {pytorch_us:7.1f} us")
    print(f"  Speedup               : {pytorch_us / triton_us:5.2f}x")


if __name__ == "__main__":
    print("=== transpose_copy ===")
    test_small()
    test_non_divisible()
    test_dtypes()
    test_exact_shape()
    test_dst_not_zeroed()
    bench()

    print("\n=== transpose_add ===")
    test_add_small()
    test_add_non_divisible()
    test_add_exact_shape()
    test_add_zeros()
    bench_add()

    print("\nAll tests passed.")
