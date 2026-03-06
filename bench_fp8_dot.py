import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor
import time

@triton.jit
def matmul_kernel(a_desc, b_desc, c_desc, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  NUM_SMS: tl.constexpr, CAST_A: tl.constexpr):
    start_pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = tl.cdiv(M, BLOCK_M) * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([pid_m * BLOCK_M, ki * BLOCK_K])
            b = b_desc.load([pid_n * BLOCK_N, ki * BLOCK_K])
            if CAST_A:
                acc = tl.dot(a.to(tl.float8e4nv), b.T, acc)
            else:
                acc = tl.dot(a, b.T, acc)
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.bfloat16))

M, K, N = 16384, 768, 3072
x_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
W_bf16 = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)
x_f8 = x_bf16.to(torch.float8_e4m3fn)
W_f8 = W_bf16.to(torch.float8_e4m3fn)
out = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
BM, BN, BK = 128, 256, 64

def run(a, b, cast_a):
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(b, [BN, BK])
    c_desc = TensorDescriptor.from_tensor(out, [BM, BN])
    grid = lambda META: (min(NUM_SMS, triton.cdiv(M, BM) * triton.cdiv(N, BN)),)
    matmul_kernel[grid](a_desc, b_desc, c_desc, M, N, K,
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                        NUM_SMS=NUM_SMS, CAST_A=cast_a, num_warps=8, num_stages=3)

# Warmup all 3 variants
for _ in range(5):
    run(x_bf16, W_bf16, False)  # BF16 x BF16
    run(x_bf16, W_f8, True)     # Cast BF16->FP8 x FP8
    run(x_f8, W_f8, False)      # FP8 x FP8 (both pre-cast)
torch.cuda.synchronize()

configs = [
    ('BF16xBF16', x_bf16, W_bf16, False),
    ('Cast+FP8 ', x_bf16, W_f8, True),
    ('FP8xFP8  ', x_f8, W_f8, False),
]
for label, a, b, cast_a in configs:
    t0 = time.perf_counter()
    for _ in range(200):
        run(a, b, cast_a)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 200 * 1000
    print(f'{label}: {ms:.3f}ms')
