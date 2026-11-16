"""GPU self-check of the column-weighted CE kernel. One GPU, about a minute, before any timed run.

    python records/track_1_short/2026-11-15_StratifiedSoftmaxTail/gpu_check.py     (from the repository root)

1. With every log weight zero, the weighted kernel must reproduce the record's kernel at the same width.
2. With random log weights, its losses and its fp8 gradient must match the weighted softcapped cross
   entropy written out in torch (fp64), within the fp16 / fp8 rounding the kernel itself applies.
Exits non-zero on a mismatch: do not start a run then.
"""
import os
import sys

sys.path[0] = os.getcwd()  # the repository root, not this folder (its files must not shadow the stdlib)
import torch

from track_1_short.perf.kernels import sampled_cross_entropy as sce
from track_1_short.perf.kernels.cross_entropy import (
    CE_KERNEL_BLOCK_SIZE,
    SOFTCAP_A,
    SOFTCAP_B,
    SOFTCAP_C,
    compile_ce_kernel,
)
from track_1_short.stratified_softmax import WIDTH as P


def fp8_table(exponent_bits: int, mantissa_bits: int) -> torch.Tensor:
    """Value of each byte of an fp8 format (e4m3: 4, 3; e5m2: 5, 2). NaN / inf bytes decode to NaN."""
    bias = (1 << (exponent_bits - 1)) - 1
    values = []
    for byte in range(256):
        sign = -1.0 if byte >> 7 else 1.0
        exponent = (byte >> mantissa_bits) & ((1 << exponent_bits) - 1)
        mantissa = (byte & ((1 << mantissa_bits) - 1)) / (1 << mantissa_bits)
        if exponent == 0:
            values.append(sign * mantissa * 2.0 ** (1 - bias))
        else:
            values.append(sign * (1 + mantissa) * 2.0 ** (exponent - bias))
    table = torch.tensor(values, dtype=torch.float64)
    if exponent_bits == 4:
        table[[0x7F, 0xFF]] = float("nan")
    else:
        table[[0x7C, 0x7D, 0x7E, 0x7F, 0xFC, 0xFD, 0xFE, 0xFF]] = float("nan")
    return table


def decode(tensor: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return table.to(tensor.device)[tensor.view(torch.uint8).long()]


def main() -> int:
    assert sce.STRATIFIED_CE_KERNEL is not None, "STRATIFIED_SOFTMAX=0: nothing to check"
    torch.manual_seed(0)
    dev = torch.device("cuda", 0)
    N, grad_s = 2048, 2.0 ** -10
    A, B, C = float(SOFTCAP_A), float(SOFTCAP_B), float(SOFTCAP_C)
    logits = (torch.randn(N, P, device=dev) * 6).to(torch.float8_e4m3fn)
    targets = torch.randint(0, P, (N,), device=dev)
    no_prefix = torch.full((N,), -1, dtype=torch.int64, device=dev)
    mtp_weights = torch.ones(1, dtype=torch.float32, device=dev)
    prefix_weight = torch.zeros(1, dtype=torch.float32, device=dev)

    def outputs():
        return (torch.empty(N, dtype=torch.float32, device=dev),
                torch.empty((N, P), dtype=torch.float8_e5m2, device=dev))

    def weighted(col_logw):
        losses, grad = outputs()
        sce.stratified_ce_fwd_bwd(logits, targets, mtp_weights, no_prefix, prefix_weight, col_logw, losses, grad,
                                  N, 1, A, B, C, grad_s)
        torch.cuda.synchronize()
        return losses, grad

    # 1. Zero weights against the record's kernel compiled at the same width.
    record_kernel = compile_ce_kernel(P)
    record_losses, record_grad = outputs()
    record_kernel((N, 1, 1), (CE_KERNEL_BLOCK_SIZE, 1, 1),
                  (logits, targets, mtp_weights, no_prefix, prefix_weight, record_losses, record_grad,
                   N, 1, A, B, C, grad_s), shared_mem=P * 2)
    torch.cuda.synchronize()
    zero_losses, zero_grad = weighted(torch.zeros(P, dtype=torch.float32, device=dev))
    loss_gap = (zero_losses - record_losses).abs().max().item()
    byte_gap = (zero_grad.view(torch.uint8) != record_grad.view(torch.uint8)).float().mean().item()
    ok_zero = loss_gap < 1e-3 and byte_gap < 1e-3
    print(f"1. zero weights vs the record kernel: max loss gap {loss_gap:.2e}, gradient bytes differing "
          f"{byte_gap:.2e} -> {'ok' if ok_zero else 'MISMATCH'}")

    # 2. Random weights on a random quarter of the columns against the fp64 reference.
    col_logw = (torch.rand(P, device=dev) * 4 * (torch.rand(P, device=dev) < 0.25)).to(torch.float32)
    losses, grad = weighted(col_logw)
    sigmoid = torch.sigmoid((decode(logits, fp8_table(4, 3)).float() + B) / C).half().double()  # the kernel keeps fp16
    z = A * sigmoid
    lse = torch.logsumexp(z + col_logw.double(), dim=1)
    ref_losses = lse - z.gather(1, targets[:, None])[:, 0]
    grad_z = torch.exp(z + col_logw.double() - lse[:, None])
    grad_z.scatter_add_(1, targets[:, None], -torch.ones(N, 1, dtype=torch.float64, device=dev))
    ref_grad = (A / C / grad_s) * grad_z * sigmoid * (1 - sigmoid)
    got_grad = decode(grad, fp8_table(5, 2))
    loss_gap = (losses.double() - ref_losses).abs().max().item()
    within = ((got_grad - ref_grad).abs() <= 0.13 * ref_grad.abs() + 2.0 ** -15).float().mean().item()
    # What ignoring the weights would give, to show the check can tell the difference.
    unweighted_gap = (torch.logsumexp(z, dim=1) - lse).abs().mean().item()
    ok_weighted = loss_gap < 0.05 and within > 0.999
    print(f"2. random weights vs the fp64 reference: max loss gap {loss_gap:.2e} (ignoring the weights would "
          f"be off by {unweighted_gap:.2f}), gradient entries within fp8 rounding {within:.5f} "
          f"-> {'ok' if ok_weighted else 'MISMATCH'}")

    passed = ok_zero and ok_weighted
    print(f"GPU CHECK {'PASSED' if passed else 'FAILED'} on {torch.cuda.get_device_name(0)}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
