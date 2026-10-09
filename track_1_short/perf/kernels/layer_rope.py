"""LayerRoPE gains for the fp8 weight refresh: every (site, layer) row the caches fold in, in one launch.

What it replaces: GPT.layer_rope_gamma_eff() (about fifteen small elementwise ops) plus the gathers and
stacks that reorder its rows into bank / slot order, all of which the refresh would otherwise run
uncompiled inside its CUDA graph, every step.

Why it is faster: one program per output row computes that row from the 16 gate scalars, the four
gamma vectors and the shared frequencies, and stores it straight into its bank / slot position.

The math is GPT.layer_rope_gamma_eff()'s, in fp32, rounded to bf16 at the store; only the transcendental
implementations may differ from the compiled forward's (by an ulp-level amount). The forward's copy of
the gains only carries their gradients, so the two never have to agree bitwise.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_rope_gain_rows_kernel(gates_ptr, gammas_ptr, freqs_ptr, sites_ptr, layers_ptr, out_ptr,
                                 D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    site = tl.load(sites_ptr + row)
    u = tl.log(tl.load(layers_ptr + row).to(tl.float32) + 1.0)
    alpha = tl.load(gates_ptr + site * 4 + 0)
    beta = tl.load(gates_ptr + site * 4 + 1)
    alpha_rot = tl.load(gates_ptr + site * 4 + 2)
    beta_rot = tl.load(gates_ptr + site * 4 + 3)
    mag = tl.exp(alpha + beta * u)
    rot = tl.exp(alpha_rot + beta_rot * u)
    j = tl.arange(0, BLOCK)
    mask = j < HALF
    theta = rot * tl.load(freqs_ptr + j, mask=mask, other=0.0)
    c = tl.cos(theta)
    s = tl.sin(theta)
    gu = tl.load(gammas_ptr + site * D + 2 * j, mask=mask, other=0.0)
    gv = tl.load(gammas_ptr + site * D + 2 * j + 1, mask=mask, other=0.0)
    tl.store(out_ptr + row * D + 2 * j, (mag * (gu * c - gv * s)).to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + row * D + 2 * j + 1, (mag * (gu * s + gv * c)).to(tl.bfloat16), mask=mask)


def layer_rope_gain_rows(gates: torch.Tensor, gammas: torch.Tensor, freqs: torch.Tensor,
                         sites: torch.Tensor, layers: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[r] = the bf16 gain of site sites[r] at layer layers[r]; gates (4, 4) and gammas (4, D) fp32."""
    rows, D = out.shape
    assert gates.shape == (4, 4) and gammas.shape == (4, D) and gates.dtype == gammas.dtype == torch.float32
    assert gates.is_contiguous() and gammas.is_contiguous() and out.is_contiguous()
    assert sites.shape == layers.shape == (rows,)
    _layer_rope_gain_rows_kernel[(rows,)](gates, gammas, freqs, sites, layers, out,
                                          D=D, HALF=D // 2, BLOCK=triton.next_power_of_2(D // 2), num_warps=4)
    return out
