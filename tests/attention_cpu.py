"""Load actual attention control/shape code with CPU reference GPU-kernel boundaries.

This deliberately does not emulate FP8 rounding, FA3 numerics, Triton or CUDA graphs.
It exercises the production method, rather than a rewritten attention layer.
"""
import ast
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import types

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def qk_reference(qk, factor1, factor2, heads, rotary_dim, paired, key_offset):
    tokens, _, width = qk.shape
    assert not (paired and key_offset)
    work = qk.double() if qk.dtype == torch.float64 else qk.float()
    norm = work * (work.square().mean(-1, keepdim=True) + 2**-23).rsqrt()
    q, k = norm.split(heads, dim=1)
    def rotate(x):
        flipped = x.reshape(tokens, heads, width//2, 2).flip(-1).reshape_as(x)
        if paired:
            idx = torch.arange(heads) % 2
            f1, f2 = (f.reshape(tokens, 2, width)[:, idx].to(work.dtype) for f in (factor1, factor2))
        else:
            f1, f2 = (f[:, None, :].to(work.dtype) for f in (factor1, factor2))
        return f1*x + f2*flipped
    out_q, out_k = rotate(q), rotate(k)
    if key_offset:
        stationary = torch.cat((k[:1, :, rotary_dim:], k[:-1, :, rotary_dim:]), dim=0)
        out_k = torch.cat((out_k[..., :rotary_dim], stationary), dim=-1)
    shape = (2*tokens, heads//2, width) if paired else (tokens, heads, width)
    return out_q.to(qk.dtype).reshape(shape).contiguous(), out_k.to(qk.dtype).reshape(shape).contiguous()


def packed_reference(x, qk_w, v_w, weight, weight_t, weight_scale, scale,
                     x_scale, grad_scale, factor1, factor2, heads, paired, key_offset,
                     rotary_dim, x_f8, x_f8_t):
    # Match the production ABI: independent Q/K allocations; nonpaired V is a strided suffix,
    # while paired V is dense so token/head axes can be merged without materialization.
    tokens, width = x.shape[1], qk_w.shape[0]//(2*heads)
    qkv = F.linear(x, scale*torch.cat((qk_w, v_w)))
    qk, v = qkv.split((qk_w.shape[0], v_w.shape[0]), dim=-1)
    qk = qk.reshape(tokens, 2*heads, width)
    v = v.reshape(tokens, heads, -1)
    if paired:
        v = v.contiguous()
    q, k = qk_reference(qk, factor1, factor2, heads, rotary_dim, paired, key_offset)
    return q, k, v


def attention_reference(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                        max_seqlen_k, causal, softmax_scale, window_size):
    assert q.ndim == k.ndim == v.ndim == 3
    assert causal and torch.equal(cu_seqlens_q, cu_seqlens_k)
    assert q.shape[:2] == k.shape[:2] == v.shape[:2]
    ends = sorted(set(cu_seqlens_q.tolist()))
    pieces = []
    for lo, hi in zip(ends, ends[1:]):
        if hi <= lo:
            continue
        pos = torch.arange(hi-lo)
        mask = (pos[:, None] >= pos[None, :]) & (pos[:, None]-pos[None, :] <= window_size[0])
        pieces.append(F.scaled_dot_product_attention(q[lo:hi].transpose(0, 1), k[lo:hi].transpose(0, 1),
                      v[lo:hi].transpose(0, 1), attn_mask=mask, scale=softmax_scale).transpose(0, 1))
    return torch.cat(pieces, dim=0)


def load_attention(*, legacy=False):
    """Only skip imports/initializers that require CUDA; execute unmodified class definitions."""
    path = Path(__file__).resolve().parents[1] / "track_1_short/model/attention.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
             and n.name in ("Yarn", "AttnArgs", "CausalSelfAttention")]
    name = "tests._legacy_attention_cpu" if legacy else "tests._attention_cpu"
    module = types.ModuleType(name)
    sys.modules[name] = module  # dataclass resolves annotations through the defining module
    module.__dict__.update(torch=torch, nn=nn, Tensor=Tensor, F=F, math=math, dataclass=dataclass,
                           PackedFP8QKV=packed_reference, qk_norm_rope_forward=qk_reference,
                           flash_attn_interface=types.SimpleNamespace(flash_attn_varlen_func=attention_reference))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    if legacy:
        path = Path(__file__).with_name("reference_attention.py")
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "legacy_forward")
        # Annotations use the same runtime classes; only the frozen method's body is substituted.
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), module.__dict__)
        module.CausalSelfAttention.forward = module.legacy_forward
    return module


def case(module, *, qk=64, value=128, paired=False, aux=False, gate=False,
         key_offset=False, fp8=True, dtype=torch.float64, strided=False, seed=396, tokens=32):
    generator = torch.Generator().manual_seed(seed)
    heads, dim = 6, 24
    def leaf(shape, scale=1.):
        raw = torch.randn((*shape[:-1], shape[-1]*(2 if strided else 1)), generator=generator, dtype=dtype)
        raw.mul_(scale)
        if strided:
            raw = raw[..., ::2]
        return raw.detach().requires_grad_()
    leaves = dict(x=leaf((1, tokens, dim)), qk=leaf((2*heads*qk, dim), .1),
                  v=leaf((heads*value, dim), .1), o=leaf((dim, heads*value), .1), scale=leaf((2,), .3))
    if aux:
        leaves["aux"] = leaf((1, tokens, heads*128), .1)
    if gate:
        leaves["gate"] = leaf((1, tokens, heads), .3)
    if not paired:
        leaves["alpha"] = leaf((1, tokens, heads), .1)
    leaves["o_gain"] = torch.tensor(.9, dtype=dtype, requires_grad=True)
    model = module.CausalSelfAttention(heads, 128, qk, value, tokens, paired)
    yarn = module.Yarn(qk, tokens, paired, attn_scale=.12, device="cpu")
    args = module.AttnArgs(sa_lambdas=leaves["scale"], seqlens=torch.tensor([0, tokens//4, 3*tokens//4, tokens, tokens], dtype=torch.int32),
                          bm_size=7, yarn=yarn, key_offset=key_offset, attn_gate_w=leaves.get("gate"),
                          aux_v=leaves.get("aux"), xsa_alpha=leaves.get("alpha"), train_max_seq_len=torch.tensor(tokens),
                          o_gain=leaves["o_gain"])
    model.train(fp8)
    return model, args, leaves, (None,)*7 if fp8 else None


def forward(bundle):
    model, args, leaves, fp8 = bundle
    return model(leaves["x"], args, leaves["qk"], leaves["v"], leaves["o"], fp8)
