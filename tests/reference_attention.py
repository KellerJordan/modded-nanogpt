"""Attention forward frozen at 42c9714; external GPU kernel boundaries are supplied by the CPU harness."""
from __future__ import annotations

def legacy_forward(self, x: Tensor, attn_args: AttnArgs, qk_w: Tensor, v_w: Tensor, o_w: Tensor, qkv_fp8=None):
    """qkv_fp8 (training only; None under validation): (weight_f8, weight_f8_t, weight_scale, x_scale,
        grad_scale, x_f8, x_f8_t), this layer's packed weight cache, its static fp8 scales and its
        already-quantized input."""
    B, T = (x.size(0), x.size(1))
    assert B == 1, 'varlen sequences requires B == 1'
    assert T % 16 == 0
    H = self.num_heads
    aux_v, attn_gate_w = (attn_args.aux_v, attn_args.attn_gate_w)
    sa_lambdas, key_offset = (attn_args.sa_lambdas, attn_args.key_offset)
    seqlens, bm_size = (attn_args.seqlens, attn_args.bm_size)
    train_max_seq_len, yarn = (attn_args.train_max_seq_len, attn_args.yarn)
    assert yarn.head_dim == self.qk_dim
    assert not key_offset or self.qk_dim > Yarn.ROTARY_DIM
    if qkv_fp8 is not None:
        weight_f8, weight_f8_t, weight_scale, x_scale, grad_scale, x_f8, x_f8_t = qkv_fp8
        q, k, v = PackedFP8QKV(x, qk_w.type_as(x), v_w.type_as(x), weight_f8, weight_f8_t, weight_scale, sa_lambdas[0], x_scale, grad_scale, yarn.factor1[:T], yarn.factor2[:T], H, self.paired, key_offset, Yarn.ROTARY_DIM, x_f8, x_f8_t)
        q, k, v = (q[None], k[None], v[None])
    else:
        qkv_weight = sa_lambdas[0] * torch.cat((qk_w, v_w)).type_as(x)
        qkv = F.linear(x, qkv_weight)
        qk, v = qkv.split((2 * H * self.qk_dim, H * self.v_dim), dim=-1)
        qk = qk.view(B, T, 2 * H, self.qk_dim)
        v = v.view(B, T, H, self.v_dim)
        q, k = qk_norm_rope_forward(qk[0], yarn.factor1[:T], yarn.factor2[:T], H, Yarn.ROTARY_DIM, self.paired, key_offset)
        q, k = (q[None], k[None])
    max_len = train_max_seq_len if self.training else self.val_max_seq_len
    if aux_v is not None:
        aux_v = aux_v.view(B, T, H, self.head_dim)[..., :self.v_dim]
    if not self.paired:
        if aux_v is not None:
            v = v + aux_v
    else:
        v = v.reshape(B, T * 2, H // 2, self.v_dim)
        if aux_v is not None:
            v = v + aux_v.reshape(v.shape)
        seqlens = 2 * seqlens
        max_len = 2 * max_len
    y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens, max_seqlen_q=max_len, max_seqlen_k=max_len, causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0))
    y = y.view(B, T, H, self.v_dim)
    if attn_args.xsa_alpha is not None and (not self.paired):
        dot = (y * v).sum(-1, keepdim=True)
        denom = v.square().sum(-1, keepdim=True).clamp_min(1e-08)
        alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(B, T, H, 1)
        y = y - alpha * (dot / denom) * v
    if attn_gate_w is not None:
        y = y * attn_gate_w.type_as(y).view(B, T, H, 1)
    y = y.contiguous().view(B, T, H * self.v_dim)
    o_scale = sa_lambdas[1] if attn_args.o_gain is None else sa_lambdas[1] * attn_args.o_gain
    return F.linear(y, o_scale * o_w.type_as(y))
