import torch
import torch.nn as nn
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from model.flex_mods import generate_tanh_softcap
from model.utils import norm, Linear


class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            t = torch.arange(seq_len, device=x.device)
            freqs = torch.outer(t, self.inv_freq)
            self.seq_len_cached = seq_len
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        # apply_rotary_emb(x, cos, sin)
        x1, x2 = x.chunk(2, dim=3)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head = self.hidden_size // self.n_heads

        assert self.hidden_size % self.n_heads == 0
        self.Wq = Linear(self.hidden_size, self.hidden_size)
        self.Wk = Linear(self.hidden_size, self.hidden_size)
        self.Wv = Linear(self.hidden_size, self.hidden_size)
        self.rotary = Rotary(self.d_head) # dim // num_attention_heads = head_dim
        self.Wo = Linear(self.hidden_size, self.hidden_size)
        self.Wo.weight.data.zero_() # zero init suggested by @Grad6230497
        
        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))

        if config.attention_soft_cap:
            self.soft_cap_mod = generate_tanh_softcap(config.attention_soft_cap, approx=True)
        else:
            self.soft_cap_mod = None
        self.unet = config.unet

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            vi: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        l, d = x.size() # batch size must be 1 for FlexAttention
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        q = q.view(1, l, self.n_heads, self.d_head)
        k = k.view(1, l, self.n_heads, self.d_head)
        v = v.view(1, l, self.n_heads, self.d_head)

        if self.unet and vi is not None:
            # Reshape vi from (l, d) to (1, l, n_heads, d_head) to match v's shape
            v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v)
        
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            score_mod=self.soft_cap_mod,
            block_mask=attention_mask,
            enable_gqa=True,
        )
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.Wo(y)
        return y


class PAttention(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, n_tokens) ->  (b, L, d)
    """
    def __init__(self, config):
        super(PAttention, self).__init__()
        self.config = config
        self.n_tokens = config.num_att_tokens
        self.Wq = Linear(config.hidden_size, config.hidden_size)
        self.Pk = nn.Parameter(torch.randn(1, self.n_tokens, config.hidden_size))
        self.Pv = nn.Parameter(torch.randn(1, self.n_tokens, config.hidden_size))
        self.sliding_window_size = config.sliding_window_size

    def forward(
            self,
            x: torch.Tensor,
            sliding_window_size: Optional[int] = None,
        ) -> torch.Tensor:
        Q_len, d = x.size() # batch size must be 1 for FlexAttention

        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            return bidirectional_sliding_window_mask

        KV_len = self.n_tokens
        attention_mask = create_block_mask(doc_mask_mod, 1, 1, Q_len, KV_len)

        q = self.Wq(x).unsqueeze(0).unsqueeze(1) # (1, 1, Q_len, d)
        k = self.Pk.unsqueeze(1) # (1, 1, n_tokens, d)
        v = self.Pv.unsqueeze(1) # (1, 1, n_tokens, d)
        y = flex_attention(q, k, v, block_mask=attention_mask) # (1, 1, Q_len, d)
        return y.squeeze(1) # (1, Q_len, d)


class MultiHeadPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.d_head = self.hidden_size // self.n_heads
        self.Wq = PAttention(config)
        self.Wk = PAttention(config)
        self.Wv = PAttention(config)
        self.Wo = Linear((self.hidden_size // self.n_heads) * self.n_heads, self.hidden_size)
        self.rotary = Rotary(self.d_head)

        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.unet = config.unet

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            vi: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        l, d = x.size()
        q = self.Wq(x) # (1, l, d)
        k = self.Wk(x) # (1, l, d)
        v = self.Wv(x) # (1, l, d)

        if self.unet and vi is not None:
            # Reshape vi from (l, d) to (1, l, d) to match v's shape before applying it
            vi_reshaped = vi.unsqueeze(0)  # (1, l, d)
            v = self.lambdas[0] * v + self.lambdas[1] * vi_reshaped

        q = q.view(1, l, self.n_heads, self.d_head)
        k = k.view(1, l, self.n_heads, self.d_head)
        v = v.view(1, l, self.n_heads, self.d_head)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=attention_mask,
        ).transpose(1, 2) # (1, n_heads, l, d_head)
        
        y = y.contiguous().view(1, l, self.n_heads * self.d_head) # (1, l, n_heads * d_head)
        return self.Wo(y).squeeze(0) # (l, hidden_size)
