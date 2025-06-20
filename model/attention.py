import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention

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
    def __init__(self, hidden_size: int, n_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads

        corrected_dim = n_heads * self.d_head
        std = 0.5 * (self.d_head ** -0.5)
        bound = (3 ** 0.5) * std

        self.Wqkv = nn.Parameter(torch.empty(3, hidden_size, corrected_dim).uniform_(-bound, bound))
        self.Wo = nn.Linear(hidden_size, hidden_size)
        self.rotary = Rotary(self.d_head)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.Wqkv.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.n_heads, self.d_head).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=attention_mask).transpose(1, 2)
        y = y.contiguous().view(B, T, self.n_heads * self.d_head)
        y = self.Wo(y)
        return y
    

class PAttention(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, n_tokens) ->  (b, L, d)
    """
    def __init__(
            self,
            hidden_size: int,
            n_tokens: int,
            dropout: float = 0.2,
    ):
        super(PAttention, self).__init__()
        self.n_tokens = n_tokens
        self.Wq = Linear(hidden_size, hidden_size)
        self.Pk = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.Pv = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, _ = x.size()
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, :].expand(b, self.n_token, self.L).bool()
        
        q = self.Wq(x) # (b, L, d)
        out = F.scaled_dot_product_attention(q, self.Pk, self.Pv, attn_mask=attention_mask, is_causal=False) # (b, L, d)
        return self.dropout(out)