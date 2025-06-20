import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

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
        l, d = x.size() # batch size must be 1 for FlexAttention
        q, k, v = F.linear(x, self.Wqkv.flatten(end_dim=1).type_as(x)).view(1, l, 3 * self.n_heads, self.d_head).chunk(3, dim=-2)
        # (1, l, n_heads, d_head), (1, l, n_heads, d_head), (1, l, n_heads, d_head)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=attention_mask).transpose(1, 2)
        # (1, n_heads, l, d_head), (1, n_heads, l, d_head), (1, n_heads, l, d_head)
        # -> (1, l, n_heads, d_head)
        y = y.contiguous().view(1, l, self.n_heads * self.d_head) # (1, l, n_heads * d_head)
        y = self.Wo(y).squeeze(0) # (l, hidden_size)
        return y


class PAttention(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, n_tokens) ->  (b, L, d)
    """
    def __init__(self, hidden_size: int, n_tokens: int, sliding_window_size: int):
        super(PAttention, self).__init__()
        self.n_tokens = n_tokens
        self.Wq = Linear(hidden_size, hidden_size)
        self.Pk = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.Pv = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.sliding_window_size = sliding_window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q_len, d = x.size() # batch size must be 1 for FlexAttention

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < self.sliding_window_size
            return bidirectional_sliding_window_mask

        KV_len = self.n_tokens
        attention_mask = create_block_mask(doc_mask_mod, 1, 1, Q_len, KV_len)

        q = self.Wq(x).unsqueeze(0).unsqueeze(1) # (1, 1, Q_len, d)
        k = self.Pk.unsqueeze(1) # (1, 1, n_tokens, d)
        v = self.Pv.unsqueeze(1) # (1, 1, n_tokens, d)
        y = flex_attention(q, k, v, block_mask=attention_mask) # (1, 1, Q_len, d)
        return y.squeeze(1) # (1, Q_len, d)
    

class MultiHeadPAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, n_tokens: int, sliding_window_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.Wq = PAttention(hidden_size, n_tokens=n_tokens, sliding_window_size=sliding_window_size)
        self.Wk = PAttention(hidden_size, n_tokens=n_tokens, sliding_window_size=sliding_window_size)
        self.Wv = PAttention(hidden_size, n_tokens=n_tokens, sliding_window_size=sliding_window_size)
        self.Wo = Linear((hidden_size // n_heads) * n_heads, hidden_size)
        self.rotary = Rotary(self.d_head)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        l, d = x.size()
        q = self.Wq(x) # (1, l, d)
        k = self.Wk(x) # (1, l, d)
        v = self.Wv(x) # (1, l, d)
        q = q.view(1, l, self.n_heads, self.d_head)
        k = k.view(1, l, self.n_heads, self.d_head)
        v = v.view(1, l, self.n_heads, self.d_head)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=attention_mask).transpose(1, 2) # (1, n_heads, l, d_head)
        y = y.contiguous().view(1, l, self.n_heads * self.d_head) # (1, l, n_heads * d_head)
        return self.Wo(y).squeeze(0) # (l, hidden_size)
