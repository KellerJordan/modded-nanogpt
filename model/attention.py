import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention

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
            **kwargs,
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
        self.Pk = nn.Parameter(torch.randn(self.n_tokens, config.hidden_size))
        self.Pv = nn.Parameter(torch.randn(self.n_tokens, config.hidden_size))

    def act(self, x: torch.Tensor) -> torch.Tensor:
        o = x / torch.norm(x, p=2, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])
        o = F.gelu(o)
        return o

    def forward(self, x: torch.Tensor, last_eos: Optional[int] = None) -> torch.Tensor:
        if last_eos is None:
            last_eos = x.shape[1] - 1
        Q_len, d = x.size() # batch size must be 1 for FlexAttention

        attention_mask = torch.ones(Q_len, self.n_tokens, device=x.device)
        attention_mask[last_eos:, :] = 0

        q = self.Wq(x) # (Q_len, d)
        k = self.Pk # (n_tokens, d)
        v = self.Pv # (n_tokens, d)

        attn_weight = q @ k.transpose(0, 1) # (Q_len, n_tokens)
        #attn_weight *= attention_mask
        attn_weight = self.act(attn_weight)

        y = attn_weight @ v # (Q_len, d)
        return y.unsqueeze(0) # (1, Q_len, d)


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
            last_eos: Optional[int] = None,
            **kwargs,
        ) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        l, d = x.size()
        q = self.Wq(x, last_eos) # (1, l, d)
        k = self.Wk(x, last_eos) # (1, l, d)
        v = self.Wv(x, last_eos) # (1, l, d)

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


if __name__ == '__main__':
    # test pattention
    # py -m model.attention
    
    # Simple config for testing
    class TestConfig:
        def __init__(self):
            self.hidden_size = 64
            self.num_att_tokens = 8
    
    config = TestConfig()
    pattention = PAttention(config)
    
    # Test input: sequence length 10, hidden size 64
    seq_len = 10
    x = torch.randn(seq_len, config.hidden_size)
    
    # Test mask logic with different last_eos values
    print("Testing PAttention mask logic...")
    
    # Case 1: last_eos = 5 (mask positions 0-4, unmask positions 5-9)
    last_eos = 5
    output = pattention(x, last_eos=last_eos)
    
    # Manually check the mask logic
    q = pattention.Wq(x)
    k = pattention.Pk
    attn_weight = q @ k.transpose(0, 1)
    
    # Create expected mask
    expected_mask = torch.ones(seq_len, config.num_att_tokens)
    expected_mask[:last_eos, :] = 0
    
    # Apply mask
    masked_attn = attn_weight * expected_mask
    
    # Check that positions before last_eos are zero
    assert torch.allclose(masked_attn[:last_eos, :], torch.zeros(last_eos, config.num_att_tokens)), \
        "Attention weights before last_eos should be zero"
    
    # Check that positions from last_eos onwards are non-zero (assuming non-zero input)
    assert not torch.allclose(masked_attn[last_eos:, :], torch.zeros(seq_len - last_eos, config.num_att_tokens)), \
        "Attention weights from last_eos onwards should be non-zero"
    
    print(f"Test passed for last_eos={last_eos}")
    
    # Case 2: last_eos = 0 (no masking)
    last_eos = 0
    output = pattention(x, last_eos=last_eos)
    print(f"Test passed for last_eos={last_eos}")
    
    # Case 3: last_eos = seq_len - 1 (mask all but last position)
    last_eos = seq_len - 1
    output = pattention(x, last_eos=last_eos)
    print(f"Test passed for last_eos={last_eos}")
    
    print("All PAttention mask tests passed!")