import math
from utils.lmath import project, distance
import torch
from torch import nn
import torch.nn.functional as F
from lib.lorentz.manifold import CustomLorentz

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_heads
        assert self.n_embd % self.n_heads == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_heads, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_heads, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_heads, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class HyperbolicSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_heads
        assert self.n_embd % self.n_heads == 0
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        if not config.k_lr:
            self.register_buffer('c', torch.tensor(float(config.curvature)))
        else:
            x = torch.randn(1, config.n_heads, 1, 1, device=self.c_attn.weight.device)
            init_c = torch.exp(x) * config.curvature
            self.k = nn.Parameter(init_c)
        
        self.register_buffer('p', torch.tensor(2.0))
        self.register_buffer('eps', torch.tensor(1e-3))
        
        self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length)).view(1, 1, config.sequence_length, config.sequence_length))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        lq = project(q, k=self.k, dim=-1).unsqueeze(-2)
        lk = project(k, k=self.k, dim=-1).unsqueeze(-3)

        dis = distance(lq, lk, k=self.k, dim=-1)

        wei = 1 / (self.eps + dis**self.p)
        wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, 0.) 
        wei = wei / wei.sum(dim=-1, keepdim=True)
        y = wei @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """
    def __init__(
            self, 
            manifold: CustomLorentz, 
            num_features: int, 
            num_classes: int
        ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.zeros(num_classes,))
        self.z = torch.nn.Parameter(F.pad(torch.zeros(num_classes, num_features-2), pad=(1,0), value=1)) # z should not be (0,0)

        self.init_weights()

    def forward(self, x):
        # x: (B, T, num_features)

        # Hyperplane parameters
        sqrt_mK = 1 / self.manifold.k.sqrt()  # scalar
        norm_z = torch.norm(self.z, dim=-1)  # (num_classes,)
        w_t = torch.sinh(sqrt_mK * self.a) * norm_z  # (num_classes,)
        w_s = torch.cosh(sqrt_mK * self.a).unsqueeze(-1) * self.z  # (num_classes, num_features -1)

        beta = torch.sqrt(-w_t**2 + torch.norm(w_s, dim=-1)**2)  # (num_classes,)

        x0 = x.narrow(-1, 0, 1)  # (B, T, 1)
        x_rest = x.narrow(-1, 1, x.shape[-1]-1)  # (B, T, num_features -1)
        inner_prod = torch.matmul(x_rest, self.z.T)  # (B, T, num_classes)
        alpha = -x0 * w_t.view(1, 1, -1) + torch.cosh(sqrt_mK * self.a).view(1, 1, -1) * inner_prod  # (B, T, num_classes)
        sqrt_mK_alpha_over_beta = sqrt_mK * alpha / beta.view(1, 1, -1)
        d = self.manifold.k.sqrt() * torch.abs(torch.asinh(sqrt_mK_alpha_over_beta))  # (B, T, num_classes)

        logits = torch.sign(alpha) * beta.view(1, 1, -1) * d  # (B, T, num_classes)

        return logits

    def init_weights(self):
        stdv = 1. / math.sqrt(1 + self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.attn_mode == 'euc':
            self.attn = CausalSelfAttention(config)
        elif config.attn_mode == 'hyp':
            self.attn = HyperbolicSelfAttention(config)
        else:
            raise ValueError("Invalid attn_mode, use 'euc'/'hyp'.")
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
        ))

        if config.head_mode == 'euc':
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            stdv = 1. / math.sqrt(config.n_embd)
            nn.init.uniform_(self.lm_head.weight.data, -stdv, stdv)

        elif config.head_mode == 'hyp':
            self.manifold = CustomLorentz(k=torch.tensor([config.curvature]))
            self.lm_head = LorentzMLR(
                manifold=self.manifold,
                num_features=config.n_embd,
                num_classes=config.vocab_size
            )
        else:
            raise ValueError("Invalid head_mode, choose 'euc'/'hyp'.")

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),))
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def generate_text(self, context, max_length=200, temperature=1.0, top_k=50):
        self.eval()
        generated = context.clone()
        for _ in range(max_length):
            with torch.no_grad():
                logits, _ = self(generated, return_logits=True)
                logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits[logits < values[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def model_size(self):
        """Calculate the model size in millions or thousands, based on parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        if total_params >= 1e6:
            return f"{total_params / 1e6:.2f}M"
        else:
            return f"{total_params / 1e3:.2f}K"