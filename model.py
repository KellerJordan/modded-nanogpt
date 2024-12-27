import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers import EsmTokenizer
from utils import ProteinMasker


"""
TODO
add type setting
"""

class ModelConfig:
    def __init__(self, args):
        # 33 tokens: https://huggingface.co/Synthyra/ESMplusplus_large/blob/main/modeling_esm_plusplus.py#L868-L874
        # Depth of the number of layers is typically more important than the depth of the hidden dimension for PLMs
        # ESM2-8M has 6 layers, 20 heads, 320 hidden dim: https://huggingface.co/facebook/esm2_t6_8M_UR50D/blob/main/config.json
        # ESM2-35M has 12 layers, 20 heads, 480 hidden dim: https://huggingface.co/facebook/esm2_t12_35M_UR50D/blob/main/config.json
        # ESM2-150M has 30 layers, 20 heads, 640 hidden dim: https://huggingface.co/facebook/esm2_t30_150M_UR50D/blob/main/config.json
        # ESM2-650M has 33 layers, 20 heads, 1280 hidden dim: https://huggingface.co/facebook/esm2_t33_650M_UR50D/blob/main/config.json
        self.vocab_size = args.vocab_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.model_dim = args.model_dim


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer('inv_freq', (1 / base) ** (torch.arange(0, dim, 2) / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
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
    """
    TODO
    Add F.spda option
    Add causal option (flex and sdpa)
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads) # dim // num_heads = head_dim
        self.c_proj = CastedLinear(dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward_sdpa(self, x, vi, attention_mask=None):
        pass

    def forward(self, x, vi, block_mask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v) # @KoszarskyB & @Grad62304977
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc   = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config.model_dim, config.num_heads)
        self.mlp = MLP(config.model_dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, vi, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), vi, block_mask)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.model_dim)
            for _ in range(config.num_layers // 2)
        ])

    def forward(self, inputs) -> "list[torch.Tensor]":
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


class ESM(nn.Module):
    """
    TODO
    Add causal option (flex and sdpa)
    """
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.masker = ProteinMasker(tokenizer, 0.20) # 20% masking rate https://arxiv.org/abs/2301.06568
        self.cls_id = tokenizer.cls_token_id
        self.vocab_size = tokenizer.vocab_size
        self.num_layers = config.num_layers

        # U-net design by @brendanh0gan
        assert config.num_layers % 2 == 0, "Number of layers should be even for U-net design"
        self.num_encoder_layers = config.num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.embed = nn.Embedding(self.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        # U-net structure on token value embeddings by @leloykun
        self.value_embeds = ValueEmbedding(config)
        self.lm_head = CastedLinear(config.model_dim, self.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977
        self.cross_entropy = nn.CrossEntropyLoss()

    def get_logits(self, input_ids: torch.Tensor, sliding_window_size: torch.Tensor):
        input_ids = input_ids.flatten() # flex_attention needs batch 1
        docs = (input_ids == self.cls_id).cumsum(0)

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            return bidirectional_sliding_window_mask & doc_mask

        S = len(input_ids)
        block_mask = create_block_mask(doc_mask_mod, None, None, S, S)

        x = self.embed(input_ids[None])
        x = norm(x) # @Grad62304977
        x0 = x
        ve = self.value_embeds(input_ids)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            # U-net structure on token value embeddings by @leloykun
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        return logits

    def inference(self, input_ids, labels=None): # premaked input_ids
        sliding_window_size = torch.tensor(1024, dtype=torch.int32, device="cuda")
        logits = self.get_logits(input_ids, sliding_window_size)
        loss = None
        if labels is not None:
            loss = self.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1).long())
        return loss, logits

    def forward(self, input_ids, sliding_window_size: torch.Tensor):
        input_ids, labels = self.masker(input_ids)
        logits = self.get_logits(input_ids, sliding_window_size)
        return self.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1).long())
