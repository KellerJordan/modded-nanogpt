import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
from functools import partial
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput
try:
    from .utils import ProteinMasker
except ImportError:
    from utils import ProteinMasker


@dataclass
class ModelConfig(PretrainedConfig):
    """
    33 tokens: https://huggingface.co/Synthyra/ESMplusplus_large/blob/main/modeling_esm_plusplus.py#L868-L874
    ESM2-8M has 6 layers, 20 heads, 320 hidden dim: https://huggingface.co/facebook/esm2_t6_8M_UR50D/blob/main/config.json
    ESM2-35M has 12 layers, 20 heads, 480 hidden dim: https://huggingface.co/facebook/esm2_t12_35M_UR50D/blob/main/config.json
    ESM2-150M has 30 layers, 20 heads, 640 hidden dim: https://huggingface.co/facebook/esm2_t30_150M_UR50D/blob/main/config.json
    ESM2-650M has 33 layers, 20 heads, 1280 hidden dim: https://huggingface.co/facebook/esm2_t33_650M_UR50D/blob/main/config.json
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int =  8,
        num_hidden_layers: int = 12,
        num_att_tokens: int = 128,
        vocab_size: int = 33,
        expansion_ratio: float = 2.0,
        dropout: float = 0.1,
        soft_logit_cap: float = 16.0,
        tokenformer: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_att_tokens = num_att_tokens
        self.vocab_size = vocab_size
        self.expansion_ratio = expansion_ratio
        self.dropout = dropout
        self.soft_logit_cap = soft_logit_cap
        self.tokenformer = tokenformer


@dataclass
class ESMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_encoder_layers: int):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_size)
            for _ in range(num_encoder_layers)
        ])

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


class LMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, soft_logit_cap: float = 30.0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.soft_logit_cap = soft_logit_cap
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(norm(x))
        x = self.act(x)
        x = self.decoder(x) + self.bias
        return self.soft_logit_cap * torch.tanh(x / self.soft_logit_cap)


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
        self.QKV = Linear(hidden_size, hidden_size * 3)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.O = Linear((hidden_size // n_heads) * n_heads, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = Rotary(hidden_size // n_heads)

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        qkv = self.QKV(x) # (bs, seq_len, hidden_size * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (bs, seq_len, hidden_size)
        q, k = self.rotary(norm(q)), self.rotary(norm(k))

        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, seq_len, d_head)
        a = flex_attention(q, k, v, block_mask=attention_mask, enable_gqa=True)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.O(a) # (bs, seq_len, hidden_size)


class TokenParamAttention(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, n_tokens) ->  (b, L, d)
    """
    def __init__(
            self,
            hidden_size: int,
            num_att_tokens: int = 128,
            num_attention_heads: int = 16,
    ):
        super(TokenParamAttention, self).__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.num_att_tokens = num_att_tokens
        self.d_head = hidden_size // num_attention_heads
        self.Pk = nn.Parameter(torch.randn(1, num_att_tokens, hidden_size))
        self.Pv = nn.Parameter(torch.randn(1, num_att_tokens, hidden_size))
        self.Q = Linear(hidden_size, hidden_size)
        self.K = Linear(hidden_size, hidden_size)
        self.V = Linear(hidden_size, hidden_size)
        self.O = Linear((hidden_size // num_attention_heads) * num_attention_heads, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=num_attention_heads)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, _ = x.size()
        q = self.Q(x) # (b, L, d)
        k = self.K(self.Pk).expand(b, -1, -1) # (b, num_att_tokens, d)
        v = self.V(self.Pv).expand(b, -1, -1) # (b, num_att_tokens, d)
        q, k = norm(q), norm(k)
        q, k, v = map(self.reshaper, (q, k, v))  # (b, num_attention_heads, L, d_head), (b, num_attention_heads, num_att_tokens, d_head), (b, num_attention_heads, num_att_tokens, d_head)
        a = flex_attention(q, k, v, block_mask=attention_mask, enable_gqa=True)
        a = rearrange(a, "b h s d -> b s (h d)")  # (b, L, n_heads * d_head)
        return self.O(a)  # (b, L, d)


def correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class MLP(nn.Module):
    def __init__(self, dim, expansion_ratio):
        super().__init__()
        self.up   = Linear(dim, correction_fn(expansion_ratio, dim))
        self.down = Linear(correction_fn(expansion_ratio, dim), dim)
        self.down.weight.data.zero_() # zero init suggested by @Grad62304977
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # https://arxiv.org/abs/2109.08668v2
        # ReLU squared ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        return self.down(self.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SelfAttention(config.hidden_size, config.num_attention_heads)
        self.mlp_1 = MLP(config.hidden_size, config.expansion_ratio)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.tokenformer = config.tokenformer
        if self.tokenformer:
            self.token_param_attn = TokenParamAttention(config.hidden_size, config.num_att_tokens, config.num_attention_heads)
            self.mlp_2 = MLP(config.hidden_size, config.expansion_ratio)
    
    def forward(
        self,
        x: torch.Tensor,
        vi: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.self_attn(x, vi, attention_mask)
        x = x + self.mlp_1(norm(x))
        if self.tokenformer:
            x = x + self.token_param_attn(norm(x), attention_mask)
            x = x + self.mlp_2(norm(x))
        return x


class ESM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.num_hidden_layers % 2 == 0, "num_hidden_layers must be even"
        self.num_encoder_layers = config.num_hidden_layers // 2
        self.num_decoder_layers = config.num_hidden_layers // 2
        self.skip_weights = nn.Parameter(torch.ones(self.num_encoder_layers))
        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            x: torch.Tensor,
            x0: torch.Tensor,
            ve: List[torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.layers[i](x, ve_enc[i], x0, attention_mask)
            skip_connections.append(x)
        
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.layers[self.num_encoder_layers + i](x, ve_dec[i], x0, attention_mask)
        return x


class ESMForMaskedLM(PreTrainedModel):
    config_class = ModelConfig
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.masker = ProteinMasker(tokenizer)
        self.cls_id = tokenizer.cls_token_id
        self.vocab_size = tokenizer.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.value_embeds = ValueEmbedding(config.vocab_size, config.hidden_size, config.num_hidden_layers // 2)
        self.bert = ESM(config)
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, config.soft_logit_cap)
        self.lm_head.decoder.weight = self.embedding.weight
        self.vocab_size = config.vocab_size
        self.ce = nn.CrossEntropyLoss()

    def _embed(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        x = norm(self.embedding(input_ids[None]))
        x0 = x
        ve = self.value_embeds(input_ids)
        return x, x0, ve

    def _get_last_hidden_state(self, input_ids: torch.Tensor, sliding_window_size: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.flatten() # flex_attention needs batch 1
        docs = (input_ids == self.cls_id).cumsum(0)

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            return bidirectional_sliding_window_mask & doc_mask

        S = len(input_ids)
        attention_mask = create_block_mask(doc_mask_mod, None, None, S, S)

        x, x0, ve = self._embed(input_ids)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.layers[i](x, ve_enc[i], x0, attention_mask)
            skip_connections.append(x)
        
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.layers[self.num_encoder_layers + i](x, ve_dec[i], x0, attention_mask)
        return x

    def get_vector_embeddings(self, input_ids: torch.Tensor, sliding_window_size: torch.Tensor) -> torch.Tensor:
        docs = (input_ids == self.cls_id).cumsum(0)
        x = self._get_last_hidden_state(input_ids, sliding_window_size)
        x = x.view(-1, self.config.hidden_size) # (S, hidden_size)
        # At this point, x is shape [S, hidden_size]
        # We want to mean-pool across each document index.
        # Convert docs to 0-based so we can do nice indexing
        num_docs = docs.max().item()
        doc_ids = docs - 1  # Now documents are labeled [0, 1, 2, ...]
        # Mean-pool across tokens belonging to each doc
        doc_embeds = []
        for doc_idx in range(num_docs):
            mask = (doc_ids == doc_idx)
            # Collect all token embeddings for this doc and average
            doc_embeds.append(x[mask].mean(dim=0))
        # Stack into [num_documents, hidden_size]
        return torch.stack(doc_embeds, dim=0)

    def forward(
            self,
            input_ids: torch.Tensor,
            sliding_window_size: torch.Tensor,
            mask_prob: torch.Tensor,
            keep_replace_prob: torch.Tensor) -> torch.Tensor:
        input_ids, labels = self.masker(input_ids, mask_prob, keep_replace_prob)
        x = self._get_last_hidden_state(input_ids, sliding_window_size)
        logits = self.lm_head(x)
        loss = self.ce(logits.view(-1, self.vocab_size).float(), labels.view(-1).long())
        return ESMOutput(loss=loss, logits=logits, last_hidden_state=x)


if __name__ == "__main__":
    config = ModelConfig()
    model = ESMForMaskedLM(config).cuda()
    print(model)

    input_ids = torch.randint(0, 33, (2, 100)).to(torch.int32).cuda()
    sliding_window_size = torch.tensor(10).cuda()
    mask_prob = torch.tensor(0.1).cuda()
    keep_replace_prob = torch.tensor(0.1).cuda()
    output = model(input_ids, sliding_window_size, mask_prob, keep_replace_prob)
    print(output)
