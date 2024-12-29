import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from typing import Optional, Tuple, List, Any
try:
    from .utils import ProteinMasker
except ImportError:
    from utils import ProteinMasker


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
        vocab_size=33,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        expansion_ratio=8/3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.expansion_ratio = expansion_ratio


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))


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
    """
    TODO
    Add F.spda option
    Add causal option (flex and sdpa)
    """
    def __init__(self, dim, num_attention_heads):
        super().__init__()
        assert dim % num_attention_heads == 0
        self.num_attention_heads = num_attention_heads
        self.qkv = CastedLinear(dim, 3 * dim)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_attention_heads) # dim // num_attention_heads = head_dim
        self.o_proj = CastedLinear(dim, dim)
        self.o_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward_sdpa(self, x: torch.Tensor, vi: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        TODO
        Question? Is this output actually different than flex attention output?
        Likely yes because of scoremod and / or soft capping
        Would be good to be able to do inference this way for typical PLM inference pipelines
        https://pytorch.org/blog/flexattention/
        """
        B, T = x.size(0), x.size(1) # batch size, sequence length
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_attention_heads, -1)
        k = k.view(B, T, self.num_attention_heads, -1)
        v = v.view(B, T, self.num_attention_heads, -1)
        v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v) # @KoszarskyB & @Grad62304977
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=True
        )
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.o_proj(y)
        return y

    def forward(self, x: torch.Tensor, vi: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_attention_heads, -1)
        k = k.view(B, T, self.num_attention_heads, -1)
        v = v.view(B, T, self.num_attention_heads, -1)
        v = self.lambdas[0] * v + self.lambdas[1] * vi.view_as(v) # @KoszarskyB & @Grad62304977
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, enable_gqa=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.o_proj(y)
        return y


def correction_fn(expansion_ratio: float, d_model: int) -> int:
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class MLP(nn.Module):
    def __init__(self, dim, expansion_ratio):
        super().__init__()
        self.up   = CastedLinear(dim, correction_fn(expansion_ratio, dim))
        self.down = CastedLinear(correction_fn(expansion_ratio, dim), dim)
        self.down.weight.data.zero_() # zero init suggested by @Grad62304977
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # https://arxiv.org/abs/2109.08668v2
        # ReLU squared ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        return self.down(self.relu(self.up(x)).square())


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config.hidden_size, config.num_attention_heads)
        self.mlp = MLP(config.hidden_size, config.expansion_ratio)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def sdpa_forward(self, x: torch.Tensor, vi: torch.Tensor, x0: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn.forward_sdpa(norm(x), vi, attention_mask)
        x = x + self.mlp(norm(x))
        return x

    def forward(self, x: torch.Tensor, vi: torch.Tensor, x0: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(norm(x), vi, block_mask)
        x = x + self.mlp(norm(x))
        return x


class ValueEmbedding(nn.Module):
    def __init__(self, config: "ModelConfig"):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_hidden_layers // 2)
        ])

    def forward(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        ve = [emb(inputs) for emb in self.embed]
        ve += reversed(ve)
        return ve


class ESM(PreTrainedModel):
    """
    TODO
    Add causal option (flex and sdpa)
    """
    config_class = ModelConfig
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config
        tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.masker = ProteinMasker(tokenizer, 0.20) # 20% masking rate https://arxiv.org/abs/2301.06568
        self.inference_masker = ProteinMasker(tokenizer, 0.15) # 15% masking rate for inference, ESM2
        self.cls_id = tokenizer.cls_token_id
        self.vocab_size = tokenizer.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # U-net design by @brendanh0gan
        assert config.num_hidden_layers % 2 == 0, "Number of layers should be even for U-net design"
        self.num_encoder_layers = config.num_hidden_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = config.num_hidden_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.embed = nn.Embedding(self.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
        # U-net structure on token value embeddings by @leloykun
        self.value_embeds = ValueEmbedding(config)
        self.lm_head = CastedLinear(config.hidden_size, self.vocab_size)
        self.lm_head.weight.data.zero_() # @Grad62304977
        self.cross_entropy = nn.CrossEntropyLoss()

    def embed_forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        x = self.embed(input_ids[None])
        x = norm(x) # @Grad62304977
        x0 = x
        ve = self.value_embeds(input_ids)
        return x, x0, ve

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        return logits

    def sdpa_forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].bool()
        
        x, x0, ve = self.embed_forward(input_ids)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]

        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i].sdpa_forward(x, ve_enc[i], x0, attention_mask)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i].sdpa_forward(x, ve_dec[i], x0, attention_mask)

        return self.get_logits(x)

    def flex_forward(self, input_ids: torch.Tensor, sliding_window_size: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.flatten() # flex_attention needs batch 1
        docs = (input_ids == self.cls_id).cumsum(0)

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            return bidirectional_sliding_window_mask & doc_mask

        S = len(input_ids)
        block_mask = create_block_mask(doc_mask_mod, None, None, S, S)

        x, x0, ve = self.embed_forward(input_ids)
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

        return self.get_logits(x)

    def get_vector_embeddings(self, input_ids: torch.Tensor, sliding_window_size: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.flatten()
        docs = (input_ids == self.cls_id).cumsum(dim=0)  # shape: [S]
        
        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            return bidirectional_sliding_window_mask & doc_mask
        
        S = len(input_ids)
        block_mask = create_block_mask(doc_mask_mod, None, None, S, S)
        x, x0, ve = self.embed_forward(input_ids)  # x shape: [S, hidden_size]
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_mask)
            skip_connections.append(x)

        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_mask)
        
        x = x.view(-1, self.config.hidden_size)
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

    def inference(self, input_ids: torch.Tensor, sliding_window_size: torch.Tensor = None) -> Tuple[torch.Tensor, Any, Any]:
        input_ids, labels = self.inference_masker(input_ids)
        logits = self.flex_forward(input_ids, sliding_window_size)
        loss = None
        if labels is not None:
            loss = self.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1).long())
        return logits, loss, labels

    def forward(self, input_ids: torch.Tensor, sliding_window_size: torch.Tensor) -> torch.Tensor:
        input_ids, labels = self.masker(input_ids)
        logits = self.flex_forward(input_ids, sliding_window_size)
        return self.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1).long())


if __name__ == '__main__':
    """
    TODO
    look at MSE between flex attention outputs and sdpa outputs
    """