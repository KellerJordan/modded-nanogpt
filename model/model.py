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
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_att_tokens = num_att_tokens
        self.vocab_size = vocab_size
        self.expansion_ratio = expansion_ratio
        self.dropout = dropout
        self.soft_logit_cap = soft_logit_cap


@dataclass
class ESMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


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





class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = SelfAttention(config.hidden_size, config.num_attention_heads)
        self.mlp = MLP(config.hidden_size, config.expansion_ratio)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.self_attn(norm(x), attention_mask)
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class PLM(PreTrainedModel):
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
        self.bert = Transformer(config)
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
    model = PLM(config).cuda()
    print(model)

    input_ids = torch.randint(0, 33, (2, 100)).to(torch.int32).cuda()
    sliding_window_size = torch.tensor(10).cuda()
    mask_prob = torch.tensor(0.1).cuda()
    keep_replace_prob = torch.tensor(0.1).cuda()
    output = model(input_ids, sliding_window_size, mask_prob, keep_replace_prob)
    print(output)
