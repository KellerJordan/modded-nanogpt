import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass
from torch.nn.attention.flex_attention import create_block_mask
from transformers import EsmTokenizer, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from model.attention import SelfAttention, MultiHeadPAttention
from model.utils import norm, MLP


@dataclass
class PLMConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 512,
        num_attention_heads: int =  8,
        num_hidden_layers: int = 12,
        num_att_tokens: int = 512,
        vocab_size: int = 33,
        expansion_ratio: float = 2.0,
        attention_soft_cap: float = 64.0,
        add_att_soft_cap: bool = True,
        soft_logit_cap: float = 16.0,
        sliding_window_size: int = 2048,
        p_attention: bool = False,
        tie_embeddings: bool = False,
        unet: bool = False,
        mlm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_att_tokens = num_att_tokens
        self.vocab_size = vocab_size
        self.expansion_ratio = expansion_ratio
        self.soft_logit_cap = soft_logit_cap
        self.attention_soft_cap = attention_soft_cap
        self.add_att_soft_cap = add_att_soft_cap
        self.sliding_window_size = sliding_window_size
        self.p_attention = p_attention
        self.tie_embeddings = tie_embeddings
        self.unet = unet
        self.mlm = mlm
    

@dataclass
class ESMOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None


class ValueEmbedding(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_hidden_layers // 2)
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


class TransformerBlock(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.config = config
        if config.p_attention:
            self.attn = MultiHeadPAttention(config)
        else:
            self.attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.unet = config.unet
        if config.unet:
            self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
    
    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            vi: Optional[torch.Tensor] = None,
            x0: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        if self.unet:
            x = self.lambdas[0] * x + self.lambdas[1] * x0
            x = x + self.attn(norm(x), attention_mask, vi)
        else:
            x = x + self.attn(norm(x), attention_mask)
        x = x + self.mlp(norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x
    

class UnetTransformer(nn.Module):
    def __init__(self, config: PLMConfig):
        super().__init__()
        assert config.num_hidden_layers % 2 == 0
        self.num_encoder_layers = config.num_hidden_layers // 2
        self.num_decoder_layers = config.num_hidden_layers // 2

        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor, ve: List[torch.Tensor], attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x0 = x
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.layers[i](x, attention_mask, ve_enc[i], x0)
            skip_connections.append(x)
        
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.layers[self.num_encoder_layers + i](x, attention_mask, ve_dec[i], x0)
        return x


class PLM(PreTrainedModel):
    config_class = PLMConfig
    def __init__(self, config: PLMConfig):
        super().__init__(config)
        self.config = config
        self.tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        self.cls_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

        self.vocab_size = config.vocab_size
        self.n_heads = config.num_attention_heads
        self.sliding_window_size = config.sliding_window_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        self.unet = config.unet
        if config.unet:
            self.transformer = UnetTransformer(config)
            self.value_embeds = ValueEmbedding(config)
        else:
            self.transformer = Transformer(config)
    
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, config.soft_logit_cap)
        if config.tie_embeddings:
            self.lm_head.decoder.weight = self.embedding.weight

        self.mlm = config.mlm
        self.ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')

    def get_last_hidden_state(self, input_ids: torch.Tensor, sliding_window_size: int) -> torch.Tensor: # (l,)
        docs = (input_ids == self.cls_token_id).cumsum(0)
        last_eos = (input_ids == self.eos_token_id).nonzero()[-1].squeeze()
        seq_len = len(input_ids)

        def doc_mask_mod(b, h, q_idx, kv_idx):
            bidirectional_sliding_window_mask = torch.abs(q_idx - kv_idx) < sliding_window_size
            doc_mask = docs[q_idx] == docs[kv_idx]
            pad_mask = (q_idx <= last_eos) & (kv_idx <= last_eos)
            return bidirectional_sliding_window_mask & doc_mask & pad_mask

        attention_mask = create_block_mask(
            mask_mod=doc_mask_mod,
            B=1,
            H=self.n_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=input_ids.device,
        )

        x = self.embedding(input_ids)
        x = norm(x)
        if self.unet:
            ve = self.value_embeds(input_ids)
            x = self.transformer(x, ve, attention_mask)
        else:
            x = self.transformer(x, attention_mask)
        return x

    def get_vector_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        docs = (input_ids == self.cls_token_id).cumsum(0)
        x = self.get_last_hidden_state(input_ids)
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
        labels: torch.Tensor,
        mask_rate: torch.Tensor,
        sliding_window_size: Optional[int] = None,
        ) -> torch.Tensor:
        if sliding_window_size is None:
            sliding_window_size = self.sliding_window_size

        last_hidden_state = self.get_last_hidden_state(input_ids, sliding_window_size)

        lm_logits = self.lm_head(norm(last_hidden_state)) # (l, v)

        loss = self.ce(
            lm_logits.view(-1, self.vocab_size),
            labels.view(-1).long()
        )
        if self.training and not self.mlm:
            loss = loss / mask_rate

        return loss


if __name__ == "__main__":
    # py -m model.model
    from torchinfo import summary
    config = PLMConfig(
        hidden_size=768,
        num_attention_heads=6,
        num_hidden_layers=24,
        expansion_ratio=8/3,
        unet=True,
    )
    model = PLM(config).cuda()
    summary(model)

    input_ids = torch.randint(0, 33, (1, 100)).cuda()
    output = model(input_ids)
    print(f"loss: {output.loss}")
    print(f"logits: {output.logits[0].shape}")
    print(f"labels: {output.logits[1].shape}")
    print(f"last_hidden_state: {output.last_hidden_state.shape}")