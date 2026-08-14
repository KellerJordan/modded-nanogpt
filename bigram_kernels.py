# bigram_kernels.py — bigram embedding backward rewrite (M90B). Standalone
# module imported by the trainer; baked on/off by the trainer's configuration
# constants.
#
# M90B — bigram embedding backward as bf16 index_add_.
#   Stock aten embedding_dense_backward zero-fills a full-table fp32 buffer,
#   scatter-adds in fp32 and casts down to bf16. The table is 377280x768 so
#   that is >1.7 GB of avoidable traffic per (odd) step. This replaces it with
#   a dynamo-traceable autograd.Function whose backward is a bf16 zeros +
#   index_add_ (atomic bf16 adds). Gradient support and amax are identical;
#   per-element values may differ by ~1 bf16 ulp (accumulation order/precision)
#   vs the aten path — not bit-identical to it.

import torch
import torch.nn.functional as F


# ================================================================ M90B


class _M90BigramEmbed(torch.autograd.Function):
    """Embedding with bf16 index_add_ backward (no fp32 full-table zero-fill).

    Dynamo traces autograd.Functions, so this stays inside the fullgraph
    compile; inductor sees plain zeros + index_add_.
    """

    @staticmethod
    def forward(ctx, weight, idx):
        ctx.save_for_backward(idx)
        ctx.num_rows = weight.shape[0]
        return F.embedding(idx, weight)

    @staticmethod
    def backward(ctx, g):
        (idx,) = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        gw = torch.zeros((ctx.num_rows, g2.shape[-1]), dtype=g2.dtype,
                         device=g2.device)
        gw.index_add_(0, idx.reshape(-1).long(), g2)
        return gw, None


def m90b_bigram_embed(weight: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return _M90BigramEmbed.apply(weight, idx)
