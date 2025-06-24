# https://github.com/pytorch-labs/attention-gym/blob/main/attn_gym/mods/softcapping.py

import torch
import numpy as np
import math
from torch import Tensor
from torch._inductor.virtualized import ops
from functools import partial
from typing import Optional
from pathlib import Path
from torch._inductor.lowering import make_pointwise, register_lowering
from torch.nn.attention.flex_attention import (
    _score_mod_signature,
    _mask_mod_signature,
    _vmap_for_bhqkv,
    _ModificationType,
)

try:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
except ImportError:
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
from contextlib import nullcontext


def create_score_mod(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: Optional[_score_mod_signature],
    mask_mod: Optional[_mask_mod_signature],
    device: str = "cuda",
    _compile: bool = False,
    scale: Optional[float] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    B = 1
    H = 1
    M = query.shape[0]
    N = key.shape[0]

    b = torch.arange(0, B, device=device) + batch_idx
    h = torch.arange(0, H, device=device) + head_idx
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    type = _ModificationType.SCORE_MOD if score_mod is not None else _ModificationType.MASK_MOD
    if _compile:
        ctx = nullcontext()
    else:
        ctx = TransformGetItemToIndex()

    with ctx:
        mod_fn = score_mod if type == _ModificationType.SCORE_MOD else mask_mod
        prefix = (0,) if type == _ModificationType.SCORE_MOD else ()
        mod = _vmap_for_bhqkv(mod_fn, prefix=prefix)
        scores = query @ key.transpose(-2, -1)
        scores *= scale_factor
        scores = scores.view(1, 1, M, N)
        if type == _ModificationType.SCORE_MOD:
            out = mod(scores, b, h, m, n)
        else:
            out = mod(b, h, m, n)

    return out


@torch.library.custom_op("approx::tanh", mutates_args=())
def _tanh_approx(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


@_tanh_approx.register_fake
def _(inp: torch.Tensor) -> torch.Tensor:
    return torch.tanh(inp)


def _tanh_approx_lowering(inp):
    fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
    return make_pointwise(fn)(inp)


register_lowering(torch.ops.approx.tanh)(_tanh_approx_lowering)


class _TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(x):
        return torch.ops.approx.tanh(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result = output
        ctx.save_for_backward(result)

    @staticmethod
    def backward(ctx, grad_output):
        (result,) = ctx.saved_tensors
        return grad_output * (1 - result * result)

    @staticmethod
    def vmap(info, in_dims, x):
        return torch.tanh(x), 0


_tanh_approx = _TanhApprox.apply


def generate_tanh_softcap(soft_cap: int, approx: bool = False) -> _score_mod_signature:
    """Returns an tanh bias score_mod given the number of heads H

    Args:
        soft_cap: The soft cap value to use for normalizing logits
        approx: Whether to use the `tanh.approx.` ptx instruction

    Returns:
        tanh_softcap: score_mod
    """
    tanh = _tanh_approx if approx else torch.tanh

    def tanh_softcap(score, b, h, q_idx, kv_idx):
        return soft_cap * tanh(score / soft_cap)

    prefix = "tanh_softcap_approx" if approx else "tanh_softcap"
    tanh_softcap.__name__ = f"{prefix}_{soft_cap}"

    return tanh_softcap


def generate_dilated_sliding_window(window_size: int, dilation: int) -> _mask_mod_signature:
    """Generates a dilated sliding window attention mask.
    Args:
        window_size: The size of the sliding window.
        dilation: The dilation factor for the sliding window.

    Note:
        Query at position i can only attend to keys within a window of size `window_size`
        centered around i, where the keys are at positions j such that:
        * abs(i - j) <= window_size
        * abs(i - j) % dilation == 0
    """

    def dilated_sliding_window(b, h, q_idx, kv_idx):
        diff = torch.abs(q_idx - kv_idx)
        in_window = diff <= window_size
        is_dilated = (diff % dilation) == 0
        return in_window & is_dilated

    dilated_sliding_window.__name__ = f"dilated_sliding_window_{window_size}_dilation_{dilation}"
    return dilated_sliding_window


def _name_to_title(name: str) -> str:
    title = name.replace("_", " ")
    title = " ".join(word.capitalize() for word in title.split())
    return title


def visualize_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    score_mod: Optional[_score_mod_signature] = None,
    mask_mod: Optional[_mask_mod_signature] = None,
    device: str = "cuda",
    name: str = "attention_scores",
    path: Optional[Path] = None,
    batch_idx: int = 0,
    head_idx: int = 0,
    scale: Optional[float] = None,
):
    """
    Generate and save a visualization of attention scores.

    Args:
        query (Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        key (Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        score_mod (Optional[Callable]): If this is set this will take precedence over the mask_mod.
        mask_mod (Optional[Callable]): The mask_mod function used to create block_mask
        device (str): Device to run computations on (default: "cuda").
        name (str): Base name for the file and title (default: 'attention_scores').
        path (Path): Path to save the visualization. If None, will be saved to the current working directory.
        batch_idx (int): Index of the batch to visualize (default: 0).
        head_idx (int): Index of the head to visualize (default: 0).
        scale (float): Scale factor to apply to the attention scores. If None, will be set to 1 / sqrt(head_dim).

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    assert score_mod is not None or mask_mod is not None, (
        "Must provide either score_mod or mask_mod"
    )
    query = query[batch_idx, head_idx, :, :]
    key = key[batch_idx, head_idx, :, :]
    scores_viz = create_score_mod(
        query,
        key,
        score_mod=score_mod,
        mask_mod=mask_mod,
        scale=scale,
        device=device,
        batch_idx=batch_idx,
        head_idx=head_idx,
    )
    # If both score_mod and mask_mod are provided, apply both
    if score_mod is not None and mask_mod is not None:
        mask_viz = create_score_mod(
            query,
            key,
            score_mod=None,
            mask_mod=mask_mod,
            scale=scale,
            device=device,
            batch_idx=batch_idx,
            head_idx=head_idx,
        )
        # Apply mask by setting masked positions to -inf
        scores_viz = torch.where(mask_viz == 0, float("-inf"), scores_viz)

    suffix_title = f"Batch {batch_idx}, Head {head_idx}" if batch_idx != 0 or head_idx != 0 else ""

    fig, ax = plt.subplots(figsize=(12, 10))
    color = "viridis" if score_mod is not None else "cividis"
    if score_mod is not None and mask_mod is not None:
        color = "plasma"
    im = ax.imshow(scores_viz.cpu().detach()[0, 0, :, :], aspect="auto", cmap=color)
    fig.colorbar(im)

    title = _name_to_title(name)
    file_path = Path(name).with_suffix(".png") if path is None else path.with_suffix(".png")
    ax.set_title(f"{title}\n{suffix_title}", fontsize=20)

    ax.set_xlabel("Key Tokens", fontsize=18)
    ax.set_ylabel("Query Tokens", fontsize=18)

    # Move y-axis ticks and labels to the top
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    # Add tick labels if the number of tokens is manageable
    num_query_tokens, num_kv_tokens = scores_viz.shape[-2:]
    if num_query_tokens <= 32 and num_kv_tokens <= 32:
        ax.set_xticks(range(num_kv_tokens))
        rotation = 45 if num_kv_tokens > 12 else 0
        ax.set_xticklabels(
            [f"KV{i}" for i in range(num_kv_tokens)], fontsize=16, rotation=rotation
        )
        ax.set_yticks(range(num_query_tokens))
        ax.set_yticklabels([f"Q{i}" for i in range(num_query_tokens)], fontsize=16)
        # Align grid with pixel boundaries
        ax.set_xticks(np.arange(-0.5, num_kv_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_query_tokens, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=2)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory

    print(f"Visualization saved as {file_path}")


def main(device: str = "cpu"):
    """Visualize the attention scores of dilated sliding window mask mod.

    Args:
        device (str): Device to use for computation.
    """
    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 24, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    dilated_sliding_window_mask = generate_dilated_sliding_window(window_size=8, dilation=4)
    visualize_attention_scores(
        query,
        key,
        mask_mod=dilated_sliding_window_mask,
        device=device,
        name="dilated_sliding_window_mask",
    )


if __name__ == "__main__":
    main()