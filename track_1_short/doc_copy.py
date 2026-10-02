"""Document-local copy feature: a candidate next token per position from the longest exact earlier match
of its context in the same document, fed to the model as an extra input.

For each match length L in COPY_LENGTHS, every position with L tokens of document history hashes its
context x[t-L+1..t] into a 64-bit key mixed with L and the document index. One stable sort of all keys
puts each position right after the latest earlier position with the same key, so one sort finds every
match. The longest L with a match wins; the candidate is x[j + 1] for that match's end j < t. The bucket
encodes L and how often the context occurred earlier in the document (1, 2, 3+); bucket 0 = no match.
A 64-bit key collision (probability about (len(COPY_LENGTHS) * T)^2 / 2^65 per forward: ~1e-8 for a rank's
training batch, ~2e-7 for its validation batch) can only cost a match; it cannot reach past t (see below).

Contexts hash PR #375's normalized ids (token_norm.py), so a context matches across case, accent and
whitespace variants; the candidate is the raw next token and documents split on the raw BOS id.

Only tokens at or before t are read (the match must end before t, enforced explicitly), so the model
stays a causal next-token predictor. The model embeds each candidate with the token embedding (tied to
lm_head for most of the run), scales it per bucket, adds a per-bucket embedding and
injects the sum at the input and before the final norm (model/gpt.py).

Cost: three prefix sums, one sort of len(COPY_LENGTHS) x T keys and some scatters and gathers inside the
compiled forward (captured in the step's CUDA graph like the rest of it): no host syncs.
"""
import torch
from torch import Tensor

COPY_LENGTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
COUNT_BUCKETS = 3  # earlier occurrences of the matched context: 1, 2, 3+
COPY_BUCKETS = len(COPY_LENGTHS) * COUNT_BUCKETS + 1  # 0 = no match
BOS_ID = 50256
_HASH_MUL = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64; products wrap
_DOC_MUL = 0x5851F42D4C957F2D
_LEN_MUL = 0x2545F4914F6CDD1D


def hash_power_tables(T: int, device) -> tuple[Tensor, Tensor]:
    """(M^t, M^-t) for t < T in int64 arithmetic mod 2^64 (M is odd, hence invertible): model buffers."""
    inv = pow(_HASH_MUL % 2**64, -1, 2**64)
    inv = inv - 2**64 if inv >= 2**63 else inv
    ones = torch.ones(1, dtype=torch.int64, device=device)
    fwd = torch.cat([ones, torch.full((T - 1,), _HASH_MUL, dtype=torch.int64, device=device).cumprod(0)])
    bwd = torch.cat([ones, torch.full((T - 1,), inv, dtype=torch.int64, device=device).cumprod(0)])
    return fwd, bwd


def _run_starts(is_start: Tensor, run_id: Tensor, index: Tensor) -> Tensor:
    """For each element, the index of the latest element at or before it that starts a run (0 before the
    first start). run_id = cumsum(is_start). Each run's start index is scattered once and gathered back:
    torch.cummax gives the same values but is ~25-100x slower on GPU."""
    starts = torch.zeros(index.numel() + 1, dtype=torch.int64, device=index.device)
    starts.scatter_add_(0, run_id, torch.where(is_start, index, 0))
    return starts[run_id]


def _window_hashes(x: Tensor, lengths: Tensor, powers: tuple[Tensor, Tensor], norm_map: Tensor):
    """Hash of every L-token window ending at every position, for all lengths at once: ([nL, T] hash,
    [nL, T] valid = the window lies inside the position's document, [T] document index, [T] position).

    Window hash of y = norm(x) + 1 over [t-L+1, t]: sum_i y[i] M^(t-i) = M^t (S[t] - S[t-L]) with
    S[t] = sum_{i<=t} y[i] M^-i, all mod 2^64, mixed with the length (lengths collide only by chance).
    """
    T = x.numel()
    pos = torch.arange(T, device=x.device)
    is_bos = x == BOS_ID
    doc = torch.cumsum(is_bos.long(), 0)
    doc_start = _run_starts(is_bos, doc, pos)
    fwd, bwd = powers[0][:T], powers[1][:T]
    S = torch.cumsum((norm_map[x].long() + 1) * bwd, 0)
    S_pad = torch.cat([S.new_zeros(1), S])                                  # S_pad[t + 1] = S[t]
    start = pos[None, :] - lengths[:, None] + 1                             # [nL, T]
    valid = start >= doc_start[None, :]
    window = fwd[None, :] * (S_pad[pos + 1][None, :] - S_pad[start.clamp_min(0)])
    return window ^ (lengths[:, None] * _LEN_MUL), valid, doc, pos


@torch.no_grad()
def doc_copy_features(x: Tensor, powers: tuple[Tensor, Tensor], norm_map: Tensor) -> tuple[Tensor, Tensor]:
    """x [T] token ids -> (candidate [T] int64, -1 if none; bucket [T] int64, 0 if none).

    Equal (length, document, window) keys sort next to each other, positions ascending (stable), so each
    position's predecessor in the sorted order is its latest earlier match."""
    x = x.long()
    T = x.numel()
    dev = x.device
    lengths = torch.tensor(COPY_LENGTHS, device=dev)
    nL = lengths.numel()
    window, valid, doc, pos = _window_hashes(x, lengths, powers, norm_map)
    key = window ^ (doc[None, :] * _DOC_MUL)
    flat_pos = pos.repeat(nL)
    # Invalid windows get key -1 - t, which a valid 64-bit hash is vanishingly unlikely to equal; invalid
    # entries never count as matches (valid below). No large constants: inductor.
    key = torch.where(valid, key, -1 - flat_pos.view(nL, T)).flatten()
    sorted_key, order = torch.sort(key, stable=True)
    same = sorted_key[1:] == sorted_key[:-1]
    prev = torch.full((nL * T,), -1, dtype=torch.int64, device=dev)
    prev[order[1:]] = torch.where(same, flat_pos[order[:-1]], -1)
    prev = prev.view(nL, T)
    # prev < t keeps the feature causal by construction: equal keys from different lengths sort by length
    # row, not by position, so a 64-bit collision across lengths could otherwise pair t with a later position.
    found = (prev >= 0) & (prev < pos[None, :]) & valid
    # The longest length with a match: the last found row per column.
    rank = torch.where(found, torch.arange(1, nL + 1, device=dev)[:, None], 0)
    length_bucket = rank.max(0).values
    best_prev = prev.gather(0, (length_bucket - 1).clamp_min(0)[None, :])[0]
    candidate = torch.where(length_bucket > 0, x[(best_prev + 1).clamp(max=T - 1)], -1)
    # Earlier occurrences of each entry's context: its offset within its run of equal sorted keys.
    idx = torch.arange(nL * T, device=dev)
    new_run = torch.cat([same.new_ones(1), ~same])
    run_start = _run_starts(new_run, torch.cumsum(new_run.long(), 0), idx)
    earlier = torch.empty(nL * T, dtype=torch.int64, device=dev)
    earlier[order] = idx - run_start
    earlier = earlier.view(nL, T).gather(0, (length_bucket - 1).clamp_min(0)[None, :])[0]
    count_bucket = earlier.clamp(1, COUNT_BUCKETS) - 1
    bucket = torch.where(length_bucket > 0, (length_bucket - 1) * COUNT_BUCKETS + count_bucket + 1, 0)
    return candidate, bucket
