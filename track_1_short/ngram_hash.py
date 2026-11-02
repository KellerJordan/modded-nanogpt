"""Fused CPU construction of the record's signed-int32 n-gram row hashes."""
import torch

# 224x the old 377,280-row table. The two equal channels reserve their last row for missing history.
NGRAM_VOCAB_SIZE = 84_602_880
BIGRAM_ROW_MULS = (36313, 27191)
TRIGRAM_ROW_MULS = (17351, 60961, 45259)


def _hash_rows(x: torch.Tensor) -> torch.Tensor:
    """Functional form lets Inductor fuse shifts, wrapped products, xor and signed remainder.

    The bigram at position one intentionally reads the reserved row as its previous token,
    preserving #360's in-place construction. Products wrap in int32 before the positive remainder.
    """
    half = NGRAM_VOCAB_SIZE // 2
    mod = half - 1
    positions = torch.arange(x.numel(), dtype=torch.int32, device=x.device)
    previous = torch.where(positions == 1, mod, x.roll(1))
    bigram = ((x * BIGRAM_ROW_MULS[0]) ^ (previous * BIGRAM_ROW_MULS[1])) % mod
    bigram = torch.where(positions == 0, mod, bigram)
    trigram = ((x * TRIGRAM_ROW_MULS[0]) ^ (x.roll(1) * TRIGRAM_ROW_MULS[1])
               ^ (x.roll(2) * TRIGRAM_ROW_MULS[2])) % mod + half
    trigram = torch.where(positions < 2, NGRAM_VOCAB_SIZE - 1, trigram)
    return torch.cat((bigram, trigram))


_compiled_hash_rows = torch.compile(_hash_rows, dynamic=False, fullgraph=True)


@torch.no_grad()
def ngram_row_ids(x: torch.Tensor) -> torch.Tensor:
    """Host int32 rows [2T], ordered bigrams then trigrams, without modifying the input.

    The loader already supplies contiguous CPU int32 tokens. Other widths/strides are normalized
    once; evaluation may supply device tokens, in which case there is one explicit D2H copy.
    """
    if x.ndim != 1:
        raise ValueError("n-gram tokens must be one-dimensional")
    if x.numel() == 0:
        return torch.empty(0, dtype=torch.int32)
    if x.numel() == 1:
        return torch.tensor([NGRAM_VOCAB_SIZE // 2 - 1, NGRAM_VOCAB_SIZE - 1], dtype=torch.int32)
    return _compiled_hash_rows(x.to(device="cpu", dtype=torch.int32).contiguous())


def prime_ngram_hash(token_counts):
    """Compile every train/validation geometry on dummy tokens before the training clock starts."""
    for count in sorted(set(token_counts)):
        ngram_row_ids(torch.zeros(count, dtype=torch.int32))
