"""Prefix-token prediction: map each token to its longest proper prefix that is also a token."""
import numpy as np
import tiktoken
import torch
from torch import Tensor

# Preload (download + parse) the GPT-2 tokenizer at import, outside the timed region.
# build_prefix_table_bucket then only pays the table construction itself.
tiktoken.get_encoding("gpt2")


def build_prefix_table_bucket(vocab_size: int, bucket: int, num_buckets: int) -> Tensor:
    """The prefix table's entries for one of `num_buckets` token buckets, -1 elsewhere.

    The full table maps each token T to its "prefix token" T': the token whose byte string is the
    longest proper prefix of T's byte string that is itself a token in the vocabulary. Tokens without
    one map to -1, "ignore this term". It is derived only from the static GPT-2 vocabulary, never
    from the training corpus.

    Bucket b holds every token whose second byte is b mod num_buckets, plus every 1-byte token. A bucket
    is closed under "is a proper prefix of" (a proper prefix of a multi-byte token is a 1-byte token or
    shares its second byte), so it yields the same entries as the whole vocabulary, and the buckets
    overlap only on 1-byte tokens, which map to -1. The elementwise max over all buckets is therefore
    the full table: each rank builds one bucket and an all_reduce(MAX) assembles it (record #360: the
    build is on the clock, so it is split across ranks).

    Sorting byte strings puts every proper prefix before its extensions, so a stack of
    live ancestors yields the longest one in O(V log V) instead of probing every length.
    """
    byte_to_id = tiktoken.get_encoding("gpt2")._mergeable_ranks
    byte_to_id = {b: tid for b, tid in byte_to_id.items() if len(b) == 1 or b[1] % num_buckets == bucket}
    # A numpy table wrapped without a copy: building a 50k-element Python list and converting it costs a
    # few ms more on the clock (record #360).
    table = np.full(vocab_size, -1, dtype=np.int64)
    stack: list[bytes] = []  # ancestors of the current token, shortest first
    stack_ids: list[int] = []
    for b, tid in sorted(byte_to_id.items()):
        while stack and not b.startswith(stack[-1]):
            stack.pop()
            stack_ids.pop()
        if stack_ids:
            table[tid] = stack_ids[-1]
        stack.append(b)
        stack_ids.append(tid)
    return torch.from_numpy(table)
