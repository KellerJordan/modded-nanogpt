"""Independent CPU preprocessing references frozen from upstream 4ea6b937."""
import threading
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

BOS_ID = 50256
NGRAM_VOCAB_SIZE = 84602880
BIGRAM_ROW_MULS = (36313, 27191)
TRIGRAM_ROW_MULS = (17351, 60961, 45259)

class LegacyShard:

    def __init__(self, tokens: Tensor, world_size: int=1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0
        self.bos_idx = (tokens[:6000000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]
        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration('Insufficient BOS ahead; hit tail of shard.')
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                idx += 1
                end = min(self.bos_idx[idx] if idx < n else self.size, cur + max_seq_len, cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
            assert cur_len == num_tokens_local + 1
        self.i = idx
        return (starts, ends)

def legacy_ngram_row_ids(x: Tensor) -> Tensor:
    """Token ids [T] (host) -> table row ids [2T] int32: out[:T] bigram rows, out[T:] trigram rows.

    Positions without enough history get the channel's reserved last row. Note the bigram hash at
    t = 1 reads the reserved id, not x[0], as its previous token (it is computed in place after
    out[0] is set, exactly as in record #360).
    """
    x = x.to(torch.int32)
    half = NGRAM_VOCAB_SIZE // 2
    bigram_mod, trigram_mod = (half - 1, NGRAM_VOCAB_SIZE - half - 1)
    n = x.numel()
    out = torch.empty(2 * n, dtype=torch.int32)
    bigram, trigram = (out[:n], out[n:])
    bigram.copy_(x)
    bigram[0] = bigram_mod
    bigram[1:] = torch.bitwise_xor(BIGRAM_ROW_MULS[0] * bigram[1:], BIGRAM_ROW_MULS[1] * bigram[:-1]) % bigram_mod
    trigram[0] = trigram[1] = half + trigram_mod
    trigram[2:] = torch.bitwise_xor(torch.bitwise_xor(TRIGRAM_ROW_MULS[0] * x[2:], TRIGRAM_ROW_MULS[1] * x[1:-1]), TRIGRAM_ROW_MULS[2] * x[:-2]) % trigram_mod + half
    return out
