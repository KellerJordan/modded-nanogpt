"""PR #375's token normalization map: GPT-2 token id -> normalized id, applied before the n-gram hashes.

The hashed n-gram rows (ngram_table.ngram_row_ids) and the sign hashes (GPT.ng_norm_map) both read a token's
normalized id instead of its own, so tokens the map merges share their n-gram rows. The map is int32 [50304]
(the padded vocabulary); the sha256 pins its 50257 real entries.
"""
import hashlib
import os

import numpy as np
import torch

NORM_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "norm_map_pr375.npy")
NORM_MAP_SHA256 = "79fb57ac259495ba36f1afb9b0d5da35f3048b5067fa59c101d5a3c1ed15c2a2"  # of the first 50257 entries

NORM_MAP = torch.from_numpy(np.load(NORM_MAP_PATH))
assert NORM_MAP.dtype == torch.int32 and NORM_MAP.numel() == 50304
assert hashlib.sha256(NORM_MAP[:50257].numpy().astype("<i4").tobytes()).hexdigest() == NORM_MAP_SHA256, \
    "token normalization map mismatch (PR #375)"
