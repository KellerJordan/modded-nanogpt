"""The n-gram hashes' token normalization: GPT-2 id -> the id of its normalization class.

What: tokens whose text normalizes to the same string (NFKC, accents stripped, lowercased, runs of
whitespace collapsed to one space, outer whitespace stripped) share one id -- the smallest id of the
class -- so " The", "THE" and " the" hash to the same bigram / trigram row and sign row.

Only the n-gram hashes read the mapped ids (ngram_table.ngram_row_ids and the sign-pool hashes of the
model's ngram_embedding call). The token embedding, value embeddings, lm_head, targets and the data
stream all keep raw GPT-2 ids.

Tokens that keep their own id: <|endoftext|>, the byte tokens that are not whole UTF-8, and the 9
tokens that normalize to "" (the vertical-tab and form-feed controls, lone combining marks, the emoji
variation selector U+FE0F). Whitespace-only tokens do not: the sentinel keeps them as " ". The map is
padded to the model's 50,304-row vocab with identity.

Checks: 32,980 classes over the 50,257 tokens, 17,277 ids remapped, and NORM_MAP_SHA256 below.
"""
import hashlib

import numpy as np
import tiktoken
import torch
from tokenizers import Regex, normalizers

NORM_MAP_SIZE = 50304
# sha256 of the map's first 50,257 entries as little-endian int32.
NORM_MAP_SHA256 = "79fb57ac259495ba36f1afb9b0d5da35f3048b5067fa59c101d5a3c1ed15c2a2"

# Private-use sentinel: a whitespace-only token becomes " ", then this, so Strip keeps it.
_SENTINEL = "\ue000"


def _normalizer():
    return normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), _SENTINEL),
        normalizers.Strip(),
        normalizers.Replace(_SENTINEL, " "),
    ])


def build_norm_map() -> np.ndarray:
    """[NORM_MAP_SIZE] int32: each id -> the smallest id with the same normalized text."""
    enc = tiktoken.get_encoding("gpt2")
    norm = _normalizer()
    out = np.arange(NORM_MAP_SIZE, dtype=np.int32)
    first: dict[str, int] = {}
    for i in range(enc.n_vocab):
        if i == enc.eot_token:
            continue
        try:
            text = enc.decode_single_token_bytes(i).decode("utf-8")
        except UnicodeDecodeError:
            continue
        key = norm.normalize_str(text)
        if key == "":
            continue
        out[i] = first.setdefault(key, i)
    digest = hashlib.sha256(out[:enc.n_vocab].astype("<i4").tobytes()).hexdigest()
    assert digest == NORM_MAP_SHA256, f"token normalization map mismatch (sha256 {digest}); check the tokenizers version"
    return out


# Host copy, built once at import: the data loader maps every batch's tokens before hashing.
NORM_MAP = torch.from_numpy(build_norm_map())
