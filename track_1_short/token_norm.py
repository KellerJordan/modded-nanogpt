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
import re

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


# The ASCII fast path's equivalent of the normalizer: on ASCII text NFKC, NFD and StripAccents are the
# identity, Lowercase is bytes.lower(), and Strip removes exactly these (Rust's char::is_whitespace).
_WS_RUN = re.compile(rb"[ \t\r\n]+")
_ASCII_WHITESPACE = b" \t\n\x0b\x0c\r"


def _key(token: bytes, norm) -> bytes:
    """A token's normalized text as UTF-8; b"" for the tokens that keep their own id."""
    if token.isascii():  # 49,383 of the 50,256 tokens
        key = _WS_RUN.sub(b" ", token.lower())
        return key if key == b" " else key.strip(_ASCII_WHITESPACE)
    try:
        return norm.normalize_str(token.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError:  # a partial UTF-8 byte token
        return b""


def build_norm_map() -> np.ndarray:
    """[NORM_MAP_SIZE] int32: each id -> the smallest id with the same normalized text.

    ~70 ms on a laptop CPU: only the 873 non-ASCII tokens go through the (per-call slow) HF normalizer. <|endoftext|>
    is not a mergeable token, so it keeps its id with the padding.
    """
    ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks  # token bytes -> id, ids 0..50255
    tokens = [b""] * len(ranks)
    for token, i in ranks.items():
        tokens[i] = token
    norm = _normalizer()
    first: dict[bytes, int] = {}
    ids = [first.setdefault(key, i) if key else i for i, key in enumerate(_key(t, norm) for t in tokens)]
    out = np.arange(NORM_MAP_SIZE, dtype=np.int32)
    out[:len(ids)] = ids
    digest = hashlib.sha256(out[:len(ranks) + 1].astype("<i4").tobytes()).hexdigest()
    assert digest == NORM_MAP_SHA256, f"token normalization map mismatch (sha256 {digest}); check the tokenizers version"
    return out


# Host copy, built once at import: the data loader maps every batch's tokens before hashing.
NORM_MAP = torch.from_numpy(build_norm_map())
