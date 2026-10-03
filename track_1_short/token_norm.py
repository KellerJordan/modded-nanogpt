"""PR #375's n-gram hash token normalization: each GPT-2 id -> the smallest id whose text normalizes the same (NFKC,
accents stripped, lowercased, whitespace runs to one space, outer whitespace stripped). #375 builds it with the HF
`tokenizers` normalizers; this is the same map in pure Python, checked against #375's sha256 of it."""
import hashlib
import re
import unicodedata

import numpy as np
import tiktoken
import torch

NORM_MAP_SIZE = 50304
NORM_MAP_SHA256 = "79fb57ac259495ba36f1afb9b0d5da35f3048b5067fa59c101d5a3c1ed15c2a2"  # #375's, of the first 50,257 entries
_WS_RUN = re.compile(rb"[ \t\r\n]+")
_ASCII_WHITESPACE = b" \t\n\x0b\x0c\r"
_WS_RUN_STR = re.compile(r"[ \t\r\n]+")


def _rust_strip(s):  # Rust's str::trim (Unicode White_Space), as HF's Strip normalizer
    ws = lambda ch: ch in "\t\n\x0b\x0c\r \x85\xa0                　"
    i, j = 0, len(s)
    while i < j and ws(s[i]):
        i += 1
    while j > i and ws(s[j - 1]):
        j -= 1
    return s[i:j]


def _key(token):  # a token's normalized text as UTF-8; b"" for the tokens that keep their own id
    if token.isascii():
        key = _WS_RUN.sub(b" ", token.lower())
        return key if key == b" " else key.strip(_ASCII_WHITESPACE)
    try:
        s = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", token.decode("utf-8")))
        s = _WS_RUN_STR.sub(" ", "".join(ch for ch in s if unicodedata.category(ch) != "Mn").lower())
        return (" " if s == " " else _rust_strip(s)).encode("utf-8")
    except UnicodeDecodeError:  # a partial UTF-8 byte token
        return b""


def build_norm_map():
    ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks  # token bytes -> id, ids 0..50255
    tokens = [b""] * len(ranks)
    for token, i in ranks.items():
        tokens[i] = token
    first = {}
    out = np.arange(NORM_MAP_SIZE, dtype=np.int32)
    out[:len(ranks)] = [first.setdefault(key, i) if key else i for i, key in enumerate(_key(t) for t in tokens)]
    assert hashlib.sha256(out[:len(ranks) + 1].astype("<i4").tobytes()).hexdigest() == NORM_MAP_SHA256
    return out


NORM_MAP = torch.from_numpy(build_norm_map())
