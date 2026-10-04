"""GPT-2 normalization-class IDs for n-gram hashes only.

Call build_norm_map() inside the training stopwatch; importing builds no map.
Raw token streams, embeddings, targets and logits must retain their original IDs.
"""

import hashlib
import re
import unicodedata

import numpy as np
import tiktoken


NORM_MAP_SIZE = 50304
NORM_MAP_SHA256 = '79fb57ac259495ba36f1afb9b0d5da35f3048b5067fa59c101d5a3c1ed15c2a2'

_BYTE_WS_RUN = re.compile(rb'[ \t\r\n]+')
_TEXT_WS_RUN = re.compile(r'[ \t\r\n]+')
_ASCII_WHITESPACE = b' \t\n\v\f\r'
# Rust char::is_whitespace uses Unicode White_Space, unlike Python's extra U+001C..001F.
_UNICODE_WHITESPACE = (
    '\t\n\v\f\r \x85\xa0\u1680'
    '\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a'
    '\u2028\u2029\u202f\u205f\u3000'
)
_SENTINEL = '\ue000'


def _key(token: bytes) -> bytes:
    if token.isascii():
        key = _BYTE_WS_RUN.sub(b' ', token.lower())
        return key if key == b' ' else key.strip(_ASCII_WHITESPACE)
    try:
        text = token.decode('utf-8')
    except UnicodeDecodeError:
        return b''
    text = unicodedata.normalize('NFD', unicodedata.normalize('NFKC', text))
    # The reference lowercases each character, without contextual final-sigma handling.
    text = ''.join(c.lower() for c in text if unicodedata.category(c) != 'Mn')
    text = _TEXT_WS_RUN.sub(' ', text)
    if text == ' ':
        text = _SENTINEL
    return text.strip(_UNICODE_WHITESPACE).replace(_SENTINEL, ' ').encode('utf-8')


def build_norm_map() -> np.ndarray:
    """Build fresh int32 IDs; official GPT-2 tokenizer assets must already be cached."""
    ranks = tiktoken.get_encoding('gpt2')._mergeable_ranks
    tokens = [b''] * len(ranks)
    for token, token_id in ranks.items():
        tokens[token_id] = token
    out = np.arange(NORM_MAP_SIZE, dtype=np.int32)
    first: dict[bytes, int] = {}
    for token_id, token in enumerate(tokens):
        key = _key(token)
        if key:
            out[token_id] = first.setdefault(key, token_id)
    digest = hashlib.sha256(out[:50257].astype('<i4', copy=False).tobytes()).hexdigest()
    if digest != NORM_MAP_SHA256:
        raise AssertionError(f'token normalization map mismatch: {digest} (Unicode {unicodedata.unidata_version})')
    return out
