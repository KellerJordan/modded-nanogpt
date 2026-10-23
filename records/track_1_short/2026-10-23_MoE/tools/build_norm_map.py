#!/usr/bin/env python3
"""Rebuild norm_map_pr375.npy from the GPT-2 vocabulary alone, byte for byte.

norm_map_pr375.npy is the token normalization map of PR #375 (token_norm.build_norm_map): it sends every GPT-2 id to
the smallest id of its normalization class, so " The", "THE" and " the" share one n-gram hash row. The record trainer
loads it (NG_NORM=1) instead of importing the `tokenizers` package, and checks the sha256 of its first 50,257 entries.
This script is the audit trail: it regenerates the file from tiktoken's GPT-2 ranks and compares it with the shipped one.

Normalization of a token's text (the PR's rules):
  NFKC, NFD, strip accents (drop combining marks), lowercase, collapse runs of [ \\t\\r\\n] to one space, strip outer
  whitespace. A token that is only whitespace becomes " " (a private-use sentinel keeps Strip from eating it).
  Tokens that keep their own id: <|endoftext|>, byte tokens that are not whole UTF-8, and the 9 tokens that normalize to
  "" (vertical-tab / form-feed controls, lone combining marks, the emoji variation selector U+FE0F).
  The map is padded from 50,257 to the model's 50,304 rows with the identity.

Two backends build the 873 non-ASCII keys (the 49,383 ASCII tokens go through an exact fast path in both):
  tokenizers    HuggingFace `normalizers` (what PR #375 used; needs `pip install tokenizers`)
  unicodedata   standard library only; same rules written out, with Rust's White_Space set for Strip
--backend auto uses `tokenizers` when importable, else `unicodedata`; --backend both builds with each and demands that
they agree. Only tiktoken (and numpy) are needed otherwise; tiktoken reads the GPT-2 vocabulary from its cache
(TIKTOKEN_CACHE_DIR) or downloads it.

    python build_norm_map.py [--out norm_map_rebuilt.npy] [--reference path/to/norm_map_pr375.npy] [--backend auto]

Exit status 0 only if every check passes.
"""
import argparse
import hashlib
import os
import re
import sys
import unicodedata

import numpy as np
import tiktoken

VOCAB = 50257                     # GPT-2 ids 0..50255 plus <|endoftext|> (50256)
NORM_MAP_SIZE = 50304             # the model's padded vocabulary
NUM_CLASSES = 32980               # distinct values among the first 50,257 entries (PR #375)
NUM_REMAPPED = 17277              # ids that map to a smaller id
# sha256 of the first 50,257 entries as little-endian int32 (the check the trainer itself makes at load time)
NORM_MAP_SHA256 = "79fb57ac259495ba36f1afb9b0d5da35f3048b5067fa59c101d5a3c1ed15c2a2"
# sha256 of the whole shipped file (npy header included)
NORM_MAP_FILE_SHA256 = "ab8ce1ae7b4e9d6df52d79b393d1ac9fae52fd2c4bdfac92849a2850c17d3152"

_SENTINEL = ""  # private use: a whitespace-only token becomes " ", then this, so Strip keeps it

# ASCII fast path: on ASCII text NFKC, NFD and StripAccents are the identity, Lowercase is bytes.lower(), and Strip
# removes exactly these (Rust's char::is_whitespace on ASCII).
_WS_RUN = re.compile(rb"[ \t\r\n]+")
_ASCII_WHITESPACE = b" \t\n\x0b\x0c\r"

# Rust's char::is_whitespace (Unicode White_Space): what the HF `Strip` normalizer removes. Not str.isspace(), which
# also counts U+001C..U+001F.
_RUST_WS = frozenset(
    list(range(0x09, 0x0E)) + [0x20, 0x85, 0xA0, 0x1680] + list(range(0x2000, 0x200B))
    + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000])
_TEXT_WS_RUN = re.compile(r"[ \t\r\n]+")


def _hf_normalizer():
    from tokenizers import Regex, normalizers
    return normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), _SENTINEL),
        normalizers.Strip(),
        normalizers.Replace(_SENTINEL, " "),
    ]).normalize_str


def _unicodedata_normalize(text: str) -> str:
    """The same eight steps with the standard library."""
    text = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", text))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")          # StripAccents
    text = "".join(c.lower() for c in text)                                   # Lowercase: per character, as in Rust
    text = _TEXT_WS_RUN.sub(" ", text)
    if text == " ":
        text = _SENTINEL
    lo, hi = 0, len(text)
    while lo < hi and ord(text[lo]) in _RUST_WS:
        lo += 1
    while hi > lo and ord(text[hi - 1]) in _RUST_WS:
        hi -= 1
    return text[lo:hi].replace(_SENTINEL, " ")


def _key(token: bytes, normalize) -> bytes:
    """A token's normalized text as UTF-8; b"" for the tokens that keep their own id."""
    if token.isascii():  # 49,383 of the 50,256 mergeable tokens
        key = _WS_RUN.sub(b" ", token.lower())
        return key if key == b" " else key.strip(_ASCII_WHITESPACE)
    try:
        return normalize(token.decode("utf-8")).encode("utf-8")
    except UnicodeDecodeError:  # a partial UTF-8 byte token
        return b""


def build_norm_map(normalize) -> np.ndarray:
    """[NORM_MAP_SIZE] int32: each id -> the smallest id with the same normalized text."""
    ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks  # token bytes -> id, ids 0..50255
    assert len(ranks) == VOCAB - 1, len(ranks)
    tokens = [b""] * len(ranks)
    for token, i in ranks.items():
        tokens[i] = token
    first: dict[bytes, int] = {}
    ids = [first.setdefault(key, i) if key else i for i, key in enumerate(_key(t, normalize) for t in tokens)]
    out = np.arange(NORM_MAP_SIZE, dtype=np.int32)  # <|endoftext|> and the padding rows keep their own id
    out[:len(ids)] = ids
    return out


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check_map(m: np.ndarray, label: str) -> bool:
    """The map's own invariants, independent of any file."""
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'ok' if passed else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")

    print(f"{label}:")
    report("dtype int32, shape (50304,)", m.dtype == np.int32 and m.shape == (NORM_MAP_SIZE,), f"{m.dtype}, {m.shape}")
    head = m[:VOCAB]
    ident = np.arange(NORM_MAP_SIZE, dtype=np.int32)
    report("tail 50257..50303 is the identity", np.array_equal(m[VOCAB:], ident[VOCAB:]),
           f"{NORM_MAP_SIZE - VOCAB} entries")
    report("<|endoftext|> (50256) keeps its id", m[50256] == 50256)
    report("every id maps to a smaller-or-equal id", bool((head <= ident[:VOCAB]).all()))
    report("map is idempotent (a class representative maps to itself)", bool((head[head] == head).all()))
    classes, remapped = len(np.unique(head)), int((head != ident[:VOCAB]).sum())
    report(f"{NUM_CLASSES} classes over {VOCAB} tokens", classes == NUM_CLASSES, f"got {classes}")
    report(f"{NUM_REMAPPED} ids remapped", remapped == NUM_REMAPPED, f"got {remapped}")
    digest = sha256_bytes(head.astype("<i4").tobytes())
    report("sha256 of the first 50,257 entries (trainer's load check)", digest == NORM_MAP_SHA256, digest)
    return ok


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "norm_map_pr375.npy"), os.path.join(here, "..", "norm_map_pr375.npy"),
                  os.path.join(os.getcwd(), "norm_map_pr375.npy")]
    default_ref = next((p for p in candidates if os.path.exists(p)), None)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="norm_map_rebuilt.npy", help="where to write the rebuilt map")
    ap.add_argument("--reference", default=default_ref, help="the shipped norm_map_pr375.npy to compare byte for byte")
    ap.add_argument("--backend", choices=("auto", "tokenizers", "unicodedata", "both"), default="auto")
    args = ap.parse_args()

    backends = {"tokenizers": None, "unicodedata": _unicodedata_normalize}
    try:
        backends["tokenizers"] = _hf_normalizer()
    except ImportError:
        backends.pop("tokenizers")
    if args.backend == "auto":
        names = ["tokenizers" if "tokenizers" in backends else "unicodedata"]
    elif args.backend == "both":
        names = list(backends)
    else:
        names = [args.backend]
    for n in names:
        if n not in backends:
            sys.exit(f"backend {n!r} unavailable (pip install tokenizers)")

    ok = True
    maps = {}
    for n in names:
        maps[n] = build_norm_map(backends[n])
        ok &= check_map(maps[n], f"map built with {n}")
    if len(maps) == 2:
        same = np.array_equal(maps["tokenizers"], maps["unicodedata"])
        print(f"tokenizers and unicodedata backends agree: {same}")
        ok &= same
    m = maps[names[0]]

    np.save(args.out, m)
    with open(args.out, "rb") as f:
        rebuilt = f.read()
    print(f"wrote {args.out}: {len(rebuilt)} bytes, sha256 {sha256_bytes(rebuilt)}")
    same = sha256_bytes(rebuilt) == NORM_MAP_FILE_SHA256
    print(f"  [{'ok' if same else 'FAIL'}] whole-file sha256 equals the one recorded in this script")
    ok &= same
    if args.reference:
        with open(args.reference, "rb") as f:
            shipped = f.read()
        same = rebuilt == shipped
        print(f"reference {args.reference}: {len(shipped)} bytes, sha256 {sha256_bytes(shipped)}")
        print(f"  [{'ok' if same else 'FAIL'}] rebuilt file is byte-identical to the reference "
              f"(sha256 of the whole file {'equal' if sha256_bytes(rebuilt) == sha256_bytes(shipped) else 'DIFFERENT'})")
        ok &= same
        ref_map = np.load(args.reference)
        ok &= check_map(ref_map, "reference map")
    else:
        print("no --reference given and none found next to this script: byte comparison skipped")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
