"""Canonical token masking for the final validation (record #350).

GPT-2's tokenizer never emits certain (prev, cur) token pairs. At the final validation those
tokens are masked out of the softmax, which can only lower the loss. The mask is built in a
forked child process during training so its cost stays on the clock without stalling it.
"""
import os
import signal
import time
import traceback
import unicodedata
import warnings

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch import Tensor

# Whether a merge may span the seam between a prev-token end and a cur-token start, by the
# pretokenizer class of the character on either side of it.
_CLS = "SNLO?"  # whitespace, number, letter, other, inside a character
_SEAM_OK = np.ones((len(_CLS), len(_CLS)), dtype=bool)
_SEAM_OK[:, _CLS.index("?")] = False  # cur's pretoken keeps going past cur's end
_SEAM_OK[_CLS.index("S"), _CLS.index("S")] = False  # a whitespace run is one pretoken
_SEAM_OK[_CLS.index("O"), _CLS.index("L")] = False  # contractions, see build_canonical_mask

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

def _char_cls(ch: str) -> str:
    if ch.isspace() and not "\x1c" <= ch <= "\x1f":
        return "S"
    cat = unicodedata.category(ch)[0]
    return cat if cat in "LN" else "O"

def _edge_cls(b: bytes, first: bool) -> int:
    for n in range(1, 5):
        try:
            s = (b[:n] if first else b[-n:]).decode()
        except UnicodeDecodeError:
            continue
        return _CLS.index(_char_cls(s[0] if first else s[-1]))
    return _CLS.index("?")

def _ends_contraction(text: str) -> bool:
    for c in _CONTRACTIONS:
        if text.endswith(c):
            before = text[:-len(c)]
            return not before or _char_cls(before[-1]) in "LN"
    return False

def build_canonical_mask(vocab_size: int, ranks: dict | None = None) -> np.ndarray:
    """Bit-packed (vocab_size, vocab_size // 8) mask of non-canonical (prev, cur) pairs.

    Bit x of row p is set when the GPT-2 tokenizer would never emit token x directly after
    token p, i.e. encode(decode([..., p, x])) != [..., p, x], so softmax can drop it.

    Since GPT-2 is not pure BPE we need to consider the pretokenization rules.

    Set bits have to hold for the pair *in context*, which is stricter than proving
    encode(decode([p, x])) != [p, x]: the mask is applied mid-document, so text on either
    side of the pair gets a vote. Pairs whose answer depends on it are left unset.

    * Left. The seven contraction rules ('s, 't, ...) are dropped and tokens ending in a
      contraction pretoken mask nothing at all, because whether a "'" opens a pretoken
      depends on what precedes the previous token.
    * Right. x's first-piece trajectory ends in an unbounded interval, which assumes the
      pretoken stops at x. When x ends mid-character it demonstrably does not, and a merge
      inside the continuation can preempt the seam merge at a lower rank -- so the pair
      survives re-encoding after all.
    """
    ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks if ranks is None else ranks
    tok = {v: k for k, v in ranks.items()}
    never = 1 << 30

    # Trajectory of each token's first and last BPE piece as (start_rank, piece) intervals,
    # plus the merge rule that finally forms the token. Keyed by token id, which is also the
    # rank of that final merge -- tiktoken numbers a merged token by its own rank.
    firsts, lasts, rules = {}, {}, {}
    for tid, b in tok.items():
        pieces = [bytes([c]) for c in b]
        first_traj, last_traj = [(0, pieces[0])], [(0, pieces[-1])]
        while len(pieces) > 1:
            best = best_i = None
            for i in range(len(pieces) - 1):
                r = ranks.get(pieces[i] + pieces[i + 1])
                if r is not None and (best is None or r < best):
                    best, best_i = r, i
            if best is None:
                break
            rules[tid] = (pieces[best_i], pieces[best_i + 1])
            pieces[best_i:best_i + 2] = [pieces[best_i] + pieces[best_i + 1]]
            if best_i == 0:
                first_traj.append((best + 1, pieces[0]))
            if best_i == len(pieces) - 1:
                last_traj.append((best, pieces[-1]))
        firsts[tid], lasts[tid] = first_traj, last_traj

    def by_piece(trajs):
        idx = {}
        for tid, traj in trajs.items():
            for i, (start, piece) in enumerate(traj):
                end = traj[i + 1][0] if i + 1 < len(traj) else never
                if start < end:
                    idx.setdefault(piece, []).append((start, end, tid))
        return idx

    by_last, by_first = by_piece(lasts), by_piece(firsts)

    # Pretokenizer class of each token end, and whether a token ends in a contraction
    # pretoken -- nothing can extend one of those, so it may not mask anything.
    end_cls = np.full(vocab_size, _CLS.index("?"), dtype=np.intp)
    start_cls = np.full(vocab_size, _CLS.index("?"), dtype=np.intp)
    closed = np.zeros(vocab_size, dtype=bool)
    for tid, b in tok.items():
        end_cls[tid], start_cls[tid] = _edge_cls(b, first=False), _edge_cls(b, first=True)
        try:
            closed[tid] = _ends_contraction(b.decode())
        except UnicodeDecodeError:
            pass  # ends mid-character, so it cannot end in a contraction

    mask = np.zeros((vocab_size, vocab_size), dtype=bool)
    for rank, (a, b) in rules.items():
        ps = np.array([p for s, e, p in by_last.get(a, ()) if s <= rank < e], dtype=np.intp)
        ps = ps[~closed[ps]]
        xs = np.array([x for s, e, x in by_first.get(b, ()) if s <= rank < e], dtype=np.intp)
        if len(ps) and len(xs):
            mask[np.ix_(ps, xs)] |= _SEAM_OK[np.ix_(end_cls[ps], start_cls[xs])]

    return np.packbits(mask, axis=1, bitorder="little")

class BackgroundCanonicalMask:
    """Builds the canonical mask concurrently with training, in a forked child process.

    The build is a few seconds of pure-Python work, so running it in a thread would hold the GIL
    and throttle the training loop's kernel launches. A forked child has its own interpreter and
    only computes, writing the result into shared memory.
    """

    def __init__(self, vocab_size: int, owner: bool, print0):
        self.vocab_size = vocab_size
        self.print0 = print0
        self.buf = self.ranks = self.pid = None
        self.pinned = False
        if owner:
            self.buf = torch.empty(vocab_size, vocab_size // 8, dtype=torch.uint8).share_memory_()
            self.ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks  # cached, but takes a lock
            # Page-lock the buffer so that collect's H2D is a direct DMA rather than a staged
            # copy, roughly 10ms instead of 50. Registering is itself slow, which is why it
            # belongs here, before the clock. The mapping is MAP_SHARED, so the child still
            # writes to these same pages and the fork stays safe.
            cudart = torch.cuda.cudart()
            err = cudart.cudaHostRegister(self.buf.data_ptr(), self.buf.nbytes, 0)
            self.pinned = err == cudart.cudaError.success
            if not self.pinned:
                print0(f"NOTE: could not page-lock the canonical mask buffer ({err}), "
                       "so its copy to device will be slower", console=True)

    def start(self):
        if self.buf is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # fork out of a threaded process
            self.pid = os.fork()
        if self.pid == 0:
            code = 1
            try:
                np.copyto(self.buf.numpy(), build_canonical_mask(self.vocab_size, self.ranks))
                code = 0
            except BaseException:
                traceback.print_exc()
            os._exit(code)  # skips atexit/CUDA/NCCL teardown, which is not ours to run

    def wait(self, timeout=60.0):
        """Block until the mask is ready. Called from the timed region."""
        if self.pid is None:
            return
        deadline = time.perf_counter() + timeout
        while not (reaped := os.waitpid(self.pid, os.WNOHANG))[0]:
            if time.perf_counter() > deadline:
                os.kill(self.pid, signal.SIGKILL)
                reaped = os.waitpid(self.pid, 0)
                break
            time.sleep(0.05)
        self.pid = None
        if reaped[1]:
            self.print0(f"WARNING: background canonical mask build failed ({reaped[1]}), building it inline", console=True)
            np.copyto(self.buf.numpy(), build_canonical_mask(self.vocab_size, self.ranks))

    def collect(self, out: Tensor):
        """Fill `out` on every rank, then release the shared buffer."""
        assert self.pid is None, "collect before wait"
        if self.buf is not None:
            out.copy_(self.buf)  # kept blocking: the source is released just below
            if self.pinned:
                torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        dist.broadcast(out, 0)
        self.buf = self.ranks = None
