"""Canonical token masking for the final validation (record #350).

GPT-2's tokenizer never emits certain (prev, cur) token pairs. At the final validation those
tokens are masked out of the softmax, which can only lower the loss. The mask is built on the
clock in a background thread during training, by the exact_match extension's port of
build_canonical_mask below (bit-identical, ~0.1 s on a few cores instead of 9 s).
"""
import threading
import time
import traceback
import unicodedata

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch import Tensor

import exact_match

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

def _edges(vocab_size: int, tok: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pretokenizer class of each token start and end, and whether a token ends in a contraction
    pretoken -- nothing can extend one of those, so it may not mask anything."""
    end_cls = np.full(vocab_size, _CLS.index("?"), dtype=np.intp)
    start_cls = np.full(vocab_size, _CLS.index("?"), dtype=np.intp)
    closed = np.zeros(vocab_size, dtype=bool)
    for tid, b in tok.items():
        end_cls[tid], start_cls[tid] = _edge_cls(b, first=False), _edge_cls(b, first=True)
        try:
            closed[tid] = _ends_contraction(b.decode())
        except UnicodeDecodeError:
            pass  # ends mid-character, so it cannot end in a contraction
    return start_cls, end_cls, closed

def tokenizer_tables(vocab_size: int, ranks: dict) -> tuple[np.ndarray, ...]:
    """The inputs of exact_match.canonical_mask, from the tokenizer alone (~0.13 s, before the
    clock): the tokens' bytes concatenated in id order and their offsets, each token's start
    class, and the bit set of start classes a seam merge may follow each token with."""
    tok = {v: k for k, v in ranks.items()}
    assert sorted(tok) == list(range(len(tok))) and len(tok) <= vocab_size, "token ids must be 0, 1, ..."
    start_cls, end_cls, closed = _edges(vocab_size, tok)
    allowed = np.packbits(_SEAM_OK[end_cls], axis=1, bitorder="little")[:, 0]
    allowed[closed] = 0
    lens = np.array([len(tok[t]) for t in range(len(tok))])
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    data = np.frombuffer(b"".join(tok[t] for t in range(len(tok))), dtype=np.uint8)
    return data, offsets, start_cls.astype(np.uint8), allowed

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

    start_cls, end_cls, closed = _edges(vocab_size, tok)
    mask = np.zeros((vocab_size, vocab_size), dtype=bool)
    for rank, (a, b) in rules.items():
        ps = np.array([p for s, e, p in by_last.get(a, ()) if s <= rank < e], dtype=np.intp)
        ps = ps[~closed[ps]]
        xs = np.array([x for s, e, x in by_first.get(b, ()) if s <= rank < e], dtype=np.intp)
        if len(ps) and len(xs):
            mask[np.ix_(ps, xs)] |= _SEAM_OK[np.ix_(end_cls[ps], start_cls[xs])]

    return np.packbits(mask, axis=1, bitorder="little")

class BackgroundCanonicalMask:
    """Builds the canonical mask concurrently with training, in a background thread.

    The build is exact_match.canonical_mask, which releases the GIL, so the training loop keeps
    launching kernels; its threads run on the rank's cores other than the main thread's.
    """

    def __init__(self, vocab_size: int, owner: bool, print0):
        self.vocab_size = vocab_size
        self.print0 = print0
        self.buf = self.tables = self.thread = self.error = self.seconds = None
        self.pinned = False
        if owner:
            self.buf = torch.empty(vocab_size, vocab_size // 8, dtype=torch.uint8)
            self.tables = tokenizer_tables(vocab_size, tiktoken.get_encoding("gpt2")._mergeable_ranks)
            # Page-lock the buffer so that collect's H2D is a direct DMA rather than a staged
            # copy, roughly 10ms instead of 50. Registering is itself slow, which is why it
            # belongs here, before the clock.
            cudart = torch.cuda.cudart()
            err = cudart.cudaHostRegister(self.buf.data_ptr(), self.buf.nbytes, 0)
            self.pinned = err == cudart.cudaError.success
            if not self.pinned:
                print0(f"NOTE: could not page-lock the canonical mask buffer ({err}), "
                       "so its copy to device will be slower", console=True)
        # Every rank's: collect's copy and broadcast run on it, beside the last training steps.
        self.stream = torch.cuda.Stream()

    def start(self, cpus: set[int]):
        """Start the build on len(cpus) threads pinned to `cpus`."""
        if self.buf is None:
            return

        def build():
            t = time.perf_counter()
            try:
                exact_match.canonical_mask(self.buf.numpy(), *self.tables, len(cpus), sorted(cpus))
            except BaseException as e:
                traceback.print_exc()
                self.error = e
            self.seconds = time.perf_counter() - t

        self.thread = threading.Thread(target=build, name="canonical-mask", daemon=True)
        self.thread.start()

    def wait(self):
        """Block until the mask is ready. Called from the timed region."""
        if self.thread is None:
            return
        self.thread.join()
        self.thread = None
        if self.error is not None:
            raise RuntimeError("the canonical mask build failed") from self.error
        self.print0(f"canonical mask built in {1000 * self.seconds:.0f} ms")

    def collect(self, out: Tensor):
        """Fill `out` on every rank: rank 0's copy to the device and the broadcast, queued on a stream of their own,
        so that they run beside the device's last training steps and the host waits for neither (a blocking copy
        waited for the device to finish training, then unpinning took 5 ms). Later work on the current stream
        waits for them. The buffer stays pinned and alive until release()."""
        assert self.thread is None, "collect before wait"
        with torch.cuda.stream(self.stream):
            if self.buf is not None:
                out.copy_(self.buf, non_blocking=True)
            dist.broadcast(out, 0)
        torch.cuda.current_stream().wait_stream(self.stream)

    def release(self):
        """After the clock stopped (the copy is done): unpin and drop the buffer."""
        if self.buf is not None and self.pinned:
            torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        self.buf = self.tables = None
