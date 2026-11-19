"""Canonical token masking for the final validation (record #350).

GPT-2's tokenizer never emits certain (prev, cur) token pairs. At the final validation those
tokens are masked out of the softmax, which can only lower the loss. The mask is built in a
spawned child process during training so its cost stays on the clock without stalling it.
"""
import os
import signal
import sys
import tempfile
import time

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
from torch import Tensor

from track_1_short.canonical_mask_build import build_canonical_mask


class BackgroundCanonicalMask:
    """Builds the canonical mask concurrently with training, in a forked child process.

    The build is a few seconds of pure-Python work, so running it in a thread would hold the GIL
    and throttle the training loop's kernel launches. A forked child has its own interpreter and
    only computes, writing the result into shared memory.

    The child is a fresh interpreter started with posix_spawn (vfork + exec), not os.fork(): forking this
    process copies its page tables (GBs of mappings: pinned rings, CUDA), measured at ~60 ms of every rank's
    host time at the clock start, while a spawn returns in about a millisecond. The child imports numpy and
    tiktoken itself (~0.2-0.5 s more child time, all of it on the clock and overlapped with training) and
    writes the mask into a /dev/shm file that this process maps shared (the same pages, page-locked here).

    Every rank builds its own copy (the build is deterministic, so the bytes are identical) and
    uploads it to the device mid-run: poll_upload(), called between training steps, notices the
    finished child and queues the H2D in chunks on a side stream, one chunk per call. At the tail,
    collect() only makes the current stream wait on the upload's event: no blocking copy, no
    broadcast, no device sync. release() unregisters the pinned buffer after the clock stops.
    """

    # H2D chunks of the upload, one per poll_upload() call (i.e. per training step): ~32 MB each, so
    # one step's copy never hogs the copy engine the batch and row uploads also use.
    UPLOAD_CHUNKS = 10

    def __init__(self, vocab_size: int, owner: bool, print0):
        self.vocab_size = vocab_size
        self.print0 = print0
        self.buf = self.ranks = self.pid = None
        self.pinned = False
        self.out = self.stream = self.uploaded = None
        self.next_chunk = 0
        self.path = None
        if owner:
            # A named tmpfs file, mapped MAP_SHARED here and by the spawned child: both see the same pages.
            nbytes = vocab_size * (vocab_size // 8)
            fd, self.path = tempfile.mkstemp(prefix="canon_mask_", dir="/dev/shm")
            try:
                os.ftruncate(fd, nbytes)
            finally:
                os.close(fd)
            self.buf = torch.from_file(self.path, shared=True, size=nbytes, dtype=torch.uint8).view(vocab_size, vocab_size // 8)
            self.ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks  # inline fallback only; also warms tiktoken's cache file
            # Page-lock the buffer so that the H2D is a true async DMA (from pageable memory it would be
            # staged and block the host). Registering is itself slow, which is why it belongs here,
            # before the clock. The mapping is MAP_SHARED, so the child still writes to these same
            # pages and the fork stays safe.
            cudart = torch.cuda.cudart()
            err = cudart.cudaHostRegister(self.buf.data_ptr(), self.buf.nbytes, 0)
            self.pinned = err == cudart.cudaError.success
            if not self.pinned:
                print0(f"NOTE: could not page-lock the canonical mask buffer ({err}), "
                       "so its copy to device will block the host", console=True)
            self.stream = torch.cuda.Stream()
            self.uploaded = torch.cuda.Event()

    def start(self):
        """On the clock: spawn the builder (python -m track_1_short.canonical_mask_build <path> <vocab_size>)."""
        if self.buf is None:
            return
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # The child must not touch CUDA or NCCL; nothing it imports does, but hide the devices anyway.
        env["CUDA_VISIBLE_DEVICES"] = ""
        self.pid = os.posix_spawn(sys.executable, [sys.executable, "-m", "track_1_short.canonical_mask_build",
                                                   self.path, str(self.vocab_size)], env)

    def _reaped(self, status: int):
        self.pid = None
        self._unlink()
        if status:
            self.print0(f"WARNING: background canonical mask build failed ({status}), building it inline", console=True)
            np.copyto(self.buf.numpy(), build_canonical_mask(self.vocab_size, self.ranks))

    def poll_upload(self, out: Tensor):
        """Between training steps (eager, never under capture): once the child is done, queue the next
        chunk of the H2D into `out` on the side stream. Never blocks: a WNOHANG reap and async copies."""
        if self.buf is None or self.next_chunk >= self.UPLOAD_CHUNKS:
            return
        if self.pid is not None:
            reaped = os.waitpid(self.pid, os.WNOHANG)
            if not reaped[0]:
                return
            self._reaped(reaped[1])
        self._upload_chunk(out)

    def _upload_chunk(self, out: Tensor):
        if self.next_chunk == 0:
            self.out = out
            # Order the writes after whatever the device has queued so far (nothing reads `out`
            # until the final validation, but this costs no host time).
            self.stream.wait_stream(torch.cuda.current_stream())
        assert out is self.out
        rows = self.vocab_size
        lo = rows * self.next_chunk // self.UPLOAD_CHUNKS
        hi = rows * (self.next_chunk + 1) // self.UPLOAD_CHUNKS
        with torch.cuda.stream(self.stream):
            out[lo:hi].copy_(self.buf[lo:hi], non_blocking=True)
        self.next_chunk += 1
        if self.next_chunk == self.UPLOAD_CHUNKS:
            self.uploaded.record(self.stream)

    def wait(self, timeout=60.0):
        """Block until the mask is ready. Called from the timed region; a no-op once poll_upload reaped it."""
        if self.pid is None:
            return
        deadline = time.perf_counter() + timeout
        while not (reaped := os.waitpid(self.pid, os.WNOHANG))[0]:
            if time.perf_counter() > deadline:
                os.kill(self.pid, signal.SIGKILL)
                reaped = os.waitpid(self.pid, 0)
                break
            time.sleep(0.05)
        self._reaped(reaped[1])

    def _unlink(self):
        """Drop the /dev/shm name once the child is reaped (the mapping keeps the pages until release)."""
        if self.path is not None:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self.path = None

    def collect(self, out: Tensor):
        """On the clock: finish queueing the upload if poll_upload has not, then make the current stream
        wait for it. No host sync; every rank holds its own identical copy, so there is no broadcast."""
        assert self.pid is None, "collect before wait"
        if self.buf is None:
            return
        while self.next_chunk < self.UPLOAD_CHUNKS:
            self._upload_chunk(out)
        torch.cuda.current_stream().wait_event(self.uploaded)

    def release(self, out: Tensor):
        """After the clock stops: unpin and drop the host buffer, and check that every rank holds the
        same mask (row and column popcount-weighted sums; a mismatch is fatal)."""
        if self.buf is not None:
            self.uploaded.synchronize()  # the H2D reads the buffer until here
            if self.pinned:
                torch.cuda.cudart().cudaHostUnregister(self.buf.data_ptr())
        self._unlink()
        self.buf = self.ranks = None
        sums = torch.cat([out.sum(1, dtype=torch.int64), out.sum(0, dtype=torch.int64)])
        lo, hi = sums.clone(), sums.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        assert torch.equal(lo, hi), "canonical mask differs across ranks"
        assert int(lo.sum()) > 0, "canonical mask is empty"
