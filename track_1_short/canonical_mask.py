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
    """Builds the canonical mask concurrently with training, in a spawned child process.

    The build is a few seconds of pure-Python work, so running it in a thread would hold the GIL
    and throttle the training loop's kernel launches. A spawned child has its own interpreter and
    only computes, writing the result into shared memory. posix_spawn, not os.fork(): forking copies
    this process's page tables (~60 ms on the clock). Every rank builds and uploads its own copy.
    """

    UPLOAD_CHUNKS = 10  # H2D chunks, one per poll_upload() call

    def __init__(self, vocab_size: int, owner: bool, print0):
        self.vocab_size = vocab_size
        self.print0 = print0
        self.buf = self.ranks = self.pid = None
        self.pinned = False
        self.out = self.stream = self.uploaded = None
        self.next_chunk = 0
        self.path = None
        if owner:
            # /dev/shm file shared with the child
            nbytes = vocab_size * (vocab_size // 8)
            fd, self.path = tempfile.mkstemp(prefix="canon_mask_", dir="/dev/shm")
            try:
                os.ftruncate(fd, nbytes)
            finally:
                os.close(fd)
            self.buf = torch.from_file(self.path, shared=True, size=nbytes, dtype=torch.uint8).view(vocab_size, vocab_size // 8)
            self.ranks = tiktoken.get_encoding("gpt2")._mergeable_ranks  # inline fallback
            # Page-lock the buffer so that the H2D is a true async DMA. Registering is itself slow,
            # which is why it belongs here, before the clock.
            cudart = torch.cuda.cudart()
            err = cudart.cudaHostRegister(self.buf.data_ptr(), self.buf.nbytes, 0)
            self.pinned = err == cudart.cudaError.success
            if not self.pinned:
                print0(f"NOTE: could not page-lock the canonical mask buffer ({err}), "
                       "so its copy to device will block the host", console=True)
            self.stream = torch.cuda.Stream()
            self.uploaded = torch.cuda.Event()

    def start(self):
        if self.buf is None:
            return
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # the child must not touch CUDA
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
        """Between steps: once the child is done, queue the next H2D chunk into `out`. Never blocks."""
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
        self._reaped(reaped[1])

    def _unlink(self):
        if self.path is not None:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass
            self.path = None

    def collect(self, out: Tensor):
        """Finish queueing the upload and make the current stream wait for it. No host sync."""
        assert self.pid is None, "collect before wait"
        if self.buf is None:
            return
        while self.next_chunk < self.UPLOAD_CHUNKS:
            self._upload_chunk(out)
        torch.cuda.current_stream().wait_event(self.uploaded)

    def release(self, out: Tensor):
        """Off the clock: release the host buffer and check that every rank holds the same mask."""
        if self.buf is not None:
            self.uploaded.synchronize()
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
