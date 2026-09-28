"""Data loading: FineWeb .bin shards -> BOS-aligned varlen batches, plus the n-gram table row ids."""
import glob
import os
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import numpy as np

import torch
import torch.distributed as dist
from torch import Tensor

from track_1_short.config import VIRTUAL_SEQ_CAP
from track_1_short.model.layers import next_multiple_of_n
from track_1_short.ngram_table import ngram_row_ids
from track_1_short.perf.pinned_batches import PinnedBatchStaging
from track_1_short.schedule import TrainingSchedule

HEADER_BYTES = 256 * 4
# Parallel pread threads for a shard read. The first training shard and every validation shard are read
# on the clock; one thread's readinto leaves the page cache / disk bandwidth underused (record #360).
SHARD_READ_THREADS = 4


def pread_parallel(fd: int, out: np.ndarray, offset: int, nbytes: int) -> int:
    """Read nbytes at `offset` of fd into `out` (a contiguous array of exactly nbytes), split
    SHARD_READ_THREADS ways over disjoint byte ranges. Returns the bytes read (short on EOF)."""
    view = memoryview(out).cast("B")
    assert len(view) == nbytes, f"buffer holds {len(view)} bytes, reading {nbytes}"

    def read_range(lo: int, hi: int) -> int:
        start = lo
        while lo < hi:
            n = os.preadv(fd, [view[lo:hi]], offset + lo)
            if n == 0:
                break  # EOF
            lo += n
        return lo - start

    edges = [(nbytes * i) // SHARD_READ_THREADS for i in range(SHARD_READ_THREADS + 1)]
    with ThreadPoolExecutor(SHARD_READ_THREADS) as pool:
        return sum(pool.map(read_range, edges, edges[1:]))


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        # straight into the array: avoids a bytes->array copy (@YouJiacheng)
        nbytes = pread_parallel(f.fileno(), tokens.numpy(), HEADER_BYTES, 2 * num_tokens)
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256


class Batch(NamedTuple):
    inputs: Tensor         # [T] int32, device
    targets: Tensor        # [T] int64, device
    cum_seqlens: Tensor    # [max_num_docs] int32, device
    ngram_ids: Tensor      # [2T] int32 n-gram table rows (ngram_row_ids), device: the forward's cache lookup
    ngram_ids_cpu: np.ndarray  # the same, host: the row-pull want lists are built from it
    targets_cpu: Tensor    # host copy: the sampled-softmax candidate build reads targets without a D2H
    inputs_cpu: np.ndarray  # host copy: the value_embeds row-pull want lists are built from it

# Rows of the packed cu_seqlens table per local batch size (tokens per rank); 40960 is the batch-20 taper.
TRAIN_MAX_NUM_DOCS = {16384: 64, 32768: 96, 40960: 128, 49152: 128}


def cu_seqlens_rows(tokens_per_rank: int) -> int:
    """Length of the packed cu_seqlens table for a batch of this many tokens per rank (a fixed shape,
    so the compiled model and its CUDA graphs see one shape per batch size)."""
    return TRAIN_MAX_NUM_DOCS.get(tokens_per_rank, next_multiple_of_n(tokens_per_rank // 300, n=128))

class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration("Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                idx += 1
                end = min(self.bos_idx[idx] if idx < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur

            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

def split_attention_segments(cum_lengths: Tensor, cap: int) -> Tensor:
    """Document ends (cumulative) -> attention-segment ends, with a boundary every `cap` tokens inside
    each document. Only attention sees the extra boundaries; the tokens are unchanged."""
    ends = [end for start, stop in pairwise([0, *cum_lengths.tolist()])
            for end in (*range(start + cap, stop, cap), stop)]
    return torch.tensor(ends, dtype=cum_lengths.dtype)

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, staging: PinnedBatchStaging,
                               align_to_bos: bool = True):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    # staging: the pinned slots every batch's H2D copies go through (perf/pinned_batches.py)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert num_tokens % world_size == 0, "Batch size must be divisible by world size"

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        shard = Shard(tokens, world_size)
        next_shard_getter = Shard.load_async(next(file_iter), world_size)
    else:
        pos = 0  # for unaligned case

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = cu_seqlens_rows(num_tokens_local)

        if align_to_bos:
            try:
                seq_starts, seq_ends = shard.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                # This shard is exhausted, load the next one in the next loop iteration.
                shard = next_shard_getter()
                tokens = shard.tokens
                try:
                    next_shard_getter = Shard.load_async(next(file_iter), world_size)
                except StopIteration:
                    next_shard_getter = None  # no more shards to preload
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
            if max_seq_len > VIRTUAL_SEQ_CAP:
                cum_lengths = split_attention_segments(cum_lengths, VIRTUAL_SEQ_CAP)
                assert len(cum_lengths) < max_num_docs, f"{len(cum_lengths)} attention segments overflow the {max_num_docs}-row cu_seqlens table"

        else:
            if pos + num_tokens + 1 >= len(tokens):  # should not occur for val data
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens


        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _ngram_ids = ngram_row_ids(_inputs)
        inputs, targets, cum_seqlens, ngram_ids = staging.upload(_inputs, _targets, _cum_lengths, _ngram_ids)

        new_params = yield Batch(
            inputs=inputs,
            targets=targets,
            cum_seqlens=cum_seqlens,
            ngram_ids=ngram_ids,
            ngram_ids_cpu=_ngram_ids.numpy(),
            targets_cpu=_targets,
            inputs_cpu=_inputs.numpy(),
        )

        if new_params is not None:
            # makes it possible for generator to receive new (num_tokens, max_seq_len) via .send()
            num_tokens, max_seq_len = new_params
            assert num_tokens % world_size == 0, "Num tokens must be divisible by world size"


class ScheduledBatches:
    """Training batches by step, over the sequence of steps a run trains (one microbatch per step).

    The batch geometry (tokens, max sequence length) of each step comes from the schedule and is sent to
    the loader when it changes. Batches are fetched in step order and held until taken, so peek() can
    look ahead: the sampled-softmax candidate build peeks one step ahead, the row prefetch up to
    MAX_CYCLE_STEPS + PREP_SLACK_STEPS (perf/row_prefetch.py). `steps` is range(total_steps) for the
    timed run and the sampled, non-consecutive steps for warmup.
    """
    def __init__(self, loader, schedule: TrainingSchedule, steps: Sequence[int]):
        self.loader = loader
        self.schedule = schedule
        self.steps = list(steps)
        stage, _ = schedule.lookup(0)
        self.geometry = (stage.batch_size, stage.train_max_seq_len)  # what the loader was created with
        self.next_index = 0  # position in `steps` of the next batch to fetch
        self.ahead: dict[int, Batch] = {}  # step -> batch, fetched and not yet taken, in step order

    def _fetch_next(self):
        step = self.steps[self.next_index]
        self.next_index += 1
        stage, _ = self.schedule.lookup(step)
        geometry = (stage.batch_size, stage.train_max_seq_len)
        new_params = geometry if geometry != self.geometry else None
        self.geometry = geometry
        self.ahead[step] = self.loader.send(new_params)

    def peek(self, step: int) -> Batch:
        while step not in self.ahead:
            assert self.next_index < len(self.steps) and self.steps[self.next_index] <= step, f"step {step} is not ahead"
            self._fetch_next()
        return self.ahead[step]

    def take(self, step: int) -> Batch:
        batch = self.peek(step)
        assert next(iter(self.ahead)) == step, f"step {step} taken out of order"
        del self.ahead[step]
        return batch
