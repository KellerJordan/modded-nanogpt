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


def _load_data_shard(file: Path, prefix_tokens: int | None = None, pin_memory: bool = False):
    """The shard's tokens; with `prefix_tokens`, only its first prefix_tokens tokens (the shard must hold
    more than that, so a caller reading the prefix sees exactly the tokens a whole-shard reader would).
    Pageable by default: see _read_shard."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    if prefix_tokens is not None:
        assert num_tokens > prefix_tokens, f"shard {file} holds {num_tokens} tokens, the prefix needs more than {prefix_tokens}"
        num_tokens = prefix_tokens
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=pin_memory)
        # straight into the array: avoids a bytes->array copy (@YouJiacheng)
        nbytes = pread_parallel(f.fileno(), tokens.numpy(), HEADER_BYTES, 2 * num_tokens)
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


# The first training shard is read on the clock, before step 0's batch. Only this many leading tokens are
# read before the first batch; the rest streams in on a background thread (ShardRead). Shard's partial
# BOS index scans the first 6M tokens, and a batch read from the partial index ends at most one batch
# (<= 49153 tokens per rank) past its last entry, so 8M covers every read until the full index is in.
FIRST_SHARD_SYNC_TOKENS = 8_000_000
PARTIAL_INDEX_TOKENS = 6_000_000


class ShardRead:
    """A shard's tokens, possibly still being read: tokens[:readable] are final; wait() for the rest."""
    def __init__(self, tokens: Tensor, readable: int, thread: threading.Thread | None = None):
        self.tokens = tokens
        self.readable = readable
        self.thread = thread
        self.error: BaseException | None = None

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None
            if self.error is not None:
                raise RuntimeError("background read of the training shard failed") from self.error
            self.readable = self.tokens.numel()
        return self.tokens


def _read_shard(file: Path, sync_tokens: int | None = None) -> ShardRead:
    """Read a shard. With sync_tokens, only its first sync_tokens tokens are read before returning; a
    background thread reads the rest (ShardRead.wait). Same bytes in the same places either way.

    Pageable memory: the batches' H2D copies go through PinnedBatchStaging's own pinned slots, and a shard
    is only ever sliced and cast on the host, so pinning a 200 MB shard (an on-clock cudaHostAlloc for the
    first one) bought nothing."""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    tokens = torch.empty(num_tokens, dtype=torch.uint16)
    arr = tokens.numpy()
    f = file.open("rb", buffering=0)
    head = num_tokens if sync_tokens is None else min(sync_tokens, num_tokens)
    try:
        # straight into the array: avoids a bytes->array copy (@YouJiacheng)
        nbytes = pread_parallel(f.fileno(), arr[:head], HEADER_BYTES, 2 * head)
        assert nbytes == 2 * head, "number of tokens read does not match header"
    except BaseException:
        f.close()
        raise
    if head == num_tokens:
        f.close()
        return ShardRead(tokens, num_tokens)

    def read_rest():
        try:
            n = pread_parallel(f.fileno(), arr[head:], HEADER_BYTES + 2 * head, 2 * (num_tokens - head))
            assert n == 2 * (num_tokens - head), "number of tokens read does not match header"
        except BaseException as e:
            read.error = e
        finally:
            f.close()

    read = ShardRead(tokens, head, threading.Thread(target=read_rest))
    read.thread.start()
    return read


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
    def __init__(self, read: ShardRead, world_size: int = 1):
        self.read = read
        self.tokens = tokens = read.tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0

        # Partial index now, full index async (after the rest of the shard is read, if it is still streaming)
        assert read.readable >= min(PARTIAL_INDEX_TOKENS, self.size), "partial BOS index past the bytes read"
        # numpy's flatnonzero: the same int64 positions as torch's nonzero, ~3x faster on the host
        self.bos_idx = np.flatnonzero(tokens.numpy()[:PARTIAL_INDEX_TOKENS] == BOS_ID).astype(np.int64, copy=False)
        self._full_idx = None
        self._scan_error: BaseException | None = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        try:
            self.read.wait()
            self._full_idx = np.flatnonzero(self.tokens.numpy() == BOS_ID).astype(np.int64, copy=False)
        except BaseException as e:
            self._scan_error = e
        self._ready.set()

    def _wait_full(self):
        """Block until the full BOS index (and so every byte of the shard) is in, and switch to it."""
        self._ready.wait()
        if self._scan_error is not None:
            raise RuntimeError("full BOS index of the training shard failed") from self._scan_error
        self._maybe_switch()

    def readable_end(self) -> int:
        """tokens[:readable_end()] are final (the whole shard once the full index is in)."""
        return self.size if self.bos_idx is self._full_idx else self.read.readable

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
                if idx + 1 >= n and self.bos_idx is not self._full_idx:
                    # The partial index cannot say where this document ends (or has no document left): wait for
                    # the full one rather than cut the document at the end of the shard or skip to the next shard,
                    # so the batches are the full index's whatever the scan's timing.
                    self._wait_full()
                    n = len(self.bos_idx)
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
    def load_async(file: Path, world_size: int = 1, sync_tokens: int | None = None):
        """Returns getter function for async shard loading (sync_tokens: see _read_shard)"""
        result = {}
        ready = threading.Event()
        def load():
            try:
                result['shard'] = Shard(_read_shard(file, sync_tokens=sync_tokens), world_size)
            except BaseException as e:
                result['error'] = e
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            if 'error' in result:
                raise RuntimeError(f"loading {file} failed") from result['error']
            return result['shard']
        return get


class FirstShardStart:
    """The timed run's first training shard, started at the clock start rather than at the first batch.

    Created before the clock (it only lists the files); start() right after the clock starts begins reading
    the shard's head and building its partial BOS index on a background thread, under the clock-start work
    that precedes step 0 (the canonical-mask fork, the prefix table). The loader takes it at its first batch."""
    def __init__(self, filename_pattern: str):
        self.files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
        self._get = None

    def start(self):
        if self._get is None:
            self._get = Shard.load_async(self.files[0], dist.get_world_size(), sync_tokens=FIRST_SHARD_SYNC_TOKENS)

    def get(self) -> "Shard":
        self.start()
        return self._get()

def split_attention_segments(cum_lengths: Tensor, cap: int) -> Tensor:
    """Document ends (cumulative) -> attention-segment ends, with a boundary every `cap` tokens inside
    each document. Only attention sees the extra boundaries; the tokens are unchanged."""
    ends = [end for start, stop in pairwise([0, *cum_lengths.tolist()])
            for end in (*range(start + cap, stop, cap), stop)]
    return torch.tensor(ends, dtype=cum_lengths.dtype)

def _unaligned_slice(tokens: Tensor, pos: int, rank: int, num_tokens_local: int):
    """The unaligned (validation) loader's slice for one rank: inputs, targets and the BOS positions."""
    pos_local = pos + rank * num_tokens_local
    buf = tokens[pos_local: pos_local + num_tokens_local + 1]
    _inputs = buf[:-1].view(num_tokens_local, )
    _targets = buf[1:].view(num_tokens_local, )

    cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
    return _inputs, _targets, cum_lengths

def _host_batch(_inputs: Tensor, _targets: Tensor, cum_lengths: Tensor, num_tokens_local: int, max_num_docs: int):
    """A batch's host tensors as uploaded: int32 inputs, int64 targets, the packed int32 cu_seqlens table
    and the int32 n-gram row ids."""
    _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
    _cum_lengths[0] = 0
    _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

    # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
    _inputs = _inputs.to(dtype=torch.int32)
    _targets = _targets.to(dtype=torch.int64)
    _cum_lengths = _cum_lengths.to(dtype=torch.int32)
    _ngram_ids = ngram_row_ids(_inputs)
    return _inputs, _targets, _cum_lengths, _ngram_ids

class ValPrefetch:
    """The final validation's batches, prepared on the host by a background thread during training.

    Exactly the batches distributed_data_generator(val_files, val_batch_size, -1, ..., align_to_bos=False)
    yields for the first val_tokens tokens: the same shard, slices, casts, cu_seqlens and n-gram ids (the
    same helpers build them). Only the needed prefix of the shard is read (val_tokens + 1 tokens), into
    pageable memory; each batch's tensors are copied into pinned buffers allocated in the constructor (off
    the clock, like the loaders' pinned slots), so at the tail batches() is just one async H2D per tensor.
    start() is called inside the timed region, so the read and prep stay on the clock, overlapped with
    training instead of serialized after the final device sync.
    """

    def __init__(self, filename_pattern: str, num_tokens: int, val_tokens: int, device: torch.device):
        self.rank, world_size = dist.get_rank(), dist.get_world_size()
        assert num_tokens % world_size == 0, "Batch size must be divisible by world size"
        assert val_tokens % num_tokens == 0
        files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        if not files:
            raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")
        self.file, self.device = files[0], device
        self.num_tokens, self.val_tokens = num_tokens, val_tokens
        self.num_tokens_local = num_tokens // world_size
        self.max_num_docs = cu_seqlens_rows(self.num_tokens_local)
        pinned = lambda n, dtype: torch.empty(n, dtype=dtype, pin_memory=True)
        ntl = self.num_tokens_local
        self.host = [(pinned(ntl, torch.int32), pinned(ntl, torch.int64), pinned(self.max_num_docs, torch.int32),
                      pinned(2 * ntl, torch.int32)) for _ in range(val_tokens // num_tokens)]
        self.thread = None
        self.error = None

    def start(self):
        assert self.thread is None
        self.thread = threading.Thread(target=self._prepare, daemon=True)
        self.thread.start()

    def _prepare(self):
        try:
            # The generator would switch shards when pos + num_tokens + 1 >= len(tokens); the prefix
            # read asserts the shard is longer than val_tokens + 1, so that never happens, as before.
            tokens = _load_data_shard(self.file, prefix_tokens=self.val_tokens + 1, pin_memory=False)
            for b, host in enumerate(self.host):
                _inputs, _targets, cum_lengths = _unaligned_slice(tokens, b * self.num_tokens, self.rank, self.num_tokens_local)
                for buf, t in zip(host, _host_batch(_inputs, _targets, cum_lengths, self.num_tokens_local, self.max_num_docs)):
                    assert buf.shape == t.shape and buf.dtype == t.dtype, f"{t.dtype}{tuple(t.shape)} != {buf.dtype}{tuple(buf.shape)}"
                    buf.copy_(t)
        except BaseException as e:
            self.error = e

    def batches(self) -> list[Batch]:
        """Wait for the host prep (normally long done) and queue the batches' H2D copies on the current
        stream. Starts the prep inline if start() was never called."""
        if self.thread is None:
            self.start()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("validation prefetch failed") from self.error
        out = []
        for _inputs, _targets, _cum_lengths, _ngram_ids in self.host:
            out.append(Batch(
                inputs=_inputs.to(self.device, non_blocking=True),
                targets=_targets.to(self.device, non_blocking=True),
                cum_seqlens=_cum_lengths.to(self.device, non_blocking=True),
                ngram_ids=_ngram_ids.to(self.device, non_blocking=True),
                ngram_ids_cpu=_ngram_ids.numpy(),
                targets_cpu=_targets,
                inputs_cpu=_inputs.numpy(),
            ))
        return out

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, staging: PinnedBatchStaging,
                               align_to_bos: bool = True, first_shard: FirstShardStart | None = None):
    # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
    # staging: the pinned slots every batch's H2D copies go through (perf/pinned_batches.py)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert num_tokens % world_size == 0, "Batch size must be divisible by world size"

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)  # Use itertools.cycle(files) for multi-epoch training
    if align_to_bos:
        # On the clock: only the leading FIRST_SHARD_SYNC_TOKENS are read before the first batch (started at the
        # clock start if the caller passed first_shard).
        first_file = next(file_iter)
        if first_shard is not None:
            assert first_shard.files[0] == first_file, "first_shard is not this loader's first file"
            shard = first_shard.get()
        else:
            shard = Shard(_read_shard(first_file, sync_tokens=FIRST_SHARD_SYNC_TOKENS), world_size)
        tokens = shard.tokens
        next_shard_getter = Shard.load_async(next(file_iter), world_size)
    else:
        tokens = _load_data_shard(next(file_iter))
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

            assert int(end_idxs[-1]) <= shard.readable_end(), "batch reads training-shard bytes not yet read"
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

            _inputs, _targets, cum_lengths = _unaligned_slice(tokens, pos, rank, num_tokens_local)
            pos += num_tokens

        _inputs, _targets, _cum_lengths, _ngram_ids = _host_batch(_inputs, _targets, cum_lengths, num_tokens_local, max_num_docs)
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
