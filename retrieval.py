"""Next-token embeddings, the prefetching online cache, and the sharded validation cache."""
import os
import glob
import numpy as np
import torch
import torch.distributed as dist
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("RAYON_NUM_THREADS", "4")
from exact_match import Cache

def merge_top2(keys, group=None):
    """Rank shard-local candidates by length/count/token, excluding duplicate tokens.

    Counts are shard-local, not exact global frequencies.
    """
    length = keys.max(1).values >> 40
    if group is not None:
        dist.all_reduce(length, op=dist.ReduceOp.MAX, group=group)
    keys = torch.where((keys >> 40) == length[:, None], keys, 0)
    ids = torch.full((len(keys), 4), -1, dtype=torch.int32)
    for i in range(2):
        best = keys.max(1).values
        if group is not None:
            dist.all_reduce(best, op=dist.ReduceOp.MAX, group=group)
        ids[:, i] = torch.where(best > 0, 0xFFFF - (best & 0xFFFF), -1)
        keys = torch.where((keys & 0xFFFF) == (best[:, None] & 0xFFFF), 0, keys)
    return length.numpy(), ids.numpy()

def retrieval_vector(model, length, candidates, device):
    to_device = lambda x: torch.as_tensor(x, device=device, dtype=torch.int64)
    length, ids = to_device(length), to_device(candidates)
    bucket = torch.bucketize(length, to_device([1, 24, 32, 48, 64, 128, 256]), right=True)
    valid = ids >= 0
    gathered = model.embed.weight[ids.clamp_min(0)].float()
    mean = (gathered * valid[..., None]).sum(1, dtype=torch.float32) / valid.sum(1, keepdim=True).clamp_min(1)
    values = mean * model.ret_next_scale[bucket, None] + model.ret_bucket_embed[bucket]
    return (values * (length > 0)[:, None]).bfloat16()

class OnlineCache:
    """Drives the training loader ahead of the training loop on a worker thread.

    For each step it queries every microbatch, stages the matches on the device with a
    non-blocking copy, then inserts the step's tokens. The queue holds `depth` steps, to
    overlap CPU lookup and device copies with training.
    Each queued microbatch holds about 2 MB on the device, so a deep queue is cheap.
    """
    def __init__(self, min_context, capacity, depth=16):
        self.cache = Cache(min_context, 512)
        self.cache.reserve(capacity)
        self.depth = depth
        self.thread = None
        self.error = None

    def prefetch(self, loader, loader_args, steps, microbatches, device):
        queue = Queue(self.depth * microbatches)

        def run():
            try:
                if torch.device(device).type == "cuda":
                    torch.cuda.set_device(device)
                prev = loader_args(0)
                for step in range(steps):
                    args, groups = loader_args(step), []
                    for idx in range(microbatches):
                        *batch, (inputs, bounds, group) = loader.send(args if idx == 0 and args != prev else None)
                        matches = tuple((t.pin_memory() if torch.cuda.is_available() else t).to(device, non_blocking=True)
                                        for t in map(torch.from_numpy, self.cache.query(inputs, bounds)))
                        queue.put((*batch, matches))
                        groups.append(group)
                    prev = args
                    if step == steps - 1:
                        break

                    for tokens, starts, ends in groups:
                        for ss, ee in zip(starts, ends):
                            buf = np.concatenate([tokens[s:e] for s, e in zip(ss, ee)]).astype(np.int32)
                            bounds = np.minimum(np.cumsum([0] + [e-s for s, e in zip(ss, ee)]), len(buf)-1).astype(np.int32)
                            self.cache.insert(buf[:-1], buf[1:], bounds)
            except BaseException as exc:
                self.error = exc
                queue.put(exc)

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()
        for _ in range(steps * microbatches):
            item = queue.get()
            if isinstance(item, BaseException):
                raise item
            yield item

    def finish(self):
        if self.thread is not None:
            self.thread.join()
        if self.error is not None:
            raise self.error

class ValidationCache:
    """Each rank indexes a slice of the training shards, queries the validation inputs against
    it, and the results are merged across ranks by (match length, mode count, smaller token)."""
    def __init__(self, train_pattern, val_pattern, min_context, local_tokens, total_tokens,
                 group=None, expected_shards=103):
        self.group = group
        self.local_tokens, self.total_tokens = local_tokens, total_tokens
        if total_tokens <= 0 or local_tokens <= 0 or total_tokens % local_tokens:
            raise ValueError("validation tokens must be a positive multiple of local_tokens")
        files = sorted(glob.glob(train_pattern))
        if len(files) != expected_shards:
            raise ValueError(f"expected {expected_shards} training shards, found {len(files)}")

        tokens_per_shard = 101_000_000
        self.training_token_capacity = len(files) * tokens_per_shard
        rank, world = (dist.get_rank(group), dist.get_world_size(group)) if group is not None else (0, 1)
        self.length = np.zeros(total_tokens, np.int32)
        self.next = np.full((total_tokens, 4), -1, np.int32)
        self.counts = np.zeros((total_tokens, 4), np.int32)
        self.status = torch.zeros(1, dtype=torch.int32)
        self.future = None
        self.ready = False
        val_files = sorted(glob.glob(val_pattern))
        if not val_files:
            raise ValueError("no validation shard")
        self.val_file = open(val_files[0], "rb")
        header = np.fromfile(self.val_file, dtype="<i4", count=256)
        if len(header) != 256 or tuple(header[:2]) != (20240520, 1) or header[2] < total_tokens + 1:
            self.val_file.close()
            raise ValueError("invalid or undersized validation shard")
        self.inputs = np.empty(total_tokens, dtype="<u2")
        try:
            local_files = files[rank::world]
            self.cache = Cache.allocate_offline(
                local_files, min_context, 512,
                token_capacity=len(local_files) * tokens_per_shard)
        except Exception:
            self.val_file.close()
            raise
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.worker.submit(lambda: None).result()

    def start(self):
        if self.future is not None:
            raise RuntimeError("validation cache already started")
        self.future = self.worker.submit(self._build)

    def _build(self):
        error = None

        try:
            self.cache.build()
            if self.val_file.readinto(self.inputs) != self.inputs.nbytes:
                raise ValueError("truncated validation shard")
            for offset in range(0, self.total_tokens, self.local_tokens):
                end = offset + self.local_tokens
                x = self.inputs[offset:end].astype(np.int32)
                self.length[offset:end], candidates = self.cache.query(
                    x, np.array([0, len(x)], np.int32))
                self.next[offset:end] = candidates[:, :4]
                self.counts[offset:end] = candidates[:, 4:]
        except Exception as exc:
            error = exc
            self.status[0] = 1
        finally:
            self.val_file.close()
            self.cache.release()
            del self.cache, self.inputs

        if self.group is not None:
            dist.all_reduce(self.status, op=dist.ReduceOp.MAX, group=self.group)
        if self.status.item():
            raise RuntimeError("offline validation cache build failed") from error
        for offset in range(0, self.total_tokens, self.local_tokens):
            end = offset + self.local_tokens
            length = torch.from_numpy(self.length[offset:end]).long()
            ids = torch.from_numpy(self.next[offset:end]).long()
            counts = torch.from_numpy(self.counts[offset:end]).long()
            key = (length[:, None] << 40) | (counts.clamp(0, (1 << 24)-1) << 16) | (0xFFFF - ids.clamp_min(0))
            key *= (ids >= 0) & (length[:, None] > 0)
            self.length[offset:end], self.next[offset:end] = merge_top2(key, self.group)
        del self.counts
        self.ready = True

    def finish(self):
        if self.future is None:
            raise RuntimeError("validation cache has not been started")
        try:
            self.future.result()
        finally:
            self.worker.shutdown()

    def query(self, offset, batch):
        if not self.ready:
            raise RuntimeError("validation cache is not ready")
        end = offset + len(batch[0])
        if offset < 0 or end > self.total_tokens or offset % self.local_tokens or end-offset != self.local_tokens:
            raise ValueError("validation batch does not match cache layout")
        return self.length[offset:end], self.next[offset:end]
