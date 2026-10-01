"""Exact-match retrieval: each position sees the next tokens that followed its context before.

Training rows come from a StreamIndex, strictly causal in the token stream: at the start of training it plans every
step's documents and resolves each step's rows, from the documents of earlier steps, in phases ahead of training.
Validation rows come from a SlotIndex over all training shards, built on the clock by rank 0 while training runs and
looked up by every rank after training, when the validation tokens are read.

A row is a cell (the match length bucket, and bins of the matches' total count and of the top token's share of it)
and the top two next tokens with their counts (token | count << 16, 0 if none); all zero where nothing matched.
"""
import glob
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import torch
import torch.distributed as dist
from exact_match import CELLS as RET_CELLS, SlotIndex, StreamIndex

VAL_BUILD_THREADS = 16  # rank 0's validation index build, on cores every rank sets aside

@dataclass(frozen=True)
class CpuPlan:
    main: set[int]          # the main thread's core, its own from step 1
    rest: set[int]          # every other thread of this rank from step 1 (NCCL, gloo, loaders, the online index)
    val_build: list[int]    # this rank's share of the cores rank 0's validation index build runs on
    val_lookup: list[int]   # this rank's validation lookups, after training


def pin_cpus(local_rank: int, world_size: int) -> CpuPlan:
    """Each rank gets an equal share of the physical cores local to its GPU (within the allowed set: Slurm hands out
    arbitrary CPU ids), and is pinned to it before the process group starts, so every thread inherits it. Its last
    cores run rank 0's validation index build, one thread per core; after training both hyperthreads of every core but
    the first run this rank's validation lookups. From step 1 the main thread has the first core to itself."""
    cpulist = lambda path: {c for r in open(path).read().split(",") for a, _, b in [r.partition("-")] for c in range(int(a), int(b or a) + 1)}
    allowed = os.sched_getaffinity(0)
    cores = lambda cpus: sorted({frozenset(cpulist(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")) & allowed
                                 for c in cpus}, key=min)
    local = [frozenset(cpulist("/sys/bus/pci/devices/{0.pci_domain_id:04x}:{0.pci_bus_id:02x}:{0.pci_device_id:02x}.0/local_cpulist"
                               .format(torch.cuda.get_device_properties(i)))) & allowed for i in range(torch.cuda.device_count())]
    if min(len(cores(cpus)) / local.count(cpus) for cpus in local) < 2:  # too few allowed cores near some GPU: ignore locality
        local = [frozenset(allowed)] * len(local)
    near = cores(local[local_rank])
    k, n = local[:local_rank].count(local[local_rank]), local.count(local[local_rank])
    mine = near[k * len(near) // n:(k + 1) * len(near) // n]
    val_lookup = sorted(set().union(*mine[1:]))
    spare = -(-VAL_BUILD_THREADS // world_size)
    val_build = [min(c) for c in mine[-spare:]] if len(mine) - spare >= 2 else []
    mine = mine[:len(mine) - len(val_build)]
    os.sched_setaffinity(0, set().union(*mine))
    return CpuPlan(main=set(mine[0]), rest=set().union(*mine[1:]) or set(mine[0]), val_build=val_build, val_lookup=val_lookup)


def pin_threads(plan: CpuPlan, child: int | None):
    """At step 1: the main thread to its own core, every other thread but the validation index's pools (pinned
    already) and the forked `child` process, if any, to the rest."""
    for tid in map(int, os.listdir("/proc/self/task")):
        try:
            if not open(f"/proc/self/task/{tid}/comm").read().startswith("slotindex-"):
                os.sched_setaffinity(tid, plan.main if tid == threading.get_native_id() else plan.rest)
        except (ProcessLookupError, FileNotFoundError):
            pass
    if child:
        try:
            os.sched_setaffinity(child, plan.rest)
        except ProcessLookupError:
            pass


class OnlineCache:
    """Training rows, resolved on the clock ahead of training into a pinned ring allocated before the clock (a
    cudaHostAlloc on the clock holds the driver lock). A step's job is submitted when the loader fetches its batch, at
    most `lookahead` steps before the step trains, so a ring deeper than that never rewrites a slot whose step is still
    to train; a slot is rewritten only after its previous H2D retired (one event per slot)."""
    def __init__(self, files, schedule, rank, world, device, lookahead):
        self.index = StreamIndex(sorted(glob.glob(files)), schedule, rank, world)
        self.builder = ThreadPoolExecutor(max_workers=1)
        self.worker = ThreadPoolExecutor(max_workers=1, initializer=torch.cuda.set_device, initargs=(device,))
        self.ring = torch.zeros((lookahead + 2, max(n for n, _ in schedule), 3), dtype=torch.int32, pin_memory=True)
        self.uploaded = [torch.cuda.Event() for _ in range(len(self.ring))]
        self.device = device
        self.jobs = {}  # step -> its rows' future, from fetch until the step trains
        self.step0 = None

    def start(self):
        self.ready = self.builder.submit(self.index.build)

    def submit(self, step, batch):
        """ScheduledBatches' fetch hook: `batch.docs` is the loader's (shard tokens, every rank's document starts, ends)."""
        self.jobs[step] = self.worker.submit(self._rows, step, *batch.docs)

    def _rows(self, step, tokens, starts, ends):
        rows = self.index.rows(step, len(tokens), starts, ends)
        i = step % len(self.ring)
        self.uploaded[i].synchronize()
        slot = self.ring[i, :len(rows)]
        slot.numpy()[:] = rows
        return slot

    def rows(self, step, n):
        """This step's rows on the device. Zero during warmup (no jobs) and at step 0: nothing was trained before it, so
        its rows are only checked to be empty, once step 1's are taken."""
        job = self.jobs.pop(step, None)
        if job is None or step == 0:
            self.step0 = job
            return torch.zeros((n, 3), dtype=torch.int32, device=self.device)
        if step == 1:
            assert not self.step0.result().any(), "step 0 has retrieval matches"
        rows = job.result().to(self.device, non_blocking=True)  # the job also checks the loader's documents against the plan
        self.uploaded[step % len(self.ring)].record()
        return rows

class ValidationCache:
    """Validation rows from one SlotIndex over all training shards, a table in /dev/shm shared by the ranks.

    Before the clock rank 0 creates the table and faults in its build memory, and the other ranks open it. On the
    clock, while training runs, rank 0 builds it on threads pinned to the `build_cpus` every rank sets aside. After
    training each rank reads and looks up just the `chunk`-token validation chunks it evaluates, on its `lookup_cpus`.
    """
    def __init__(self, train_files, val_files, chunk, total, group, build_cpus, lookup_cpus):
        files = sorted(glob.glob(train_files))
        assert len(files) == 103, f"the validation index covers all 103 training shards, found {len(files)}"
        self.val_file, self.chunk, self.group = sorted(glob.glob(val_files))[0], chunk, group
        self.rank, world = dist.get_rank(group), dist.get_world_size(group)
        self.offsets = range(self.rank * chunk, total, world * chunk)
        path = [f"/dev/shm/exact-match-{uuid.uuid4().hex}"]
        dist.broadcast_object_list(path, src=0, group=group)
        cpus = [None] * world
        dist.all_gather_object(cpus, sorted(build_cpus), group=group)
        if self.rank == 0:
            self.index = SlotIndex.create(path[0], files, VAL_BUILD_THREADS, sum(cpus, [])[:VAL_BUILD_THREADS], lookup_cpus)
        dist.barrier(group)
        if self.rank != 0:
            self.index = SlotIndex.attach(path[0], lookup_cpus)
        dist.barrier(group)
        if self.rank == 0:
            os.unlink(path[0])
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.worker.submit(lambda: None).result()  # its thread starts before the clock

    def start(self, after):
        """During training, once `after` (a future) is done: rank 0 builds the table."""
        self.built = self.worker.submit(self._build, after)

    def _build(self, after):
        after.result()
        if self.rank == 0:
            t = time.perf_counter()
            self.index.build()
            print(f"validation index built in {time.perf_counter() - t:.2f} s", flush=True)
        dist.barrier(self.group)
        if self.rank == 0:
            self.index.release()

    def lookup(self):
        """After training: read this rank's validation chunks and look them up, beside the main thread."""
        def rows():
            tokens = np.concatenate([np.fromfile(self.val_file, "<u2", self.chunk, offset=1024 + 2 * o) for o in self.offsets])
            return np.split(self.index.query(tokens, self.chunk), len(self.offsets))
        self.looked_up = self.worker.submit(rows)

    def rows(self):
        """Wait for the build and the lookup: the rows of each of this rank's validation chunks."""
        self.built.result()
        return self.looked_up.result()
