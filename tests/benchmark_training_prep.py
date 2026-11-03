"""python -m tests.benchmark_training_prep --shard data/fineweb10B/fineweb_train_000001.bin"""
import argparse
import platform
import statistics
from time import perf_counter

import numpy as np
import torch

from tests.reference_training_prep import LegacyShard, legacy_ngram_row_ids
from track_1_short.data import Shard
from track_1_short.ngram_hash import ngram_row_ids, prime_ngram_hash


def measure_pair(functions, iterations, repeats):
    samples = [[], []]
    for rep in range(repeats):
        for index in (0, 1) if rep % 2 == 0 else (1, 0):
            begin = perf_counter()
            for _ in range(iterations):
                functions[index]()
            samples[index].append((perf_counter() - begin) * 1e6 / iterations)
    medians = [statistics.median(s) for s in samples]
    formatted = [f"{m:.2f} [{min(s):.2f},{max(s):.2f}]" for m, s in zip(medians, samples)]
    return f"{formatted[0]} | {formatted[1]} | {medians[0] / medians[1]:.3f}x"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--threads", type=int, default=1, help="torchrun normally sets OMP_NUM_THREADS=1")
    args = parser.parse_args()
    if min(args.iterations, args.threads) < 1 or args.repeats < 2:
        parser.error("positive iterations/threads and at least two repeats are required")
    torch.set_num_threads(args.threads)
    header = np.fromfile(args.shard, dtype=np.int32, count=256)
    if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
        parser.error("unsupported FineWeb shard")
    raw = np.memmap(args.shard, dtype=np.uint16, mode="r", offset=1024)
    if raw.size < 262144:
        parser.error("at least 262144 tokens are needed")
    tokens = torch.from_numpy(np.array(raw[:6_000_000], dtype=np.int32))
    counts = (16384, 32768, 40960, 49152, 262144)
    prime_ngram_hash(counts)
    print(f"torch={torch.__version__} threads={torch.get_num_threads()} hardware={platform.platform()}")
    print("operation | geometry | original median [min,max] us | optimized median [min,max] us | speedup")
    for count in counts:
        x = tokens[:count]
        torch.testing.assert_close(ngram_row_ids(x), legacy_ngram_row_ids(x), rtol=0, atol=0)
        timings = measure_pair((lambda: legacy_ngram_row_ids(x), lambda: ngram_row_ids(x)), args.iterations, args.repeats)
        print(f"hash | {count} | {timings}", flush=True)
    for budget, cap in ((16384, 896), (32768, 2048), (49152, 3072), (40960, 3072)):
        shards = [cls(tokens, world_size=8) for cls in (LegacyShard, Shard)]
        for shard in shards:
            shard._loader_thread.join()
            shard._maybe_switch()
        def plan(shard):
            try:
                return shard.next_batch(budget, cap)
            except StopIteration:
                shard.i = 0
                return shard.next_batch(budget, cap)
        for _ in range(24):
            old, new = (plan(shard) for shard in shards)
            for a, b in zip(old, new):
                for rank in range(8):
                    np.testing.assert_array_equal(a[rank], b[rank])
        functions = [lambda s=s: plan(s) for s in shards]
        timings = measure_pair(functions, args.iterations, args.repeats)
        print(f"BOS-plan | {budget}/{cap} | {timings}", flush=True)
        def first_plan(shard):
            shard.i = 0
            if isinstance(shard, Shard):
                shard._batch_cap = None  # include rebuilding the prefix at each cap/index change
            return shard.next_batch(budget, cap)
        timings = measure_pair([lambda s=s: first_plan(s) for s in shards], args.iterations, args.repeats)
        print(f"BOS-first-plan | {budget}/{cap} | {timings} | prefix_bytes={shards[1]._batch_prefix.nbytes}", flush=True)


if __name__ == "__main__":
    main()
