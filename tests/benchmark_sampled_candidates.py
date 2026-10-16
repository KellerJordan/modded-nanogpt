"""python -m tests.benchmark_sampled_candidates [--shard path/to/fineweb_train_000001.bin]"""
import argparse
import platform
import statistics
from time import perf_counter

import numpy as np

from tests.reference_sampled_candidates import BufferedCandidateBuilder
from track_1_short.perf.sampled_candidates import CandidateBuilder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", help="FineWeb .bin shard (a token-aligned prefix is also accepted)")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    opts = parser.parse_args()
    if opts.iterations < 1 or opts.repeats < 2:
        parser.error("iterations must be positive and repeats at least two")
    rng = np.random.default_rng(360)
    tokens = None
    if opts.shard:
        header = np.fromfile(opts.shard, dtype=np.int32, count=256)
        if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
            parser.error("not a supported FineWeb shard")
        tokens = np.memmap(opts.shard, dtype=np.uint16, mode="r", offset=1024)
    prefix = rng.integers(0, 50304, 50304, dtype=np.int64)
    prefix[::3] = -1
    print(f"numpy={np.__version__} hardware={platform.platform()} iterations={opts.iterations} repeats={opts.repeats}")
    print(f"targets={'FineWeb shard prefix' if tokens is not None else 'seeded Zipf(1.25)'}; synthetic prefix table, 1/3 absent")
    print("P | T | buffered median [min,max] us | direct median [min,max] us | speedup")
    for P, T in ((10240, 16384), (10240, 32768), (14336, 49152), (24576, 49152)):
        if tokens is not None and tokens.size < 32 * T + 1:
            parser.error("shard must contain at least 32 complete microbatches plus one token")
        batches = [(np.array(tokens[1 + i*T:1 + (i+1)*T], dtype=np.int64) if tokens is not None
                    else rng.zipf(1.25, T).astype(np.int64) % 50304) for i in range(32)]
        builders = [cls(50304, 24576, 49152) for cls in (BufferedCandidateBuilder, CandidateBuilder)]
        for targets in batches:
            results = [b.build(P, targets, prefix) for b in builders]
            for old, new in zip(*results):
                np.testing.assert_array_equal(old, new)
        samples = [[], []]
        for repeat in range(opts.repeats):
            for variant in (0, 1) if repeat % 2 == 0 else (1, 0):
                builder = builders[variant]
                builder.reset(0, 8)
                begin = perf_counter()
                for i in range(opts.iterations):
                    builder.build(P, batches[i % len(batches)], prefix)
                samples[variant].append((perf_counter() - begin) * 1e6 / opts.iterations)
        medians = [statistics.median(s) for s in samples]
        intervals = [f"{m:.2f} [{min(s):.2f},{max(s):.2f}]" for m, s in zip(medians, samples)]
        print(f"{P} | {T} | {intervals[0]} | {intervals[1]} | {medians[0] / medians[1]:.3f}x", flush=True)


if __name__ == "__main__":
    main()
