import unittest

import numpy as np
import torch

from tests.reference_training_prep import LegacyShard, legacy_ngram_row_ids
from track_1_short.data import BOS_ID, Shard
from track_1_short.ngram_hash import _hash_rows, ngram_row_ids, prime_ngram_hash


def indexed_shard(cls, tokens, world_size=1):
    shard = cls(tokens, world_size)
    shard._loader_thread.join()
    shard._maybe_switch()
    return shard


class TrainingPrepTests(unittest.TestCase):
    def test_compiled_hash_all_production_shapes_and_wrapped_int32_products(self):
        torch.compiler.reset()
        rng = np.random.default_rng(387)
        for n in (16384, 32768, 40960, 49152, 262144):
            with self.subTest(n=n):
                x = torch.from_numpy(rng.integers(-2**31, 2**31, n, dtype=np.int32))
                before = x.clone()
                torch.testing.assert_close(ngram_row_ids(x), legacy_ngram_row_ids(x), rtol=0, atol=0)
                self.assertTrue(torch.equal(x, before))
        prime_ngram_hash([16384, 32768, 40960, 49152, 262144])

    def test_hash_reserved_history_empty_and_one_token(self):
        self.assertEqual(ngram_row_ids(torch.tensor([], dtype=torch.int32)).numel(), 0)
        torch.testing.assert_close(ngram_row_ids(torch.tensor([7])), torch.tensor([42301439, 84602879], dtype=torch.int32))
        # Position one's previous bigram token is the reserved row, not token zero.
        x = torch.tensor([50256, 50256, 1, 0, 50256], dtype=torch.int32)
        torch.testing.assert_close(_hash_rows(x), legacy_ngram_row_ids(x), rtol=0, atol=0)

    def test_primed_hash_does_not_compile_during_later_batches(self):
        from torch._dynamo.utils import counters
        torch.compiler.reset()
        counts = (16384, 32768, 40960, 49152, 262144)
        prime_ngram_hash(counts)
        before = counters["stats"]["unique_graphs"]
        for context in (torch.enable_grad, torch.no_grad):
            with context():
                for count in counts:
                    x = torch.arange(count, dtype=torch.int32) % 50257
                    torch.testing.assert_close(ngram_row_ids(x), legacy_ngram_row_ids(x), rtol=0, atol=0)
        self.assertEqual(counters["stats"]["unique_graphs"], before)

    def test_hash_math_for_strides_widths_and_boundary_values(self):
        for dtype in (torch.int32, torch.int64, torch.uint16):
            x = torch.tensor([0, 1, 50256, 50257, 65535, 7, 13, 3], dtype=dtype)[::2]
            x32 = x.to(torch.int32).contiguous()
            torch.testing.assert_close(_hash_rows(x32), legacy_ngram_row_ids(x), rtol=0, atol=0)
        rng = np.random.default_rng(360)
        for n in range(2, 65):
            x = torch.from_numpy(rng.integers(-2**31, 2**31, n, dtype=np.int32))
            torch.testing.assert_close(_hash_rows(x), legacy_ngram_row_ids(x), rtol=0, atol=0)

    def test_bos_plans_preserve_every_rank_span_across_geometry_changes(self):
        rng = np.random.default_rng(387)
        lengths = rng.integers(1, 80, 8000)
        tokens = torch.zeros(int(lengths.sum()), dtype=torch.int32)
        tokens[np.r_[0, np.cumsum(lengths[:-1])]] = BOS_ID
        for world in (1, 2, 8):
            actual, old = (indexed_shard(cls, tokens, world) for cls in (Shard, LegacyShard))
            for step in range(150):
                budget, cap = (127, 255, 63)[step % 3], (32, 96, 16)[step % 3]
                try:
                    expected = old.next_batch(budget, cap)
                except StopIteration:
                    with self.assertRaises(StopIteration):
                        actual.next_batch(budget, cap)
                    break
                got = actual.next_batch(budget, cap)
                self.assertEqual(actual.i, old.i)
                for rank in range(world):
                    for a, b in zip(got, expected):
                        np.testing.assert_array_equal(a[rank], b[rank])
                    starts, ends = (torch.tensor(x[rank]) for x in got)
                    old_starts, old_ends = (torch.tensor(x[rank]) for x in expected)
                    torch.testing.assert_close((ends-starts).cumsum(0), (old_ends-old_starts).cumsum(0))

    def test_bos_partial_to_full_index_invalidates_cached_lengths(self):
        tokens = torch.zeros(23, dtype=torch.int32)
        tokens[[0, 3, 8, 10, 19]] = BOS_ID
        shard = indexed_shard(Shard, tokens)
        shard.bos_idx = shard._full_idx[:3]
        shard._ready.clear()
        shard.next_batch(6, 4)
        shard._ready.set()
        starts, ends = shard.next_batch(6, 4)
        np.testing.assert_array_equal(starts[0], [8, 10, 19])
        np.testing.assert_array_equal(ends[0], [10, 14, 20])

    def test_bos_empty_exact_budget_and_failure_does_not_consume(self):
        empty = indexed_shard(Shard, torch.zeros(10, dtype=torch.int32))
        with self.assertRaises(StopIteration):
            empty.next_batch(0, 4)
        tokens = torch.zeros(9, dtype=torch.int32)
        tokens[[0, 3, 7]] = BOS_ID
        shard = indexed_shard(Shard, tokens, 2)
        with self.assertRaises(StopIteration):
            shard.next_batch(6, 4)
        self.assertEqual(shard.i, 0)
        one = indexed_shard(Shard, tokens)
        starts, ends = one.next_batch(6, 4)
        np.testing.assert_array_equal(starts[0], [0, 3])
        np.testing.assert_array_equal(ends[0], [3, 7])


if __name__ == "__main__":
    unittest.main()
