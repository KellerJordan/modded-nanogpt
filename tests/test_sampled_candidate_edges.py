from itertools import product
import unittest

import numpy as np

from track_1_short.perf.sampled_candidates import CandidateBuilder
from tests.reference_sampled_candidates import BufferedCandidateBuilder


class CandidateBoundaryTests(unittest.TestCase):
    def assert_matches_reference(self, actual, expected, P, targets, prefix):
        inputs_before = targets.copy()
        prefix_before = None if prefix is None else prefix.copy()
        want = expected.build(P, targets, prefix)
        result = actual.build(P, targets, prefix)
        for got, reference in zip(result, want):
            if reference is None:
                self.assertIsNone(got)
            else:
                np.testing.assert_array_equal(got, reference)
                self.assertEqual(got.dtype, reference.dtype)
                self.assertEqual(got.shape, reference.shape)
                self.assertTrue(got.flags.c_contiguous)
        candidates, target_pos, _ = result
        self.assertEqual(len(candidates), P)
        self.assertTrue(np.all(candidates[1:] > candidates[:-1]))
        np.testing.assert_array_equal(candidates[target_pos], targets)
        self.assertEqual(actual.sweep_offset, expected.sweep_offset)
        self.assertEqual(actual.pos[actual.vocab_size], -1)
        np.testing.assert_array_equal(targets, inputs_before)
        if prefix is not None:
            np.testing.assert_array_equal(prefix, prefix_before)
        return result

    def test_row_capacity_rejects_oversized_batches_including_one_into_zero(self):
        for capacity, count in ((0, 1), (0, 2), (1, 2), (2, 3)):
            with self.subTest(capacity=capacity, count=count):
                builder = CandidateBuilder(17, 5, capacity)
                for prefix in (None, np.full(17, -1, dtype=np.int64)):
                    with self.assertRaises(ValueError):
                        builder.build(1, np.ones(count, dtype=np.int64), prefix)

    def test_non_vector_targets_are_rejected(self):
        for shape in ((), (1, 1), (2, 1), (1, 1, 1)):
            with self.subTest(shape=shape):
                builder = CandidateBuilder(17, 5, 3)
                with self.assertRaises(ValueError):
                    builder.build(1, np.ones(shape, dtype=np.int64), None)

    def test_candidate_count_cannot_broadcast_past_scratch_or_vocabulary(self):
        for capacity, P in ((0, 1), (1, 2), (2, 3), (18, 18), (5, -1)):
            with self.subTest(capacity=capacity, P=P):
                builder = CandidateBuilder(17, capacity, 2)
                before = builder.sweep_offset
                with self.assertRaises(ValueError):
                    builder.build(P, np.array([0, 1]), None)
                self.assertEqual(builder.sweep_offset, before)

    def test_zero_capacity_accepts_empty_input_and_zero_candidates(self):
        for prefix in (None, np.full(1, -1, dtype=np.int64)):
            self.assert_matches_reference(CandidateBuilder(1, 0, 0), BufferedCandidateBuilder(1, 0, 0),
                                          0, np.array([], dtype=np.int64), prefix)

    def test_empty_input_can_still_select_negatives(self):
        for P in (0, 1, 5, 17):
            self.assert_matches_reference(CandidateBuilder(17, 17, 0), BufferedCandidateBuilder(17, 17, 0),
                                          P, np.array([], dtype=np.int64), np.full(17, -1, dtype=np.int64))

    def test_full_vocabulary_and_repeated_extreme_ids(self):
        targets = np.array([0, 16, 16, 0, 0, 16], dtype=np.int64)
        candidates, positions, prefixes = self.assert_matches_reference(
            CandidateBuilder(17, 17, 6), BufferedCandidateBuilder(17, 17, 6),
            17, targets, np.arange(16, -1, -1, dtype=np.int64))
        np.testing.assert_array_equal(candidates, np.arange(17))
        np.testing.assert_array_equal(positions, targets)
        np.testing.assert_array_equal(prefixes, 16 - targets)

    def test_exact_distinct_budget_does_not_advance_negatives(self):
        actual, expected = CandidateBuilder(1009, 4, 6), BufferedCandidateBuilder(1009, 4, 6)
        actual.reset(7, 8)
        expected.reset(7, 8)
        self.assert_matches_reference(actual, expected, 3, np.array([0, 8, 1008, 8, 0, 1008]), None)
        self.assertEqual(actual.sweep_offset, 882)

    def test_sweep_wrap_and_fallback_together(self):
        actual, expected = CandidateBuilder(50304, 1000, 800), BufferedCandidateBuilder(50304, 1000, 800)
        actual.sweep_offset = expected.sweep_offset = 50304 - 10
        targets = actual.sweep[50304 - 10:50304 - 10 + 800].copy()
        self.assert_matches_reference(actual, expected, 1000, targets, np.full(50304, -1, dtype=np.int64))
        self.assertEqual(actual.sweep_offset, 586)

    def test_sweep_wrap_without_fallback(self):
        actual, expected = CandidateBuilder(1009, 4, 1), BufferedCandidateBuilder(1009, 4, 1)
        actual.reset(7, 8)
        expected.reset(7, 8)
        self.assert_matches_reference(actual, expected, 4, np.array([0]), None)
        self.assertEqual(actual.sweep_offset, 134)

    def test_stale_prefix_positions_and_unused_scratch_are_not_exposed(self):
        actual, expected = CandidateBuilder(17, 5, 6), BufferedCandidateBuilder(17, 5, 6)
        self.assert_matches_reference(actual, expected, 5, np.array([0, 1, 16, 1, 0, 16]), np.arange(17))
        actual.target_pos.fill(-77)
        actual.prefix_pos.fill(-88)
        prefix = np.full(17, -1, dtype=np.int64)
        prefix[8] = 16  # selected in the previous build, absent from this one
        _, targets, prefixes = self.assert_matches_reference(actual, expected, 1, np.array([8, 8, 8]), prefix)
        np.testing.assert_array_equal(targets, [0, 0, 0])
        np.testing.assert_array_equal(prefixes, [-1, -1, -1])
        np.testing.assert_array_equal(actual.target_pos[3:], [-77, -77, -77])
        np.testing.assert_array_equal(actual.prefix_pos[3:], [-88, -88, -88])
        self.assertTrue(np.shares_memory(targets, actual.target_pos))
        self.assertTrue(np.shares_memory(prefixes, actual.prefix_pos))
        before = actual.prefix_pos.copy()
        _, _, no_prefix = self.assert_matches_reference(actual, expected, 1, np.array([8]), None)
        self.assertIsNone(no_prefix)
        np.testing.assert_array_equal(actual.prefix_pos, before)

    def test_reset_replays_after_many_builds(self):
        actual = CandidateBuilder(1009, 128, 32)
        targets = np.arange(32, dtype=np.int64)
        prefix = np.arange(1008, -1, -1, dtype=np.int64)
        for world in (1, 2, 8, 16):
            for rank in range(world):
                actual.reset(rank, world)
                first = tuple(x.copy() for x in actual.build(128, targets, prefix))
                for _ in range(20):
                    actual.build(128, targets, prefix)
                actual.reset(rank, world)
                for got, want in zip(actual.build(128, targets, prefix), first):
                    np.testing.assert_array_equal(got, want)

    def test_integer_width_byte_order_and_strided_readonly_inputs(self):
        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                      np.dtype(">i8")):
            for stride in (1, 2, -1, -2):
                with self.subTest(dtype=dtype, stride=stride):
                    targets = np.array([0, 16, 1, 8, 1, 16], dtype=dtype)[::stride]
                    # Negative-stride prefix table with both absent and present prefixes.
                    prefix = np.arange(16, -1, -1, dtype=np.int64)
                    prefix[::3] = -1
                    prefix = prefix[::-1]
                    targets.flags.writeable = prefix.flags.writeable = False
                    self.assert_matches_reference(CandidateBuilder(17, 8, 6), BufferedCandidateBuilder(17, 8, 6),
                                                  8, targets, prefix)

    def test_unsigned64_targets_have_known_positions_on_older_numpy_too(self):
        # NumPy 1.x np.take rejects uint64 indices that integer indexing accepts. This is outside
        # the trainer's signed-int64 contract, so check literal results rather than claim parity.
        for stride in (1, 2, -1, -2):
            with self.subTest(stride=stride):
                targets = np.array([0, 16, 1], dtype=np.uint64)[::stride]
                candidates, positions, prefixes = CandidateBuilder(17, 3, 3).build(3, targets, np.arange(17))
                self.assertEqual(candidates.dtype, np.dtype(np.int64))
                np.testing.assert_array_equal(candidates[positions], targets)
                np.testing.assert_array_equal(prefixes, positions)
        candidates, positions, prefixes = CandidateBuilder(17, 3, 3).build(
            3, np.array([0, 16, 1], dtype=np.uint64), np.arange(17))
        np.testing.assert_array_equal(candidates, [0, 1, 16])
        np.testing.assert_array_equal(positions, [0, 2, 1])
        np.testing.assert_array_equal(prefixes, [0, 2, 1])

    def test_failed_budget_does_not_advance_sweep_and_recovery_matches(self):
        actual, expected = CandidateBuilder(17, 5, 6), BufferedCandidateBuilder(17, 5, 6)
        for builder in (actual, expected):
            builder.reset(7, 8)
            with self.assertRaises(AssertionError):
                builder.build(5, np.arange(6), None)
        self.assertEqual(actual.sweep_offset, 14)
        self.assert_matches_reference(actual, expected, 5, np.array([0, 1, 1, 16]), None)

    def test_invalid_prefix_indices_raise_then_next_build_recovers(self):
        for bad_id in (18, -19):
            with self.subTest(bad_id=bad_id):
                actual, expected = CandidateBuilder(17, 5, 3), BufferedCandidateBuilder(17, 5, 3)
                prefix = np.full(17, -1, dtype=np.int64)
                prefix[1] = bad_id
                for builder in (actual, expected):
                    with self.assertRaises(IndexError):
                        builder.build(5, np.array([1, 1, 1]), prefix)
                prefix[1] = -1
                self.assert_matches_reference(actual, expected, 5, np.array([1, 1, 1]), prefix)

    def test_invalid_target_indices_still_raise(self):
        for bad_id in (18, -19):
            for builder_type in (CandidateBuilder, BufferedCandidateBuilder):
                with self.subTest(bad_id=bad_id, builder=builder_type.__name__):
                    with self.assertRaises(IndexError):
                        builder_type(17, 5, 3).build(5, np.array([bad_id]), None)

    def test_exhaustive_tiny_vocabularies(self):
        for V in range(1, 6):
            actual, expected = CandidateBuilder(V, V, 3), BufferedCandidateBuilder(V, V, 3)
            tables = (None, np.full(V, -1, dtype=np.int64), np.arange(V, dtype=np.int64),
                      np.arange(V - 1, -1, -1, dtype=np.int64))
            for T in range(4):
                for values in product(range(V), repeat=T):
                    targets = np.array(values, dtype=np.int64)
                    for P in range(len(set(values)), V + 1):
                        for rank in (0, 7):
                            for prefix in tables:
                                with self.subTest(V=V, targets=values, P=P, rank=rank,
                                                  prefix=prefix is not None):
                                    actual.reset(rank, 8)
                                    expected.reset(rank, 8)
                                    self.assert_matches_reference(actual, expected, P, targets, prefix)

    def test_two_thousand_seeded_state_transitions(self):
        rng = np.random.default_rng(387)
        for V in (1, 2, 3, 5, 17, 257, 1009, 4093, 50304, 50304):
            actual, expected = CandidateBuilder(V, V, 128), BufferedCandidateBuilder(V, V, 128)
            prefix = rng.integers(0, V, V, dtype=np.int64)
            prefix[::3] = -1
            for step in range(200):
                if step % 11 == 0:
                    world = int(rng.choice([1, 2, 8, 16]))
                    rank = int(rng.integers(world))
                    actual.reset(rank, world)
                    expected.reset(rank, world)
                T = int(rng.integers(129))
                storage = rng.integers(0, V, 2 * T, dtype=np.int64)
                targets = storage[::2] if step % 2 else storage[::-2]
                P = int(rng.integers(len(set(targets.tolist())), V + 1))
                table = None if step % 3 == 0 else prefix
                with self.subTest(V=V, step=step, T=T, P=P):
                    self.assert_matches_reference(actual, expected, P, targets, table)


if __name__ == "__main__":
    unittest.main()
