import unittest

import numpy as np

from tests.reference_sampled_candidates import BufferedCandidateBuilder
from track_1_short.perf.sampled_candidates import CandidateBuilder


class SampledCandidateTests(unittest.TestCase):
    def test_missing_prefix_and_absent_candidate_keep_minus_one(self):
        builder = CandidateBuilder(17, 5, 5)
        prefix = np.full(17, -1, dtype=np.int64)
        prefix[0], prefix[1] = 4, 3
        candidates, targets, prefixes = builder.build(5, np.array([16, 1, 1, 0, 16]), prefix)
        np.testing.assert_array_equal(candidates, [0, 1, 2, 4, 16])
        np.testing.assert_array_equal(targets, [4, 1, 1, 0, 4])
        np.testing.assert_array_equal(prefixes, [-1, -1, -1, 3, -1])

    def test_warmup_without_prefix_table(self):
        builder = CandidateBuilder(17, 5, 3)
        candidates, targets, prefixes = builder.build(3, np.array([0, 8, 16]), None)
        np.testing.assert_array_equal(candidates, [0, 8, 16])
        np.testing.assert_array_equal(targets, [0, 1, 2])
        self.assertIsNone(prefixes)

    def test_positions_match_original_across_ranks_counts_and_resets(self):
        rng = np.random.default_rng(360)
        for rank in range(8):
            actual = CandidateBuilder(50304, 24576, 49152)
            expected = BufferedCandidateBuilder(50304, 24576, 49152)
            prefix = rng.integers(0, 50304, 50304, dtype=np.int64)
            prefix[::3] = -1
            for _ in range(2):
                actual.reset(rank, 8)
                expected.reset(rank, 8)
                for P, T in ((10240, 16384), (10240, 32768), (14336, 49152), (24576, 49152)):
                    targets = rng.zipf(1.25, T).astype(np.int64) % 50304
                    for table in (None, prefix):
                        result, want = actual.build(P, targets, table), expected.build(P, targets, table)
                        for got, reference in zip(result, want):
                            if reference is None:
                                self.assertIsNone(got)
                            else:
                                np.testing.assert_array_equal(got, reference)
                        self.assertEqual(actual.sweep_offset, expected.sweep_offset)

    def test_negative_selection_fallback_matches_original(self):
        actual = CandidateBuilder(50304, 1000, 800)
        expected = BufferedCandidateBuilder(50304, 1000, 800)
        # Block the entire initial negative draw to force the free-class fallback.
        targets = actual.sweep[:800].copy()
        prefix = np.full(50304, -1, dtype=np.int64)
        for got, want in zip(actual.build(1000, targets, prefix), expected.build(1000, targets, prefix)):
            np.testing.assert_array_equal(got, want)

    def test_candidate_budget_still_rejects_too_many_targets(self):
        builder = CandidateBuilder(17, 5, 6)
        with self.assertRaises(AssertionError):
            builder.build(5, np.arange(6), None)


if __name__ == "__main__":
    unittest.main()
