"""Real host staging/future code with CPU tensors and already-completed upload events.

These tests cover host copies and scratch lifetimes, not CUDA event ordering or DMA.
"""
from concurrent.futures import ThreadPoolExecutor
import unittest

import numpy as np
import torch

from tests.reference_sampled_candidates import BufferedCandidateBuilder
from track_1_short.perf.sampled_candidates import CandidateBuilder
from track_1_short.perf.sampled_softmax_overlap import CandidateStaging, PinnedSlot, PINNED_SLOTS


class CompletedUpload:
    def synchronize(self):
        pass


def host_staging(pool):
    # Avoid the CUDA-only constructor while exercising the actual fill_slot, submit and take methods.
    staging = CandidateStaging.__new__(CandidateStaging)
    staging.slots = [PinnedSlot(*(torch.full((n,), -99, dtype=torch.int64) for n in (17, 6, 6)),
                                CompletedUpload()) for _ in range(PINNED_SLOTS)]
    staging.next_slot = 0
    staging.prep = pool
    staging.pending = None
    return staging


def copied(result):
    return tuple(None if x is None else x.copy() for x in result)


def assert_slot(test, staged, expected):
    candidates, targets, prefix = expected
    test.assertEqual(staged.num_candidates, len(candidates))
    test.assertEqual(staged.num_rows, len(targets))
    test.assertEqual(staged.num_prefix_rows, 0 if prefix is None else len(prefix))
    np.testing.assert_array_equal(staged.slot.candidates[:staged.num_candidates].numpy(), candidates)
    np.testing.assert_array_equal(staged.slot.target_pos[:staged.num_rows].numpy(), targets)
    if prefix is not None:
        np.testing.assert_array_equal(staged.slot.prefix_pos[:staged.num_prefix_rows].numpy(), prefix)


class CandidateHostStagingTests(unittest.TestCase):
    def test_slot_copy_survives_next_build_overwriting_scratch(self):
        builder = CandidateBuilder(17, 17, 6)
        with ThreadPoolExecutor(max_workers=1) as pool:
            staging = host_staging(pool)
            result = builder.build(5, np.array([0, 1, 16, 1, 0, 16]), np.arange(17))
            expected = copied(result)
            staged = staging.fill_slot(*result)
            builder.build(1, np.full(6, 8, dtype=np.int64), np.full(17, -1, dtype=np.int64))
            assert_slot(self, staged, expected)

    def test_worker_handoff_and_ring_wrap_with_changing_sizes_and_prefixes(self):
        builder, reference = CandidateBuilder(17, 17, 6), BufferedCandidateBuilder(17, 17, 6)
        prefix = np.arange(16, -1, -1, dtype=np.int64)
        prefix[::3] = -1
        prefix.flags.writeable = False
        held = []
        with ThreadPoolExecutor(max_workers=1) as pool:
            staging = host_staging(pool)
            for step in range(16):
                targets = np.array([0, 16, 1, 1, 8, 16], dtype=np.int64)[:(6, 1, 0, 4)[step % 4]]
                targets.flags.writeable = False
                table = None if step % 3 == 0 else prefix
                P = (5, 17, 2, 4)[step % 4]
                expected = copied(reference.build(P, targets, table))
                def build_and_stage(P=P, targets=targets, table=table):
                    return staging.fill_slot(*builder.build(P, targets, table))
                staging.submit(step, build_and_stage)
                staged = staging.take(step)
                assert_slot(self, staged, expected)
                held = [(s, slot, want) for s, slot, want in held if s > step - PINNED_SLOTS]
                for _, slot, want in held:
                    assert_slot(self, slot, want)
                held.append((step, staged, expected))
            self.assertIsNone(staging.pending)
            self.assertIsNone(staging.take(16))
            staging.reset()

    def test_failed_worker_build_can_be_joined_and_retried(self):
        builder = CandidateBuilder(17, 17, 0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            staging = host_staging(pool)
            staging.submit(0, builder.build, 1, np.array([0]), None)
            with self.assertRaises(AssertionError):
                staging.submit(1, int)
            with self.assertRaises(ValueError):
                staging.take(0)
            self.assertIsNone(staging.pending)
            staging.submit(1, lambda: staging.fill_slot(*builder.build(0, np.array([], dtype=np.int64), None)))
            staged = staging.take(1)
            assert_slot(self, staged, (np.array([], dtype=np.int64), np.array([], dtype=np.int64), None))


if __name__ == "__main__":
    unittest.main()
