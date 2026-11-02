from pathlib import Path
import tempfile
import unittest

from torch.multiprocessing.spawn import ProcessRaisedException

from tests.training_loop import run_training_parity


class CoupledTrainingTests(unittest.TestCase):
    def test_two_rank_losses_gradients_weights_moments_and_resume(self):
        Path("work").mkdir(exist_ok=True)
        for seed in (0, 387):
            with self.subTest(seed=seed), tempfile.TemporaryDirectory(dir="work") as directory:
                reports = run_training_parity(directory, world=2, steps=24, seed=seed)
                self.assertEqual(len(reports), 2)
                for report in reports:
                    self.assertTrue(report["checkpoint_roundtrip"])
                    self.assertTrue(report["tail_averaged"])
                    self.assertEqual(len(report["trace"]), 24)
                    self.assertEqual(sum(t["adam_update"] for t in report["trace"]), 12)
                    self.assertEqual(sum(t["row_update"] for t in report["trace"]), 9)
                    self.assertGreater(report["changed_parameters"], 0)

    def test_corrupted_candidate_labels_fail_actual_forward_loss_comparison(self):
        Path("work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir="work") as directory:
            with self.assertRaisesRegex(ProcessRaisedException, "forward-loss mismatch"):
                run_training_parity(directory, world=1, steps=8, fault=True)


if __name__ == "__main__":
    unittest.main()
