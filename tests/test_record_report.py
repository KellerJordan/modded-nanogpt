from pathlib import Path
import subprocess
import tempfile
import unittest

from tests.validate_record_training import checkout_revisions, final_metrics


class RecordReportTests(unittest.TestCase):
    def test_checkout_preflight_rejects_same_path_revision_and_dirty_training_code(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {n: Path(directory, n) for n in ("baseline", "candidate")}
            def git(path, *args):
                return subprocess.check_output(["git", "-c", "user.name=Fixture", "-c",
                                                "user.email=fixture@example.invalid", *args], cwd=path)
            for path in paths.values():
                path.mkdir()
                git(path, "init", "-q")
                (path / "train_gpt.py").write_text("# baseline\n")
                git(path, "add", "train_gpt.py")
                git(path, "commit", "-qm", "baseline")
            with self.assertRaisesRegex(ValueError, "separate checkouts"):
                checkout_revisions(dict(baseline=paths["baseline"], candidate=paths["baseline"]))
            # Force identical commits rather than relying on timestamp equality during git commit.
            git(paths["candidate"], "fetch", "-q", str(paths["baseline"]), "HEAD")
            git(paths["candidate"], "reset", "--hard", "FETCH_HEAD")
            with self.assertRaisesRegex(ValueError, "different commits"):
                checkout_revisions(paths)
            (paths["candidate"] / "train_gpt.py").write_text("# candidate\n")
            with self.assertRaisesRegex(ValueError, "tracked changes"):
                checkout_revisions(paths)
            git(paths["candidate"], "commit", "-qam", "candidate")
            revisions = checkout_revisions(paths)
            self.assertNotEqual(revisions["baseline"], revisions["candidate"])

    def test_only_completed_final_validation_counts(self):
        text = "step:0/1194 val_loss:10.0 train_time:1ms\nstep:1194/1194 val_loss:3.2773 train_time:39914ms"
        self.assertEqual(final_metrics(text), dict(steps=1194, val_loss=3.2773, train_time_ms=39914.))

    def test_incomplete_or_nonfinite_training_is_rejected(self):
        for text in ("", "step:10/1194 val_loss:3.2 train_time:10ms",
                     "step:1194/1194 val_loss:nan train_time:39914ms",
                     "step:1194/1194 val_loss:3.27 train_time:infms",
                     "step:1194/1194 val_loss:3.27 train_time:0ms"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                final_metrics(text)


if __name__ == "__main__":
    unittest.main()
