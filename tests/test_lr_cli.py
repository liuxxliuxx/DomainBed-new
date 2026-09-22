"""Test shared learning-rate forwarding without importing PyTorch."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

import run_all


class LearningRateCLITest(unittest.TestCase):
    def parse(self, *extra):
        with mock.patch.object(sys, "argv", [
            "run_all.py", "--algorithm", "ERM", "--gpu", "0", *extra
        ]):
            return run_all.parse_args()

    def test_default_and_explicit(self):
        self.assertIsNone(self.parse().lr)
        self.assertEqual(self.parse("--lr", "1e-4").lr, 1e-4)

    def test_invalid_values(self):
        for value in ("0", "-1", "nan", "inf", "abc"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse("--lr", value)

    def test_forwarding_all_recipes(self):
        root = Path(__file__).resolve().parents[1]
        for name in run_all.METHODS:
            for backbone in ("resnet", "vit"):
                with self.subTest(method=name, backbone=backbone):
                    args = (root, name, 0, 32, "HTP", "5000", backbone)
                    original = run_all.build_command(*args)
                    self.assertEqual(original, run_all.build_command(*args, lr=None))
                    self.assertEqual(run_all.build_command(*args, lr=1e-4),
                                     original + ["--lr", "0.0001"])

    def test_main_forwards_lr(self):
        args = self.parse("--lr", "1e-4")
        with mock.patch.object(run_all, "parse_args", return_value=args), \
                mock.patch.object(Path, "is_dir", return_value=True), \
                mock.patch.object(run_all.subprocess, "run") as run, \
                contextlib.redirect_stdout(io.StringIO()):
            run_all.main()
        self.assertEqual(run.call_count, len(run_all.SEEDS))
        for call in run.call_args_list:
            self.assertEqual(call.args[0][-2:], ["--lr", "0.0001"])


if __name__ == "__main__":
    unittest.main()
