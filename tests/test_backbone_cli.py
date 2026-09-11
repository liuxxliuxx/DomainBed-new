"""Runner/backbone compatibility tests; these do not import PyTorch."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import run_all
from domainbed.backbones import normalize_backbone


class BackboneCLITest(unittest.TestCase):
    def test_names_and_invalid_values(self):
        for name in ("vit", "ViT", "vit_b_16", "VIT_B_16"):
            self.assertEqual(normalize_backbone(name), "vit")
        self.assertEqual(normalize_backbone("ResNet"), "resnet")
        with self.assertRaises(ValueError):
            normalize_backbone("not_a_backbone")

    def test_parser_default_and_alias(self):
        with mock.patch.object(sys, "argv", ["run_all.py", "--algorithm", "ERM", "--gpu", "0"]):
            self.assertEqual(run_all.parse_args().backbone, "resnet")
        with mock.patch.object(sys, "argv", ["run_all.py", "--algorithm", "ERM", "--gpu", "0",
                                             "--backbone", "ViT_B_16"]):
            self.assertEqual(run_all.parse_args().backbone, "vit")

    def test_every_runner_recipe(self):
        root = Path(__file__).resolve().parents[1]
        for name in run_all.METHODS:
            with self.subTest(method=name):
                old = run_all.build_command(root, name, 2, 8, "PACS", "5000")
                explicit = run_all.build_command(root, name, 2, 8, "PACS", "5000", "ResNet")
                new = run_all.build_command(root, name, 2, 8, "PACS", "5000", "ViT")
                self.assertEqual(old, explicit)
                self.assertNotIn("--backbone", old)
                self.assertEqual(new[-2:], ["--backbone", "vit"])
                self.assertEqual(new[2], f"PACS_{name}_vit_b_16_seed2")
                self.assertEqual(new[:2] + new[3:-2], old[:2] + old[3:])

    def test_optional_pre_edit_runner_reference(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "tmp/vit_backbone_reference_20260907/run_all.py"
        if not path.exists():
            self.skipTest("Pre-edit source snapshot is not present")
        spec = importlib.util.spec_from_file_location("old_run_all", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        # New recipes may be added; every pre-existing recipe must stay unchanged.
        self.assertEqual({name: run_all.METHODS[name] for name in old.METHODS}, old.METHODS)
        for name in old.METHODS:
            for seed in old.SEEDS:
                args = (root, name, seed, 32, "HTP", "5000")
                self.assertEqual(old.build_command(*args), run_all.build_command(*args))


if __name__ == "__main__":
    unittest.main()
