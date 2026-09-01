import copy
import unittest

import torch
import torch.nn as nn
import torchvision

import run_all
from domainbed import hparams_registry
from domainbed.lib.swa_utils import AveragedModel
from domainbed.models.aloft import ALOFT, resnet_aloft
from domainbed.models.aloft_structure import (
    SketchStructureTargets,
    StructureProbe,
    collect_structure_losses,
    find_structure_probes,
    resnet_aloft_structure,
    structure_scale_at_step,
)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(image):
    return (image - MEAN.to(image)) / STD.to(image)


def square_image(open_top=False, size=224):
    image = torch.ones(1, 3, size, size)
    lo, hi, thickness = 48, 176, 4
    image[:, :, lo:lo + thickness, lo:hi] = 0.0
    image[:, :, hi - thickness:hi, lo:hi] = 0.0
    image[:, :, lo:hi, lo:lo + thickness] = 0.0
    image[:, :, lo:hi, hi - thickness:hi] = 0.0
    if open_top:
        image[:, :, lo:lo + thickness, 80:144] = 1.0
    return image


class SketchStructureTargetsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.builder = SketchStructureTargets()

    def test_horizontal_vertical_and_diagonal_double_angles(self):
        cases = []
        horizontal = torch.ones(1, 3, 128, 128)
        horizontal[:, :, 62:66, 16:112] = 0.0
        cases.append((horizontal, (1.0, 0.0)))

        vertical = torch.ones(1, 3, 128, 128)
        vertical[:, :, 16:112, 62:66] = 0.0
        cases.append((vertical, (-1.0, 0.0)))

        diagonal = torch.ones(1, 3, 128, 128)
        for index in range(16, 112):
            diagonal[:, :, index - 2:index + 2, index] = 0.0
        cases.append((diagonal, (0.0, 1.0)))

        for image, expected in cases:
            targets = self.builder(normalize(image))
            direction = targets["direction"]
            confidence = targets["direction_confidence"]
            self.assertGreater((confidence > 0.05).float().mean().item(), 0.02)
            mean_direction = (
                direction * confidence).sum(dim=(0, 2, 3)) / confidence.sum()
            self.assertAlmostEqual(mean_direction[0].item(), expected[0], delta=0.08)
            self.assertAlmostEqual(mean_direction[1].item(), expected[1], delta=0.08)

    def test_targets_are_invariant_to_foreground_polarity(self):
        dark_on_light = square_image()
        light_on_dark = 1.0 - dark_on_light
        first = self.builder(normalize(dark_on_light))
        second = self.builder(normalize(light_on_dark))

        for key in ("direction", "direction_confidence", "stroke", "closure"):
            self.assertTrue(torch.allclose(first[key], second[key], atol=1e-5))
        self.assertTrue(torch.equal(first["closure_valid"], second["closure_valid"]))

    def test_closed_square_is_valid_and_large_gap_is_open(self):
        images = torch.cat([square_image(), square_image(open_top=True)])
        targets = self.builder(normalize(images))
        areas = targets["closure"].sum(dim=(1, 2, 3))

        self.assertEqual(targets["closure_valid"].tolist(), [1.0, 0.0])
        self.assertGreater(areas[0].item(), 4.0)
        self.assertEqual(areas[1].item(), 0.0)

    def test_faint_watermark_and_border_bar_do_not_make_closure(self):
        image = torch.ones(1, 3, 224, 224)
        image[:, :, -24:, :] = 0.0
        for row in (40, 100, 150):
            image[:, :, row:row + 2, 40:80] = 0.9
            image[:, :, row + 20:row + 22, 40:80] = 0.9
            image[:, :, row:row + 22, 40:42] = 0.9
            image[:, :, row:row + 22, 78:80] = 0.9

        targets = self.builder(normalize(image))
        self.assertEqual(targets["closure_valid"].item(), 0.0)
        self.assertEqual(targets["closure"].sum().item(), 0.0)
        self.assertLess(targets["stroke"].mean().item(), 0.1)

    def test_targets_are_detached_and_finite(self):
        image = normalize(square_image()).requires_grad_()
        targets = self.builder(image)
        for target in targets.values():
            self.assertFalse(target.requires_grad)
            self.assertTrue(torch.isfinite(target).all().item())


class StructureProbeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    @staticmethod
    def _targets(batch=2, height=16, width=16):
        direction = torch.zeros(batch, 2, height, width)
        direction[:, 0] = 1.0
        confidence = torch.ones(batch, 1, height, width)
        stroke = torch.zeros(batch, 1, height, width)
        stroke[:, :, height // 2] = 1.0
        closure = torch.zeros_like(stroke)
        closure[:, :, 4:-4, 4:-4] = 1.0
        return {
            "direction": direction,
            "direction_confidence": confidence,
            "stroke": stroke,
            "closure": closure,
            "closure_valid": torch.ones(batch),
        }

    def test_probe_returns_same_tensor_and_has_finite_gradients(self):
        for kind, expected_names in (
                ("direction", {"dir"}),
                ("topology", {"stroke", "closure", "cldice"})):
            probe = StructureProbe(
                channels=8, kind=kind, hidden_channels=8,
                skeleton_iters=3).train()
            feature = torch.randn(2, 8, 16, 16, requires_grad=True)
            probe.set_batch_context(self._targets(), 1.0)
            output = probe(feature)
            probe.clear_batch_context()

            self.assertIs(output, feature)
            losses = collect_structure_losses(nn.Sequential(probe))
            self.assertEqual(set(losses), expected_names)
            total = torch.stack(list(losses.values())).sum()
            self.assertTrue(torch.isfinite(total).item())
            total.backward()
            self.assertTrue(torch.isfinite(feature.grad).all().item())
            head_gradients = [
                parameter.grad for parameter in probe.head.parameters()
                if parameter.grad is not None]
            self.assertTrue(head_gradients)
            self.assertTrue(all(torch.isfinite(item).all() for item in head_gradients))

    def test_eval_and_missing_context_are_exact_identity(self):
        probe = StructureProbe(8, "direction", hidden_channels=8)
        feature = torch.randn(2, 8, 8, 8)
        self.assertIs(probe.train()(feature), feature)
        probe.set_batch_context(self._targets(height=8, width=8), 1.0)
        self.assertIs(probe.eval()(feature), feature)
        self.assertIsNone(collect_structure_losses(nn.Sequential(probe)).get("dir"))

    def test_schedule_boundaries(self):
        expected = {
            0: 0.0,
            200: 0.0,
            201: 0.002,
            450: 0.5,
            700: 1.0,
            701: 1.0,
        }
        for step, value in expected.items():
            self.assertAlmostEqual(structure_scale_at_step(step), value, places=7)


class ALOFTStructureNetworkTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_stage_layout_for_r18_and_r50_and_both_bands(self):
        for constructor, channels in (
                (torchvision.models.resnet18, (64, 128)),
                (torchvision.models.resnet50, (256, 512))):
            for reverse in (False, True):
                network = constructor(weights=None)
                resnet_aloft_structure(
                    network,
                    direction=True,
                    topology=True,
                    hidden_channels=16,
                    skeleton_iters=3,
                    mode="E",
                    alpha=1.0,
                    mask_ratio=0.7,
                    perturb_prob=1.0,
                    rev=reverse,
                )
                probes = find_structure_probes(network)
                aloft_modules = [
                    item for item in network.modules() if isinstance(item, ALOFT)]
                self.assertEqual(len(aloft_modules), 3)
                self.assertTrue(all(item.rev is reverse for item in aloft_modules))
                self.assertTrue(all(item.mode == "E" for item in aloft_modules))
                self.assertEqual(len(probes), 2)
                self.assertEqual(
                    (probes[0].channels, probes[1].channels), channels)
                self.assertIsInstance(network.layer1[-1], StructureProbe)
                self.assertIsInstance(network.layer2[-1], StructureProbe)
                self.assertIsInstance(network.layer3[-1], ALOFT)
                self.assertFalse(any(
                    isinstance(item, StructureProbe)
                    for item in network.layer3.modules()))

    def test_classification_forward_matches_original_aloft(self):
        raw = torchvision.models.resnet18(weights=None)
        baseline = resnet_aloft(
            copy.deepcopy(raw), mode="E", alpha=1.0, mask_ratio=0.7,
            perturb_prob=1.0, rev=False).train()
        structured = resnet_aloft_structure(
            copy.deepcopy(raw), direction=True, topology=True,
            hidden_channels=8, skeleton_iters=3, mode="E", alpha=1.0,
            mask_ratio=0.7, perturb_prob=1.0, rev=False).train()
        feature_targets = SketchStructureTargets()(normalize(
            torch.rand(2, 3, 64, 64)))
        for probe in find_structure_probes(structured):
            probe.set_batch_context(feature_targets, 1.0)
        network_input = torch.randn(2, 3, 64, 64)

        torch.manual_seed(101)
        expected = baseline(network_input)
        torch.manual_seed(101)
        actual = structured(network_input)
        for probe in find_structure_probes(structured):
            probe.clear_batch_context()
        self.assertTrue(torch.equal(expected, actual))

    def test_swad_eval_is_deterministic_and_bypasses_probes(self):
        network = torchvision.models.resnet18(weights=None)
        network = resnet_aloft_structure(
            network, hidden_channels=8, skeleton_iters=3, mode="E",
            alpha=1.0, mask_ratio=0.7, perturb_prob=1.0, rev=True)
        averaged = AveragedModel(network)
        network.eval()
        averaged.eval()
        x = torch.randn(2, 3, 64, 64)
        self.assertTrue(torch.equal(network(x), network(x)))
        self.assertTrue(torch.equal(network(x), averaged(x)))

    def test_hparams_and_run_all_entries(self):
        algorithms = [
            "ALOFT_StructLF_E", "ALOFT_StructLF_Dir_E",
            "ALOFT_StructLF_Topo_E", "ALOFT_StructHF_E",
            "ALOFT_StructHF_Dir_E", "ALOFT_StructHF_Topo_E",
        ]
        for algorithm in algorithms:
            hparams = hparams_registry.default_hparams(algorithm, "SKET")
            self.assertEqual(hparams["aloft_mask_ratio"], 0.7)
            self.assertEqual(hparams["aloft_positions"], [
                "layer1", "layer2", "layer3"])
            self.assertEqual(hparams["aloft_struct_head_channels"], 64)
            self.assertIn(algorithm, run_all.METHODS)

        self.assertEqual(
            run_all.METHODS["ALOFT_LF_E_mask07"]["algorithm"], "ALOFT_E")
        self.assertEqual(
            run_all.METHODS["ALOFT_HF_E_mask07"]["algorithm"], "ALOFT_HF_E")


if __name__ == "__main__":
    unittest.main()
