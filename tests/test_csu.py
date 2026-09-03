import copy
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch
import torchvision

import run_all
from domainbed import hparams_registry

# The repository imports open_clip from domainbed.networks even when testing a
# ResNet-only algorithm. Keep this unit test runnable in the lightweight local
# environment where that optional package is not installed.
try:
    import open_clip  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open_clip"] = types.ModuleType("open_clip")

from domainbed.algorithms import get_algorithm_class
from domainbed.lib.swa_utils import AveragedModel
from domainbed.models.csu import (
    CorrelatedDistributionUncertainty,
    find_csu_modules,
    resnet_csu,
)


def reference_csu(x, alpha=0.3, eps=1e-6):
    """The released CSU computation, used for a fixed-seed parity check."""

    batch, channels = x.shape[:2]
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = (x.var(dim=(2, 3), keepdim=True) + eps).sqrt()
    normalized = (x - mean) / std
    factor = torch.distributions.Beta(alpha, alpha).sample(
        (batch, 1, 1, 1)).to(x.device)

    mean_flat = torch.squeeze(mean)
    mean_center = mean_flat - mean_flat.mean(dim=0, keepdim=True)
    covariance_mean = mean_center.T @ mean_center / batch
    std_flat = torch.squeeze(std)
    std_center = std_flat - std_flat.mean(dim=0, keepdim=True)
    covariance_std = std_center.T @ std_center / batch

    with torch.no_grad():
        _, mean_vectors = torch.linalg.eigh(
            channels * covariance_mean
            + eps * torch.eye(channels, device=x.device))
        _, std_vectors = torch.linalg.eigh(
            channels * covariance_std
            + eps * torch.eye(channels, device=x.device))

    mean_root = (
        mean_vectors
        @ torch.diag(torch.sqrt(torch.clip(torch.diag(
            mean_vectors.T @ covariance_mean @ mean_vectors), min=1e-12)))
        @ mean_vectors.T
    )
    std_root = (
        std_vectors
        @ torch.diag(torch.sqrt(torch.clip(torch.diag(
            std_vectors.T @ covariance_std @ std_vectors), min=1e-12)))
        @ std_vectors.T
    )
    mean_noise = torch.randn(batch, 1, channels, device=x.device) @ mean_root
    std_noise = torch.randn(batch, 1, channels, device=x.device) @ std_root
    sampled_mean = mean + factor * mean_noise.reshape(batch, channels, 1, 1)
    sampled_std = std + factor * std_noise.reshape(batch, channels, 1, 1)
    return normalized * sampled_std + sampled_mean


class CorrelatedDistributionUncertaintyTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

    def test_eval_probability_zero_and_singleton_are_identity(self):
        x = torch.randn(4, 8, 7, 7)
        self.assertIs(CorrelatedDistributionUncertainty(p=1.0).eval()(x), x)
        self.assertIs(CorrelatedDistributionUncertainty(p=0.0).train()(x), x)
        singleton = x[:1]
        self.assertIs(
            CorrelatedDistributionUncertainty(p=1.0).train()(singleton),
            singleton,
        )

    def test_matches_released_computation_for_fixed_seed(self):
        x = torch.randn(4, 8, 7, 7)
        module = CorrelatedDistributionUncertainty(
            p=1.0, alpha=0.3).train()

        torch.manual_seed(19)
        np.random.seed(19)
        actual = module(x)
        torch.manual_seed(19)
        np.random.seed(19)
        np.random.random()
        expected = reference_csu(x)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-5))

    def test_training_output_and_gradients_are_finite(self):
        x = torch.randn(4, 8, 7, 7, requires_grad=True)
        output = CorrelatedDistributionUncertainty(
            p=1.0, alpha=0.3).train()(x)

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertFalse(torch.equal(output, x))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())


class CSUIntegrationTest(unittest.TestCase):
    def test_resnet50_layout_and_duplicate_guard(self):
        network = torchvision.models.resnet50(weights=None)
        resnet_csu(
            network,
            positions=("maxpool", "layer1"),
            p=0.5,
            alpha=0.3,
        )

        self.assertIsInstance(
            network.maxpool[-1], CorrelatedDistributionUncertainty)
        self.assertIsInstance(
            network.layer1[-1], CorrelatedDistributionUncertainty)
        self.assertEqual(len(find_csu_modules(network)), 2)
        with self.assertRaises(RuntimeError):
            resnet_csu(network, positions=("layer1",))

    def test_eval_and_swad_are_deterministic(self):
        network = torchvision.models.resnet50(weights=None)
        resnet_csu(network, p=1.0, alpha=0.3)
        restored = torchvision.models.resnet50(weights=None)
        resnet_csu(restored, p=1.0, alpha=0.3)
        restored.load_state_dict(network.state_dict())
        averaged = AveragedModel(copy.deepcopy(network))
        network.eval()
        restored.eval()
        averaged.eval()
        x = torch.randn(2, 3, 32, 32)

        self.assertTrue(torch.equal(network(x), network(x)))
        self.assertTrue(torch.equal(network(x), restored(x)))
        self.assertTrue(torch.equal(network(x), averaged(x)))

    def test_algorithm_hparams_and_run_all_entries(self):
        hparams = hparams_registry.default_hparams("CSU", "SKET")
        self.assertEqual(hparams["csu_p"], 0.5)
        self.assertEqual(hparams["csu_alpha"], 0.3)
        self.assertEqual(hparams["csu_positions"], ["maxpool", "layer1"])
        self.assertIsNotNone(get_algorithm_class("CSU"))

        self.assertEqual(run_all.METHODS["CSU"]["algorithm"], "CSU")
        self.assertEqual(run_all.METHODS["CSU"]["swad"], "False")
        self.assertEqual(run_all.METHODS["CSU_SWAD"]["algorithm"], "CSU")
        self.assertEqual(run_all.METHODS["CSU_SWAD"]["swad"], "LossValley")

        command = run_all.build_command(
            Path.cwd(), "CSU_SWAD", seed=2, batch=8,
            dataset="SKET", steps="10")
        self.assertIn("CSU", command)
        self.assertIn("LossValley", command)
        self.assertEqual(command[command.index("--csu_p") + 1], "0.5")
        self.assertEqual(command[command.index("--csu_alpha") + 1], "0.3")

    def test_algorithm_update_smoke(self):
        hparams = hparams_registry.default_hparams("CSU", "SKET")
        hparams.update({
            "pretrained": False,
            "csu_p": 1.0,
            "csu_positions": ["maxpool", "layer1"],
        })
        algorithm = get_algorithm_class("CSU")(
            (3, 32, 32), num_classes=3, num_domains=2, hparams=hparams)
        result = algorithm.update(
            [torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)],
            [torch.tensor([0, 1]), torch.tensor([1, 2])],
        )

        self.assertTrue(np.isfinite(result["loss"]))


if __name__ == "__main__":
    unittest.main()
