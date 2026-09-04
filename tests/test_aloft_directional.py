import copy
import math
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch
import torchvision

import run_all
from domainbed import hparams_registry

# networks.py imports this optional dependency even for ResNet-only tests.
try:
    import open_clip  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open_clip"] = types.ModuleType("open_clip")

from domainbed.algorithms import get_algorithm_class
from domainbed.lib.swa_utils import AveragedModel
from domainbed.models.aloft import (
    ALOFT,
    find_aloft_modules,
    resnet_aloft,
)


def reference_aloft_e(module, x):
    """ALOFT-E implementation before directional noise was added."""
    batch, channels, height, width = x.shape
    spectrum = torch.fft.fftshift(
        torch.fft.fft2(x.float(), dim=(2, 3), norm="ortho"), dim=(2, 3))
    mask = module._mask(height, width, x.device)
    values = torch.view_as_real(spectrum)
    values = values.permute(0, 1, 4, 2, 3).reshape(
        batch, 2 * channels, height, width)
    sigma = (
        values.var(dim=0, unbiased=False, keepdim=True) + module.eps
    ).sqrt()
    changed = torch.where(
        mask, values + torch.randn_like(values) * module.alpha * sigma, values)
    changed = changed.reshape(
        batch, channels, 2, height, width
    ).permute(0, 1, 3, 4, 2).contiguous()
    spectrum = torch.view_as_complex(changed)
    spectrum = torch.fft.ifftshift(spectrum, dim=(2, 3))
    return torch.fft.ifft2(
        spectrum, dim=(2, 3), norm="ortho").real.to(x.dtype)


class DirectionalALOFTMathTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_iid_default_matches_previous_aloft_e(self):
        module = ALOFT(
            mode="E", alpha=1.0, mask_ratio=0.7,
            perturb_prob=1.0).train()
        x = torch.randn(5, 4, 10, 10)

        torch.manual_seed(29)
        expected = reference_aloft_e(module, x)
        torch.manual_seed(29)
        actual = module(x)

        self.assertEqual(module.noise_mode, "iid")
        self.assertTrue(torch.equal(actual, expected))

    def test_covariance_noise_uses_batch_residual_basis(self):
        module = ALOFT(noise_mode="covariance", eps=1e-4)
        flat = torch.tensor([
            [1.0, 2.0, -1.0],
            [2.0, 4.0, 0.0],
            [4.0, 8.0, 1.0],
            [5.0, 10.0, 2.0],
        ])

        torch.manual_seed(37)
        actual = module._covariance_noise(flat)
        torch.manual_seed(37)
        coefficients = torch.randn(4, 4)
        centered = flat - flat.mean(0, keepdim=True)
        expected = coefficients @ centered / math.sqrt(4)
        expected = expected + torch.randn_like(flat) * math.sqrt(module.eps)

        self.assertTrue(torch.equal(actual, expected))
        covariance = centered.T @ centered / flat.shape[0]
        self.assertGreater(covariance[0, 1].item(), 0.0)
        self.assertTrue(torch.allclose(
            covariance.diagonal(), flat.var(0, unbiased=False)))

    def test_domain_noise_matches_anova_decomposition(self):
        module = ALOFT(noise_mode="domain", eps=1e-4)
        flat = torch.tensor([
            [0.0, 0.0, 1.0],
            [2.0, 2.0, 3.0],
            [4.0, 4.0, 2.0],
            [6.0, 6.0, 4.0],
            [8.0, 8.0, 3.0],
        ])
        domain_ids = torch.tensor([2, 2, 7, 7, 7])
        module.set_domain_context(domain_ids)

        weights = torch.tensor([2.0 / 5.0, 3.0 / 5.0])
        means = torch.stack((flat[:2].mean(0), flat[2:].mean(0)))
        global_mean = flat.mean(0, keepdim=True)
        basis = (means - global_mean) * weights.sqrt().unsqueeze(1)
        within_variance = (
            weights[0] * (flat[:2] - means[0]).square().mean(0)
            + weights[1] * (flat[2:] - means[1]).square().mean(0)
        )

        torch.manual_seed(41)
        actual = module._domain_noise(flat)
        torch.manual_seed(41)
        expected = torch.randn(5, 2) @ basis
        expected = expected + torch.randn_like(flat) * (
            within_variance + module.eps).sqrt().unsqueeze(0)

        total_variance = flat.var(0, unbiased=False)
        decomposed = basis.square().sum(0) + within_variance
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.allclose(
            decomposed, total_variance, atol=1e-6, rtol=1e-6))

    def test_context_validation_and_disabled_paths(self):
        x = torch.randn(4, 3, 8, 8)
        domain_module = ALOFT(noise_mode="domain").train()
        with self.assertRaises(RuntimeError):
            domain_module(x)

        domain_module.set_domain_context(torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            domain_module(x)
        domain_module.set_domain_context(torch.zeros(4, dtype=torch.long))
        with self.assertRaises(ValueError):
            domain_module(x)

        with self.assertRaises(ValueError):
            ALOFT(mode="S", noise_mode="covariance")
        with self.assertRaises(ValueError):
            ALOFT(noise_mode="unknown")
        with self.assertRaises(ValueError):
            domain_module.set_domain_context(torch.zeros(2, 2))

        self.assertIs(ALOFT(noise_mode="covariance").eval()(x), x)
        self.assertIs(ALOFT(
            noise_mode="covariance", alpha=0.0).train()(x), x)
        singleton = x[:1]
        self.assertIs(
            ALOFT(noise_mode="covariance").train()(singleton), singleton)

    def test_directional_forward_preserves_unselected_band_and_gradients(self):
        x = torch.randn(4, 3, 12, 12, requires_grad=True)
        module = ALOFT(
            noise_mode="covariance", alpha=1.0,
            mask_ratio=0.7, perturb_prob=1.0).train()
        output = module(x)

        before = torch.fft.fftshift(
            torch.fft.fft2(x.detach(), norm="ortho"), dim=(2, 3))
        after = torch.fft.fftshift(
            torch.fft.fft2(output.detach(), norm="ortho"), dim=(2, 3))
        mask = module._mask(12, 12, x.device)[0, 0]

        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.isfinite(output).all().item())
        self.assertFalse(torch.equal(output, x))
        self.assertTrue(torch.allclose(
            before[..., ~mask], after[..., ~mask], atol=2e-5, rtol=2e-5))

        output.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_domain_forward_is_finite_and_context_can_be_cleared(self):
        x = torch.randn(4, 3, 12, 12, requires_grad=True)
        module = ALOFT(
            noise_mode="domain", mask_ratio=0.7,
            perturb_prob=1.0).train()
        module.set_domain_context(torch.tensor([0, 0, 1, 1]))
        try:
            output = module(x)
        finally:
            module.clear_domain_context()

        self.assertIsNone(module._domain_context)
        self.assertTrue(torch.isfinite(output).all().item())
        output.mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())


class DirectionalALOFTIntegrationTest(unittest.TestCase):
    def test_resnet_layout_for_both_direction_modes(self):
        for constructor in (
                torchvision.models.resnet18, torchvision.models.resnet50):
            for noise_mode in ("covariance", "domain"):
                network = resnet_aloft(
                    constructor(weights=None), noise_mode=noise_mode,
                    mode="E", alpha=1.0, mask_ratio=0.7,
                    perturb_prob=1.0)
                modules = find_aloft_modules(network)
                self.assertEqual(len(modules), 3)
                self.assertTrue(all(
                    item.noise_mode == noise_mode for item in modules))
                with self.assertRaises(RuntimeError):
                    resnet_aloft(network, positions=("layer3",))

    def test_eval_and_swad_are_deterministic(self):
        network = resnet_aloft(
            torchvision.models.resnet18(weights=None),
            noise_mode="covariance", mask_ratio=0.7)
        restored = resnet_aloft(
            torchvision.models.resnet18(weights=None),
            noise_mode="covariance", mask_ratio=0.7)
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
        for algorithm in ("ALOFT_CovLF_E", "ALOFT_DomainLF_E"):
            hparams = hparams_registry.default_hparams(algorithm, "SKET")
            self.assertEqual(hparams["aloft_alpha"], 1.0)
            self.assertEqual(hparams["aloft_mask_ratio"], 0.7)
            self.assertEqual(hparams["aloft_perturb_prob"], 1.0)
            self.assertEqual(
                hparams["aloft_positions"], ["layer1", "layer2", "layer3"])
            self.assertIsNotNone(get_algorithm_class(algorithm))
            self.assertEqual(run_all.METHODS[algorithm]["algorithm"], algorithm)
            self.assertEqual(run_all.METHODS[algorithm]["swad"], "LossValley")

            command = run_all.build_command(
                Path.cwd(), algorithm, seed=2, batch=8,
                dataset="SKET", steps="10")
            self.assertIn(algorithm, command)
            self.assertEqual(
                command[command.index("--aloft_mask_ratio") + 1], "0.7")

    def test_domain_algorithm_update_clears_context(self):
        hparams = hparams_registry.default_hparams(
            "ALOFT_DomainLF_E", "SKET")
        hparams.update({"pretrained": False, "resnet18": True})
        algorithm = get_algorithm_class("ALOFT_DomainLF_E")(
            (3, 32, 32), num_classes=3, num_domains=2, hparams=hparams)
        result = algorithm.update(
            [torch.randn(2, 3, 32, 32), torch.randn(3, 3, 32, 32)],
            [torch.tensor([0, 1]), torch.tensor([1, 2, 0])],
        )

        self.assertTrue(np.isfinite(result["loss"]))
        self.assertTrue(all(
            module._domain_context is None
            for module in find_aloft_modules(algorithm)))


if __name__ == "__main__":
    unittest.main()
