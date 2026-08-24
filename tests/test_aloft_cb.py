import unittest

import torch
import torch.nn as nn
import torchvision

from domainbed.lib.swa_utils import AveragedModel
from domainbed.models.aloft import ALOFT as OriginalALOFT
from domainbed.models.aloft_cb import (
    ALOFT,
    BandStatsCodebook,
    codebook_strength_at_step,
    find_band_codebooks,
    resnet_aloft_cb,
)


def make_codebook():
    return BandStatsCodebook(
        channels=8,
        mask_ratio=0.7,
        n_bands=3,
        codebook=4,
        group_size=4,
        strength_max=0.2,
        decay=0.99,
        dead_patience=20,
        reservoir_size=8,
    )


class BandStatsCodebookTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_resnet_stage_layout_for_r18_and_r50(self):
        for constructor, layer3_channels in (
                (torchvision.models.resnet18, 256),
                (torchvision.models.resnet50, 1024)):
            network = constructor(weights=None)
            resnet_aloft_cb(
                network,
                positions=("layer1", "layer2", "layer3"),
                mode="E",
                rev=True,
                alpha=1.0,
                mask_ratio=0.7,
                perturb_prob=1.0,
                codebook=4,
                group_size=32,
                n_bands=3,
                strength_max=0.2,
                decay=0.99,
                dead_patience=20,
                reservoir_size=8,
            )

            self.assertIsInstance(network.layer1[-1], ALOFT)
            self.assertIsInstance(network.layer2[-1], ALOFT)
            self.assertTrue(network.layer1[-1].rev)
            self.assertTrue(network.layer2[-1].rev)
            self.assertEqual(network.layer1[-1].mode, "E")
            self.assertEqual(network.layer2[-1].mode, "E")
            self.assertIsInstance(network.layer3[-1], BandStatsCodebook)
            self.assertEqual(network.layer3[-1].channels, layer3_channels)
            self.assertEqual(len(find_band_codebooks(network)), 1)

    def test_copied_aloft_matches_original(self):
        kwargs = dict(
            mode="E", alpha=1.0, mask_ratio=0.7, perturb_prob=1.0, rev=True)
        original = OriginalALOFT(**kwargs).train()
        copied = ALOFT(**kwargs).train()
        x = torch.randn(5, 16, 14, 14)

        torch.manual_seed(19)
        expected = original(x)
        torch.manual_seed(19)
        actual = copied(x)
        self.assertTrue(torch.equal(expected, actual))

    def test_disabled_module_is_identity_and_does_not_collect(self):
        module = make_codebook().train()
        x = torch.randn(4, 8, 14, 14)
        y = module(x)

        self.assertTrue(torch.equal(x, y))
        self.assertEqual(module.reservoir_count.sum().item(), 0)
        self.assertFalse(module.initialized.any().item())

    def test_strength_schedule(self):
        values = [
            codebook_strength_at_step(step, 2000, 500, 0.2)
            for step in (1999, 2000, 2250, 2500, 3000)
        ]
        self.assertEqual(values, [0.0, 0.0, 0.1, 0.2, 0.2])

    def test_warmup_is_exact_identity_and_uses_real_vectors(self):
        module = make_codebook().train()
        module.set_collection(True)
        x = torch.randn(4, 8, 14, 14)
        y = module(x)

        self.assertTrue(torch.equal(x, y))
        self.assertTrue(module.initialized.all().item())
        self.assertTrue(torch.equal(module.emb, module.reservoir[:, :module.K]))
        self.assertEqual(module.reservoir_count.tolist(), [4, 4])

        before = module.emb.clone()
        x2 = torch.randn_like(x)
        y2 = module(x2)
        self.assertTrue(torch.equal(x2, y2))
        self.assertFalse(torch.equal(before, module.emb))

    def test_eval_is_deterministic_and_independent_of_other_samples(self):
        module = make_codebook().train()
        module.set_collection(True)
        module(torch.randn(4, 8, 14, 14))
        module.set_collection(False)
        module.set_quantization(True, 0.2)
        module.eval()

        x = torch.randn(2, 8, 14, 14)
        other = torch.randn(3, 8, 14, 14)
        y1 = module(x)
        y2 = module(x)
        y_mixed = module(torch.cat((x[:1], other), dim=0))[:1]

        self.assertTrue(torch.equal(y1, y2))
        self.assertTrue(torch.allclose(y1[:1], y_mixed, atol=1e-6, rtol=1e-5))
        self.assertEqual(y1.shape, x.shape)
        self.assertEqual(y1.dtype, x.dtype)
        self.assertEqual(y1.device, x.device)
        self.assertTrue(torch.isfinite(y1).all().item())
        self.assertFalse(torch.equal(y1, x))

    def test_quantized_path_has_finite_gradients(self):
        module = make_codebook().train()
        module.set_collection(True)
        module(torch.randn(4, 8, 14, 14))
        module.set_collection(False)
        module.set_quantization(True, 0.2)
        module.eval()

        x = torch.randn(2, 8, 14, 14, requires_grad=True)
        y = module(x)
        y.square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())

    def test_quantization_preserves_low_frequencies_dc_and_phase(self):
        module = make_codebook().train()
        module.set_collection(True)
        module(torch.randn(4, 8, 14, 14))
        module.set_collection(False)
        module.set_quantization(True, 0.2)
        module.eval()

        x = torch.randn(2, 8, 14, 14)
        y = module(x)
        before = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(2, 3))
        after = torch.fft.fftshift(torch.fft.fft2(y, norm="ortho"), dim=(2, 3))
        high = module._bands(14, 14, x.device).any(0)

        self.assertTrue(torch.allclose(
            before[..., ~high], after[..., ~high], atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(
            before[..., 7, 7], after[..., 7, 7], atol=2e-5, rtol=2e-5))
        valid = before.abs() > 1e-5
        phase_before = before[valid] / before[valid].abs()
        phase_after = after[valid] / after[valid].abs().clamp_min(1e-8)
        self.assertTrue(torch.allclose(
            phase_before, phase_after, atol=2e-5, rtol=2e-5))

    def test_strength_clamps_and_freeze_stops_codebook_updates(self):
        module = make_codebook().train()
        module.set_collection(True)
        module(torch.randn(4, 8, 14, 14))
        module.set_quantization(True, 1.0)
        self.assertEqual(module.strength, module.strength_max)

        module.freeze_codebook()
        module.set_collection(True)
        names = (
            "emb", "ema_count", "ema_sum", "inactive_steps", "reservoir",
            "reservoir_count", "initialized",
        )
        before = {name: getattr(module, name).clone() for name in names}
        module(torch.randn(4, 8, 14, 14))

        self.assertFalse(module.collect_enabled)
        for name in names:
            self.assertTrue(torch.equal(before[name], getattr(module, name)), name)

    def test_swad_copy_keeps_the_frozen_codebook(self):
        codebook = make_codebook().train()
        codebook.set_collection(True)
        codebook(torch.randn(4, 8, 14, 14))
        codebook.freeze_codebook()
        model = nn.Sequential(nn.Conv2d(8, 8, 1), codebook)
        averaged = AveragedModel(model)

        with torch.no_grad():
            model[0].weight.add_(1.0)
        averaged.update_parameters(model)

        copied = averaged.module[1]
        self.assertTrue(copied.frozen)
        for name in ("emb", "ema_count", "ema_sum", "reservoir"):
            self.assertTrue(torch.equal(getattr(model[1], name), getattr(copied, name)))

    def test_checkpoint_preserves_quantization_state(self):
        module = make_codebook().train()
        module.set_collection(True)
        module(torch.randn(4, 8, 14, 14))
        module.set_quantization(True, 0.2)
        module.freeze_codebook()

        restored = make_codebook()
        restored.load_state_dict(module.state_dict())
        self.assertTrue(restored.enabled)
        self.assertTrue(restored.frozen)
        self.assertEqual(restored.strength, 0.2)
        self.assertTrue(torch.equal(restored.emb, module.emb))


if __name__ == "__main__":
    unittest.main()
