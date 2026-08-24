import unittest

import torch
import torch.nn as nn
import torchvision

from domainbed.lib.swa_utils import AveragedModel
from domainbed.models.aloft_sketch import (
    SketchSpectrumPerturb,
    collect_sketch_topology_loss,
    find_sketch_spectrum_modules,
    resnet_aloft_sketch,
)


def make_module(topology=False, warmup=2, ramp=2, min_count=1,
                ready_ratio=0.5):
    return SketchSpectrumPerturb(
        channels=8,
        num_classes=4,
        alpha=1.0,
        mask_ratio=0.5,
        perturb_prob=1.0,
        group_size=4,
        radial_bands=3,
        orientation_bins=6,
        strength_max=0.3,
        warmup_steps=warmup,
        ramp_steps=ramp,
        class_decay=0.99,
        class_min_count=min_count,
        ready_ratio=ready_ratio,
        gate_power=0.5,
        topology=topology,
        skeleton_iters=3,
    )


def run_with_context(module, x, labels, step):
    module.set_batch_context(labels, step)
    try:
        return module(x)
    finally:
        module.clear_batch_context()


class SketchSpectrumPerturbTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_resnet_stage_layout_for_r18_and_r50(self):
        for constructor, channels in (
                (torchvision.models.resnet18, (64, 128)),
                (torchvision.models.resnet50, (256, 512))):
            network = constructor(weights=None)
            original_layer3_last = network.layer3[-1]
            resnet_aloft_sketch(
                network,
                num_classes=203,
                positions=("layer1", "layer2"),
                alpha=1.0,
                mask_ratio=0.7,
                perturb_prob=1.0,
                group_size=32,
                radial_bands=3,
                orientation_bins=6,
                strength_max=0.3,
                warmup_steps=500,
                ramp_steps=500,
                class_decay=0.99,
                class_min_count=20,
                ready_ratio=0.5,
                gate_power=0.5,
            )

            modules = find_sketch_spectrum_modules(network)
            self.assertEqual(len(modules), 2)
            self.assertIsInstance(network.layer1[-1], SketchSpectrumPerturb)
            self.assertIsInstance(network.layer2[-1], SketchSpectrumPerturb)
            self.assertEqual(
                (network.layer1[-1].channels, network.layer2[-1].channels), channels)
            self.assertIs(network.layer3[-1], original_layer3_last)
            self.assertFalse(any(
                isinstance(item, SketchSpectrumPerturb)
                for item in network.layer3.modules()))

    def test_warmup_is_exact_identity_and_updates_class_buffers(self):
        module = make_module(warmup=2).train()
        x = torch.randn(8, 8, 28, 28)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        before = module.class_mean.clone()

        output = run_with_context(module, x, labels, step=0)

        self.assertTrue(torch.equal(output, x))
        self.assertEqual(module.class_count.tolist(), [2, 2, 2, 2])
        self.assertFalse(torch.equal(before, module.class_mean))
        self.assertEqual(module.activation_step.item(), -1)
        self.assertEqual(module.last_strength.item(), 0.0)

    def test_readiness_delays_activation_and_ramp_is_linear(self):
        module = make_module(warmup=2, ramp=2, min_count=3,
                             ready_ratio=0.5).train()
        labels = torch.tensor([0, 1])
        for step in (0, 1):
            x = torch.randn(2, 8, 28, 28)
            self.assertTrue(torch.equal(
                run_with_context(module, x, labels, step), x))
        self.assertEqual(module.activation_step.item(), -1)

        x = torch.randn(2, 8, 28, 28)
        self.assertTrue(torch.equal(run_with_context(module, x, labels, 2), x))
        self.assertEqual(module.activation_step.item(), 2)
        self.assertEqual(module.last_strength.item(), 0.0)

        run_with_context(module, torch.randn_like(x), labels, 3)
        self.assertAlmostEqual(module.last_strength.item(), 0.15, places=6)
        run_with_context(module, torch.randn_like(x), labels, 4)
        self.assertAlmostEqual(module.last_strength.item(), 0.3, places=6)

    def test_fisher_gate_suppresses_discriminative_dimension(self):
        module = make_module()
        module.class_count.fill_(10)
        module.class_mean.zero_()
        module.class_second.fill_(4.0)
        module.class_mean[:, :, 0] = torch.tensor(
            [0.0, 3.0, 6.0, 9.0]).view(4, 1)
        module.class_second[:, :, 0] = module.class_mean[:, :, 0].square() + 0.01

        fisher, gate, ready_ratio = module._fisher_gate()

        self.assertEqual(ready_ratio.item(), 1.0)
        self.assertGreater(fisher[:, 0].mean().item(), 0.99)
        self.assertLess(gate[:, 0].mean().item(), 0.11)
        self.assertLess(fisher[:, 1].mean().item(), 1e-6)
        self.assertGreater(gate[:, 1].mean().item(), 0.99)

    def _active_output(self, topology=False, requires_grad=False):
        module = make_module(
            topology=topology, warmup=0, ramp=1, min_count=1).train()
        module.class_count.fill_(10)
        module.class_mean.zero_()
        module.class_second.fill_(1.0)
        module.activation_step.fill_(0)
        x = torch.randn(8, 8, 28, 28, requires_grad=requires_grad)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        torch.manual_seed(31)
        output = run_with_context(module, x, labels, step=1)
        return module, x, output

    def test_frequency_constraints_and_conjugate_pair_masks(self):
        module, x, output = self._active_output()
        self.assertFalse(torch.equal(output, x))
        before = torch.fft.fftshift(
            torch.fft.fft2(x, norm="ortho"), dim=(2, 3))
        after = torch.fft.fftshift(
            torch.fft.fft2(output, norm="ortho"), dim=(2, 3))
        cells, orientations, high = module._masks(28, 28, x.device)

        self.assertTrue(torch.allclose(
            before[..., ~high], after[..., ~high], atol=3e-5, rtol=3e-5))
        self.assertTrue(torch.allclose(
            before[..., 14, 14], after[..., 14, 14], atol=3e-5, rtol=3e-5))

        valid = high.view(1, 1, 28, 28) & (before.abs() > 1e-5)
        phase_delta = torch.angle(after[valid] * before[valid].conj()).abs()
        # A float32 FFT -> real IFFT -> FFT round trip adds a small numerical
        # error.  The pre-IFFT spectrum diagnostic below checks the hard phase
        # constraint itself at the stricter threshold.
        self.assertLess(phase_delta.max().item(), 1e-4)

        before_cells = module._cell_energy(before.abs(), cells)
        after_cells = module._cell_energy(after.abs(), cells)
        before_direction = before_cells.sum(2)
        after_direction = after_cells.sum(2)
        self.assertTrue(torch.allclose(
            before_direction, after_direction, atol=3e-4, rtol=3e-4))

        pair_rows = module._conjugate_indices(28, x.device)
        pair_cols = module._conjugate_indices(28, x.device)
        for mask in cells.flatten(0, 1):
            paired = mask.index_select(0, pair_rows).index_select(1, pair_cols)
            self.assertTrue(torch.equal(mask, paired))

        self.assertLess(module.last_phase_drift.item(), 1e-5)
        self.assertLess(module.last_orientation_drift.item(), 3e-4)
        self.assertGreater(module.last_radial_shift.item(), 0.0)

    def test_plain_variant_has_no_auxiliary_loss(self):
        module, _, _ = self._active_output(topology=False)
        model = nn.Sequential(module)
        self.assertIsNone(collect_sketch_topology_loss(model))
        self.assertEqual(module.last_topology_loss.item(), 0.0)

    def test_topology_loss_and_fft_path_have_finite_gradients(self):
        module, x, output = self._active_output(
            topology=True, requires_grad=True)
        topology_loss = module.pop_topology_loss()

        self.assertIsNotNone(topology_loss)
        self.assertTrue(torch.isfinite(topology_loss).item())
        (output.square().mean() + 0.05 * topology_loss).backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertGreaterEqual(module.last_topology_loss.item(), 0.0)

    def test_eval_is_exact_identity_and_does_not_update_statistics(self):
        module = make_module().eval()
        module.class_count.fill_(5)
        before_count = module.class_count.clone()
        before_mean = module.class_mean.clone()
        x = torch.randn(4, 8, 28, 28)
        labels = torch.tensor([0, 1, 2, 3])

        first = run_with_context(module, x, labels, 1000)
        second = run_with_context(module, x, labels, 1000)

        self.assertTrue(torch.equal(first, x))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(module.class_count, before_count))
        self.assertTrue(torch.equal(module.class_mean, before_mean))

    def test_checkpoint_buffers_and_swad_eval_behavior(self):
        module = make_module().train()
        labels = torch.tensor([0, 1, 2, 3])
        run_with_context(module, torch.randn(4, 8, 28, 28), labels, 0)
        module.activation_step.fill_(7)

        restored = make_module()
        restored.load_state_dict(module.state_dict())
        self.assertTrue(torch.equal(restored.class_count, module.class_count))
        self.assertTrue(torch.equal(restored.class_mean, module.class_mean))
        self.assertEqual(restored.activation_step.item(), 7)

        model = nn.Sequential(nn.Conv2d(8, 8, 1), module)
        averaged = AveragedModel(model)
        module.class_mean.add_(1.0)
        module.class_count.add_(2)
        averaged.update_parameters(model)
        # DomainBed's AveragedModel averages parameters only.  These buffers
        # are training-only, and both copies are exact identities in eval.
        model.eval()
        averaged.eval()
        eval_input = torch.randn(2, 8, 28, 28)
        self.assertTrue(torch.equal(model(eval_input), averaged(eval_input)))

    def test_shape_dtype_and_missing_context_identity(self):
        module = make_module().train()
        x = torch.randn(2, 8, 28, 28)
        output = module(x)
        self.assertTrue(torch.equal(output, x))
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(output.device, x.device)
        self.assertTrue(torch.isfinite(output).all().item())


if __name__ == "__main__":
    unittest.main()
