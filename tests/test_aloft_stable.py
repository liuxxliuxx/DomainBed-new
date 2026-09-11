"""Offline tests for the clean-view ALOFT/StableNet combination."""
import copy
import math
from pathlib import Path
import random
import sys
import types
import unittest
from unittest import mock

import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

try:
    import open_clip  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open_clip"] = types.ModuleType("open_clip")

import run_all
from domainbed import hparams_registry
from domainbed.algorithms import get_algorithm_class
from domainbed.lib.aloft_stable import (
    clean_features, domain_balanced_weights, stable_mix_at_step,
)
from domainbed.lib.stablenet import learn_weights
from domainbed.lib.swa_utils import AveragedModel
from domainbed.models.aloft import ALOFT, find_aloft_modules


def hparams(**overrides):
    hp = hparams_registry.default_hparams("ALOFT_Stable_E", "HTP")
    hp.update(pretrained=False, resnet18=True, batch_size=2,
              stable_epochb=2, stable_warmup_steps=0, stable_ramp_steps=0)
    hp.update(overrides)
    return hp


def make_model(**overrides):
    return get_algorithm_class("ALOFT_Stable_E")(
        (3, 32, 32), 2, 2, hparams(**overrides))


def batches(device="cpu"):
    return ([torch.randn(2, 3, 32, 32, device=device) for _ in range(2)],
            [torch.tensor([0, 1], device=device), torch.tensor([1, 0], device=device)])


class StableWeightMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_boundaries_and_disabled_paths(self):
        expected = {0: 0.0, 99: 0.0, 100: 0.001, 199: 0.1, 299: 0.2, 300: 0.2}
        for step, value in expected.items():
            self.assertAlmostEqual(stable_mix_at_step(step, 0.2, 100, 200), value)
        self.assertEqual(stable_mix_at_step(100, 0.2, 100, 0), 0.2)
        self.assertEqual(stable_mix_at_step(100, 0.0, 0, 0), 0.0)
        for args in ((0, -0.1, 0, 0), (0, 1.1, 0, 0), (0, float("nan"), 0, 0),
                     (-1, 0.2, 0, 0), (0, 0.2, -1, 0), (0, 0.2, 0, 1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                stable_mix_at_step(*args)

    def test_domain_mass_shrinkage_and_detach(self):
        raw = torch.tensor([[0.0], [1.0], [3.0], [-1.0], [0.0]], requires_grad=True)
        q = raw.detach().softmax(dim=0)
        balanced = torch.cat([q[:2] / q[:2].sum() * 0.4, q[2:] / q[2:].sum() * 0.6])
        for mix in (0.0, 0.2, 1.0):
            weights = domain_balanced_weights(raw, [2, 3], mix)
            torch.testing.assert_close(weights, balanced * mix + (1.0 - mix) / 5)
            self.assertFalse(weights.requires_grad)
            self.assertAlmostEqual(weights.sum().item(), 1.0)
            self.assertAlmostEqual(weights[:2].sum().item(), 0.4)
            self.assertAlmostEqual(weights[2:].sum().item(), 0.6)
            self.assertTrue((weights >= (1.0 - mix) / 5).all())

    def test_extreme_logits_do_not_underflow_a_domain(self):
        raw = torch.tensor([[1e6], [1e6 - 1], [-1e6], [-1e6 + 1]])
        weights = domain_balanced_weights(raw, [2, 2], 0.2)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(weights[:2].sum().item(), 0.5)
        self.assertAlmostEqual(weights[2:].sum().item(), 0.5)

    def test_invalid_weights_and_batches(self):
        for raw, sizes, mix in ((torch.ones(4), [2, 2], 0.2),
                                (torch.ones(4, 1), [2, 1], 0.2),
                                (torch.ones(4, 1), [0, 4], 0.2),
                                (torch.ones(4, 1), [2, 2], 1.1),
                                (torch.full((4, 1), float("nan")), [2, 2], 0.2)):
            with self.assertRaises(ValueError):
                domain_balanced_weights(raw, sizes, mix)

    def test_original_inner_loop_does_not_backpropagate_to_features(self):
        features = torch.randn(4, 8, requires_grad=True)
        weights, raw = learn_weights(features, features.detach(), torch.ones(4, 1), hparams())
        self.assertIsNone(features.grad)
        self.assertFalse(weights.requires_grad)
        self.assertFalse(raw.requires_grad)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(weights.sum().item(), 1.0)

    def test_default_inner_loop_at_htp_batch_and_resnet50_dimension(self):
        hp = hparams_registry.default_hparams("ALOFT_Stable_E", "HTP")
        torch.manual_seed(11)
        features = torch.randn(64, 2048)
        _, raw = learn_weights(features, features.clone(), torch.ones(64, 1), hp, epoch=1)
        weights = domain_balanced_weights(raw, [32, 32], hp["stable_mix_max"])
        self.assertTrue(torch.isfinite(weights).all())
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
        torch.testing.assert_close(weights[:32].sum(), torch.tensor(0.5))
        torch.testing.assert_close(weights[32:].sum(), torch.tensor(0.5))


class CleanViewTest(unittest.TestCase):
    def test_training_batchnorm_buffers_are_not_updated(self):
        net = nn.Sequential(nn.BatchNorm2d(3), ALOFT()).train()
        buffers = {name: value.clone() for name, value in net.named_buffers()}
        clean_features(net, torch.randn(4, 3, 8, 8))
        self.assertTrue(all(module.training for module in net.modules()))
        for name, value in net.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]))

    def test_disables_perturbation_dropout_and_bn_then_restores_modes(self):
        net = nn.Sequential(nn.BatchNorm2d(3), ALOFT(), nn.Dropout(0.5), nn.Flatten()).train()
        net[0].eval()  # Deliberately mixed flags must survive.
        states = [module.training for module in net.modules()]
        buffers = {name: value.clone() for name, value in net.named_buffers()}
        x = torch.randn(4, 3, 8, 8, requires_grad=True)
        rng = torch.get_rng_state().clone()
        actual = clean_features(net, x)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(actual.requires_grad)
        self.assertEqual(states, [module.training for module in net.modules()])
        for name, value in net.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]))
        torch.testing.assert_close(actual, net[0](x).flatten(1))

    def test_restores_modes_after_forward_exception(self):
        net = nn.Sequential(nn.BatchNorm2d(3), ALOFT(), nn.Dropout(0.5)).train()
        net[0].eval()
        states = [module.training for module in net.modules()]
        with mock.patch.object(net, "forward", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                clean_features(net, torch.randn(4, 3, 8, 8))
        self.assertEqual(states, [module.training for module in net.modules()])


class StableALOFTIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(31)
        random.seed(31)

    def test_hparams_registration_and_runner_preserve_baseline(self):
        hp = hparams_registry.default_hparams("ALOFT_Stable_E", "HTP")
        original = hparams_registry.default_hparams("StableNet", "HTP")
        for name, value in original.items():
            if name.startswith("stable_"):
                self.assertEqual(hp[name], value)
        self.assertEqual(hp["aloft_mask_ratio"], 0.7)
        self.assertEqual(hp["aloft_positions"], ["layer1", "layer2", "layer3"])
        self.assertEqual(hp["stable_mix_max"], 0.2)
        self.assertEqual(hp["stable_warmup_steps"], 100)
        self.assertEqual(hp["stable_ramp_steps"], 200)
        self.assertEqual(run_all.METHODS["ALOFT_Stable_E"]["swad"], "LossValley")
        for backbone in ("resnet", "vit"):
            cmd = run_all.build_command(Path.cwd(), "ALOFT_Stable_E", 3, 32, "HTP", "5000", backbone)
            self.assertEqual(cmd[cmd.index("--aloft_mask_ratio") + 1], "0.7")
            self.assertEqual(cmd[cmd.index("--stable_mix_max") + 1], "0.2")
            self.assertEqual(cmd[cmd.index("--trial_seed") + 1], "3")
        self.assertEqual(run_all.METHODS["ALOFT_E"]["extra_args"][3], "0.5")
        self.assertEqual(run_all.METHODS["ALOFT_LF_E_mask07"]["extra_args"][3], "0.7")

    def test_invalid_combination_parameters(self):
        for hp in ({"stable_mix_max": 1.2}, {"stable_warmup_steps": -1},
                   {"stable_ramp_steps": -1}, {"stable_presave_ratio": 1.0}):
            with self.subTest(hp=hp), self.assertRaises(ValueError):
                make_model(**hp)

    def test_disabled_and_warmup_match_baseline_updates_and_rng_exactly(self):
        xs, ys = batches()
        for overrides in ({"stable_mix_max": 0.0}, {"stable_warmup_steps": 10}):
            with self.subTest(overrides=overrides):
                combined = make_model(resnet_dropout=0.3, **overrides).train()
                base = get_algorithm_class("ALOFT_E")((3, 32, 32), 2, 2, combined.hparams).train()
                base.network.load_state_dict(combined.network.state_dict(), strict=True)
                with mock.patch("domainbed.algorithms.algorithms.learn_weights") as learner:
                    for _ in range(2):
                        torch.manual_seed(71)
                        expected = base.update(xs, ys)
                        expected_rng = torch.get_rng_state().clone()
                        torch.manual_seed(71)
                        actual = combined.update(xs, ys)
                        self.assertEqual(expected["loss"], actual["loss"])
                        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                        self.assertEqual(actual["stable_mix"], 0.0)
                        self.assertEqual(actual["weight_ess"], 4.0)
                        for name, value in base.network.state_dict().items():
                            self.assertTrue(torch.equal(value, combined.network.state_dict()[name]), name)
                    learner.assert_not_called()

    def test_clean_view_is_used_for_weights_and_history(self):
        model = make_model(resnet_dropout=0.4).train()
        xs, ys = batches()
        forwards = []
        def capture(module, inputs, output):
            forwards.append((module.training, output.requires_grad, output.detach().clone()))
        handle = model.featurizer.register_forward_hook(capture)
        self.addCleanup(handle.remove)
        predictions = []
        handle = model.classifier.register_forward_hook(
            lambda module, inputs, output: predictions.append(output.detach().clone()))
        self.addCleanup(handle.remove)
        raw = torch.tensor([[-2.0], [1.0], [2.0], [-1.0]])
        with mock.patch("domainbed.algorithms.algorithms.learn_weights",
                        return_value=(raw.softmax(0), raw)) as learner:
            result = model.update(xs, ys, epoch=7)
        self.assertEqual(len(forwards), 2)
        self.assertEqual(forwards[0][:2], (False, False))
        self.assertEqual(forwards[1][:2], (True, True))
        self.assertTrue(torch.equal(learner.call_args.args[0], forwards[0][2]))
        self.assertTrue(torch.equal(learner.call_args.args[1], forwards[0][2]))
        self.assertEqual(learner.call_args.kwargs["epoch"], 7)
        self.assertTrue(torch.equal(model.pre_features, forwards[0][2]))
        self.assertTrue(torch.equal(model.pre_logits, raw))
        self.assertFalse(model.pre_features.requires_grad)
        self.assertTrue(all(module.training for module in find_aloft_modules(model)))
        weights = domain_balanced_weights(raw, [2, 2], 0.2).flatten()
        expected_loss = (weights * nn.functional.cross_entropy(
            predictions[0], torch.cat(ys), reduction="none")).sum().item()
        self.assertEqual(result["loss"], expected_loss)
        self.assertAlmostEqual(result["weight_ess"], (1 / weights.square().sum()).item())
        self.assertAlmostEqual(result["weight_domain_0"], 0.5)
        self.assertAlmostEqual(result["weight_domain_1"], 0.5)
        self.assertAlmostEqual(result["weight_class_0"], weights[[0, 3]].sum().item())
        self.assertAlmostEqual(result["batch_class_0"], 0.5)

    def test_history_warmup_ramp_and_ema_use_real_clean_features(self):
        model = make_model(stable_warmup_steps=1, stable_ramp_steps=2).train()
        xs, ys = batches()
        first = model.update(xs, ys)
        self.assertEqual(first["stable_mix"], 0.0)
        self.assertEqual(model.history_count.item(), 1)
        previous = model.pre_features.clone()
        clean = clean_features(model.featurizer, torch.cat(xs))
        second = model.update(xs, ys)
        self.assertAlmostEqual(second["stable_mix"], 0.1)
        torch.testing.assert_close(model.pre_features, (previous + clean) / 2)
        self.assertAlmostEqual(model.update(xs, ys)["stable_mix"], 0.2)
        model.history_count.fill_(10)
        previous = model.pre_features.clone()
        clean = clean_features(model.featurizer, torch.cat(xs))
        model.update(xs, ys)
        torch.testing.assert_close(model.pre_features, previous * 0.9 + clean * 0.1)

    def test_invalid_batch_rejected_before_training(self):
        model = make_model()
        xs, ys = batches()
        for bad_x, bad_y in ((xs[:1], ys[:1]), (xs, ys[:1]),
                             ([xs[0][:1], xs[1]], ys), (xs, [ys[0][:1], ys[1]])):
            with self.assertRaisesRegex(ValueError, "per source domain"):
                model.update(bad_x, bad_y)
        self.assertEqual(model.update_count.item(), 0)

    def test_resnet18_active_update_has_finite_backbone_gradients(self):
        model = make_model().train()
        before = model.featurizer.network.conv1.weight.detach().clone()
        result = model.update(*batches())
        self.assertTrue(all(math.isfinite(value) for value in result.values()))
        self.assertGreater(result["stable_mix"], 0)
        self.assertTrue(torch.isfinite(model.featurizer.network.conv1.weight.grad).all())
        self.assertFalse(torch.equal(before, model.featurizer.network.conv1.weight))
        self.assertEqual(model.update_count.item(), 1)

    def test_resnet50_full_feature_dimension_active_update(self):
        model = make_model(resnet18=False, stable_epochb=1).train()
        result = model.update(*batches())
        self.assertEqual(model.pre_features.shape, (4, 2048))
        self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_small_vit_active_update(self):
        def small_vit(weights=None, image_size=32, **kwargs):
            return VisionTransformer(image_size=image_size, patch_size=16, num_layers=12,
                                     num_heads=4, hidden_dim=32, mlp_dim=64, num_classes=1000)
        with mock.patch("torchvision.models.vit_b_16", side_effect=small_vit):
            model = make_model(backbone="vit").train()
        result = model.update(*batches())
        self.assertEqual(model.pre_features.shape, (4, 32))
        self.assertTrue(all(math.isfinite(value) for value in result.values()))

    def test_checkpoint_resume_restores_history_schedule_and_predictions(self):
        model = make_model(stable_warmup_steps=1, stable_ramp_steps=2).train()
        xs, ys = batches()
        model.update(xs, ys)
        restored = make_model(stable_warmup_steps=1, stable_ramp_steps=2).train()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        restored.optimizer.load_state_dict(copy.deepcopy(model.optimizer.state_dict()))
        torch.manual_seed(19)
        expected = model.update(xs, ys)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(19)
        actual = restored.update(xs, ys)
        self.assertEqual(expected, actual)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
        model.eval()
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(xs[0]), restored(xs[0])))

    def test_eval_and_swad_do_not_use_weights_or_change_history(self):
        model = make_model().train()
        xs, ys = batches()
        model.update(xs, ys)
        averaged = AveragedModel(model)
        averaged.update_parameters(model, step=0)
        model.eval()
        averaged.eval()
        history = model.pre_features.clone()
        with mock.patch("domainbed.algorithms.algorithms.learn_weights") as learner, torch.no_grad():
            prediction = model(xs[0])
            self.assertTrue(torch.equal(prediction, model(xs[0])))
            self.assertTrue(torch.equal(prediction, averaged(xs[0])))
            # Prediction must be independent of the train-only buffers.
            model.pre_features.zero_()
            model.pre_logits.fill_(100)
            self.assertTrue(torch.equal(prediction, model(xs[0])))
            learner.assert_not_called()
        self.assertFalse(torch.equal(history, model.pre_features))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable locally")
    def test_cuda_active_update(self):
        model = make_model().cuda().train()
        result = model.update(*batches("cuda"))
        self.assertTrue(all(math.isfinite(value) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
