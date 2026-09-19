"""Offline SFT formula, gradient, state, backbone and SWAD tests.

No dataset or pretrained weights are downloaded. Finite differences keep the
projection target fixed, exactly as the alternating PCE update requires.
"""
import copy
import io
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import VisionTransformer

try:
    import open_clip  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open_clip"] = types.ModuleType("open_clip")

import run_all
from domainbed import hparams_registry
from domainbed.algorithms import get_algorithm_class
from domainbed.lib import sft
from domainbed.lib.swa_utils import AveragedModel, update_bn
from domainbed.swad import LossValley


def hparams(**overrides):
    hp = hparams_registry.default_hparams("SFT", "PACS")
    hp.update(pretrained=False, resnet18=True, batch_size=2, image_size=64)
    hp.update(overrides)
    return hp


class ToyFeatures(nn.Sequential):
    n_outputs = 4

    def __init__(self, dropout=0.0, bn=False):
        super().__init__(nn.Linear(2, 4), nn.BatchNorm1d(4) if bn else nn.Identity(),
                         nn.Tanh(), nn.Dropout(dropout))


def toy(num_domains=3, dropout=0.0, bn=False, **overrides):
    with mock.patch("domainbed.networks.Featurizer",
                    side_effect=lambda *a: ToyFeatures(dropout, bn)):
        return get_algorithm_class("SFT")((2,), 3, num_domains, hparams(**overrides)).double()


def batches(device="cpu"):
    return ([torch.randn(n, 2, device=device, dtype=torch.double) for n in (3, 2, 4)],
            [torch.arange(n, device=device) % 3 for n in (3, 2, 4)])


def small_vit(weights=None, image_size=64, **kwargs):
    return VisionTransformer(image_size=image_size, patch_size=16, num_layers=12,
                             num_heads=4, hidden_dim=32, mlp_dim=64, num_classes=1000)


class SFTTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(140)

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name in expected:
            self.assertTrue(torch.equal(expected[name], actual[name]), name)


class ProjectionTests(SFTTestCase):
    def test_identity_constraints_ties_permutations_and_detach(self):
        logits = torch.tensor([[6., 0., -1.], [0., 0., 0.], [0., 3., 3.]],
                              dtype=torch.double, requires_grad=True)
        labels = torch.tensor([0, 1, 0])
        q = sft.project_labels(logits, labels, 3.0)
        torch.testing.assert_close(q[0], logits[0].softmax(-1))
        torch.testing.assert_close(q[1], torch.tensor([.2, .6, .2], dtype=torch.double))
        torch.testing.assert_close(q[2], torch.tensor([.6, .2, .2], dtype=torch.double))
        self.assertFalse(q.requires_grad)
        perm = torch.tensor([2, 0, 1])
        inverse = perm.argsort()
        qp = sft.project_labels(logits[:, perm], inverse[labels], 3.0)
        torch.testing.assert_close(qp[:, inverse], q)
        uniform = sft.project_labels(torch.zeros(3, 3), torch.arange(3), 1.)
        torch.testing.assert_close(uniform, torch.full((3, 3), 1 / 3))

    def test_matches_independent_constrained_solver(self):
        from scipy.optimize import minimize
        for classes in (2, 4, 7):
            for alpha in (1., 2., 10.):
                logits = torch.randn(4, classes, dtype=torch.double)
                labels = torch.arange(4) % classes
                actual = sft.project_labels(logits, labels, alpha).numpy()
                for index, (p, label) in enumerate(zip(logits.softmax(-1).numpy(), labels.tolist())):
                    matrix = np.zeros((classes - 1, classes))
                    for row, other in enumerate(i for i in range(classes) if i != label):
                        matrix[row, label], matrix[row, other] = 1., -alpha
                    start = np.full(classes, 1. / (alpha + classes - 1))
                    start[label] *= alpha
                    result = minimize(
                        lambda q: np.sum(q * np.log(q / p)), start,
                        jac=lambda q: np.log(q / p) + 1,
                        bounds=[(1e-12, 1)] * classes,
                        constraints=[
                            {"type": "eq", "fun": lambda q: q.sum() - 1,
                             "jac": lambda q: np.ones_like(q)},
                            {"type": "ineq", "fun": lambda q: matrix @ q,
                             "jac": lambda q: matrix}],
                        method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
                    self.assertTrue(result.success, result.message)
                    np.testing.assert_allclose(actual[index], result.x, atol=2e-6, rtol=2e-6)

    def test_extreme_logits_many_classes_and_pce_gradient(self):
        for dtype in (torch.float32, torch.float64):
            for alpha in (1., 10., 1000.):
                logits = (torch.randn(5, 345, dtype=dtype) * 2000).requires_grad_()
                labels = torch.tensor([0, 1, 50, 200, 344])
                q = sft.project_labels(logits, labels, alpha)
                self.assertTrue(torch.isfinite(q).all())
                self.assertTrue((q >= 0).all())
                torch.testing.assert_close(q.sum(1), torch.ones(5, dtype=dtype))
                true = q.gather(1, labels[:, None])
                other = q.scatter(1, labels[:, None], 0.)
                self.assertTrue((true + 2e-4 >= alpha * other).all())
                loss = sft.projection_cross_entropy(logits, labels, alpha)
                grad = torch.autograd.grad(loss, logits)[0]
                self.assertTrue(torch.isfinite(loss))
                torch.testing.assert_close(grad, (logits.softmax(1) - q) / len(logits))

    def test_feasible_pce_has_zero_logit_gradient(self):
        logits = torch.tensor([[5., 0., -2.]], dtype=torch.double, requires_grad=True)
        loss = sft.projection_cross_entropy(logits, torch.tensor([0]), 2.)
        torch.testing.assert_close(torch.autograd.grad(loss, logits)[0], torch.zeros_like(logits))

    def test_large_logit_gap_does_not_round_away_alpha_ratio(self):
        logits = torch.tensor([[-1e8, 0., 0.], [0., -1e8, 0.]])
        labels = torch.tensor([0, 1])
        for alpha in (1., 10., 1000.):
            q = sft.project_labels(logits, labels, alpha)
            expected = torch.full_like(q, 1. / (alpha + 2))
            expected.scatter_(1, labels[:, None], alpha / (alpha + 2))
            torch.testing.assert_close(q, expected)

    def test_invalid_projection(self):
        logits, labels = torch.zeros(2, 3), torch.tensor([0, 1])
        for alpha in (0., .9, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                sft.project_labels(logits, labels, alpha)
        for bad in (torch.tensor([-1, 0]), torch.tensor([0, 3]), labels.float(), labels[:1]):
            with self.assertRaises(ValueError):
                sft.project_labels(logits, bad, 2.)
        with self.assertRaises(ValueError):
            sft.project_labels(logits + float("nan"), labels, 2.)


class FeedbackTests(SFTTestCase):
    def test_global_perturbation_norm_unused_and_zero_gradients(self):
        parameters = [torch.zeros(2), torch.zeros(3), torch.zeros(1)]
        gradients = [torch.tensor([3., 0.]), torch.tensor([0., 4., 0.]), None]
        delta = sft.normalized_perturbation(parameters, gradients, .2)
        torch.testing.assert_close(delta[0], torch.tensor([.12, 0.]))
        torch.testing.assert_close(delta[1], torch.tensor([0., .16, 0.]))
        self.assertEqual(delta[2].item(), 0.)
        zero = torch.zeros(3, requires_grad=True)
        eps = sft.normalized_perturbation([zero], [zero], .1)[0]
        eps.sum().backward()
        self.assertTrue(torch.equal(eps, torch.zeros(3)))
        self.assertTrue(torch.isfinite(zero.grad).all())

    def test_refinement_finite_difference_including_epsilon(self):
        model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 3)).double()
        xs = [torch.randn(3, 2, dtype=torch.double),
              torch.randn(2, 2, dtype=torch.double) * 2 + 1]
        labels = torch.tensor([0, 1, 2, 0, 1])
        phi = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
        fixed_q = sft.project_labels(phi, labels, 3.)

        def objective(value):
            targets = value.softmax(-1).split([3, 2])
            a = sft.sharpness(model, xs[0], targets[0], .15)
            b = sft.sharpness(model, xs[1], targets[1], .15)
            return sft.soft_cross_entropy(value, fixed_q) + .5 * a + .7 * (a - b).abs()

        actual = torch.autograd.grad(objective(phi), phi)[0]
        expected = torch.empty_like(phi)
        eps = 1e-5
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                plus, minus = phi.detach().clone(), phi.detach().clone()
                plus[i, j] += eps
                minus[i, j] -= eps
                expected[i, j] = (objective(plus) - objective(minus)).item() / (2 * eps)
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_stochastic_pair_replays_rng_and_isolates_buffers(self):
        model = nn.Sequential(ToyFeatures(dropout=.4, bn=True), nn.Linear(4, 3)).double()
        expected = copy.deepcopy(model)
        x = torch.randn(6, 2, dtype=torch.double)
        targets = torch.randn(6, 3, dtype=torch.double).softmax(-1).requires_grad_()
        state = copy.deepcopy(model.state_dict())
        modes = [m.training for m in model.modules()]
        rng = torch.get_rng_state()
        expected(x)
        after_one = torch.get_rng_state()
        torch.set_rng_state(rng)
        base, perturbed = sft.paired_losses(
            model, x, targets, .1, create_graph=True, detach_parameters=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), after_one))
        self.assert_state_equal(state, model.state_dict())
        self.assertEqual(modes, [m.training for m in model.modules()])
        gradient = torch.autograd.grad(perturbed - base, targets)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

        # Zero epsilon must give exactly zero sharpness, even with dropout and BN.
        with mock.patch.object(sft, "normalized_perturbation",
                               side_effect=lambda ps, gs, r: [torch.zeros_like(p) for p in ps]):
            a, b = sft.paired_losses(model, x, targets, .1, create_graph=True)
        self.assertEqual((a - b).item(), 0.)
        torch.set_rng_state(rng)
        sft.paired_losses(model, x, targets.detach(), .1, update_buffers=True)
        self.assert_state_equal(dict(expected.named_buffers()), dict(model.named_buffers()))
        self.assertEqual(model[0][1].num_batches_tracked.item(), 1)

    def test_probe_failure_does_not_leave_perturbed_weights_buffers_or_rng(self):
        model = nn.Sequential(ToyFeatures(dropout=.3, bn=True), nn.Linear(4, 3)).double()
        x, targets = torch.randn(5, 2, dtype=torch.double), torch.ones(5, 3, dtype=torch.double) / 3
        before = copy.deepcopy(model.state_dict())
        ids = [id(p) for p in model.parameters()]
        rng = torch.get_rng_state()
        copy.deepcopy(model)(x)
        after_one = torch.get_rng_state()
        torch.set_rng_state(rng)
        forward = model.forward
        calls = []

        def fail_second(value):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("probe failed")
            return forward(value)

        with mock.patch.object(model, "forward", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                sft.paired_losses(model, x, targets, .1, update_buffers=True)
        self.assert_state_equal(before, model.state_dict())
        self.assertEqual(ids, [id(p) for p in model.parameters()])
        self.assertTrue(torch.equal(after_one, torch.get_rng_state()))


class AlgorithmTests(SFTTestCase):
    def test_exact_sam_step_post_update_feedback_and_all_source_pce(self):
        model = toy(optimizer="sgd", lr=.03)
        x, y = batches()
        reference = copy.deepcopy(model.network)
        teacher_before = copy.deepcopy(model.refiner.state_dict())
        with torch.no_grad():
            teacher_logits = model.refiner(torch.cat(x))
            targets = teacher_logits.softmax(-1).split([len(a) for a in x])
            pce = sft.projection_cross_entropy(teacher_logits, torch.cat(y), model.sft_alpha)
        parameters = list(reference.parameters())
        originals = [p.detach().clone() for p in parameters]
        loss = -(targets[2] * reference(x[2]).log_softmax(-1)).sum(1).mean()
        gradients = torch.autograd.grad(loss, parameters)
        norm = torch.cat([g.flatten() for g in gradients]).norm()
        with torch.no_grad():
            for p, g in zip(parameters, gradients):
                p.add_(model.sft_rho * g / norm)
        perturbed_loss = -(targets[2] * reference(x[2]).log_softmax(-1)).sum(1).mean()
        sam_gradients = torch.autograd.grad(perturbed_loss, parameters)
        expected = [p - .03 * g for p, g in zip(originals, sam_gradients)]
        feedback_domains = []
        original_sharpness = sft.sharpness

        def checked_sharpness(network, xi, ti, rho, **kwargs):
            for actual, want in zip(network.parameters(), expected):
                torch.testing.assert_close(actual, want, atol=1e-12, rtol=1e-12)
            feedback_domains.append(next(i for i, tensor in enumerate(x) if tensor is xi))
            return original_sharpness(network, xi, ti, rho, **kwargs)

        with mock.patch("torch.randperm", return_value=torch.tensor([2, 0, 1])), \
                mock.patch.object(sft, "sharpness", side_effect=checked_sharpness):
            metrics = model.update(x, y)
        self.assertEqual(feedback_domains, [2, 0])
        self.assertAlmostEqual(metrics["loss"], perturbed_loss.item(), places=12)
        self.assertAlmostEqual(metrics["sft_pce"], pce.item(), places=12)
        self.assertAlmostEqual(metrics["sft_refiner_loss"], metrics["sft_pce"]
                               + .5 * metrics["sft_sharpness_train"]
                               + .5 * metrics["sft_sharpness_gap"], places=12)
        self.assertFalse(all(torch.equal(value, model.refiner.state_dict()[key])
                             for key, value in teacher_before.items()))
        self.assertTrue(all(p.grad is None for p in model.network.parameters()))

    def test_phase_optimizer_isolation(self):
        model = toy()
        x, y = batches()
        refiner_before = copy.deepcopy(model.refiner.state_dict())
        student_after = {}
        student_step, refiner_step = model.optimizer.step, model.refiner_optimizer.step

        def checked_student_step():
            self.assertTrue(all(p.grad is None for p in model.refiner.parameters()))
            self.assert_state_equal(refiner_before, model.refiner.state_dict())
            student_step()
            student_after.update(copy.deepcopy(model.network.state_dict()))

        def checked_refiner_step():
            self.assertTrue(all(p.grad is None for p in model.network.parameters()))
            refiner_step()
            self.assert_state_equal(student_after, model.network.state_dict())

        with mock.patch.object(model.optimizer, "step", side_effect=checked_student_step), \
                mock.patch.object(model.refiner_optimizer, "step", side_effect=checked_refiner_step):
            model.update(x, y)

    def test_feedback_disabled_equals_pce_only_refiner_update(self):
        model = toy(optimizer="sgd", lr=.03, sft_lambda1=0., sft_lambda2=0.)
        reference = copy.deepcopy(model.refiner)
        x, y = batches()
        optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
        loss = sft.projection_cross_entropy(reference(torch.cat(x)), torch.cat(y), model.sft_alpha)
        loss.backward()
        optimizer.step()
        metrics = model.update(x, y)
        self.assert_state_equal(reference.state_dict(), model.refiner.state_dict())
        self.assertEqual(metrics["sft_pce"], metrics["sft_refiner_loss"])

    def test_rho_zero_matches_soft_label_step_and_zero_sharpness(self):
        model = toy(optimizer="sgd", lr=.03, sft_rho=0.)
        reference = copy.deepcopy(model.network)
        x, y = batches()
        with torch.no_grad():
            targets = model.refiner(torch.cat(x)).softmax(-1)[:len(x[0])]
        optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
        sft.soft_cross_entropy(reference(x[0]), targets).backward()
        optimizer.step()
        with mock.patch("torch.randperm", return_value=torch.tensor([0, 1, 2])):
            metrics = model.update(x, y)
        self.assert_state_equal(reference.state_dict(), model.network.state_dict())
        for key in ("sft_sharpness_train", "sft_sharpness_other", "sft_sharpness_gap"):
            self.assertEqual(metrics[key], 0.)

    def test_bn_updates_once_each_and_predict_never_calls_refiner(self):
        model = toy(bn=True, dropout=.25)
        x, y = batches()
        flags = [m.training for m in model.modules()]
        model.update(x, y)
        self.assertEqual(model.network[0][1].num_batches_tracked.item(), 1)
        self.assertEqual(model.refiner[0][1].num_batches_tracked.item(), 1)
        self.assertEqual(flags, [m.training for m in model.modules()])
        model.eval()
        rng = torch.get_rng_state()
        with mock.patch.object(model.refiner, "forward", side_effect=AssertionError("inference refiner")):
            a, b = model.predict(x[0]), model.predict(x[0])
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_checkpoint_clone_optimizers_and_deterministic_next_step(self):
        model = toy()
        x, y = batches()
        model.update(x, y)
        clone = model.clone()
        self.assert_state_equal(model.state_dict(), clone.state_dict())
        for name in ("network", "refiner"):
            self.assertTrue(all(a is not b for a, b in zip(
                getattr(model, name).parameters(), getattr(clone, name).parameters())))
        for name, optimizer in (("network", clone.optimizer), ("refiner", clone.refiner_optimizer)):
            self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]},
                             {id(p) for p in getattr(clone, name).parameters()})
            original_optimizer = model.optimizer if name == "network" else model.refiner_optimizer
            for a, b in zip(original_optimizer.state.values(), optimizer.state.values()):
                for key in a:
                    if torch.is_tensor(a[key]):
                        self.assertNotEqual(a[key].data_ptr(), b[key].data_ptr())
        buffer = io.BytesIO()
        torch.save({"model_dict": model.state_dict(), "model_hparams": model.hparams}, buffer)
        buffer.seek(0)
        restored = toy()
        restored.load_state_dict(torch.load(buffer)["model_dict"], strict=True)
        self.assert_state_equal(model.state_dict(), restored.state_dict())
        rng = torch.get_rng_state()
        metrics = model.update(x, y)
        torch.set_rng_state(rng)
        self.assertEqual(metrics, clone.update(x, y))
        self.assert_state_equal(model.state_dict(), clone.state_dict())

    def test_invalid_domains_hyperparameters_and_batches_fail_before_update(self):
        with self.assertRaises(ValueError):
            toy(num_domains=1)
        for key, bad in (("sft_rho", -1), ("sft_alpha", .9), ("sft_lambda1", -1),
                         ("sft_lambda2", float("nan")), ("sft_rho", float("inf"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                toy(**{key: bad})
        model = toy()
        x, y = batches()
        before = copy.deepcopy(model.state_dict())
        for bad_x, bad_y in ((x[:1], y[:1]), (x, [y[0].float(), *y[1:]]),
                             ([x[0][:0], *x[1:]], [y[0][:0], *y[1:]]),
                             (x, [y[0] + 3, *y[1:]])):
            with self.assertRaises(ValueError):
                model.update(bad_x, bad_y)
        self.assert_state_equal(before, model.state_dict())


class IntegrationTests(SFTTestCase):
    def test_registry_runner_defaults_and_aliases(self):
        hp = hparams()
        self.assertEqual([hp[k] for k in ("sft_rho", "sft_alpha", "sft_lambda1", "sft_lambda2")],
                         [.05, 10., .5, .5])
        for key in ("lr", "weight_decay", "optimizer", "resnet_dropout"):
            self.assertEqual(hp[key], hparams_registry.default_hparams("ERM", "PACS")[key])
        root = Path(__file__).resolve().parents[1]
        for method, swad in (("SFT", "False"), ("SFT_SWAD", "LossValley")):
            for backbone in ("resnet", "vit"):
                command = run_all.build_command(root, method, 0, 8, "HTP", "5000", backbone)
                self.assertEqual(command[command.index("--algorithm") + 1], "SFT")
                self.assertEqual(command[command.index("--swad") + 1], swad)
            self.assertEqual(run_all.parse_algorithm(method.lower()), method)

    def test_swad_only_averages_classifier_and_supports_nested_segments(self):
        model = toy()
        x, y = batches()
        model.update(x, y)  # Populate both optimizers before copying.
        averaged = AveragedModel(model)
        self.assertFalse(hasattr(averaged.module, "refiner"))
        self.assertFalse(hasattr(averaged.module, "optimizer"))
        self.assertFalse(hasattr(averaged.module, "refiner_optimizer"))
        self.assertEqual(sum(p.numel() for p in averaged.parameters()),
                         sum(p.numel() for p in model.network.parameters()))
        averaged.update_parameters(model, step=0)
        before = [p.detach().clone() for p in model.network.parameters()]
        with torch.no_grad():
            for p in model.network.parameters():
                p.add_(.2)
            for p in model.refiner.parameters():
                p.add_(100.)
        averaged.update_parameters(model, step=1)
        for actual, original in zip(averaged.parameters(), before):
            torch.testing.assert_close(actual, original + .1)
        self.assertEqual((averaged.start_step, averaged.end_step), (0, 1))
        nested = AveragedModel(averaged)
        nested.update_parameters(averaged, step=2)
        cloned = nested.clone().eval()
        nested.eval()
        torch.testing.assert_close(cloned(x[0]), nested.predict(x[0]))
        restored = AveragedModel(toy())
        restored.load_state_dict(averaged.state_dict(), strict=True)
        self.assert_state_equal(averaged.state_dict(), restored.state_dict())

    def test_loss_valley_and_bn_recompute_use_inference_view(self):
        model = toy(bn=True)
        x, y = batches()
        valley = LossValley(None, n_converge=2, n_tolerance=3, tolerance_ratio=.3)
        for step in range(5):
            model.update(x, y)
            segment = AveragedModel(model)
            segment.update_parameters(model, step=step)
            valley.update_and_evaluate(segment, .5, 1., lambda *args: None)
        # The existing LossValley hard-codes .cuda() on return. Only suppress
        # this device transfer; exercise its actual queues/selection/averaging.
        with mock.patch.object(AveragedModel, "cuda", autospec=True, side_effect=lambda m: m):
            final = valley.get_final_model()
        self.assertIsNotNone(final)
        self.assertFalse(hasattr(final.module, "refiner"))
        teacher_before = copy.deepcopy(model.refiner.state_dict())
        iterator = iter([[{"x": xi, "y": yi} for xi, yi in zip(x, y)] for _ in range(2)])
        update_bn(iterator, final, 2, device="cpu")
        self.assert_state_equal(teacher_before, model.refiner.state_dict())
        self.assertTrue(torch.isfinite(final.eval()(x[0])).all())

    def test_resnet18_resnet50_and_real_transformer_multistep(self):
        xs = [torch.randn(2, 3, 64, 64) for _ in range(2)]
        ys = [torch.tensor([0, 1]), torch.tensor([1, 2])]
        for backbone, resnet18 in (("resnet", True), ("resnet", False), ("vit", False)):
            with self.subTest(backbone=backbone, resnet18=resnet18), \
                    mock.patch("torchvision.models.vit_b_16", side_effect=small_vit):
                model = get_algorithm_class("SFT")(
                    (3, 64, 64), 3, 2, hparams(backbone=backbone, resnet18=resnet18))
                first = next(model.network.parameters()).detach().clone()
                refiner_first = next(model.refiner.parameters()).detach().clone()
                buffers = {n: b.clone() for n, b in model.named_buffers()}
                for step in range(2):
                    metrics = model.update(xs, ys, step=step)
                    self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
                self.assertFalse(torch.equal(first, next(model.network.parameters())))
                self.assertFalse(torch.equal(refiner_first, next(model.refiner.parameters())))
                self.assert_state_equal(buffers, dict(model.named_buffers()))  # freeze_bn=True
                model.eval()
                averaged = AveragedModel(model)
                averaged.update_parameters(model, step=2)
                restored = get_algorithm_class("SFT")(
                    (3, 64, 64), 3, 2, hparams(backbone=backbone, resnet18=resnet18)).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    actual = model.predict(xs[0])
                    self.assertEqual(actual.shape, (2, 3))
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(actual, averaged(xs[0]))
                    torch.testing.assert_close(actual, restored(xs[0]))
                    torch.testing.assert_close(model.refiner(xs[0]), restored.refiner(xs[0]))
                del model, averaged, restored

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_update_and_rng_replay(self):
        model = toy(dropout=.3, bn=True).cuda()
        x, y = batches("cuda")
        metrics = model.update(x, y)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        with mock.patch.object(sft, "normalized_perturbation",
                               side_effect=lambda ps, gs, r: [torch.zeros_like(p) for p in ps]):
            a, b = sft.paired_losses(model.network, x[0], model.refiner(x[0]).softmax(-1),
                                     .1, create_graph=True, detach_parameters=True)
        self.assertEqual(a.item(), b.item())


if __name__ == "__main__":
    unittest.main()
