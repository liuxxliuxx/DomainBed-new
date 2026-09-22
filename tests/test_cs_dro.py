"""Offline CS-DRO integration tests: no datasets or weights downloaded."""
import copy
import itertools
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from munch import Munch
from sconf import Config

try:
    import open_clip  # noqa: F401
except ModuleNotFoundError:
    # Unrelated optional backbone; mirror the existing SFT test setup.
    sys.modules["open_clip"] = types.ModuleType("open_clip")

import run_all
from domainbed import hparams_registry, networks, trainer
from domainbed.algorithms import get_algorithm_class
from domainbed.models.cs_dro import LightEncoder, NotearsClassifier
from domainbed.lib.swa_utils import AveragedModel
from domainbed.swad import LossValley


class ToyFeatures(nn.Sequential):
    n_outputs = 4

    def __init__(self):
        super().__init__(nn.Linear(2, 4), nn.Tanh())


def toy(**overrides):
    hp = hparams_registry.default_hparams("CS_DRO", "PACS")
    hp.update(hidden_size=4, out_dim=4, adv_steps=2, gauss_k=2,
              gauss_min_count=1, pretrained=False)
    hp.update(overrides)
    with mock.patch("domainbed.networks.Featurizer", side_effect=lambda *a: ToyFeatures()):
        return get_algorithm_class("CS_DRO")((2,), 3, 2, hp)


def batches():
    return ([torch.randn(6, 2), torch.randn(6, 2)],
            [torch.tensor([0, 1, 2, 0, 1, 2])] * 2)


def prepare_dag(model, x, y):
    model.update(x, y, step=50)
    model.update_dag()


class CSDROTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(12)

    def test_registry_and_run_all(self):
        from domainbed.algorithms import algorithms as legacy_algorithms
        self.assertIs(get_algorithm_class("CS_DRO"), legacy_algorithms.CS_DRO)
        expected = {"PACS": (256, 10., 1.), "VLCS": (512, 10., 1.),
                    "OfficeHome": (512, 1., .001), "TerraIncognita": (512, .1, 10.)}
        for dataset, values in expected.items():
            hp = hparams_registry.default_hparams("CS_DRO", dataset)
            self.assertEqual((hp["out_dim"], hp["lambda_G"], hp["kappa"]), values)
            self.assertEqual(hp["hidden_size"], hp["out_dim"])
        root = Path(__file__).resolve().parents[1]
        for name, swad in (("CS_DRO", "False"), ("CS_DRO_SWAD", "LossValley")):
            command = run_all.build_command(root, name, 0, 32, "PACS", "5000")
            self.assertEqual(command[command.index("--algorithm") + 1], "CS_DRO")
            self.assertEqual(command[command.index("--swad") + 1], swad)
        config = Config(str(root / "config.yaml"), default=hp)
        self.assertEqual(config.swad, "LossValley")
        config.argv_update(["--swad", "False"])
        self.assertIs(config.swad, False)

    def test_networks_are_isolated_from_idag(self):
        self.assertIsNot(LightEncoder, networks.LightEncoder)
        self.assertIsNot(NotearsClassifier, networks.NotearsClassifier)
        enc = LightEncoder(8, 4, 4)
        self.assertEqual([type(m) for m in enc.encoder], [nn.Linear, nn.LayerNorm])
        dag = NotearsClassifier(4, 3)
        with torch.no_grad():
            dag.weight_pos.fill_(.2)
        dag.projection()
        torch.testing.assert_close(dag._adj_sub(), torch.sqrt(dag._adj() ** 2 + 1e-10))
        torch.testing.assert_close(dag.w_l1_reg(), (dag.weight_pos + dag.weight_neg).abs().sum())

    def test_validation_and_empty_statistics(self):
        for hp in ({"hidden_size": 5}, {"num_hidden_layers": 1}, {"adv_steps": 0}):
            with self.assertRaises(ValueError):
                toy(**hp)
        with self.assertRaisesRegex(RuntimeError, "Gaussian statistics"):
            toy().update_dag()

    def test_warmup_dag_robust_update_and_clone(self):
        model = toy()
        x, y = batches()
        self.assertEqual(model.SWAD_START_STEP, 301)
        model.update(x, y, step=49)
        self.assertEqual(model.gauss_count.sum().item(), 0)
        prepare_dag(model, x, y)
        self.assertEqual(model.counter, 1)
        self.assertFalse(torch.equal(model.m_ema, torch.ones_like(model.m_ema)))
        self.assertTrue(all(np.isfinite(v) for v in model.update_dag().values()))
        fixed_mask = model.m_ema.clone()
        counts = model.gauss_count.clone()
        before = model.clf.weight.detach().clone()
        for step in (300, 301, 302):
            metrics = model.update(x, y, step=step)
            self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertFalse(torch.equal(before, model.clf.weight))
        torch.testing.assert_close(model.m_ema, fixed_mask)
        torch.testing.assert_close(model.gauss_count, counts + 4)
        self.assertTrue(all(p.requires_grad for p in model.clf.parameters()))
        self.assertEqual(model.predict(x[0]).shape, (6, 3))
        cloned = model.clone()
        owned = {id(p) for p in cloned.parameters()}
        for opt in (cloned.optimizer, cloned.opt_dag):
            self.assertTrue(all(id(p) in owned for group in opt.param_groups for p in group['params']))
        original = copy.deepcopy(model.state_dict())
        cloned.update(x, y, step=303)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, original[key])
        restored = toy()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.eval().predict(x[0]), model.eval().predict(x[0]))

    def test_full_covariance_sampling(self):
        model = toy(full_cov=True)
        x, y = batches()
        prepare_dag(model, x, y)
        z, ys, ds = model.sample_from_gaussians()
        self.assertEqual(z.shape, (12, 4))
        self.assertEqual(ys.shape, ds.shape)
        self.assertTrue(torch.isfinite(z).all())

    def test_radius_search_and_grad_scope(self):
        model = toy()
        x, y = batches()
        with torch.no_grad():
            z, labels = model.get_emb(x, y)
        result = model.recompute_rho_adv_ref(z, labels)
        self.assertEqual(result.ndim, 1)
        self.assertTrue(all(p.requires_grad for p in model.clf.parameters()))
        stream = itertools.repeat([{"x": xi, "y": yi} for xi, yi in zip(x, y)])
        def radius(zz, yy):
            self.assertTrue(torch.is_grad_enabled())
            self.assertFalse(zz.requires_grad)
            self.assertFalse(model.training)
            return np.array([4., 4.])
        with mock.patch.object(model, "recompute_rho_adv_ref", side_effect=radius):
            trainer._refresh_cs_dro_radius(model, stream, 2, "cpu", 0)
        self.assertAlmostEqual(model.target_rho, .65)
        self.assertTrue(model.training)
        with mock.patch.object(model, "recompute_rho_adv_ref", return_value=np.array([np.inf])):
            trainer._refresh_cs_dro_radius(model, stream, 1, "cpu", 1)
        self.assertAlmostEqual(model.target_rho, .65)
        with mock.patch.object(model, "recompute_rho_adv_ref", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                trainer._refresh_cs_dro_radius(model, stream, 1, "cpu", 1)
        self.assertTrue(model.training)

    def test_nested_loss_valley_preserves_fixed_mask(self):
        model = toy()
        x, y = batches()
        prepare_dag(model, x, y)
        mask = model.m_ema.clone()
        valley = LossValley(None, n_converge=2, n_tolerance=3, tolerance_ratio=.3)
        for step in range(301, 306):
            model.update(x, y, step=step)
            segment = AveragedModel(model)
            segment.update_parameters(model, step=step)
            valley.update_and_evaluate(segment, .5, 1., lambda *a: None)
        # Existing LossValley returns .cuda(); suppress only the device transfer.
        with mock.patch.object(AveragedModel, "cuda", autospec=True, side_effect=lambda m: m):
            final = valley.get_final_model()
        torch.testing.assert_close(final.module.m_ema, mask)
        self.assertGreaterEqual(final.start_step, 301)
        self.assertTrue(torch.isfinite(final.predict(x[0])).all())

    def test_real_resnet_forward_and_robust_backward(self):
        hp = hparams_registry.default_hparams("CS_DRO", "PACS")
        hp.update(pretrained=False, resnet18=True, hidden_size=4, out_dim=4,
                  adv_steps=1, gauss_k=2, gauss_min_count=1)
        model = get_algorithm_class("CS_DRO")((3, 64, 64), 3, 2, hp)
        x = [torch.randn(3, 3, 64, 64) for _ in range(2)]
        y = [torch.arange(3)] * 2
        prepare_dag(model, x, y)
        metrics = model.update(x, y, step=301)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertEqual(model.predict(x[0]).shape, (3, 3))


class TrainerScheduleTests(unittest.TestCase):
    def run_trainer(self, n_steps, checkpoint_freq=100, algorithm="CS_DRO", swad="LossValley"):
        class Dataset(list):
            input_shape = (2,)
            num_classes = 3
            N_WORKERS = 0
            environments = ["a", "b", "c"]

        class Model(nn.Module):
            SWAD_START_STEP = 301
            def __init__(self, *args):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
                self.register_buffer("m_ema", torch.ones(4))
                self.dag_steps = []
                self.steps = []
            def update(self, **inputs):
                self.steps.append(inputs["step"])
                return {"loss": 1.}
            def update_dag(self):
                self.dag_steps.append(self.steps[-1])
                self.m_ema.fill_(.5)
                return {"rec": 1.}

        model = Model()
        dataset = Dataset([list(range(4)) for _ in range(3)])
        splits = [(env, None) for env in dataset]
        evaluator = mock.Mock()
        evaluator.evaluate.return_value = ({"test_in": .5, "test_out": .5},
                                           {"train_out": .5, "tr_outloss": 1.})
        observed_starts = []
        original_update = LossValley.update_and_evaluate
        def observe(valley, segment, *args):
            observed_starts.append(segment.start_step)
            if algorithm == "CS_DRO":
                torch.testing.assert_close(segment.module.m_ema, torch.full((4,), .5))
            return original_update(valley, segment, *args)

        with tempfile.TemporaryDirectory() as tmp:
            hp = Munch(batch_size=2, test_batchsize=2, indomain_test=False,
                       val_augment=False, swad=swad, freeze_bn=True, logdir=Path(tmp),
                       swad_kwargs=Munch(n_converge=2, n_tolerance=3, tolerance_ratio=.3))
            args = types.SimpleNamespace(algorithm=algorithm, evalmode="fast", prebuild_loader=False,
                                         saliency=False, saliency_every=0, out_dir=Path(tmp),
                                         debug=False, model_save=None, tb_freq=10)
            batch = {"x": torch.zeros(2, 2), "y": torch.tensor([0, 1])}
            with mock.patch.object(trainer, "get_dataset", return_value=(dataset, splits, splits)), \
                 mock.patch.object(trainer.algorithms, "get_algorithm_class", return_value=lambda *a: model), \
                 mock.patch.object(trainer, "InfiniteDataLoader", side_effect=lambda **kw: itertools.repeat(batch)), \
                 mock.patch.object(trainer, "Evaluator", return_value=evaluator), \
                 mock.patch.object(trainer, "_refresh_cs_dro_radius") as radius, \
                 mock.patch.object(trainer.torch.cuda, "max_memory_allocated", return_value=0), \
                 mock.patch.object(LossValley, "update_and_evaluate", new=observe), \
                 mock.patch.object(AveragedModel, "cuda", autospec=True, side_effect=lambda m: m):
                ret, _ = trainer.train(None, [2], args, hp, n_steps, 2000, 0,
                                       checkpoint_freq, mock.Mock(), mock.Mock())
                calls = radius.call_count
        return model, ret, observed_starts, calls

    def test_phase_boundaries_and_swad(self):
        model, ret, starts, calls = self.run_trainer(601)
        self.assertEqual(model.dag_steps, [s for s in range(101, 300) for _ in range(3)])
        self.assertEqual(starts, [301, 401, 501])
        self.assertIn("SWAD", ret)
        self.assertEqual(calls, len(range(0, 601, 8)))

    def test_short_runs_and_swad_disabled(self):
        for steps in (10, 301, 302):
            with self.subTest(steps=steps):
                _, ret, starts, _ = self.run_trainer(steps)
                self.assertNotIn("SWAD", ret)
                self.assertEqual(starts, [])
        _, ret, starts, _ = self.run_trainer(402, swad=False)
        self.assertNotIn("SWAD", ret)
        self.assertEqual(starts, [])

    def test_other_algorithms_keep_original_swad_schedule(self):
        model, ret, starts, calls = self.run_trainer(201, algorithm="ERM")
        self.assertEqual(model.dag_steps, [])
        self.assertEqual(calls, 0)
        self.assertEqual(starts, [0, 1, 101])
        self.assertIn("SWAD", ret)


if __name__ == "__main__":
    unittest.main()
