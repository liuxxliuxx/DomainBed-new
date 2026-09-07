"""Offline CPU tests plus CUDA-only smoke tests for existing CUDA-only updates.

Run: python -m unittest discover -s tests -p test_vit_backbone.py -v
Most tests use a small 12-block torchvision ViT, not a fake featurizer. A separate
test exercises the real 768-dimensional ViT-B/16. No pretrained download occurs.
"""
import ast
import copy
import inspect
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
import torchvision
from torchvision.models.vision_transformer import VisionTransformer

try:
    import open_clip  # noqa: F401
except ModuleNotFoundError:
    sys.modules["open_clip"] = types.ModuleType("open_clip")

import run_all
from domainbed import hparams_registry, networks
from domainbed.algorithms import algorithms as algorithms
from domainbed.models.vit import ViT, MathSelfAttention, TokenGridAdapter
from domainbed.models.aloft import ALOFT, resnet_aloft, find_aloft_modules
from domainbed.models.awwsl import AWWSL
from domainbed.models.frequant import FreqQuant
from domainbed.models.aloft_cb import BandStatsCodebook
from domainbed.models.aloft_sketch import SketchSpectrumPerturb
from domainbed.models.aloft_structure import StructureProbe
from domainbed.lib.swa_utils import AveragedModel, update_bn
from domainbed.quan.vit import (
    QuantLinear, QuantPatchConv, QuantSelfAttention,
    prepare_vit_quantization, load_vit_checkpoint,
)


def small_vit(weights=None, image_size=224, **kwargs):
    return VisionTransformer(
        image_size=image_size, patch_size=16, num_layers=12, num_heads=4,
        hidden_dim=32, mlp_dim=64, num_classes=1000)


def params(name, logdir, backbone="vit"):
    result = hparams_registry.default_hparams(name, "PACS")
    result.update(
        backbone=backbone, pretrained=False, resnet_dropout=0.0,
        grad_fn="sum", logdir=Path(logdir), test_env=0, image_size=64,
        batch_size=4, neighborhoodSize=1e-6, start_step=100, end_step=200,
        extra_search="n", extra_search_start=None, extra_search_end=None,
        gga_l_gamma=0.001, annealing_patience=3, swad=False,
    )
    return result


def algorithm_names():
    return sorted(name for name, cls in vars(algorithms).items()
                  if inspect.isclass(cls) and issubclass(cls, algorithms.Algorithm)
                  and cls is not algorithms.Algorithm
                  and not name.startswith(("_", "Abstract")))


CUDA_ONLY = {"DANN", "CDANN", "IRM", "CutMix", "SagNet", "RSC"}


class ViTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(23)
        np.random.seed(23)
        random.seed(23)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def make(self, name="ERM", device="cpu"):
        with mock.patch("torchvision.models.vit_b_16", side_effect=small_vit):
            return getattr(algorithms, name)(
                (3, 64, 64), 3, 2, params(name, self.temp.name)).to(device)

    def test_real_vit_b16(self):
        model = ViT((3, 224, 224), params("ERM", self.temp.name)).eval()
        self.assertEqual(model.n_outputs, 768)
        self.assertEqual(model.grid_size, (14, 14))
        with torch.no_grad():
            self.assertEqual(model(torch.randn(1, 3, 224, 224)).shape, (1, 768))
        self.assertEqual(sum(isinstance(m, MathSelfAttention) for m in model.modules()), 12)
        # Also exercise backward at the actual B/16 width, not only the small
        # 12-block fixture used by the exhaustive algorithm matrix.
        model(torch.randn(1, 3, 224, 224))[:, 0].square().mean().backward()
        self.assertTrue(torch.isfinite(model.network.conv_proj.weight.grad).all())

    def test_math_attention_matches_torchvision_and_higher_gradients(self):
        original = nn.MultiheadAttention(32, 4, batch_first=True, dropout=0.0).double()
        local = MathSelfAttention(original)
        x = torch.randn(2, 5, 32, dtype=torch.float64, requires_grad=True)
        expected = original(x, x, x)[0]
        actual = local(x, x, x)[0]
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(set(original.state_dict()), set(local.state_dict()))
        gradient = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
        gradient.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_pretrained_interpolation_and_arm_channels(self):
        hp = params("ERM", self.temp.name)
        hp["pretrained"] = True
        with mock.patch("torchvision.models.vit_b_16", side_effect=small_vit) as builder:
            model = ViT((4, 64, 64), hp)
        self.assertEqual(builder.call_count, 2)
        self.assertIs(builder.call_args_list[0].kwargs["weights"],
                      torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
        weight = model.network.conv_proj.weight
        self.assertTrue(torch.equal(weight[:, 0], weight[:, 3]))
        self.assertEqual(model.network.encoder.pos_embedding.shape[1], 17)
        self.assertEqual(model(torch.randn(2, 4, 64, 64)).shape, (2, 32))
        for shape in ((3, 28, 28), (3, 64, 80), (64,)):
            with self.assertRaises(ValueError):
                ViT(shape, hp)

    def test_adapter_identity_prefix_order_and_eval_semantics(self):
        x = torch.randn(2, 17, 32, requires_grad=True)
        identity = TokenGridAdapter(nn.Identity(), (4, 4))
        self.assertTrue(torch.equal(identity(x), x))
        low = TokenGridAdapter(ALOFT(alpha=0), (4, 4))
        self.assertTrue(torch.equal(low(x), x))
        active = TokenGridAdapter(ALOFT(alpha=1), (4, 4))
        changed = active(x)
        self.assertTrue(torch.equal(changed[:, :1], x[:, :1]))
        self.assertFalse(torch.equal(changed[:, 1:], x[:, 1:]))
        self.assertTrue(torch.equal(active.eval()(x), x))
        always = TokenGridAdapter(AWWSL(alpha=1), (4, 4)).eval()
        self.assertFalse(torch.equal(always(x), x))

    def test_stage_positions_and_module_variants(self):
        model = self.make("ALOFT_rev_E").featurizer
        for index, block in enumerate(model.network.encoder.layers):
            self.assertEqual(isinstance(block, nn.Sequential), index in (2, 5, 8))
        self.assertTrue(all(m.rev for m in find_aloft_modules(model)))
        # Deliberately preserve the pre-existing HF-named class's LF behavior.
        self.assertTrue(all(not m.rev for m in find_aloft_modules(self.make("ALOFT_HF_E"))))
        with self.assertRaises(ValueError):
            resnet_aloft(model)
        fq = self.make("FQ")
        self.assertEqual(sum(isinstance(m, FreqQuant) for m in fq.modules()), 9)
        structure = self.make("ALOFT_StructHF_E")
        self.assertEqual(sum(isinstance(m, StructureProbe) for m in structure.modules()), 2)
        self.assertTrue(all(m.channels == 32 for m in structure.modules() if isinstance(m, StructureProbe)))
        codebook = self.make("ALOFT_CB_rev_E")
        self.assertEqual(sum(isinstance(m, BandStatsCodebook) for m in codebook.modules()), 1)
        # Its independent ALOFT class must not be silently swapped for aloft.py.
        self.assertEqual(len(find_aloft_modules(codebook)), 0)
        sketch = self.make("ALOFT_SketchTopo_rev_E")
        self.assertEqual(sum(isinstance(m, SketchSpectrumPerturb) for m in sketch.modules()), 2)

    def test_all_algorithms_construct_and_predict(self):
        x = torch.randn(4, 3, 64, 64)
        for name in algorithm_names():
            with self.subTest(algorithm=name):
                model = self.make(name).eval()
                self.assertTrue(any(isinstance(m, ViT) for m in model.modules()))
                with torch.no_grad():
                    output = model.predict(x)
                self.assertEqual(output.shape, (4, 3))
                self.assertTrue(torch.isfinite(output).all())

    def test_all_six_insertion_positions_and_layer4_fq(self):
        calls = []

        class Record(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name

            def forward(self, x):
                calls.append((self.name, tuple(x.shape)))
                return x

        backbone = self.make().featurizer
        positions = ("conv1", "maxpool", "layer1", "layer2", "layer3", "layer4")
        for name in positions:
            backbone.add_stage_op(name, Record(name), name)
        backbone(torch.randn(2, 3, 64, 64))
        self.assertEqual(calls, [(name, (2, 32, 4, 4)) for name in positions])
        self.assertIsNot(type(backbone.network.encoder.layers[-1]), nn.Sequential)
        from domainbed.models.frequant import resnet_freqquant
        backbone = self.make().featurizer
        resnet_freqquant(backbone, layers=("layer4",))
        indices = [i for i, block in enumerate(backbone.network.encoder.layers)
                   if isinstance(block, nn.Sequential)]
        self.assertEqual(indices, [9, 10])

    def test_contexts_auxiliary_losses_and_head_gradients(self):
        xs = [torch.randn(4, 3, 64, 64) for _ in range(2)]
        ys = [torch.tensor([0, 1, 2, 0]) for _ in range(2)]
        model = self.make("ALOFT_DomainLF_E")
        op = find_aloft_modules(model)[0]
        with mock.patch.object(op, "set_domain_context", wraps=op.set_domain_context) as setter:
            model.update(xs, ys, step=0)
        self.assertTrue(torch.equal(setter.call_args.args[0], torch.tensor([0]*4 + [1]*4)))
        self.assertTrue(all(m._domain_context is None for m in find_aloft_modules(model)))
        model = self.make("ALOFT_StructHF_E")
        warmup = model.update(xs, ys, step=0)
        active = model.update(xs, ys, step=1001)
        self.assertEqual(warmup["struct_loss"], 0)
        self.assertGreater(active["struct_loss"], 0)
        for probe in (m for m in model.modules() if isinstance(m, StructureProbe)):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                for p in probe.parameters()))
        hp = params("FQ", self.temp.name)
        hp["fq_mode"] = "cb_freq"
        with mock.patch("torchvision.models.vit_b_16", side_effect=small_vit):
            model = algorithms.FQ((3, 64, 64), 3, 2, hp)
        for module in model.modules():
            if isinstance(module, FreqQuant):
                module.enabled, module.strength = True, module.strength_max
        self.assertGreater(model.update(xs, ys)["aux"], 0)

    def test_train_all_actual_argument_and_config_prefix(self):
        # Execute the real parser/config prefix, stopping before log writers,
        # datasets and the existing CUDA-only trainer are initialized.
        import argparse
        import munch
        import yaml
        from sconf import Config
        from domainbed.backbones import normalize_backbone
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "train_all.py").read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        prefix = []
        for node in main.body:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "args.debug":
                break
            prefix.append(node)
        prefix.extend(ast.parse("\nfor stream in keys: stream.close()\n").body)
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_cache_size")
        code = compile(ast.fix_missing_locations(ast.Module(body=[helper] + prefix, type_ignores=[])),
                       "train_all.py:config-prefix", "exec")
        for name in run_all.METHODS:
            command = run_all.build_command(root, name, 0, 8, "PACS", "2", "vit_b_16")
            namespace = dict(argparse=argparse, munch=munch, yaml=yaml, Config=Config,
                             hparams_registry=hparams_registry, normalize_backbone=normalize_backbone)
            with self.subTest(recipe=name), mock.patch.object(sys, "argv", command[1:]):
                exec(code, namespace)
                self.assertEqual(namespace["hparams"]["backbone"], "vit")
                self.assertEqual(namespace["hparams"]["batch_size"], 8)
                self.assertEqual(namespace["args"].algorithm, run_all.METHODS[name]["algorithm"])

    def test_existing_algorithm_updates_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        relative = "domainbed/algorithms/algorithms.py"
        before = root / "tmp/vit_backbone_reference_20260907" / relative
        if not before.exists():
            self.skipTest("Pre-edit source snapshot is not present")

        def methods(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            return {(cls.name, method.name): ast.dump(method)
                    for cls in tree.body if isinstance(cls, ast.ClassDef)
                    for method in cls.body if isinstance(method, ast.FunctionDef)
                    and method.name != "__init__"}

        self.assertEqual(methods(before), methods(root / relative))

    def test_legacy_resnet_checkpoints_strict_load(self):
        root = Path(__file__).resolve().parents[1]
        before = root / "tmp/vit_backbone_reference_20260907"
        if not before.exists():
            self.skipTest("Pre-edit source snapshot is not present")
        code = '''
import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.modules['open_clip'] = types.ModuleType('open_clip')
import torch
from domainbed import hparams_registry
from domainbed.algorithms import algorithms
torch.set_num_threads(1)
for name in ('ERM', 'ALOFT_rev_E', 'ALOFT_CB_rev_E', 'FQ'):
    hp = hparams_registry.default_hparams(name, 'PACS')
    hp['pretrained'] = False
    hp['grad_fn'] = 'sum'
    hp['logdir'] = Path(sys.argv[2])
    hp['test_env'] = 0
    model = getattr(algorithms, name)((3,64,64), 3, 2, hp).eval()
    x = torch.randn(2,3,64,64)
    with torch.no_grad(): prediction = model(x)
    torch.save({'model_hparams':hp, 'model_dict':model.state_dict(),
                'input':x, 'prediction':prediction}, Path(sys.argv[2]) / (name+'.pt'))
'''
        result = subprocess.run([sys.executable, "-B", "-c", code, str(before), self.temp.name],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("ERM", "ALOFT_rev_E", "ALOFT_CB_rev_E", "FQ"):
            checkpoint = torch.load(Path(self.temp.name) / (name + ".pt"), map_location="cpu")
            self.assertNotIn("backbone", checkpoint["model_hparams"])
            model = getattr(algorithms, name)((3, 64, 64), 3, 2,
                                              checkpoint["model_hparams"]).eval()
            model.load_state_dict(checkpoint["model_dict"], strict=True)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(checkpoint["input"]), checkpoint["prediction"]))

    def _update_algorithms(self, names, device):
        xs = [torch.randn(4, 3, 64, 64, device=device) for _ in range(2)]
        ys = [torch.tensor([0, 1, 2, 0], device=device) for _ in range(2)]
        previous_cwd = os.getcwd()
        try:
            # SAM writes sam_0.txt; do not leave algorithm diagnostics in the repo.
            os.chdir(self.temp.name)
            for name in names:
                with self.subTest(algorithm=name):
                    model = self.make(name)
                    model.to(device)
                    model.train()
                    # Fish constructs a new backbone inside its update.
                    with mock.patch("torchvision.models.vit_b_16", side_effect=small_vit):
                        result = model.update(xs, ys, step=0)
                    self.assertIsInstance(result, dict)
                    self.assertTrue(all(np.isfinite(float(v)) for v in result.values()))
        finally:
            os.chdir(previous_cwd)

    def test_cpu_algorithm_updates(self):
        self._update_algorithms([n for n in algorithm_names() if n not in CUDA_ONLY], "cpu")

    def test_gga_actual_search_and_perturbed_update(self):
        xs = [torch.randn(2, 3, 64, 64) for _ in range(2)]
        ys = [torch.tensor([0, 1]), torch.tensor([1, 2])]
        gga = self.make("ERM_GGA")
        result = gga.update_simulated_annealing(xs, ys, step=100)
        self.assertTrue(np.isfinite(result["loss"]))
        gga_l = self.make("GGA_L")
        self.assertTrue(np.isfinite(gga_l.update_perturbed(xs, ys, step=100)["loss"]))

    @unittest.skipUnless(torch.cuda.is_available(), "Existing algorithm updates require CUDA")
    def test_cuda_algorithm_updates(self):
        self._update_algorithms(sorted(CUDA_ONLY), "cuda")

    def test_state_roundtrip_swad_and_noop_bn(self):
        model = self.make("ALOFT_rev_E").eval()
        restored = self.make("ALOFT_rev_E").eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        averaged = AveragedModel(model)
        averaged.update_parameters(model, step=0)
        averaged.eval()
        update_bn(iter(()), averaged, 1, device="cpu")
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), restored(x)))
            self.assertTrue(torch.equal(model(x), averaged(x)))

    def test_aloft_frequency_support_and_later_cls_effect(self):
        x = torch.randn(4, 3, 14, 14)
        low = ALOFT(mask_ratio=0.7, rev=False)
        high = ALOFT(mask_ratio=0.7, rev=True)
        masks = [m._mask(14, 14, x.device).expand_as(x) for m in (low, high)]
        self.assertTrue(torch.equal(masks[0], ~masks[1]))
        for module, mask in zip((low, high), masks):
            delta = module(x) - x
            spectrum = torch.fft.fftshift(torch.fft.fft2(delta, norm="ortho"), dim=(-2, -1))
            self.assertLess(float(spectrum[~mask].abs().max()), 2e-6)
        model = self.make("ALOFT_E")
        image = torch.randn(2, 3, 64, 64)
        original = model.eval()(image)
        model.train()
        self.assertFalse(torch.equal(original, model(image)))

    def test_all_runner_recipes_and_active_stages(self):
        # Includes all codebook/FQ modes, not just the default FQ constructor.
        for label, recipe in run_all.METHODS.items():
            with self.subTest(method=label):
                hp = params(recipe["algorithm"], self.temp.name)
                args = recipe["extra_args"]
                for key, value in zip(args[::2], args[1::2]):
                    try:
                        hp[key[2:]] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        hp[key[2:]] = value
                with mock.patch("torchvision.models.vit_b_16", side_effect=small_vit):
                    model = getattr(algorithms, recipe["algorithm"])((3, 64, 64), 3, 2, hp)
                    xs = [torch.randn(4, 3, 64, 64) for _ in range(2)]
                    ys = [torch.tensor([0, 1, 2, 0]) for _ in range(2)]
                    for module in model.modules():
                        if isinstance(module, BandStatsCodebook):
                            module.set_collection(True)
                    if recipe["algorithm"] in CUDA_ONLY:
                        model.eval()
                        self.assertEqual(model.predict(xs[0]).shape, (4, 3))
                        continue
                    model.update(xs, ys, step=0)
                    for module in model.modules():
                        if isinstance(module, FreqQuant):
                            module.enabled, module.strength = True, module.strength_max
                        if isinstance(module, BandStatsCodebook):
                            # Fill with real observations before freezing; otherwise
                            # a short smoke test only exercises its identity path.
                            while not bool(module.initialized.all()):
                                module(torch.randn(8, 32, 4, 4))
                            module.set_quantization(True, module.strength_max)
                            module.freeze_codebook()
                        if isinstance(module, SketchSpectrumPerturb):
                            module.class_min_count = 1
                            module.warmup_steps = 0
                            module.activation_step.fill_(0)
                    result = model.update(xs, ys, step=1001)
                    self.assertTrue(all(np.isfinite(float(v)) for v in result.values()))
                    model.eval()
                    with torch.no_grad():
                        self.assertTrue(torch.isfinite(model.predict(xs[0])).all())

    def test_sketch_14_grid_empty_cells_are_finite_and_preserve_masks(self):
        from domainbed.models.aloft_sketch import resnet_aloft_sketch
        with mock.patch("torchvision.models.vit_b_16", side_effect=small_vit):
            backbone = ViT((3, 224, 224), params("ERM", self.temp.name))
        resnet_aloft_sketch(backbone, num_classes=3, class_min_count=1,
                           warmup_steps=0, ramp_steps=1)
        for module in backbone.modules():
            if not isinstance(module, SketchSpectrumPerturb):
                continue
            masks, orientations, high = module._masks(14, 14, torch.device("cpu"))
            self.assertTrue(torch.equal(masks.any(0).any(0), high))
            x = torch.randn(4, 32, 14, 14, requires_grad=True)
            module.set_batch_context(torch.tensor([0, 1, 2, 0]), 0)
            module(x)
            module.set_batch_context(torch.tensor([0, 1, 2, 0]), 2)
            y = module(x)
            self.assertTrue(torch.isfinite(y).all())
            self.assertFalse(torch.equal(x, y))
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_random_hparams_do_not_shift_existing_draws(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "tmp/vit_backbone_reference_20260907/domainbed/hparams_registry.py"
        if not path.exists():
            self.skipTest("Pre-edit source snapshot is not present")
        spec = importlib.util.spec_from_file_location("old_hparams", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for name in algorithm_names():
            for dataset in ("PACS", "OfficeHome", "DomainNet", "ColoredMNIST"):
                for seed in (0, 9):
                    before = old.random_hparams(name, dataset, seed)
                    after = hparams_registry.random_hparams(name, dataset, seed)
                    self.assertEqual(after.pop("backbone"), "resnet")
                    self.assertEqual(before, after)

    def test_active_special_modules_checkpoint_roundtrip(self):
        for name in ("FQ", "ALOFT_CB_rev_E", "ALOFT_SketchTopo_rev_E", "ALOFT_StructHF_E"):
            with self.subTest(algorithm=name):
                model = self.make(name)
                for module in model.modules():
                    if isinstance(module, FreqQuant):
                        module.enabled, module.strength = True, module.strength_max
                    if isinstance(module, BandStatsCodebook):
                        module.set_collection(True)
                        while not bool(module.initialized.all()):
                            module(torch.randn(8, 32, 4, 4))
                        module.set_quantization(True, module.strength_max)
                        module.freeze_codebook()
                xs = [torch.randn(4, 3, 64, 64) for _ in range(2)]
                ys = [torch.tensor([0, 1, 2, 0]) for _ in range(2)]
                model.update(xs, ys, step=1001)
                restored = self.make(name)
                path = Path(self.temp.name) / "roundtrip.pt"
                torch.save({"model_dict": model.state_dict()}, path)
                load_vit_checkpoint(restored, torch.load(path, map_location="cpu"))
                model.eval()
                restored.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(xs[0]), restored(xs[0])))

    def test_codebook_non224_empty_bands_keep_frequency_support(self):
        model = self.make("ALOFT_CB_rev_E")
        module = next(m for m in model.modules() if isinstance(m, BandStatsCodebook))
        module.set_collection(True)
        while not bool(module.initialized.all()):
            module(torch.randn(8, 32, 4, 4))
        module.freeze_codebook()
        module.set_quantization(True, module.strength_max)
        module.eval()
        for size in (4, 12, 14):
            x = torch.randn(2, 32, size, size, requires_grad=True)
            y = module(x)
            self.assertTrue(torch.isfinite(y).all())
            self.assertFalse(torch.equal(x, y))
            delta = torch.fft.fftshift(torch.fft.fft2(y-x, norm="ortho"), dim=(-2, -1))
            mask = module._bands(size, size, x.device).any(0)
            self.assertLess(float(delta[..., ~mask].abs().max()), 3e-6)
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())

    def test_quantization_weights_scales_state_and_checkpoint(self):
        config = {"weight": {"bit": 7, "mode": "lsq", "per_channel": True,
                              "symmetric": False, "all_positive": False}, "excepts": {}}
        model = self.make()
        xs = [torch.randn(4, 3, 64, 64) for _ in range(2)]
        ys = [torch.tensor([0, 1, 2, 0]) for _ in range(2)]
        model.update(xs, ys)
        weights = {name: param for name, param in model.named_parameters()}
        states = {id(p): {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                          for k, v in value.items()} for p, value in model.optimizer.state.items()}
        converted = prepare_vit_quantization(model, config)
        self.assertEqual(len(converted), 37)  # patch + 12 attention + 24 MLP modules
        self.assertEqual(sum(isinstance(m, QuantSelfAttention) for m in model.modules()), 12)
        self.assertEqual(sum(isinstance(m, QuantLinear) for m in model.modules()), 36)
        self.assertIs(type(model.classifier), nn.Linear)
        self.assertIsInstance(model.featurizer.network.conv_proj, QuantPatchConv)
        for name, param in model.named_parameters():
            if name in weights:
                self.assertIs(param, weights[name])
        for param, value in model.optimizer.state.items():
            for key, old in states[id(param)].items():
                if torch.is_tensor(old):
                    self.assertTrue(torch.equal(value[key], old))
        known = {id(p) for g in model.optimizer.param_groups for p in g["params"]}
        self.assertTrue(all(id(p) in known for p in model.parameters()))
        scales = [p for name, p in model.named_parameters() if "quantizer.s" in name]
        self.assertEqual(len(scales), 49)
        before = [p.detach().clone() for p in scales]
        model.update(xs, ys)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in scales))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, scales)))
        self.assertEqual(prepare_vit_quantization(model, config), [])
        checkpoint = {"model_dict": model.state_dict(), "vit_quantization": model._vit_quantization_config}
        restored = self.make()
        load_vit_checkpoint(restored, checkpoint)
        model.eval()
        restored.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(xs[0]), restored(xs[0])))
        averaged = AveragedModel(model).eval()
        averaged.update_parameters(model, step=1)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(xs[0]), averaged(xs[0])))


if __name__ == "__main__":
    unittest.main()
