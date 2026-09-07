"""Exact ResNet-50 regression against a pre-edit source tree, without downloads.

python tests/check_backbone_compatibility.py --reference /path/to/old/tree

Runs old/default/explicit-resnet in separate processes, comparing SHA256 of all
parameters, buffers, gradients, optimizer states, predictions and RNG states.
Synthetic CPU checks are not substitutes for long CUDA training experiments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(repo, output, explicit):
    sys.path.insert(0, str(repo))
    import copy
    import gc
    import hashlib
    import inspect
    import random
    import tempfile
    import types
    from unittest import mock
    import ast
    import numpy as np
    import torch
    from munch import munchify

    sys.modules.setdefault("open_clip", types.ModuleType("open_clip"))
    import run_all
    from domainbed.algorithms import algorithms as alg
    from domainbed import hparams_registry as registry
    from domainbed.lib.swa_utils import AveragedModel
    from domainbed.models.frequant import FreqQuant
    from domainbed.models.aloft_cb import BandStatsCodebook
    from domainbed.models.aloft_sketch import SketchSpectrumPerturb
    from domainbed.quan.utils import find_modules_to_quantize, replace_module_by_names

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cuda_only = {"DANN", "CDANN", "IRM", "CutMix", "SagNet", "RSC"}
    names = sorted(name for name, cls in vars(alg).items()
                   if inspect.isclass(cls) and issubclass(cls, alg.Algorithm)
                   and cls is not alg.Algorithm and not name.startswith(("_", "Abstract")))
    cases = [(name, name, {}) for name in names]
    for label, recipe in run_all.METHODS.items():
        overrides = {}
        args = recipe["extra_args"]
        for key, value in zip(args[::2], args[1::2]):
            try:
                overrides[key[2:]] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                overrides[key[2:]] = value
        cases.append(("runner:" + label, recipe["algorithm"], overrides))

    def digest(value):
        h = hashlib.sha256()
        def visit(item):
            if torch.is_tensor(item):
                h.update(str((item.dtype, tuple(item.shape))).encode())
                h.update(item.detach().cpu().contiguous().numpy().tobytes())
            elif isinstance(item, np.ndarray):
                h.update(str((item.dtype, item.shape)).encode())
                h.update(item.tobytes())
            elif isinstance(item, dict):
                for key in sorted(item, key=repr):
                    visit(key)
                    visit(item[key])
            elif isinstance(item, (list, tuple)):
                h.update(type(item).__name__.encode())
                for element in item:
                    visit(element)
            else:
                h.update(repr(item).encode())
        visit(value)
        return h.hexdigest()

    def snapshot(model):
        opts = {}
        for key, value in vars(model).items():
            if isinstance(value, torch.optim.Optimizer):
                opts[key] = value.state_dict()
        return {
            "state": digest(model.state_dict()),
            "gradients": digest([(n, p.grad) for n, p in model.named_parameters()]),
            "optimizers": digest(opts),
            "rng": digest((random.getstate(), np.random.get_state(), torch.get_rng_state())),
        }

    report = {"environment": {"torch": torch.__version__, "numpy": np.__version__}, "cases": {}}
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as temp, \
            mock.patch("domainbed.models.resnet_mixstyle.init_pretrained_weights"), \
            mock.patch("domainbed.models.resnet_mixstyle2.init_pretrained_weights"):
        os.chdir(temp)
        try:
            for label, name, overrides in cases:
                random.seed(120)
                np.random.seed(120)
                torch.manual_seed(120)
                hp = registry.default_hparams(name, "PACS")
                hp.update(overrides)
                hp.update(pretrained=False, resnet18=False, grad_fn="sum", logdir=Path(temp),
                          test_env=0, batch_size=2, neighborhoodSize=1e-6,
                          start_step=100, end_step=200, extra_search="n",
                          extra_search_start=None, extra_search_end=None,
                          gga_l_gamma=0.001, annealing_patience=3)
                if explicit:
                    hp["backbone"] = "resnet"
                # These original ResNet modules require populated frequency cells.
                size = 224 if "Sketch" in name or "ALOFT_CB" in name else 64
                result = {}
                report["cases"][label] = result
                try:
                    model = getattr(alg, name)((3, size, size), 3, 2, hp)
                    result["initial"] = snapshot(model)
                    xs = [torch.randn(2, 3, size, size) for _ in range(2)]
                    ys = [torch.tensor([0, 1]), torch.tensor([1, 2])]
                    model.eval()
                    with torch.no_grad():
                        result["prediction"] = digest(model.predict(xs[0]))
                    result["after_eval"] = snapshot(model)
                    if name in cuda_only:
                        result["update"] = "SKIPPED: existing update requires CUDA"
                    else:
                        model.train()
                        for module in model.modules():
                            if isinstance(module, BandStatsCodebook):
                                module.set_collection(True)
                        result["loss"] = model.update(xs, ys, step=0)
                        result["updated"] = snapshot(model)
                        if name == "ERM_GGA":
                            model.save_best_weights()
                            model.perturb_weights()
                            result["gga_perturbed"] = snapshot(model)
                            model.restore_best_weights()
                            result["gga_restored"] = snapshot(model)
                        if name == "GGA_L":
                            result["gga_loss"] = model.update_perturbed(xs, ys, step=100)
                            result["gga_updated"] = snapshot(model)
                        special = False
                        for module in model.modules():
                            if isinstance(module, FreqQuant):
                                module.enabled, module.strength = True, module.strength_max
                                special = True
                            if isinstance(module, BandStatsCodebook):
                                while not bool(module.initialized.all()):
                                    module(torch.randn(8, module.channels, size // 16, size // 16))
                                module.set_quantization(True, module.strength_max)
                                module.freeze_codebook()
                                special = True
                            if isinstance(module, SketchSpectrumPerturb):
                                module.class_min_count = 1
                                module.warmup_steps = 0
                                module.activation_step.fill_(0)
                                special = True
                        if "Struct" in name:
                            special = True
                        if overrides.get("quant"):
                            config = munchify({"weight": {"mode": "lsq", "bit": 7,
                                "per_channel": True, "symmetric": False, "all_positive": False},
                                "act": {"mode": "lsq", "bit": 8, "per_channel": False,
                                        "symmetric": False, "all_positive": False}, "excepts": {}})
                            replacement = find_modules_to_quantize(model, config)
                            replace_module_by_names(model, replacement)
                            special = True
                        if special:
                            result["active_loss"] = model.update(xs, ys, step=1001)
                            result["active"] = snapshot(model)
                    averaged = AveragedModel(model)
                    averaged.update_parameters(model, step=0)
                    averaged.update_parameters(model, step=1)
                    result["swad"] = digest(averaged.state_dict())
                    del averaged, model
                    gc.collect()
                except Exception as error:
                    result["error"] = type(error).__name__ + ": " + str(error)
                print(label, "ERROR" if "error" in result else "OK", flush=True)
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        finally:
            os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tmp/backbone_regression"))
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--explicit", action="store_true")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of independent reference workers (1-3)")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker.resolve(), args.output.resolve(), args.explicit)
        return
    if args.reference is None:
        parser.error("--reference must point to the pre-edit source tree")
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {}
    def run_worker(item):
        name, repo, explicit = item
        path = (args.output / (name + ".json")).resolve()
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                   str(repo.resolve()), "--output", str(path)]
        if explicit:
            command.append("--explicit")
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        print("Finished", name, flush=True)
        return name, json.loads(path.read_text(encoding="utf-8"))
    work = (("before", args.reference, False),
            ("after_default", root, False), ("after_resnet", root, True))
    with ThreadPoolExecutor(max_workers=max(1, min(3, args.jobs))) as pool:
        reports.update(pool.map(run_worker, work))
    baseline = reports["before"]["cases"]
    failures = []
    for label, original in baseline.items():
        if "error" in original:
            failures.append((label, "baseline error", original["error"]))
        for mode in ("after_default", "after_resnet"):
            if original != reports[mode]["cases"].get(label):
                failures.append((label, mode, "exact comparison failed"))
    summary = {"cases": len(baseline), "failures": failures,
               "environment": reports["before"]["environment"],
               "cuda_update_cases_skipped": [k for k, v in baseline.items() if "update" in v]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
