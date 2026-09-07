"""ViT-only, weight-only QTDoG extensions; legacy Conv2d mapping is untouched."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from domainbed.models.vit import MathSelfAttention, ViT
from domainbed.quan.utils import quantizer


def _config_dict(config):
    return {
        "weight": dict(config["weight"]),
        "excepts": {name: {"weight": dict(value.get("weight", {}))}
                    for name, value in config.get("excepts", {}).items()},
    }


def _weight_quantizer(weight, name, config):
    override = config["excepts"].get(name, {}).get("weight")
    result = quantizer(config["weight"], override)
    result.init_from(weight)
    return result.to(device=weight.device, dtype=weight.dtype)


class QuantLinear(nn.Module):
    def __init__(self, original, name, config):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        # Keep optimizer references and momentum; do not create new weights.
        self.weight = original.weight
        self.bias = original.bias
        self.weight_quantizer = _weight_quantizer(self.weight, name, config)
        self.train(original.training)

    def forward(self, x):
        return F.linear(x, self.weight_quantizer(self.weight), self.bias)


class QuantPatchConv(nn.Module):
    def __init__(self, original, name, config):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.padding_mode = original.padding_mode
        self._reversed_padding_repeated_twice = original._reversed_padding_repeated_twice
        self.weight_quantizer = _weight_quantizer(self.weight, name, config)
        self.train(original.training)

    def forward(self, x):
        padding = self.padding
        if self.padding_mode != "zeros":
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        return F.conv2d(x, self.weight_quantizer(self.weight), self.bias,
                        self.stride, padding, self.dilation, self.groups)


class QuantSelfAttention(MathSelfAttention):
    def __init__(self, original, name, config):
        super().__init__(original)
        self.in_proj_quantizer = _weight_quantizer(
            self.in_proj_weight, name + ".in_proj", config)
        self.out_proj = QuantLinear(original.out_proj, name + ".out_proj", config)
        self.train(original.training)

    def project_qkv(self, x):
        return F.linear(x, self.in_proj_quantizer(self.in_proj_weight), self.in_proj_bias)


def _replace(root, path, replacement):
    parent_path, _, child = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child, replacement)


def _optimizers(model):
    seen = set()
    for module in model.modules():
        for value in vars(module).values():
            if isinstance(value, torch.optim.Optimizer) and id(value) not in seen:
                seen.add(id(value))
                yield value


def prepare_vit_quantization(model, config, optimizers=None):
    """Quantize the ViT encoder, preserving weights and registering LSQ scales.

Returns converted paths for logging. Call only on the ViT branch at q_steps.
Calling twice with the same configuration is an identity operation.
"""
    config = _config_dict(config)
    previous = getattr(model, "_vit_quantization_config", None)
    if previous is not None:
        if previous != config:
            raise ValueError("Cannot change the configuration of an already quantized ViT")
        return []
    backbones = [(name, module) for name, module in model.named_modules()
                 if isinstance(module, ViT)]
    if not backbones:
        raise ValueError("ViT quantization requires a ViT backbone")
    optimizers = list(_optimizers(model) if optimizers is None else optimizers)
    additions = []
    converted = []

    def record(wrapper, path):
        if isinstance(wrapper, QuantSelfAttention):
            additions.append((wrapper.in_proj_weight, list(wrapper.in_proj_quantizer.parameters())))
            additions.append((wrapper.out_proj.weight, list(wrapper.out_proj.weight_quantizer.parameters())))
        else:
            additions.append((wrapper.weight, list(wrapper.weight_quantizer.parameters())))
        converted.append(path)
        return wrapper

    for prefix, backbone in backbones:
        base = (prefix + "." if prefix else "") + "network."
        # conv_proj may contain an inserted spatial operation. Only the patch
        # projection is selected, not convolutional auxiliary heads.
        for relative, module in list(backbone.network.conv_proj.named_modules()):
            if isinstance(module, nn.Conv2d):
                path = "conv_proj" + ("." + relative if relative else "")
                wrapper = record(QuantPatchConv(module, base + path, config), base + path)
                _replace(backbone.network, path, wrapper)
                break
        for relative, block in list(backbone.network.encoder.layers.named_modules()):
            if not (hasattr(block, "self_attention") and hasattr(block, "mlp")):
                continue
            path = base + "encoder.layers." + relative
            block.self_attention = record(QuantSelfAttention(
                block.self_attention, path + ".self_attention", config), path + ".self_attention")
            for name, linear in list(block.mlp.named_children()):
                if isinstance(linear, nn.Linear):
                    full_name = path + ".mlp." + name
                    setattr(block.mlp, name, record(QuantLinear(linear, full_name, config), full_name))

    for optimizer in optimizers:
        known = {id(p) for group in optimizer.param_groups for p in group["params"]}
        # Match each scale to the existing parameter group that owns its weight.
        for group in list(optimizer.param_groups):
            owned = {id(p) for p in group["params"]}
            extra = []
            for weight, scales in additions:
                if id(weight) in owned:
                    for scale in scales:
                        if id(scale) not in known:
                            known.add(id(scale))
                            extra.append(scale)
            if extra:
                settings = {key: value for key, value in group.items() if key != "params"}
                optimizer.add_param_group(dict(settings, params=extra))
    model._vit_quantization_config = copy.deepcopy(config)
    return converted


def load_vit_checkpoint(model, checkpoint, strict=True):
    """Reconstruct the optional QTDoG wrappers before loading a train_all checkpoint.

Construct the algorithm with checkpoint['model_hparams'] first (pretrained=False
may be used to avoid downloading weights that will immediately be overwritten).
"""
    config = checkpoint.get("vit_quantization")
    if config is not None:
        prepare_vit_quantization(model, config)
    return model.load_state_dict(checkpoint["model_dict"], strict=strict)
