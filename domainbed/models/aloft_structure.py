"""Sketch-structure supervision layered on top of the original ALOFT module.

The probes in this file are auxiliary training branches.  They never modify
the feature tensor passed to the following ResNet stage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.models.aloft import ALOFT, resnet_aloft


def structure_scale_at_step(step, warmup=200, ramp=500):
    """Linear auxiliary-loss schedule; ALOFT itself is never scheduled."""
    step = int(step)
    warmup = int(warmup)
    ramp = int(ramp)
    if step <= warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min(1.0, max(0.0, (step - warmup) / float(ramp)))


def _soft_erode(x):
    vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x, iterations=10):
    opened = _soft_open(x)
    skeleton = F.relu(x - opened)
    work = x
    for _ in range(int(iterations)):
        work = _soft_erode(work)
        opened = _soft_open(work)
        delta = F.relu(work - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(prediction, target, iterations=10, eps=1e-6):
    pred_skeleton = soft_skeletonize(prediction, iterations)
    target_skeleton = soft_skeletonize(target, iterations)
    dims = (1, 2, 3)
    precision = ((pred_skeleton * target).sum(dims) + eps) / (
        pred_skeleton.sum(dims) + eps)
    sensitivity = ((target_skeleton * prediction).sum(dims) + eps) / (
        target_skeleton.sum(dims) + eps)
    score = (2.0 * precision * sensitivity + eps) / (
        precision + sensitivity + eps)
    return (1.0 - score).mean()


def _soft_dice_loss(prediction, target, eps=1e-6):
    dims = (1, 2, 3)
    intersection = (prediction * target).sum(dims)
    denominator = prediction.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _balanced_bce_dice(logits, target, eps=1e-6):
    positive = target.sum().detach()
    negative = target.new_tensor(target.numel()) - positive
    pos_weight = (negative / (positive + eps)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight)
    dice = _soft_dice_loss(torch.sigmoid(logits), target, eps=eps)
    return 0.5 * bce + 0.5 * dice


class SketchStructureTargets(nn.Module):
    """Build direction, stroke, and stable enclosed-region pseudo-labels."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1.0, 0.0, 1.0],
                          [-2.0, 0.0, 2.0],
                          [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3) / 8.0)
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1.0, -2.0, -1.0],
                          [0.0, 0.0, 0.0],
                          [1.0, 2.0, 1.0]]).view(1, 1, 3, 3) / 8.0)

    @staticmethod
    def _feature_size(height, width, stride):
        return ((height + stride - 1) // stride,
                (width + stride - 1) // stride)

    @staticmethod
    def _border_values(gray):
        height, width = gray.shape[-2:]
        border = max(1, int(round(min(height, width) * 0.04)))
        pieces = [
            gray[:, :, :border, :].flatten(2),
            gray[:, :, height - border:, :].flatten(2),
        ]
        if height > 2 * border:
            pieces.extend([
                gray[:, :, border:height - border, :border].flatten(2),
                gray[:, :, border:height - border, width - border:].flatten(2),
            ])
        return torch.cat(pieces, dim=2)

    @staticmethod
    def _binary_close(mask):
        dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
        return 1.0 - F.max_pool2d(1.0 - dilated, 3, stride=1, padding=1)

    @staticmethod
    def _enclosed_regions(barrier):
        free = 1.0 - barrier
        border = torch.zeros_like(free)
        border[:, :, 0, :] = 1.0
        border[:, :, -1, :] = 1.0
        border[:, :, :, 0] = 1.0
        border[:, :, :, -1] = 1.0
        reached = border * free
        height, width = free.shape[-2:]
        for _ in range(height + width):
            reached = torch.maximum(
                reached, F.max_pool2d(reached, 3, stride=1, padding=1)) * free
        return free * (1.0 - reached)

    def _line_map(self, gray):
        border = self._border_values(gray)
        background = border.median(dim=2, keepdim=True).values.unsqueeze(-1)
        contrast = (gray - background).abs()
        scale = torch.quantile(
            contrast.flatten(2), 0.99, dim=2, keepdim=True).unsqueeze(-1)
        scale = scale.clamp_min(0.05)
        normalized = contrast / scale
        ink = ((normalized - 0.15) / 0.35).clamp(0.0, 1.0)

        density = F.avg_pool2d(ink, 15, stride=1, padding=7)
        keep_line = ((0.8 - density) / 0.3).clamp(0.0, 1.0)
        return (ink * keep_line).clamp(0.0, 1.0)

    def _direction_targets(self, line, size):
        small = F.interpolate(line, size=size, mode="bilinear", align_corners=False)
        grad_x = F.conv2d(small, self.sobel_x.to(small.dtype), padding=1)
        grad_y = F.conv2d(small, self.sobel_y.to(small.dtype), padding=1)

        j_xx = F.avg_pool2d(grad_x.square(), 5, stride=1, padding=2)
        j_yy = F.avg_pool2d(grad_y.square(), 5, stride=1, padding=2)
        j_xy = F.avg_pool2d(grad_x * grad_y, 5, stride=1, padding=2)
        delta = ((j_xx - j_yy).square() + 4.0 * j_xy.square() + self.eps).sqrt()

        # The structure tensor's major eigenvector is normal to a stroke.
        # Negating its double-angle representation rotates it by pi/2.
        direction = torch.cat([
            -(j_xx - j_yy) / delta,
            -(2.0 * j_xy) / delta,
        ], dim=1)
        coherence = delta / (j_xx + j_yy + self.eps)
        energy = (j_xx + j_yy + self.eps).sqrt()
        energy_scale = torch.quantile(
            energy.flatten(2), 0.95, dim=2, keepdim=True).unsqueeze(-1)
        energy = (energy / energy_scale.clamp_min(self.eps)).clamp(0.0, 1.0)
        stroke = F.adaptive_max_pool2d(line, size)
        confidence = (coherence * energy * stroke).clamp(0.0, 1.0)
        return direction, confidence

    def _topology_targets(self, line, size):
        stroke = F.adaptive_max_pool2d(line, size).clamp(0.0, 1.0)
        low_barrier = self._binary_close((stroke >= 0.35).to(stroke.dtype))
        high_barrier = self._binary_close((stroke >= 0.55).to(stroke.dtype))
        low_inside = self._enclosed_regions(low_barrier)
        high_inside = self._enclosed_regions(high_barrier)
        closure = torch.minimum(low_inside, high_inside)

        area = closure.sum(dim=(1, 2, 3))
        total = float(size[0] * size[1])
        valid = ((area >= 4.0) & (area <= 0.6 * total)).to(stroke.dtype)
        closure = closure * valid.view(-1, 1, 1, 1)
        return stroke, closure, valid

    def forward(self, normalized_input, direction=True, topology=True):
        with torch.no_grad():
            image = (normalized_input * self.image_std.to(normalized_input.dtype)
                     + self.image_mean.to(normalized_input.dtype)).clamp(0.0, 1.0)
            gray = (0.2989 * image[:, 0:1]
                    + 0.5870 * image[:, 1:2]
                    + 0.1140 * image[:, 2:3])
            line = self._line_map(gray)
            height, width = normalized_input.shape[-2:]
            direction_size = self._feature_size(height, width, 4)
            topology_size = self._feature_size(height, width, 8)
            targets = {}
            if direction:
                direction_map, direction_confidence = self._direction_targets(
                    line, direction_size)
                targets.update({
                    "direction": direction_map.detach(),
                    "direction_confidence": direction_confidence.detach(),
                })
            if topology:
                stroke, closure, closure_valid = self._topology_targets(
                    line, topology_size)
                targets.update({
                    "stroke": stroke.detach(),
                    "closure": closure.detach(),
                    "closure_valid": closure_valid.detach(),
                })
        return targets


class _StructureHead(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        groups = min(8, int(hidden_channels))
        while hidden_channels % groups:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      groups=hidden_channels, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.layers(x)


class StructureProbe(nn.Module):
    """Predict sketch structure while returning the input feature unchanged."""

    def __init__(self, channels, kind, hidden_channels=64, skeleton_iters=10,
                 eps=1e-6):
        super().__init__()
        if kind not in ("direction", "topology"):
            raise ValueError(f"unknown structure probe kind: {kind}")
        self.channels = int(channels)
        self.kind = kind
        self.skeleton_iters = int(skeleton_iters)
        self.eps = float(eps)
        self.head = _StructureHead(
            self.channels, int(hidden_channels), 3 if kind == "direction" else 2)
        self._targets = None
        self._scale = 0.0
        self._losses = None
        self._diagnostics = self._zero_diagnostics()

    @staticmethod
    def _zero_diagnostics():
        return {
            "struct_dir_valid_ratio": 0.0,
            "struct_stroke_target_ratio": 0.0,
            "struct_closure_target_ratio": 0.0,
            "struct_closure_valid_ratio": 0.0,
        }

    def set_batch_context(self, targets, scale):
        self._targets = targets
        self._scale = float(scale)

    def clear_batch_context(self):
        self._targets = None
        self._scale = 0.0

    @staticmethod
    def _resize(target, size, mode="bilinear"):
        if target.shape[-2:] == size:
            return target
        if mode == "nearest":
            return F.interpolate(target, size=size, mode=mode)
        return F.interpolate(target, size=size, mode=mode, align_corners=False)

    def _direction_losses(self, prediction, size):
        target_direction = self._resize(self._targets["direction"], size)
        target_confidence = self._resize(
            self._targets["direction_confidence"], size)
        predicted_direction = F.normalize(prediction[:, :2], dim=1, eps=self.eps)
        cosine = (predicted_direction * target_direction).sum(dim=1, keepdim=True)
        confidence_sum = target_confidence.sum().clamp_min(self.eps)
        orientation = ((1.0 - cosine) * target_confidence).sum() / confidence_sum
        confidence = F.binary_cross_entropy_with_logits(
            prediction[:, 2:3], target_confidence)
        loss = orientation + 0.5 * confidence
        self._diagnostics["struct_dir_valid_ratio"] = float(
            (target_confidence > 0.05).float().mean().detach().item())
        return {"dir": loss}

    def _topology_losses(self, prediction, size):
        stroke = self._resize(self._targets["stroke"], size)
        closure = self._resize(self._targets["closure"], size, mode="nearest")
        valid = self._targets["closure_valid"].bool()

        stroke_logits = prediction[:, 0:1]
        closure_logits = prediction[:, 1:2]
        stroke_loss = _balanced_bce_dice(stroke_logits, stroke, self.eps)
        stroke_probability = torch.sigmoid(stroke_logits)
        cldice = soft_cldice_loss(
            stroke_probability, stroke, self.skeleton_iters, self.eps)
        if valid.any():
            closure_loss = _balanced_bce_dice(
                closure_logits[valid], closure[valid], self.eps)
        else:
            closure_loss = closure_logits.sum() * 0.0

        self._diagnostics.update({
            "struct_stroke_target_ratio": float(stroke.mean().detach().item()),
            "struct_closure_target_ratio": float(closure.mean().detach().item()),
            "struct_closure_valid_ratio": float(valid.float().mean().detach().item()),
        })
        return {
            "stroke": stroke_loss,
            "closure": closure_loss,
            "cldice": cldice,
        }

    def forward(self, x):
        self._losses = None
        self._diagnostics = self._zero_diagnostics()
        if not self.training or self._targets is None or self._scale <= 0.0:
            return x

        prediction = self.head(x)
        if self.kind == "direction":
            self._losses = self._direction_losses(prediction, x.shape[-2:])
        else:
            self._losses = self._topology_losses(prediction, x.shape[-2:])
        return x

    def pop_losses(self):
        losses = self._losses
        self._losses = None
        return losses

    def diagnostics(self):
        return dict(self._diagnostics)


def _stage_out_channels(stage):
    block = stage[-1]
    return block.bn3.num_features if hasattr(block, "bn3") else block.bn2.num_features


def resnet_aloft_structure(network, direction=True, topology=True,
                            hidden_channels=64, skeleton_iters=10,
                            positions=("layer1", "layer2", "layer3"), **aloft_kwargs):
    """Attach original ALOFT at all stages and identity structure probes."""
    if getattr(network, "is_vit_backbone", False):
        if tuple(positions) != ("layer1", "layer2", "layer3"):
            raise ValueError("ALOFT structure variants require layer1/layer2/layer3")
        resnet_aloft(network, positions=positions, **aloft_kwargs)
        if direction:
            network.add_stage_op("layer1", StructureProbe(
                network.n_outputs, "direction", hidden_channels, skeleton_iters),
                "structure_direction")
        if topology:
            network.add_stage_op("layer2", StructureProbe(
                network.n_outputs, "topology", hidden_channels, skeleton_iters),
                "structure_topology")
        return network
    positions = tuple(positions)
    if positions != ("layer1", "layer2", "layer3"):
        raise ValueError("ALOFT structure variants require layer1/layer2/layer3")
    layer1_channels = _stage_out_channels(network.layer1)
    layer2_channels = _stage_out_channels(network.layer2)
    resnet_aloft(network, positions=positions, **aloft_kwargs)

    if direction:
        network.layer1.add_module(
            "structure_direction",
            StructureProbe(layer1_channels, "direction", hidden_channels,
                           skeleton_iters))
    if topology:
        network.layer2.add_module(
            "structure_topology",
            StructureProbe(layer2_channels, "topology", hidden_channels,
                           skeleton_iters))
    return network


def find_structure_probes(module):
    return [item for item in module.modules() if isinstance(item, StructureProbe)]


def collect_structure_losses(module):
    collected = {}
    for probe in find_structure_probes(module):
        losses = probe.pop_losses()
        if losses is None:
            continue
        for name, loss in losses.items():
            collected.setdefault(name, []).append(loss)
    return {
        name: torch.stack(losses).mean()
        for name, losses in collected.items()
    }


def collect_structure_diagnostics(module):
    probes = find_structure_probes(module)
    if not probes:
        return {}
    values = [probe.diagnostics() for probe in probes]
    result = {}
    for key in values[0]:
        relevant = [item[key] for item in values if item[key] != 0.0]
        result[key] = sum(relevant) / len(relevant) if relevant else 0.0
    return result


__all__ = [
    "ALOFT",
    "SketchStructureTargets",
    "StructureProbe",
    "collect_structure_diagnostics",
    "collect_structure_losses",
    "find_structure_probes",
    "resnet_aloft_structure",
    "soft_cldice_loss",
    "soft_skeletonize",
    "structure_scale_at_step",
]
