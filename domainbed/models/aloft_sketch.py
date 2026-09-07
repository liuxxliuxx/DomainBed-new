"""Sketch-specific direction/radial spectrum perturbation for ALOFT.

The module perturbs high-frequency radial statistics inside orientation
sectors.  It preserves phase and the total energy of every orientation sector,
so the perturbation changes feature-stroke width/roughness without freely
moving the stroke layout.  Class labels are used only to suppress noise along
class-discriminative statistic dimensions; domain labels are never consumed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(x):
    vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x, iterations=10):
    """Differentiable approximation of a morphological skeleton."""
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
    """Symmetric soft-clDice loss for two single-channel stroke maps."""
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


class SketchSpectrumPerturb(nn.Module):
    """Class-protected direction/radial perturbation of a feature spectrum."""

    def __init__(self, channels, num_classes, alpha=1.0, mask_ratio=0.7,
                 perturb_prob=1.0, group_size=32, radial_bands=3,
                 orientation_bins=6, strength_max=0.3, warmup_steps=500,
                 ramp_steps=500, class_decay=0.99, class_min_count=20,
                 ready_ratio=0.5, gate_power=0.5, topology=False,
                 skeleton_iters=10, eps=1e-6):
        super().__init__()
        if channels % group_size:
            raise ValueError(
                f"channels={channels} must be divisible by group_size={group_size}")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")
        if radial_bands < 1 or orientation_bins < 1:
            raise ValueError("radial_bands and orientation_bins must be positive")
        if not (0.0 <= strength_max <= 1.0):
            raise ValueError("strength_max must be in [0, 1]")
        if not (0.0 < class_decay < 1.0):
            raise ValueError("class_decay must be in (0, 1)")
        if class_min_count < 1 or not (0.0 <= ready_ratio <= 1.0):
            raise ValueError("invalid class readiness configuration")
        if gate_power <= 0.0:
            raise ValueError("gate_power must be positive")

        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.group_size = int(group_size)
        self.groups = self.channels // self.group_size
        self.alpha = float(alpha)
        self.mask_ratio = float(mask_ratio)
        self.perturb_prob = float(perturb_prob)
        self.radial_bands = int(radial_bands)
        self.orientation_bins = int(orientation_bins)
        self.cells = self.radial_bands * self.orientation_bins
        self.stat_dim = 2 * self.cells
        self.strength_max = float(strength_max)
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.class_decay = float(class_decay)
        self.class_min_count = int(class_min_count)
        self.ready_ratio = float(ready_ratio)
        self.gate_power = float(gate_power)
        self.topology = bool(topology)
        self.skeleton_iters = int(skeleton_iters)
        self.eps = float(eps)

        stats_shape = (self.num_classes, self.groups, self.stat_dim)
        self.register_buffer("class_count", torch.zeros(
            self.num_classes, dtype=torch.long))
        self.register_buffer("class_mean", torch.zeros(stats_shape))
        self.register_buffer("class_second", torch.zeros(stats_shape))
        self.register_buffer("activation_step", torch.full((), -1, dtype=torch.long))

        self.register_buffer("last_strength", torch.zeros(()))
        self.register_buffer("last_class_ready_ratio", torch.zeros(()))
        self.register_buffer("last_fisher_score", torch.zeros(()))
        self.register_buffer("last_gate_mean", torch.ones(()))
        self.register_buffer("last_radial_shift", torch.zeros(()))
        self.register_buffer("last_orientation_drift", torch.zeros(()))
        self.register_buffer("last_phase_drift", torch.zeros(()))
        self.register_buffer("last_topology_loss", torch.zeros(()))

        self._mask_cache = {}
        self._context_labels = None
        self._context_step = None
        self._topology_loss = None

    def extra_repr(self):
        return (f"C={self.channels}, classes={self.num_classes}, groups={self.groups}, "
                f"radial={self.radial_bands}, orientation={self.orientation_bins}, "
                f"mask_ratio={self.mask_ratio}, strength_max={self.strength_max}, "
                f"topology={self.topology}")

    def set_batch_context(self, labels, step):
        """Supply class labels and the global step for one training forward."""
        if labels.ndim != 1:
            raise ValueError("labels must be a one-dimensional tensor")
        self._context_labels = labels
        self._context_step = int(step)

    def clear_batch_context(self):
        self._context_labels = None
        self._context_step = None

    def pop_topology_loss(self):
        loss = self._topology_loss
        self._topology_loss = None
        return loss

    def diagnostics(self):
        return {
            "sk_strength": float(self.last_strength.item()),
            "sk_class_ready_ratio": float(self.last_class_ready_ratio.item()),
            "sk_fisher_score": float(self.last_fisher_score.item()),
            "sk_gate_mean": float(self.last_gate_mean.item()),
            "sk_radial_shift": float(self.last_radial_shift.item()),
            "sk_orientation_drift": float(self.last_orientation_drift.item()),
            "sk_phase_drift": float(self.last_phase_drift.item()),
            "sk_topo_loss": float(self.last_topology_loss.item()),
        }

    @staticmethod
    def _conjugate_indices(length, device):
        center = length // 2
        indices = torch.arange(length, device=device)
        return torch.remainder(2 * center - indices, length)

    def _masks(self, height, width, device):
        key = (height, width, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]

        row = torch.arange(height, device=device, dtype=torch.float32) - height // 2
        col = torch.arange(width, device=device, dtype=torch.float32) - width // 2
        yy, xx = torch.meshgrid(row, col, indexing="ij")
        distance = torch.maximum(yy.abs(), xx.abs())
        cutoff = self.mask_ratio * min(height, width) / 2.0
        high = distance > cutoff

        radial_width = (distance.max() - cutoff).clamp_min(self.eps)
        radial_width = radial_width / self.radial_bands
        radial_index = torch.floor((distance - cutoff) / radial_width)
        radial_index = radial_index.long().clamp(0, self.radial_bands - 1)

        angle = torch.remainder(torch.atan2(yy, xx), math.pi)
        orientation_index = torch.floor(
            angle * self.orientation_bins / math.pi).long()
        orientation_index.clamp_(0, self.orientation_bins - 1)

        # The Nyquist row/column of an even FFT does not have a geometrically
        # opposite coordinate in the shifted grid.  Assign each conjugate pair
        # from one canonical member so every mask is exactly pair-symmetric.
        pair_row = self._conjugate_indices(height, device).view(height, 1)
        pair_col = self._conjugate_indices(width, device).view(1, width)
        flat = torch.arange(height * width, device=device).view(height, width)
        pair_flat = pair_row * width + pair_col
        canonical = torch.minimum(flat, pair_flat)
        orientation_index = orientation_index.flatten()[canonical]

        cell_masks = torch.stack([
            high & (radial_index == radial) & (orientation_index == orientation)
            for radial in range(self.radial_bands)
            for orientation in range(self.orientation_bins)
        ]).view(self.radial_bands, self.orientation_bins, height, width)
        if (not bool(cell_masks.flatten(2).any(-1).all())
                and not getattr(self, "allow_empty_cells", False)):
            raise ValueError(
                f"empty direction/radial cell for shape {(height, width)}, "
                f"mask_ratio={self.mask_ratio}, radial_bands={self.radial_bands}, "
                f"orientation_bins={self.orientation_bins}")
        orientation_masks = cell_masks.any(0)
        self._mask_cache[key] = (cell_masks, orientation_masks, high)
        return self._mask_cache[key]

    def _statistics(self, log_amplitude, cell_masks):
        batch, _, height, width = log_amplitude.shape
        grouped = log_amplitude.view(
            batch, self.groups, self.group_size, height, width)
        statistics = []
        for mask in cell_masks.flatten(0, 1):
            # A 14x14 ViT grid can have empty cells in the unchanged 3x6
            # partition. They carry no samples/energy: use finite placeholders,
            # not interpolated features or a different frequency mask. All
            # subsequent matching/energy operations on this empty mask are noops.
            # The historical ResNet path never enters this branch.
            if getattr(self, "allow_empty_cells", False) and not bool(mask.any()):
                zero = grouped.new_zeros((batch, self.groups))
                statistics.extend((zero, zero))
                continue
            values = grouped[..., mask].reshape(batch, self.groups, -1)
            mean = values.mean(-1)
            std = values.var(-1, unbiased=False).add(self.eps).sqrt()
            statistics.extend((mean, std.log()))
        return torch.stack(statistics, dim=-1), grouped

    @torch.no_grad()
    def _update_class_statistics(self, statistics, labels):
        labels = labels.detach().long()
        if labels.shape[0] != statistics.shape[0]:
            raise ValueError("label count does not match the feature batch")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
            raise ValueError("class label is outside the configured range")

        detached = statistics.detach()
        for class_id in labels.unique().tolist():
            selected = detached[labels == class_id]
            batch_mean = selected.mean(0)
            batch_second = selected.square().mean(0)
            old_count = int(self.class_count[class_id].item())
            if old_count == 0:
                self.class_mean[class_id].copy_(batch_mean)
                self.class_second[class_id].copy_(batch_second)
            else:
                self.class_mean[class_id].mul_(self.class_decay).add_(
                    batch_mean, alpha=1.0 - self.class_decay)
                self.class_second[class_id].mul_(self.class_decay).add_(
                    batch_second, alpha=1.0 - self.class_decay)
            self.class_count[class_id].add_(selected.shape[0])

    def _fisher_gate(self):
        ready = self.class_count >= self.class_min_count
        ready_ratio = ready.float().mean()
        if int(ready.sum().item()) < 2:
            fisher = self.class_mean.new_zeros((self.groups, self.stat_dim))
        else:
            means = self.class_mean[ready]
            seconds = self.class_second[ready]
            between = means.var(0, unbiased=False)
            within = (seconds - means.square()).clamp_min(0.0).mean(0)
            fisher = between / (between + within + self.eps)
        gate = (1.0 - fisher).clamp(0.0, 1.0).pow(self.gate_power)
        return fisher, gate, ready_ratio

    def _strength_at_step(self, step, ready_ratio):
        if (int(self.activation_step.item()) < 0
                and step >= self.warmup_steps
                and float(ready_ratio.item()) >= self.ready_ratio):
            self.activation_step.fill_(step)
        start = int(self.activation_step.item())
        if start < 0 or step <= start:
            return 0.0
        progress = (1.0 if self.ramp_steps <= 0 else
                    min(1.0, max(0.0, (step - start) / self.ramp_steps)))
        return self.strength_max * progress

    def _match_statistics(self, grouped, statistics, targets, cell_masks, strength):
        output = grouped.clone()
        for cell, mask in enumerate(cell_masks.flatten(0, 1)):
            values = grouped[..., mask]
            mean = statistics[..., 2 * cell].unsqueeze(-1).unsqueeze(-1)
            std = statistics[..., 2 * cell + 1].exp().unsqueeze(-1).unsqueeze(-1)
            target_mean = targets[..., 2 * cell].unsqueeze(-1).unsqueeze(-1)
            target_std = targets[..., 2 * cell + 1].clamp(-8.0, 8.0).exp()
            target_std = target_std.unsqueeze(-1).unsqueeze(-1)
            matched = target_std * (values - mean) / std.clamp_min(self.eps)
            matched = matched + target_mean
            mixed = values + strength * (matched - values)
            output[..., mask] = mixed.clamp(-20.0, 20.0)
        return output

    def _preserve_orientation_energy(self, original, changed, orientation_masks):
        output = changed.clone()
        for mask in orientation_masks:
            old_energy = original[..., mask].square().sum(-1)
            new_energy = output[..., mask].square().sum(-1)
            scale = ((old_energy + self.eps) / (new_energy + self.eps)).sqrt()
            output[..., mask] = output[..., mask] * scale.unsqueeze(-1)
        return output

    def _cell_energy(self, amplitude, cell_masks):
        values = [
            amplitude[..., mask].square().sum(-1)
            for mask in cell_masks.flatten(0, 1)
        ]
        shape = (*amplitude.shape[:2], self.radial_bands, self.orientation_bins)
        return torch.stack(values, -1).view(shape)

    def _set_spectrum_diagnostics(self, before_spec, after_spec,
                                  cell_masks, orientation_masks, high):
        with torch.no_grad():
            before_amp = before_spec.abs()
            after_amp = after_spec.abs()
            before_cells = self._cell_energy(before_amp, cell_masks)
            after_cells = self._cell_energy(after_amp, cell_masks)
            before_direction = before_cells.sum(2)
            after_direction = after_cells.sum(2)
            before_profile = before_cells / before_direction.unsqueeze(2).clamp_min(self.eps)
            after_profile = after_cells / after_direction.unsqueeze(2).clamp_min(self.eps)
            self.last_radial_shift.copy_(
                (after_profile - before_profile).abs().mean())
            relative_drift = (after_direction - before_direction).abs()
            relative_drift = relative_drift / before_direction.clamp_min(self.eps)
            self.last_orientation_drift.copy_(relative_drift.mean())

            valid = high.view(1, 1, *high.shape) & (before_amp > 1e-7)
            if bool(valid.any()):
                phase_delta = torch.angle(
                    after_spec[valid] * before_spec[valid].conj()).abs()
                self.last_phase_drift.copy_(phase_delta.max())
            else:
                self.last_phase_drift.zero_()

    def _stroke_map(self, feature):
        response = feature.float().square().mean(1, keepdim=True).add(self.eps).sqrt()
        mean = response.mean((2, 3), keepdim=True)
        std = response.var((2, 3), unbiased=False, keepdim=True).add(self.eps).sqrt()
        return torch.sigmoid((response - mean) / std)

    def forward(self, x):
        self._topology_loss = None
        if (not self.training or self._context_labels is None
                or self._context_step is None or self.alpha <= 0.0):
            return x

        batch, channels, height, width = x.shape
        if channels != self.channels:
            raise ValueError(f"expected {self.channels} channels, received {channels}")

        spectrum = torch.fft.fftshift(
            torch.fft.fft2(x.float(), dim=(2, 3), norm="ortho"), dim=(2, 3))
        raw_amplitude = spectrum.abs()
        amplitude = raw_amplitude.clamp_min(self.eps)
        cell_masks, orientation_masks, high = self._masks(height, width, x.device)
        statistics, grouped = self._statistics(amplitude.log(), cell_masks)

        self._update_class_statistics(statistics, self._context_labels)
        fisher, gate, ready_ratio = self._fisher_gate()
        strength = self._strength_at_step(self._context_step, ready_ratio)
        with torch.no_grad():
            self.last_strength.fill_(strength)
            self.last_class_ready_ratio.copy_(ready_ratio)
            self.last_fisher_score.copy_(fisher.mean())
            self.last_gate_mean.copy_(gate.mean())
            self.last_radial_shift.zero_()
            self.last_orientation_drift.zero_()
            self.last_phase_drift.zero_()
            self.last_topology_loss.zero_()

        if strength <= 0.0:
            return x
        if self.perturb_prob < 1.0:
            if float(torch.rand((), device=x.device).item()) > self.perturb_prob:
                return x

        batch_scale = statistics.var(0, unbiased=False).add(self.eps).sqrt()
        targets = statistics + (
            torch.randn_like(statistics) * self.alpha
            * batch_scale.unsqueeze(0) * gate.unsqueeze(0))
        changed_log_amplitude = self._match_statistics(
            grouped, statistics, targets, cell_masks, strength)
        changed_amplitude = changed_log_amplitude.reshape(
            batch, channels, height, width).exp()
        changed_amplitude = self._preserve_orientation_energy(
            raw_amplitude, changed_amplitude, orientation_masks)

        changed_spectrum = spectrum * (changed_amplitude / amplitude).to(spectrum.dtype)
        output = torch.fft.ifft2(
            torch.fft.ifftshift(changed_spectrum, dim=(2, 3)),
            dim=(2, 3), norm="ortho").real.to(x.dtype)
        self._set_spectrum_diagnostics(
            spectrum, changed_spectrum, cell_masks, orientation_masks, high)

        if self.topology:
            target_map = self._stroke_map(x).detach()
            prediction_map = self._stroke_map(output)
            self._topology_loss = soft_cldice_loss(
                prediction_map, target_map, self.skeleton_iters, self.eps)
            self.last_topology_loss.copy_(self._topology_loss.detach())
        return output


def _stage_out_channels(stage):
    block = stage[-1]
    return block.bn3.num_features if hasattr(block, "bn3") else block.bn2.num_features


def resnet_aloft_sketch(network, num_classes,
                        positions=("layer1", "layer2"), **kwargs):
    """Attach sketch-spectrum perturbation after the selected ResNet stages."""
    if getattr(network, "is_vit_backbone", False):
        if tuple(positions) != ("layer1", "layer2"):
            raise ValueError("ALOFT sketch perturbation must use layer1 and layer2 only")
        for name in positions:
            operation = SketchSpectrumPerturb(
                network.n_outputs, num_classes=num_classes, **kwargs)
            operation.allow_empty_cells = True
            network.add_stage_op(name, operation, "aloft_sketch")
        return network
    positions = tuple(positions)
    if positions != ("layer1", "layer2"):
        raise ValueError("ALOFT sketch perturbation must use layer1 and layer2 only")
    for name in positions:
        stage = getattr(network, name)
        if not isinstance(stage, nn.Sequential) or not len(stage):
            raise TypeError(f"{name} must be a non-empty nn.Sequential")
        if isinstance(stage[-1], SketchSpectrumPerturb):
            raise RuntimeError(f"{name} already has sketch-spectrum perturbation")
        module = SketchSpectrumPerturb(
            _stage_out_channels(stage), num_classes=num_classes, **kwargs)
        setattr(network, name, nn.Sequential(stage, module))
    return network


def find_sketch_spectrum_modules(module):
    return [m for m in module.modules() if isinstance(m, SketchSpectrumPerturb)]


def collect_sketch_topology_loss(module):
    losses = []
    for perturbation in find_sketch_spectrum_modules(module):
        loss = perturbation.pop_topology_loss()
        if loss is not None:
            losses.append(loss)
    return torch.stack(losses).mean() if losses else None


def collect_sketch_diagnostics(module):
    modules = find_sketch_spectrum_modules(module)
    if not modules:
        return {}
    diagnostics = [item.diagnostics() for item in modules]
    return {
        key: sum(values[key] for values in diagnostics) / len(diagnostics)
        for key in diagnostics[0]
    }
