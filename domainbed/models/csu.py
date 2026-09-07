"""Correlated Style Uncertainty (CSU) for torchvision ResNet models.

Adapted from the classification implementation released with
"Domain Generalization with Correlated Style Uncertainty" (WACV 2024).
"""

import numpy as np
import torch
import torch.nn as nn


class CorrelatedDistributionUncertainty(nn.Module):
    """Perturb feature statistics while preserving channel correlations."""

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)

    def extra_repr(self):
        return f"p={self.p}, alpha={self.alpha}, eps={self.eps}"

    def _eigenvectors(self, covariance):
        channels = covariance.shape[0]
        identity = torch.eye(
            channels, device=covariance.device, dtype=covariance.dtype)

        with torch.no_grad():
            try:
                _, eigenvectors = torch.linalg.eigh(
                    channels * covariance + self.eps * identity)
            except RuntimeError:
                return identity

            if not torch.isfinite(eigenvectors).all():
                return identity

        return eigenvectors

    @staticmethod
    def _covariance(statistic):
        centered = statistic - statistic.mean(dim=0, keepdim=True)
        return centered.transpose(0, 1) @ centered / statistic.shape[0]

    def _correlation_root(self, covariance):
        eigenvectors = self._eigenvectors(covariance)
        projected = eigenvectors.transpose(0, 1) @ covariance @ eigenvectors
        scales = projected.diagonal().clamp_min(1e-12).sqrt()
        return (eigenvectors * scales.unsqueeze(0)) @ eigenvectors.transpose(0, 1)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        if x.ndim != 4:
            raise ValueError(f"CSU expects BCHW input, got shape {tuple(x.shape)}")
        if x.shape[0] < 2 or np.random.random() > self.p:
            return x

        batch, channels = x.shape[:2]
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            return x

        covariance_mean = self._covariance(mean.flatten(1))
        covariance_std = self._covariance(std.flatten(1))
        if not torch.isfinite(covariance_mean).all() or not torch.isfinite(covariance_std).all():
            return x

        mean_root = self._correlation_root(covariance_mean)
        std_root = self._correlation_root(covariance_std)
        # The released implementation samples the Beta strength before the
        # Gaussian offsets; keep that order for seeded reproducibility.
        factor = self.beta.sample((batch, 1, 1, 1)).to(
            device=x.device, dtype=x.dtype)
        mean_noise = torch.randn(
            batch, 1, channels, device=x.device, dtype=x.dtype) @ mean_root
        std_noise = torch.randn(
            batch, 1, channels, device=x.device, dtype=x.dtype) @ std_root

        sampled_mean = mean + factor * mean_noise.reshape(batch, channels, 1, 1)
        sampled_std = std + factor * std_noise.reshape(batch, channels, 1, 1)
        normalized = (x - mean) / std
        return normalized * sampled_std + sampled_mean


VALID_CSU_POSITIONS = (
    "conv1", "maxpool", "layer1", "layer2", "layer3", "layer4")


def resnet_csu(network, positions=("maxpool", "layer1"), **kwargs):
    """Attach CSU modules after selected torchvision ResNet components.

    Pretrained weights must be loaded before calling this function because the
    wrappers add one level to the selected components' state-dict keys.
    """

    if getattr(network, "is_vit_backbone", False):
        for name in positions:
            network.add_stage_op(name, CorrelatedDistributionUncertainty(**kwargs), "csu")
        return network
    positions = tuple(positions)
    unknown = sorted(set(positions) - set(VALID_CSU_POSITIONS))
    if unknown:
        raise ValueError(
            f"unknown CSU positions {unknown}; choose from {VALID_CSU_POSITIONS}")
    if len(set(positions)) != len(positions):
        raise ValueError(f"duplicate CSU positions: {positions}")

    for name in positions:
        component = getattr(network, name)
        if (isinstance(component, nn.Sequential) and len(component)
                and isinstance(component[-1], CorrelatedDistributionUncertainty)):
            raise RuntimeError(f"{name} is already wrapped with CSU")
        setattr(
            network,
            name,
            nn.Sequential(component, CorrelatedDistributionUncertainty(**kwargs)),
        )

    return network


def find_csu_modules(module):
    return [
        item for item in module.modules()
        if isinstance(item, CorrelatedDistributionUncertainty)
    ]
