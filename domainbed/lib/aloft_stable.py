"""Clean-view and weight helpers for ALOFT_Stable_E.

The original ALOFT perturbation and StableNet inner objective stay unchanged.
Only the combined algorithm uses these helpers.
"""
import math

import torch


@torch.no_grad()
def clean_features(featurizer, x):
    """Disable ALOFT/dropout and BN updates for one detached forward.

    Restore each module's flag, including mixed train/eval states, even if the
    forward fails. The same already-augmented input is used by both branches.
    """
    states = [(module, module.training) for module in featurizer.modules()]
    try:
        featurizer.eval()
        return featurizer(x).detach().float()
    finally:
        for module, training in states:
            module.training = training


def stable_mix_at_step(step, maximum, warmup, ramp):
    """Zero-based schedule: warmup updates, then ramp updates to maximum."""
    if not math.isfinite(maximum) or not 0.0 <= maximum <= 1.0:
        raise ValueError("stable_mix_max must be finite and in [0, 1]")
    if any(int(value) != value or value < 0 for value in (step, warmup, ramp)):
        raise ValueError("StableNet step, warmup and ramp must be non-negative integers")
    if step < warmup or maximum == 0.0:
        return 0.0
    if ramp == 0:
        return maximum
    return maximum * min(1.0, (step - warmup + 1) / ramp)


@torch.no_grad()
def domain_balanced_weights(raw, batch_sizes, mix):
    """Preserve each source domain's original batch mass, then shrink to ERM.

    A softmax within each domain is equivalent to normalizing global StableNet
    weights within that domain, but avoids division by an underflowed mass.
    The inner optimizer still uses the original, unconstrained objective.
    """
    if raw.ndim != 2 or raw.shape[1] != 1 or not torch.isfinite(raw).all():
        raise ValueError("Expected finite StableNet logits with shape [B, 1]")
    if (not batch_sizes or any(int(size) != size or size <= 0 for size in batch_sizes)
            or sum(batch_sizes) != raw.shape[0]):
        raise ValueError("Source batch sizes must be positive and sum to B")
    if not math.isfinite(mix) or not 0.0 <= mix <= 1.0:
        raise ValueError("StableNet mixing coefficient must be in [0, 1]")

    batch = raw.shape[0]
    balanced = torch.cat([
        logits.float().softmax(dim=0) * (size / batch)
        for logits, size in zip(raw.detach().split(batch_sizes), batch_sizes)
    ])
    return balanced * mix + (1.0 - mix) / batch
