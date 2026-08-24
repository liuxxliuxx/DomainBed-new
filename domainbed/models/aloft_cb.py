"""
ALOFT with a band-statistics codebook on the final FFT position.

The first positions keep the original ALOFT perturbation.  The final position
does not perturb with ALOFT; it learns a codebook over high-frequency band
statistics and optionally quantizes those statistics.  The codebook can be
warmed while its output remains an identity mapping.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


PERTURB_TARGET = "complex"


def codebook_strength_at_step(step, start_step, ramp_steps, strength_max):
    """Linear 0-to-max ramp used by the trainer for the final-stage codebook."""
    if step < start_step:
        return 0.0
    progress = (1.0 if ramp_steps <= 0 else
                min(1.0, max(0.0, (step - start_step) / ramp_steps)))
    return float(strength_max) * progress


class ALOFT(nn.Module):
    """ALOFT perturbation copied from models/aloft.py for an independent variant."""

    def __init__(self, mode="E", alpha=1.0, mask_ratio=0.5,
                 perturb_prob=1.0, eps=1e-6, rev=False):
        super().__init__()
        assert mode in ("E", "S"), f"unknown ALOFT mode: {mode}"
        self.mode = mode
        self.rev = rev
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.perturb_prob = perturb_prob
        self.eps = eps
        self._mask_cache = {}

    def extra_repr(self):
        return (f"mode={self.mode}, alpha={self.alpha}, mask_ratio={self.mask_ratio}, "
                f"rev={self.rev}, perturb_prob={self.perturb_prob}")

    def _mask(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._mask_cache:
            half = self.mask_ratio * min(h, w) / 2.0
            u = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
            v = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
            d = torch.maximum((u - h // 2).abs(), (v - w // 2).abs())
            sel = (d > half) if self.rev else (d <= half)
            self._mask_cache[key] = sel.view(1, 1, h, w)
        return self._mask_cache[key]

    def forward(self, x):
        if not self.training or self.alpha <= 0:
            return x
        if self.perturb_prob < 1.0 and random.random() > self.perturb_prob:
            return x

        b, c, h, w = x.shape
        spec = torch.fft.fft2(x.float(), dim=(2, 3), norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(2, 3))
        mask = self._mask(h, w, x.device)

        if PERTURB_TARGET == "amplitude":
            amp, phase = torch.abs(spec), torch.angle(spec)
            spec = torch.polar(self._resample(amp, mask), phase)
        else:
            real = torch.view_as_real(spec)
            real = real.permute(0, 1, 4, 2, 3).reshape(b, 2 * c, h, w)
            real = self._resample(real, mask)
            real = real.reshape(b, c, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
            spec = torch.view_as_complex(real)

        spec = torch.fft.ifftshift(spec, dim=(2, 3))
        return torch.fft.ifft2(spec, dim=(2, 3), norm="ortho").real.to(x.dtype)

    def _resample(self, f, mask):
        return self._by_element(f, mask) if self.mode == "E" else self._by_statistic(f, mask)

    def _by_element(self, f, mask):
        sigma = (f.var(dim=0, unbiased=False, keepdim=True) + self.eps).sqrt()
        noise = torch.randn_like(f) * self.alpha * sigma
        return torch.where(mask, f + noise, f)

    def _by_statistic(self, f, mask):
        mf = mask.to(f.dtype)
        n = mf.sum().clamp(min=1.0)
        mu = (f * mf).sum(dim=(2, 3), keepdim=True) / n
        var = (((f - mu) ** 2) * mf).sum(dim=(2, 3), keepdim=True) / n
        sig = (var + self.eps).sqrt()
        sig_of_mu = (mu.var(dim=0, unbiased=False, keepdim=True) + self.eps).sqrt()
        sig_of_sig = (sig.var(dim=0, unbiased=False, keepdim=True) + self.eps).sqrt()
        mu_hat = mu + torch.randn_like(mu) * self.alpha * sig_of_mu
        sig_hat = sig + torch.randn_like(sig) * self.alpha * sig_of_sig
        new = sig_hat * (f - mu) / sig + mu_hat
        return torch.where(mask, new, f)


class BandStatsCodebook(nn.Module):
    """EMA codebook over grouped high-frequency band statistics.

    Each vector contains ``[mean(log_amp), log(std(log_amp))]`` for every
    frequency band.  Quantization changes only these statistics; channel
    directions, unselected frequencies, and phase are preserved.
    """

    def __init__(self, channels, mask_ratio=0.7, n_bands=3, codebook=256,
                 group_size=32, strength_max=0.2, decay=0.99,
                 dead_patience=200, reservoir_size=1024, eps=1e-6):
        super().__init__()
        if channels % group_size:
            raise ValueError(f"channels={channels} must be divisible by group_size={group_size}")
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError("mask_ratio must be in (0, 1)")
        if n_bands < 1 or codebook < 1 or reservoir_size < codebook:
            raise ValueError("n_bands/codebook must be positive and reservoir_size >= codebook")
        if not (0.0 <= strength_max <= 1.0):
            raise ValueError("strength_max must be in [0, 1]")
        if not (0.0 < decay < 1.0):
            raise ValueError("decay must be in (0, 1)")

        self.channels = int(channels)
        self.group_size = int(group_size)
        self.groups = self.channels // self.group_size
        self.mask_ratio = float(mask_ratio)
        self.n_bands = int(n_bands)
        self.K = int(codebook)
        self.stat_dim = 2 * self.n_bands
        self.strength_max = float(strength_max)
        self.decay = float(decay)
        self.dead_patience = int(dead_patience)
        self.reservoir_size = int(reservoir_size)
        self.eps = float(eps)

        self.collect_enabled = False
        self.enabled = False
        self.frozen = False
        self.strength = 0.0
        self._band_cache = {}

        shape = (self.groups, self.K, self.stat_dim)
        self.register_buffer("emb", torch.zeros(shape))
        self.register_buffer("ema_count", torch.zeros(self.groups, self.K))
        self.register_buffer("ema_sum", torch.zeros(shape))
        self.register_buffer("inactive_steps", torch.zeros(
            self.groups, self.K, dtype=torch.long))
        self.register_buffer("reservoir", torch.zeros(
            self.groups, self.reservoir_size, self.stat_dim))
        self.register_buffer("reservoir_count", torch.zeros(
            self.groups, dtype=torch.long))
        self.register_buffer("initialized", torch.zeros(
            self.groups, dtype=torch.bool))

        self.register_buffer("last_perplexity", torch.zeros(()))
        self.register_buffer("last_active_ratio", torch.zeros(()))
        self.register_buffer("last_max_share", torch.zeros(()))
        self.register_buffer("last_quant_error", torch.zeros(()))
        self.register_buffer("last_dead_restarts", torch.zeros((), dtype=torch.long))

    def extra_repr(self):
        return (f"C={self.channels}, groups={self.groups}, group_size={self.group_size}, "
                f"K={self.K}, bands={self.n_bands}, mask_ratio={self.mask_ratio}, "
                f"strength_max={self.strength_max}, decay={self.decay}")

    def get_extra_state(self):
        # Keep inference/freeze behavior when a checkpoint is loaded later.
        return {
            "collect_enabled": self.collect_enabled,
            "enabled": self.enabled,
            "frozen": self.frozen,
            "strength": self.strength,
        }

    def set_extra_state(self, state):
        self.collect_enabled = bool(state.get("collect_enabled", False))
        self.enabled = bool(state.get("enabled", False))
        self.frozen = bool(state.get("frozen", False))
        self.strength = min(
            self.strength_max, max(0.0, float(state.get("strength", 0.0))))

    def set_collection(self, enabled):
        self.collect_enabled = bool(enabled) and not self.frozen

    def set_quantization(self, enabled, strength=0.0):
        self.enabled = bool(enabled)
        self.strength = min(self.strength_max, max(0.0, float(strength)))

    def freeze_codebook(self):
        self.frozen = True
        self.collect_enabled = False

    def diagnostics(self):
        return {
            "cb_inited": float(self.initialized.all().item()),
            "cb_strength": float(self.strength),
            "cb_perplexity": float(self.last_perplexity.item()),
            "cb_active_ratio": float(self.last_active_ratio.item()),
            "cb_max_share": float(self.last_max_share.item()),
            "cb_quant_error": float(self.last_quant_error.item()),
            "cb_dead_restarts": float(self.last_dead_restarts.item()),
        }

    def _bands(self, h, w, device):
        key = (h, w, str(device))
        if key in self._band_cache:
            return self._band_cache[key]

        u = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
        v = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
        dist = torch.maximum((u - h // 2).abs(), (v - w // 2).abs())
        cutoff = self.mask_ratio * min(h, w) / 2.0
        high = dist > cutoff
        max_dist = dist.max()
        width = (max_dist - cutoff).clamp_min(self.eps) / self.n_bands
        band_idx = torch.floor((dist - cutoff) / width).long().clamp(0, self.n_bands - 1)
        masks = torch.stack([high & (band_idx == i) for i in range(self.n_bands)])
        if not bool(masks.flatten(1).any(1).all()):
            raise ValueError(
                f"empty frequency band for shape {(h, w)}, mask_ratio={self.mask_ratio}, "
                f"n_bands={self.n_bands}")
        self._band_cache[key] = masks
        return masks

    def _statistics(self, log_amp, masks):
        b, _, h, w = log_amp.shape
        grouped = log_amp.view(b, self.groups, self.group_size, h, w)
        parts = []
        for mask in masks:
            values = grouped[..., mask].reshape(b, self.groups, -1)
            mu = values.mean(-1)
            sig = values.var(-1, unbiased=False).add(self.eps).sqrt()
            parts.extend((mu, sig.log()))
        return torch.stack(parts, dim=-1), grouped

    def _fill_reservoir(self, stats):
        b = stats.shape[0]
        newly_initialized = False
        for g in range(self.groups):
            count = int(self.reservoir_count[g].item())
            take = min(b, self.reservoir_size - count)
            if take > 0:
                self.reservoir[g, count:count + take].copy_(stats[:take, g])
                self.reservoir_count[g] += take
            if (not bool(self.initialized[g].item())
                    and int(self.reservoir_count[g].item()) >= self.K):
                available = int(self.reservoir_count[g].item())
                # Use K real observations directly and deterministically.  This
                # keeps initialization independent of the global RNG and avoids
                # synthetic/Gaussian codewords.
                indices = torch.linspace(
                    0, available - 1, self.K, device=stats.device).long()
                initial = self.reservoir[g, indices]
                self.emb[g].copy_(initial)
                self.ema_sum[g].copy_(initial)
                self.ema_count[g].fill_(1.0)
                self.initialized[g] = True
                newly_initialized = True
        return newly_initialized

    def _nearest(self, stats):
        dist = (stats.pow(2).sum(-1, keepdim=True)
                - 2.0 * torch.einsum("bgd,gkd->bgk", stats, self.emb)
                + self.emb.pow(2).sum(-1).unsqueeze(0))
        idx = dist.argmin(-1)
        quantized = torch.stack(
            [self.emb[g][idx[:, g]] for g in range(self.groups)], dim=1)
        return quantized, idx

    def _set_diagnostics(self, stats, quantized, idx, dead_restarts=0):
        with torch.no_grad():
            counts = torch.stack([
                torch.bincount(idx[:, g], minlength=self.K)
                for g in range(self.groups)
            ]).to(stats.dtype)
            probs = counts / counts.sum(-1, keepdim=True).clamp_min(1.0)
            entropy = -(probs * probs.clamp_min(self.eps).log()).sum(-1)
            self.last_perplexity.copy_(entropy.exp().mean())
            self.last_active_ratio.copy_((counts > 0).float().mean())
            self.last_max_share.copy_(probs.max(-1).values.mean())
            self.last_quant_error.copy_(F.mse_loss(stats, quantized))
            self.last_dead_restarts.fill_(int(dead_restarts))

    @torch.no_grad()
    def _update_codebook(self, stats):
        newly_initialized = self._fill_reservoir(stats)
        if not bool(self.initialized.all()):
            return

        quantized, idx = self._nearest(stats)
        if newly_initialized:
            self._set_diagnostics(stats, quantized, idx)
            return

        dead_restarts = 0
        for g in range(self.groups):
            one_hot = F.one_hot(idx[:, g], self.K).to(stats.dtype)
            counts = one_hot.sum(0)
            sums = one_hot.t() @ stats[:, g]
            self.ema_count[g].mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
            self.ema_sum[g].mul_(self.decay).add_(sums, alpha=1.0 - self.decay)
            self.emb[g].copy_(
                self.ema_sum[g] / self.ema_count[g].unsqueeze(1).clamp_min(self.eps))

            active = counts > 0
            self.inactive_steps[g][active] = 0
            self.inactive_steps[g][~active] += 1
            dead = self.inactive_steps[g] >= self.dead_patience
            n_dead = int(dead.sum().item())
            if n_dead:
                sample_idx = torch.randint(stats.shape[0], (n_dead,), device=stats.device)
                replacement = stats[sample_idx, g]
                self.emb[g][dead] = replacement
                self.ema_sum[g][dead] = replacement
                self.ema_count[g][dead] = 1.0
                self.inactive_steps[g][dead] = 0
                dead_restarts += n_dead

        quantized, idx = self._nearest(stats)
        self._set_diagnostics(stats, quantized, idx, dead_restarts)

    def _match_statistics(self, grouped, stats, targets, masks):
        out = grouped.clone()
        strength = self.strength
        for band, mask in enumerate(masks):
            values = grouped[..., mask]
            mu = stats[..., 2 * band].unsqueeze(-1).unsqueeze(-1)
            sig = stats[..., 2 * band + 1].exp().unsqueeze(-1).unsqueeze(-1)
            target_mu = targets[..., 2 * band].unsqueeze(-1).unsqueeze(-1)
            target_sig = targets[..., 2 * band + 1].exp().unsqueeze(-1).unsqueeze(-1)
            matched = target_sig * (values - mu) / sig.clamp_min(self.eps) + target_mu
            out[..., mask] = values + strength * (matched - values)
        return out

    def forward(self, x):
        should_collect = self.training and self.collect_enabled and not self.frozen
        should_quantize = self.enabled and self.strength > 0.0
        if not should_collect and not should_quantize:
            return x

        b, c, h, w = x.shape
        spec = torch.fft.fftshift(
            torch.fft.fft2(x.float(), dim=(2, 3), norm="ortho"), dim=(2, 3))
        amp = torch.abs(spec).clamp_min(self.eps)
        masks = self._bands(h, w, x.device)
        stats, grouped = self._statistics(amp.log(), masks)

        if should_collect:
            self._update_codebook(stats.detach())

        if not should_quantize or not bool(self.initialized.all()):
            return x

        targets, idx = self._nearest(stats)
        self._set_diagnostics(stats.detach(), targets.detach(), idx)
        log_amp = self._match_statistics(grouped, stats, targets.detach(), masks)
        log_amp = log_amp.reshape(b, c, h, w)
        amp_new = log_amp.exp()
        spec = spec * (amp_new / amp).to(spec.dtype)
        spec = torch.fft.ifftshift(spec, dim=(2, 3))
        return torch.fft.ifft2(spec, dim=(2, 3), norm="ortho").real.to(x.dtype)


def _stage_out_channels(stage):
    block = stage[-1]
    return block.bn3.num_features if hasattr(block, "bn3") else block.bn2.num_features


def resnet_aloft_cb(network, positions=("layer1", "layer2", "layer3"), **kwargs):
    """Attach high-frequency ALOFT to early stages and a codebook to the last."""
    positions = tuple(positions)
    if not positions:
        raise ValueError("positions must contain at least one ResNet stage")

    cb_keys = {
        "codebook", "group_size", "n_bands", "strength_max", "decay",
        "dead_patience", "reservoir_size"
    }
    aloft_kwargs = {k: v for k, v in kwargs.items() if k not in cb_keys}
    cb_kwargs = {k: v for k, v in kwargs.items() if k in cb_keys}
    cb_kwargs["mask_ratio"] = kwargs.get("mask_ratio", 0.7)

    for index, name in enumerate(positions):
        stage = getattr(network, name)
        if not isinstance(stage, nn.Sequential) or not len(stage):
            raise TypeError(f"{name} must be a non-empty nn.Sequential")
        if isinstance(stage[-1], (ALOFT, BandStatsCodebook)):
            raise RuntimeError(f"{name} already wrapped with ALOFT-CB")
        if index == len(positions) - 1:
            module = BandStatsCodebook(_stage_out_channels(stage), **cb_kwargs)
        else:
            module = ALOFT(**aloft_kwargs)
        setattr(network, name, nn.Sequential(stage, module))
    return network


def find_band_codebooks(module):
    return [m for m in module.modules() if isinstance(m, BandStatsCodebook)]
