import torch
import torch.nn as nn


class FreqQuant(nn.Module):
    """特征图高频幅度谱的量化瓶颈。与 ALOFT 不同：eval 时照样生效。"""

    def __init__(self, channels, levels=16, mask_ratio=0.5, low_gain=1.0,
                 momentum=0.1, eps=1e-6):
        super().__init__()
        self.channels = channels
        self.levels = levels
        self.mask_ratio = mask_ratio
        self.low_gain = low_gain
        self.momentum = momentum
        self.eps = eps
        self.enabled = False      # trainer 翻
        self.strength = 0.0       # 0→1 渐变
        self.aux_loss = None      # 第二、三步用，第一步一直是 None
        self.register_buffer("log_lo", torch.zeros(()))
        self.register_buffer("log_hi", torch.ones(()))
        self.register_buffer("inited", torch.zeros(()))
        self._mask_cache = {}

    def extra_repr(self):
        return (f"C={self.channels}, levels={self.levels}, "
                f"mask_ratio={self.mask_ratio}, low_gain={self.low_gain}")

    def _mask(self, h, w, device):
        """高频区：ALOFT Eq.(2) 中心方窗的补集。"""
        key = (h, w, str(device))
        if key not in self._mask_cache:
            half = self.mask_ratio * min(h, w) / 2.0
            u = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
            v = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
            d = torch.maximum((u - h // 2).abs(), (v - w // 2).abs())
            self._mask_cache[key] = (d > half).view(1, 1, h, w)
        return self._mask_cache[key]

    def forward(self, x):
        if not self.enabled or self.strength <= 0:
            return x

        B, C, H, W = x.shape
        spec = torch.fft.fft2(x.float(), dim=(2, 3), norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(2, 3))
        hi = self._mask(H, W, x.device)

        amp = torch.abs(spec).clamp_min(self.eps)
        amp_t = amp * self.low_gain if self.low_gain != 1.0 else amp
        amp_t = torch.where(hi, amp, amp_t)      # low_gain 只作用在低频

        amp_q, self.aux_loss = self.quantize(amp, hi)
        s = self.strength
        amp_new = torch.where(hi, (1 - s) * amp + s * amp_q, amp_t)

        # 关键：不用 angle/polar，直接乘一个实数增益，相位按构造保持不变
        spec = spec * (amp_new / amp).to(spec.dtype)
        spec = torch.fft.ifftshift(spec, dim=(2, 3))
        return torch.fft.ifft2(spec, dim=(2, 3), norm="ortho").real.to(x.dtype)

    def quantize(self, amp, hi):
        """第一步：对数域均匀分级 + 直通估计。返回 (量化幅度, aux_loss)。"""
        log_amp = torch.log(amp)

        if self.training:
            with torch.no_grad():
                v = log_amp.masked_select(hi)
                if v.numel() > 100_000:      # torch.quantile 有元素数上限，必须先抽样
                    idx = torch.randint(v.numel(), (100_000,), device=v.device)
                    v = v[idx]
                lo, up = v.quantile(0.01), v.quantile(0.99)
                if self.inited == 0:
                    self.log_lo.copy_(lo); self.log_hi.copy_(up); self.inited.fill_(1)
                else:
                    m = self.momentum
                    self.log_lo.mul_(1 - m).add_(m * lo)
                    self.log_hi.mul_(1 - m).add_(m * up)

        span = (self.log_hi - self.log_lo).clamp_min(self.eps)
        t = ((log_amp - self.log_lo) / span).clamp(0, 1)
        q = torch.round(t * (self.levels - 1)) / (self.levels - 1)
        q = t + (q - t).detach()                       # STE
        return torch.exp(q * span + self.log_lo), None

        
def _out_channels(block):
    return block.bn3.num_features if hasattr(block, "bn3") else block.bn2.num_features


def resnet_freqquant(network, layers=("layer1", "layer2", "layer3"), **kwargs):
    """按 block 插。必须在预训练权重加载之后调用，且不幂等。"""
    for name in layers:
        stage = getattr(network, name)
        for i, block in enumerate(stage):
            if isinstance(block, nn.Sequential) and isinstance(block[-1], FreqQuant):
                raise RuntimeError(f"{name}[{i}] already wrapped")
            stage[i] = nn.Sequential(block, FreqQuant(_out_channels(block), **kwargs))
    return network


def collect_aux_loss(module):
    """把各模块攒的辅助损失取走并清空，第一步返回 0。"""
    total = 0.0
    for m in module.modules():
        if isinstance(m, FreqQuant) and m.aux_loss is not None:
            total = total + m.aux_loss
            m.aux_loss = None
    return total