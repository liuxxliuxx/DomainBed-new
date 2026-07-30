"""
ALOFT: dynAmic LOw-Frequency spectrum Transform (CVPR 2023).

模块化实现，作用在 CNN 特征图上。凡是官方仓库 (lingeringlight/ALOFT)
与论文正文不一致的地方，一律按论文实现。
"""
import random
import torch
import torch.nn as nn


# 论文 Eq.(6)(7)(14) 直接写在复数低频谱 F_l 上；官方代码把复数谱拆成幅度谱和
# 相位谱、只扰动幅度。默认按论文走 "complex"，改成 "amplitude" 即切回官方行为。
PERTURB_TARGET = "complex"



class ALOFT(nn.Module):
    """特征图 (B, C, H, W) 上的频域扰动，无可学习参数，eval 时是恒等映射。

    forward 严格对应论文 3.3 节：
      Eq.(1)      2D DFT，零频移到中心
      Eq.(2)      二值掩码 M，切出低频区
      Eq.(6)(7)   mode="E"：按元素建模并重采样
      Eq.(8)-(14) mode="S"：按通道统计量建模并重采样
      Eq.(3)(4)   掩码外的高频分量原样保留
      Eq.(5)      逆 DFT 回空间域
    """

    def __init__(self, mode="E", alpha=1.0, mask_ratio=0.5, perturb_prob=1.0, eps=1e-6):
        super().__init__()
        assert mode in ("E", "S"), f"unknown ALOFT mode: {mode}"
        self.mode = mode
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.perturb_prob = perturb_prob
        self.eps = eps
        self._mask_cache = {}

    def extra_repr(self):
        return (f"mode={self.mode}, alpha={self.alpha}, mask_ratio={self.mask_ratio}, "
                f"perturb_prob={self.perturb_prob}")
        
    def _mask(self, h, w, device):
        """论文 Eq.(2)：以频谱中心为心、半边长 r*min(H,W)/2 的正方形二值掩码。"""
        key = (h, w, str(device))
        if key not in self._mask_cache:
            half = self.mask_ratio * min(h, w) / 2.0
            u = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
            v = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
            d = torch.maximum((u - h // 2).abs(), (v - w // 2).abs())
            self._mask_cache[key] = (d <= half).view(1, 1, h, w)
        return self._mask_cache[key]

    def forward(self, x):
        if not self.training or self.alpha <= 0:
            return x
        if self.perturb_prob < 1.0 and random.random() > self.perturb_prob:
            return x

        B, C, H, W = x.shape
        # Eq.(1)：用完整 fft2 而非 rfft2，Eq.(2) 的中心方窗才能原样落地
        spec = torch.fft.fft2(x.float(), dim=(2, 3), norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(2, 3))
        m = self._mask(H, W, x.device)

        if PERTURB_TARGET == "amplitude":
            amp, pha = torch.abs(spec), torch.angle(spec)
            spec = torch.polar(self._resample(amp, m), pha)
        else:
            # 复数谱按实部/虚部展成 2C 个实通道，Eq.(6)-(14) 逐分量作用
            s = torch.view_as_real(spec)                       # (B,C,H,W,2)
            s = s.permute(0, 1, 4, 2, 3).reshape(B, 2 * C, H, W)
            s = self._resample(s, m)
            s = s.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2).contiguous()
            spec = torch.view_as_complex(s)

        spec = torch.fft.ifftshift(spec, dim=(2, 3))
        # Eq.(5)：随机噪声会破坏共轭对称，取实部等于投影回实信号空间
        return torch.fft.ifft2(spec, dim=(2, 3), norm="ortho").real.to(x.dtype)

    def _resample(self, f, m):
        return self._by_element(f, m) if self.mode == "E" else self._by_statistic(f, m)

    def _by_element(self, f, m):
        """ALOFT-E，论文 Eq.(6)(7)。"""
        # Eq.(6)：Sigma^2 沿 batch 维按元素统计，分母是 B（有偏），照论文写法
        sigma = (f.var(dim=0, unbiased=False, keepdim=True) + self.eps).sqrt()
        # Eq.(7)：F_l_hat = F_l + eps * Sigma(F_l)，eps ~ N(0, alpha)
        noise = torch.randn_like(f) * self.alpha * sigma
        return torch.where(m, f + noise, f)     # 掩码外即 Eq.(4) 的高频，原样保留

    def _by_statistic(self, f, m):
        """ALOFT-S，论文 Eq.(8)-(14)。"""
        mf = m.to(f.dtype)
        n = mf.sum().clamp(min=1.0)             # 低频区元素个数
        # Eq.(8)(9)：逐样本逐通道，在低频区上求均值和标准差
        mu = (f * mf).sum(dim=(2, 3), keepdim=True) / n
        var = (((f - mu) ** 2) * mf).sum(dim=(2, 3), keepdim=True) / n
        sig = (var + self.eps).sqrt()
        # Eq.(10)(11)：两个统计量各自沿 batch 维的标准差，分母 B（有偏）
        sig_of_mu = (mu.var(dim=0, unbiased=False, keepdim=True) + self.eps).sqrt()
        sig_of_sig = (sig.var(dim=0, unbiased=False, keepdim=True) + self.eps).sqrt()
        # Eq.(12)(13)：各用各的 Sigma
        mu_hat = mu + torch.randn_like(mu) * self.alpha * sig_of_mu
        sig_hat = sig + torch.randn_like(sig) * self.alpha * sig_of_sig
        # Eq.(14)：标准 AdaIN 顺序，sigma_hat 缩放、mu_hat 偏置
        new = sig_hat * (f - mu) / sig + mu_hat
        return torch.where(m, new, f)


def resnet_aloft(network, positions=("layer1", "layer2", "layer3"), **kwargs):
    """把 ALOFT 挂到 torchvision ResNet 指定 stage 的输出上。

    必须在预训练权重加载完之后调用，否则 nn.Sequential 会给权重键名多插一层 ".0."。
    不幂等，重复调用会叠加模块，所以显式挡住。
    """
    for name in positions:
        stage = getattr(network, name)
        if isinstance(stage, nn.Sequential) and len(stage) and isinstance(stage[-1], ALOFT):
            raise RuntimeError(f"{name} already wrapped with ALOFT; resnet_aloft is not idempotent")
        setattr(network, name, nn.Sequential(stage, ALOFT(**kwargs)))
    return network