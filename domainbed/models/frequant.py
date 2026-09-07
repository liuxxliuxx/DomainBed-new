"""特征图上的瓶颈模块，四种机制共用一套开关。

与 ALOFT 不同：eval 时照样生效——这是架构组件不是数据增强，训练和推理必须一致。

mode:
  quant     频带内幅度谱做对数域标量量化（原行为）
  clip      只把幅度谱压回 [log_lo, log_hi]，不分级
  cb_freq   对频带内每个位置的 C 维通道幅度向量做向量量化
  cb_feat   不做 FFT，直接对特征图每个空间位置的 C 维向量做向量量化

quant 和 clip 合起来能把「量化」和「截断」两个成分分解开：
  quant + quantile=0     只量化不截断
  clip  + quantile=0.01  只截断不量化
这一步是必要的——levels 32/64/128/256 四档结果几乎相同，说明起作用的
很可能是那个和 levels 无关的截断，而不是量化本身。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqQuant(nn.Module):
    def __init__(self, channels, mode="quant", band="high", levels=16,
                 mask_ratio=0.5, low_gain=1.0, quantile=0.01,
                 codebook=256, groups=1, strength_max=1.0,
                 momentum=0.1, dead_thr=0.5, eps=1e-6):
        super().__init__()
        assert mode in ("quant", "clip", "cb_freq", "cb_feat"), f"unknown fq_mode: {mode}"
        assert band in ("high", "low"), f"unknown fq_band: {band}"
        self.channels = channels
        self.mode = mode
        self.band = band
        self.levels = levels
        self.mask_ratio = mask_ratio
        self.low_gain = low_gain
        self.quantile = quantile
        self.K = int(codebook)
        self.G = max(1, int(groups))
        self.momentum = momentum
        self.dead_thr = dead_thr
        self.eps = eps
        self.enabled = False      # trainer 翻
        self.strength = 0.0       # 0 -> strength_max 渐变
        self.strength_max = float(strength_max)   # 封顶，<1 时输出是原值和量化值的混合
        self.aux_loss = None      # commitment loss，由 collect_aux_loss 取走
        self.register_buffer("log_lo", torch.zeros(()))
        self.register_buffer("log_hi", torch.ones(()))
        self.register_buffer("inited", torch.zeros(()))
        self._mask_cache = {}

        if mode.startswith("cb_"):
            assert channels % self.G == 0, f"channels {channels} 不能被 groups {self.G} 整除"
            d = channels // self.G
            # 码本走 EMA 更新、注册成 buffer：不进 parameters()，所以算法侧
            # 已经建好的 optimizer 不需要任何改动
            self.register_buffer("emb", torch.randn(self.G, self.K, d) * 0.1)
            self.register_buffer("ema_n", torch.zeros(self.G, self.K))
            self.register_buffer("ema_sum", torch.zeros(self.G, self.K, d))
            # CNN 相邻通道相关性强，连续切分会让 PQ「各组独立」的假设破得更厉害，
            # 先用固定随机置换打散。置换必须是 buffer，训练和推理才一致。
            perm = torch.randperm(channels)
            self.register_buffer("perm", perm)
            self.register_buffer("inv", torch.argsort(perm))

    def extra_repr(self):
        s = (f"C={self.channels}, mode={self.mode}, band={self.band}, "
             f"mask_ratio={self.mask_ratio}, quantile={self.quantile}")
        if self.mode.startswith("cb_"):
            return s + f", K={self.K}, G={self.G}"
        return s + f", levels={self.levels}, low_gain={self.low_gain}"

    # ------------------------------------------------------------------ 掩码
    def _mask(self, h, w, device):
        """选中要处理的频带。band=high 是 ALOFT Eq.(2) 中心方窗的补集。"""
        key = (h, w, str(device))
        if key not in self._mask_cache:
            half = self.mask_ratio * min(h, w) / 2.0
            u = torch.arange(h, device=device, dtype=torch.float32).view(h, 1)
            v = torch.arange(w, device=device, dtype=torch.float32).view(1, w)
            d = torch.maximum((u - h // 2).abs(), (v - w // 2).abs())
            sel = (d > half) if self.band == "high" else (d <= half)
            self._mask_cache[key] = sel.view(1, 1, h, w)
        return self._mask_cache[key]

    # ------------------------------------------------------------------ 前向
    def forward(self, x):
        if not self.enabled or self.strength <= 0:
            return x
        return self._forward_feat(x) if self.mode == "cb_feat" else self._forward_freq(x)

    def _forward_freq(self, x):
        B, C, H, W = x.shape
        spec = torch.fft.fft2(x.float(), dim=(2, 3), norm="ortho")
        spec = torch.fft.fftshift(spec, dim=(2, 3))
        m = self._mask(H, W, x.device)

        amp = torch.abs(spec).clamp_min(self.eps)
        # low_gain 作用在**没被处理的那一带**上，是个独立的旋钮
        other = amp * self.low_gain if self.low_gain != 1.0 else amp
        amp_t = torch.where(m, amp, other)

        if self.mode == "quant":
            amp_q, aux = self._scalar(amp, m)
        elif self.mode == "clip":
            amp_q, aux = self._clip(amp, m)
        else:
            amp_q, aux = self._vq_amp(amp, m)
        self.aux_loss = aux

        s = self.strength
        amp_new = torch.where(m, (1 - s) * amp + s * amp_q, amp_t)
        # 不用 angle/polar，直接乘一个实数增益，相位按构造保持不变
        spec = spec * (amp_new / amp).to(spec.dtype)
        spec = torch.fft.ifftshift(spec, dim=(2, 3))
        return torch.fft.ifft2(spec, dim=(2, 3), norm="ortho").real.to(x.dtype)

    def _forward_feat(self, x):
        """DDG 式：不碰频域，直接量化特征图每个空间位置的通道向量。"""
        B, C, H, W = x.shape
        z = x.permute(0, 2, 3, 1).reshape(-1, C).float()
        zq, self.aux_loss = self._vq_normed(z)
        out = zq.view(B, H, W, C).permute(0, 3, 1, 2)
        s = self.strength
        return ((1 - s) * x.float() + s * out).to(x.dtype)

    # -------------------------------------------------------------- 区间估计
    def _update_range(self, log_amp, m):
        if not self.training:
            return
        with torch.no_grad():
            v = log_amp.masked_select(m)
            if v.numel() > 100_000:      # torch.quantile 有元素数上限，必须先抽样
                v = v[torch.randint(v.numel(), (100_000,), device=v.device)]
            if self.quantile > 0:
                lo, up = v.quantile(self.quantile), v.quantile(1.0 - self.quantile)
            else:
                lo, up = v.min(), v.max()      # quantile=0 就是完全不截断
            if self.inited == 0:
                self.log_lo.copy_(lo)
                self.log_hi.copy_(up)
                self.inited.fill_(1)
            else:
                mo = self.momentum
                self.log_lo.mul_(1 - mo).add_(mo * lo)
                self.log_hi.mul_(1 - mo).add_(mo * up)

    def _scalar(self, amp, m):
        """对数域均匀分级 + 直通估计。levels 个台阶等比分布。"""
        log_amp = torch.log(amp)
        self._update_range(log_amp, m)
        span = (self.log_hi - self.log_lo).clamp_min(self.eps)
        t = ((log_amp - self.log_lo) / span).clamp(0, 1)
        q = torch.round(t * (self.levels - 1)) / (self.levels - 1)
        q = t + (q - t).detach()                       # STE
        return torch.exp(q * span + self.log_lo), None

    def _clip(self, amp, m):
        """只截断，不分级。quantile=0 时退化成恒等。"""
        log_amp = torch.log(amp)
        self._update_range(log_amp, m)
        return torch.exp(log_amp.clamp(self.log_lo, self.log_hi)), None

    # ---------------------------------------------------------------- 码本
    def _vq_normed(self, z):
        """按位置对通道维做均值/方差归一化再查表。

        幅度谱和 post-ReLU 激活都跨数量级，直接拿原值查表会让所有向量挤到
        同一个码上。码本只学**通道之间的相对形状**，尺度由 mu/sd 原样带过去。
        """
        mu = z.mean(-1, keepdim=True)
        sd = z.std(-1, keepdim=True).clamp_min(self.eps)
        zq, commit = self._vq((z - mu) / sd)
        return zq * sd + mu, commit

    def _vq(self, z):
        """z: (N, C) -> (量化后 (N, C), commitment loss)。码本走 EMA。"""
        N, C = z.shape
        d = C // self.G
        zp = z[:, self.perm].view(N, self.G, d)
        # (N,G,K) 平方距离；emb 是 buffer，本身不带梯度
        dist = (zp.pow(2).sum(-1, keepdim=True)
                - 2.0 * torch.einsum("ngd,gkd->ngk", zp, self.emb)
                + self.emb.pow(2).sum(-1).unsqueeze(0))
        idx = dist.argmin(-1)                                    # (N,G)
        zq = torch.stack([self.emb[g][idx[:, g]] for g in range(self.G)], 1)

        if self.training:
            with torch.no_grad():
                mo = self.momentum
                for g in range(self.G):
                    oh = F.one_hot(idx[:, g], self.K).to(zp.dtype)       # (N,K)
                    self.ema_n[g].mul_(1 - mo).add_(mo * oh.sum(0))
                    self.ema_sum[g].mul_(1 - mo).add_(mo * (oh.t() @ zp[:, g]))
                    self.emb[g] = self.ema_sum[g] / self.ema_n[g].unsqueeze(1).clamp_min(1e-3)
                    # 死码重启：没人用的码换成当前 batch 里的随机向量。比 diversity
                    # loss 简单，而且 EMA+argmin 这条路线本来就不需要 Gumbel。
                    dead = self.ema_n[g] < self.dead_thr
                    n_dead = int(dead.sum())
                    if n_dead:
                        pick = zp[torch.randint(N, (n_dead,), device=z.device), g]
                        self.emb[g][dead] = pick
                        self.ema_sum[g][dead] = pick
                        self.ema_n[g][dead] = 1.0

        commit = F.mse_loss(zp, zq.detach())     # 只约束编码器一侧
        zq = zp + (zq - zp).detach()             # STE
        return zq.reshape(N, C)[:, self.inv], commit

    def _vq_amp(self, amp, m):
        """cb_freq：只量化掩码内位置的通道幅度向量，掩码外原样返回。"""
        B, C, H, W = amp.shape
        sel = m[0, 0]                                   # (H,W) bool
        v = amp.permute(0, 2, 3, 1)[:, sel]             # (B,N,C)
        zq, commit = self._vq_normed(torch.log(v).reshape(-1, C))
        out = amp.clone()
        out.permute(0, 2, 3, 1)[:, sel] = torch.exp(zq).view_as(v)
        return out, commit

    # ---------------------------------------------------------------- 诊断
    def usage(self):
        """在用的码数 / 总码数。坍缩时这个值会掉到很低，必须打进日志。"""
        if not self.mode.startswith("cb_"):
            return None
        return int((self.ema_n > self.dead_thr).sum()), self.G * self.K


class ViTFreqQuant(FreqQuant):
    """Same operation, with ViT-only persistence of the trainer's stage flags."""

    def get_extra_state(self):
        return {"enabled": self.enabled, "strength": self.strength}

    def set_extra_state(self, state):
        self.enabled = bool(state["enabled"])
        self.strength = float(state["strength"])


def _out_channels(block):
    return block.bn3.num_features if hasattr(block, "bn3") else block.bn2.num_features


def resnet_freqquant(network, layers=("layer1", "layer2", "layer3"), **kwargs):
    """按 block 插。必须在预训练权重加载之后调用，且不幂等。"""
    if getattr(network, "is_vit_backbone", False):
        for name in layers:
            for index in network.block_indices(name):
                network.add_block_op(index, ViTFreqQuant(network.n_outputs, **kwargs), "freqquant")
        return network
    for name in layers:
        stage = getattr(network, name)
        for i, block in enumerate(stage):
            if isinstance(block, nn.Sequential) and isinstance(block[-1], FreqQuant):
                raise RuntimeError(f"{name}[{i}] already wrapped")
            stage[i] = nn.Sequential(block, FreqQuant(_out_channels(block), **kwargs))
    return network


def collect_aux_loss(module):
    """把各模块攒的辅助损失取走并清空。quant / clip 模式下恒为 0。"""
    total = 0.0
    for m in module.modules():
        if isinstance(m, FreqQuant) and m.aux_loss is not None:
            total = total + m.aux_loss
            m.aux_loss = None
    return total


def codebook_usage(module):
    """全网码本使用率，给日志用。非码本模式返回 None。"""
    used = tot = 0
    for m in module.modules():
        if isinstance(m, FreqQuant):
            u = m.usage()
            if u is not None:
                used += u[0]
                tot += u[1]
    return (used, tot) if tot else None
