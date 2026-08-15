from torchvision import transforms as T

import numpy as np
import torch

_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 频带均衡默认的环带数。低频窄、高频宽（对数分档），和离线分析脚本保持一致。
N_BANDS = 12


def band_edges(size=224, k=N_BANDS):
    """返回长度 K+1 的环带边界（半径，单位像素）。edges[0] 为 1，直流不参与。"""
    nb = size // 2
    e = np.unique(np.round(np.logspace(0, np.log10(nb), k + 1)).astype(int))
    e[0] = 1
    return e


def fixed_target(size=224, k=N_BANDS, alpha=2.0):
    """解析目标剖面：功率按 1/r^alpha 衰减，再按每个环带的系数个数加权。

    用它就不需要从源域统计，也就没有泄漏和逐划分的簿记；代价是离数据的平均剖面
    远一些、图像形变大一些。
    """
    e = band_edges(size, k)
    nb = size // 2
    r = np.arange(nb, dtype=np.float64)
    ring_count = 2.0 * np.pi * np.maximum(r, 0.5)      # 半径 r 的环上大致有多少系数
    ring_power = np.maximum(r, 0.5) ** (-alpha)
    w = ring_count * ring_power
    t = np.array([w[e[i]:e[i + 1]].sum() for i in range(len(e) - 1)])
    return t / t.sum(), e


class BandEqualize:
    """逐样本径向频带均衡。

    作用在 ToTensor 之后、Normalize 之前的 (C, H, W) float 张量上：
      1. 由亮度算出各环带能量 e_k
      2. 增益 g_k = sqrt(t_k * E / e_k)，带上 E 所以只重分配频带比例、不改总能量
         （preserve_total=False 时改为归到固定绝对量级）
      3. 增益沿半径线性插值后作用于复数谱，实数增益 -> 相位原样保留，直流不动

    训练和推理都必须经过，它是归一化不是增强。
    """

    def __init__(self, target, edges, preserve_total=True, smooth=True, eps=1e-12):
        self.target = torch.as_tensor(np.asarray(target), dtype=torch.float64)
        self.edges = np.asarray(edges)
        self.preserve_total = preserve_total
        # smooth=False 时环带内增益为常数，均衡后的频带占比精确等于 target，
        # 但边界突变会带来振铃；smooth=True 用插值换掉振铃，代价是留一点残差。
        self.smooth = smooth
        self.eps = eps
        self._cache = {}

    def __repr__(self):
        return (f"BandEqualize(K={len(self.edges) - 1}, "
                f"preserve_total={self.preserve_total}, smooth={self.smooth})")

    def _geom(self, H, W, device):
        """按 (H, W, device) 缓存径向索引与插值权重，别每张图重建。"""
        key = (H, W, str(device))
        if key in self._cache:
            return self._cache[key]

        nb = min(H, W) // 2
        yy, xx = np.mgrid[0:H, 0:W]
        rad = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
        rbin = np.clip(rad.astype(np.int64), 0, nb - 1)
        valid = (rad < nb) & (rbin > 0)                 # 直流不参与统计

        edges = self.edges
        K = len(edges) - 1
        band_of_r = np.zeros(nb, dtype=np.int64)
        for k in range(K):
            band_of_r[edges[k]:min(edges[k + 1], nb)] = k
        band_of_r[edges[-1] - 1:] = K - 1
        band_idx = band_of_r[rbin]

        # 环带增益 -> 逐半径增益的线性插值系数（一次算好，之后只做取址和乘加）
        centers = np.array([(edges[k] + edges[k + 1] - 1) / 2.0 for k in range(K)])
        r = np.arange(nb, dtype=np.float64)
        i1 = np.clip(np.searchsorted(centers, r), 0, K - 1)
        i0 = np.clip(i1 - 1, 0, K - 1)
        denom = np.where(i1 > i0, centers[i1] - centers[i0], 1.0)
        w = np.clip((r - centers[i0]) / denom, 0.0, 1.0)

        g = {
            "band_idx": torch.as_tensor(band_idx[valid], device=device),
            "valid": torch.as_tensor(valid, device=device),
            "rbin": torch.as_tensor(rbin, device=device),
            "band_of_r": torch.as_tensor(band_of_r, device=device),
            "i0": torch.as_tensor(i0, device=device),
            "i1": torch.as_tensor(i1, device=device),
            "w": torch.as_tensor(w, dtype=torch.float64, device=device),
            "nb": nb,
            "K": K,
        }
        self._cache[key] = g
        return g

    def __call__(self, x):
        if x.dim() != 3:
            raise ValueError(f"BandEqualize 需要 (C,H,W)，收到 {tuple(x.shape)}")
        dtype_in = x.dtype
        xd = x.to(torch.float64)
        C, H, W = xd.shape
        g = self._geom(H, W, xd.device)
        target = self.target.to(xd.device)

        lum = 0.299 * xd[0] + 0.587 * xd[1] + 0.114 * xd[2] if C >= 3 else xd.mean(0)
        F = torch.fft.fftshift(torch.fft.fft2(lum - lum.mean()), dim=(-2, -1))
        P = (F.real ** 2 + F.imag ** 2)

        e = torch.zeros(g["K"], dtype=torch.float64, device=xd.device)
        e.index_add_(0, g["band_idx"], P[g["valid"]])
        tot = e.sum().clamp_min(self.eps)
        scale = tot if self.preserve_total else torch.ones_like(tot)
        g_band = torch.sqrt(target * scale / e.clamp_min(self.eps))

        if self.smooth:
            g_r = torch.lerp(g_band[g["i0"]], g_band[g["i1"]], g["w"])
        else:
            g_r = g_band[g["band_of_r"]]
        g_r = g_r.clone()
        g_r[0] = 1.0                                    # 直流不动
        gain = g_r[g["rbin"]]

        m = xd.mean(dim=(-2, -1), keepdim=True)
        Fx = torch.fft.fftshift(torch.fft.fft2(xd - m), dim=(-2, -1))
        out = torch.fft.ifft2(torch.fft.ifftshift(Fx * gain, dim=(-2, -1))).real + m
        return out.to(dtype_in)


def basic(size=224, band_eq=None):
    ops = [T.Resize((size, size)), T.ToTensor()]
    if band_eq is not None:
        ops.append(band_eq)          # 必须在 ToTensor 之后、_NORM 之前
    ops.append(_NORM)
    return T.Compose(ops)


def aug(size=224, band_eq=None):
    ops = [
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
    ]
    if band_eq is not None:
        ops.append(band_eq)
    ops.append(_NORM)
    return T.Compose(ops)
