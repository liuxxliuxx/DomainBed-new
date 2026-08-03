"""SmoothGrad 显著性图，对应 StableNet 论文 4.7 节。

论文引用 [1] Adebayo et al. 的结论：跨方法比较时要用只依赖参数、
不依赖结构的方法，所以这里用 SmoothGrad 而不是 CAM / Grad-CAM。

只依赖 numpy + Pillow，不引入 matplotlib。
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# 跟 domainbed/datasets/transforms.py 里的 Normalize 保持一致
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(x):
    """[3,H,W] 归一化张量 -> [H,W,3] uint8，还原成能看的原图。"""
    x = x.detach().cpu().float() * _STD + _MEAN
    return (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _jet(g):
    """[H,W] in [0,1] -> [H,W,3] uint8，matplotlib jet 的分段线性近似。"""
    g = np.clip(g, 0, 1)
    r = np.clip(1.5 - np.abs(4 * g - 3), 0, 1)
    gr = np.clip(1.5 - np.abs(4 * g - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * g - 1), 0, 1)
    return (np.stack([r, gr, b], -1) * 255).astype(np.uint8)


def _gray(g):
    v = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    return np.stack([v, v, v], -1)


@torch.enable_grad()
def smooth_grad(algorithm, x, y, n_samples=25, noise_level=0.15):
    """SmoothGrad：对输入加 n 次噪声，求类别分数对输入的梯度再平均。

    x: [B,3,H,W]（已归一化），y: [B]
    返回 [B,H,W] float32，已逐图归一化到 [0,1]。
    """
    was_training = algorithm.training
    algorithm.eval()          # 关掉 dropout；BN 本来就冻着

    # 噪声幅度按每张图自己的动态范围定，跟 SmoothGrad 原文一致
    dyn = x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(dim=(1, 2, 3), keepdim=True)
    sigma = noise_level * dyn

    total = torch.zeros_like(x)
    for _ in range(n_samples):
        xn = (x + torch.randn_like(x) * sigma).detach().requires_grad_(True)
        logits = algorithm.predict(xn)
        # 取真实类别的 logit。对 batch 求和不会串味：
        # 每个样本的梯度只流回它自己的输入
        score = logits.gather(1, y.view(-1, 1)).sum()
        # 用 autograd.grad 而不是 backward：不会往网络参数的 .grad 里累加，
        # 所以不会污染正在进行的训练
        (grad,) = torch.autograd.grad(score, xn)
        total += grad

    sal = (total / n_samples).abs().amax(dim=1)     # 通道取最大 -> [B,H,W]

    # 逐图归一化。用 99 分位而不是 max 做分母，
    # 否则个别极亮像素会把整张图压成黑的
    B = sal.size(0)
    hi = torch.quantile(sal.view(B, -1).float(), 0.99, dim=1).view(B, 1, 1)
    sal = (sal / hi.clamp_min(1e-12)).clamp(0, 1)

    algorithm.train(was_training)
    return sal.detach().cpu().numpy()


def save_pair(img_u8, sal01, path, cmap="jet", gap=4):
    """左边原图、右边显著性图，中间留一条白缝，存成一张 png。"""
    right = _jet(sal01) if cmap == "jet" else _gray(sal01)
    sep = np.full((img_u8.shape[0], gap, 3), 255, np.uint8)
    Image.fromarray(np.concatenate([img_u8, sep, right], axis=1)).save(path)


class SaliencyDumper:
    """固定取一批图，每次调用 dump 都用同一批，方便看训练过程中注意力的迁移。"""

    def __init__(self, dataset, out_dir, n_images=16, n_samples=25,
                 noise_level=0.15, cmap="jet", chunk=8, device="cuda"):
        self.out_dir = Path(out_dir)
        self.n_samples = n_samples
        self.noise_level = noise_level
        self.cmap = cmap
        self.chunk = chunk
        self.device = device

        # 直接按下标取 _SplitDataset，不走 DataLoader。
        # FastDataLoader 内部包的是无限迭代器（fast_data_loader.py:68-80），
        # 提前 break 会让它下一轮从中间接着走，把 evaluator 的评测顺序打乱。
        n = min(n_images, len(dataset))
        stride = max(1, len(dataset) // n)
        idxs = list(range(0, len(dataset), stride))[:n]
        items = [dataset[i] for i in idxs]
        self.x = torch.stack([it["x"] for it in items])
        self.y = torch.tensor([int(it["y"]) for it in items])

    def dump(self, algorithm, step):
        d = self.out_dir / f"step_{step:06d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(self.x), self.chunk):
            x = self.x[i:i + self.chunk].to(self.device)
            y = self.y[i:i + self.chunk].to(self.device)
            sal = smooth_grad(algorithm, x, y, self.n_samples, self.noise_level)
            for j in range(x.size(0)):
                save_pair(denormalize(x[j]), sal[j],
                          d / f"{i + j:03d}_y{int(y[j])}.png", self.cmap)
        return d