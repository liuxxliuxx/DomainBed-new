"""FreqQuant 开训前自检。CPU 跑，一个 batch。

用法（在仓库根目录）：
    python check_freqquant.py [--layers layer1 layer2 layer3] [--r18]

六项检查，任何一项 FAIL 都不要开训。
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.getcwd())
from domainbed.models.frequant import FreqQuant, resnet_freqquant  # noqa: E402

OK, BAD = "  [PASS]", "  [FAIL]"
failures = []


def report(name, ok, detail=""):
    print(f"{OK if ok else BAD} {name}  {detail}")
    if not ok:
        failures.append(name)


def modules_of(net):
    return [m for m in net.modules() if isinstance(m, FreqQuant)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3"])
    ap.add_argument("--r18", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--image", default=None, help="check 4 用的线稿路径")
    args = ap.parse_args()

    torch.manual_seed(0)

    # ---------- 0. 结构：包装位置、模块数、预训练权重是否还在 ----------
    ctor = torchvision.models.resnet18 if args.r18 else torchvision.models.resnet50
    ref = ctor(weights="IMAGENET1K_V1")
    ref_w = ref.layer1[0].conv1.weight.detach().clone()
    expect = sum(len(getattr(ref, n)) for n in args.layers)

    net = resnet_freqquant(ctor(weights="IMAGENET1K_V1"), layers=tuple(args.layers),
                           levels=16, mask_ratio=0.5)
    mods = modules_of(net)
    report("0a 模块数", len(mods) == expect, f"{len(mods)} (期望 {expect})")

    keys = [k for k in net.state_dict() if k.startswith(args.layers[0])]
    nested = any(k.startswith(f"{args.layers[0]}.0.0.") for k in keys)
    report("0b state_dict 键多一层 .0.", nested,
           next((k for k in keys if ".0.0." in k), "没找到"))

    same = torch.equal(net.layer1[0][0].conv1.weight.detach(), ref_w)
    report("0c 预训练权重还在", same, "包装发生在加载之后" if same else "权重丢了，包装早于加载")

    report("0d 不幂等保护", _raises(lambda: resnet_freqquant(net, tuple(args.layers))))

    net.eval()
    x = torch.randn(args.batch, 3, 224, 224)

    # ---------- 1. 关闭时必须是逐元素恒等 ----------
    for m in mods:
        m.enabled = False
    with torch.no_grad():
        y0 = net(x)
        ref_out = ctor(weights="IMAGENET1K_V1").eval()(x) if False else None
    for m in mods:
        m.enabled, m.strength = True, 0.0
    with torch.no_grad():
        y_s0 = net(x)
    report("1 enabled=False / strength=0 恒等", torch.equal(y0, y_s0),
           f"max|diff|={(y0 - y_s0).abs().max():.2e}")

    # ---------- 2. 单独验 FFT 往返（把 quantize 短路成恒等） ----------
    # 注意：不能用 levels 很大来做这个检查，因为量化区间取的是 1%/99% 分位，
    # 两端 2% 的系数会被 clamp 掉，误差不会小。必须把 quantize 整个旁路。
    orig = FreqQuant.quantize
    FreqQuant.quantize = lambda self, amp, hi: (amp, None)
    try:
        for m in mods:
            m.enabled, m.strength, m.low_gain = True, 1.0, 1.0
        with torch.no_grad():
            y1 = net(x)
        err = (y1 - y0).abs().max().item()
        report("2 FFT 往返恒等", err < 1e-4, f"max|diff|={err:.2e} (阈值 1e-4)")
    finally:
        FreqQuant.quantize = orig

    # ---------- 3. 量化器本身的相对误差 ----------
    m0 = mods[0]
    feat = torch.randn(2, m0.channels, 28, 28).abs() + 1e-3
    hi = m0._mask(28, 28, feat.device)
    m0.eval()
    la = torch.log(feat.clamp_min(m0.eps))
    m0.log_lo.copy_(la.min())
    m0.log_hi.copy_(la.max())
    m0.inited.fill_(1)
    m0.levels = 4096
    q, _ = m0.quantize(feat.clamp_min(m0.eps), hi)
    rel = ((q - feat).abs() / feat).max().item()
    report("3 levels=4096 相对误差", rel < 0.02, f"max rel={rel:.4f} (阈值 0.02)")
    m0.levels = 16

    # ---------- 4. 目视：levels=4 抹掉细节但保留结构 ----------
    if args.image and os.path.exists(args.image):
        _visual(args.image, mods[0])
    else:
        print("  [SKIP] 4 目视检查：没给 --image")

    # ---------- 4b. 掩码朝向：低频系数必须原样保留 ----------
    # 这是判断掩码有没有取反的唯一可靠依据。目视看不出来，因为傅里叶基是全局的，
    # 改任何一个高频系数都会在整幅图上铺开，差值图"均匀铺满"是正常现象。
    m0.eval()
    m0.enabled, m0.strength, m0.levels = True, 1.0, 4
    feat = torch.randn(2, m0.channels, 28, 28)
    with torch.no_grad():
        out_f = m0(feat)
    fx = torch.fft.fftshift(torch.fft.fft2(feat.float(), norm="ortho"), dim=(2, 3))
    fy = torch.fft.fftshift(torch.fft.fft2(out_f.float(), norm="ortho"), dim=(2, 3))
    himask = m0._mask(28, 28, feat.device)
    d = (fy - fx).abs()
    lo_err = d.masked_select(~himask).max().item()
    hi_err = d.masked_select(himask).max().item()
    report("4b 低频未被改动", lo_err < 1e-4 and hi_err > 10 * max(lo_err, 1e-6),
           f"低频 max|dF|={lo_err:.2e}  高频 max|dF|={hi_err:.2e}")
    dc = (fy[:, :, 14, 14] - fx[:, :, 14, 14]).abs().max().item()
    report("4c 直流分量守恒", dc < 1e-4, f"|dDC|={dc:.2e}")
    m0.levels = 16

    # ---------- 5. 梯度能穿过 STE 回到 conv1 ----------
    net.train()
    for m in mods:
        m.enabled, m.strength = True, 1.0
    net.zero_grad()
    out = net(torch.randn(2, 3, 224, 224))
    out.sum().backward()
    g = net.conv1.weight.grad
    ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    report("5 梯度回到 conv1", bool(ok),
           f"|grad|={g.abs().sum():.3e}" if g is not None else "grad is None")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("全部通过，可以上服务器测显存和步时了。")


def _raises(fn):
    try:
        fn()
    except RuntimeError:
        return True
    return False


def _visual(path, mod):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    im = Image.open(path).convert("L").resize((224, 224))
    a = 1.0 - torch.from_numpy(np.asarray(im, dtype="float32") / 255.0)
    x = a.view(1, 1, 224, 224).repeat(1, mod.channels, 1, 1)
    mod.enabled, mod.strength, mod.levels = True, 1.0, 4
    mod.inited.fill_(0)
    mod.train()          # 让它自己估一次分位区间
    with torch.no_grad():
        _ = mod(x)
    mod.eval()
    with torch.no_grad():
        y = mod(x)

    xi, yi = x[0, 0].numpy(), y[0, 0].numpy()
    di = yi - xi
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fq_visual.png")
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.1))

    # 前两幅共用同一个色标，否则 autoscale 会把输出的背景显示成灰色，看着像直流漂了
    vmin, vmax = float(xi.min()), float(xi.max())
    for a_, img, t in zip(ax[:2], [xi, yi], ["输入", "levels=4 输出（与左图同色标）"]):
        a_.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        a_.set_title(t, fontsize=10)
        a_.axis("off")
    lim = float(np.abs(di).max())
    ax[2].imshow(di, cmap="coolwarm", vmin=-lim, vmax=lim)
    ax[2].set_title(f"差值（对称色标 ±{lim:.3f}）", fontsize=10)
    ax[2].axis("off")

    # 差值的径向功率谱：能量该堆在高频侧，堆在低频就是掩码取反了
    F = np.fft.fftshift(np.fft.fft2(di))
    P = np.abs(F) ** 2
    n = 112
    yy, xx = np.mgrid[0:224, 0:224]
    rad = np.sqrt((yy - 112) ** 2 + (xx - 112) ** 2)
    rb = np.clip(rad.astype(int), 0, n - 1)
    ins = rad < n
    cnt = np.bincount(rb[ins], minlength=n)
    prof = np.bincount(rb[ins], weights=P[ins], minlength=n) / np.maximum(cnt, 1)
    w = prof * cnt
    cut = int(mod.mask_ratio * 224 / 2)
    frac_hi = w[cut:].sum() / w.sum()
    ax[3].plot(np.arange(1, n), prof[1:], lw=2, color="#2a78d6")
    ax[3].axvline(cut, color="#52514e", lw=1.2, ls=(0, (4, 3)))
    ax[3].set_yscale("log")
    ax[3].set_title(f"差值的功率谱：高频侧占 {frac_hi:.1%}", fontsize=10)
    ax[3].set_xlabel("径向频率 r", fontsize=9)
    ax[3].grid(True, color="#dedcd6", lw=0.8)
    for s in ("top", "right"):
        ax[3].spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"  [LOOK] 4 目视检查已存 -> {out}  差值高频占比={frac_hi:.1%}（应 >80%）")


if __name__ == "__main__":
    main()