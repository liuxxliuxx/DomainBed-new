"""FreqQuant 四模式自检。CPU 跑，几个 batch 的事。

用法（仓库根目录）：
    python check_freqquant.py
    python check_freqquant.py --r18 --layers layer2 layer3

任何一项 FAIL 都不要开训。
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.getcwd())
from domainbed.models.frequant import (  # noqa: E402
    FreqQuant, codebook_usage, collect_aux_loss, resnet_freqquant,
)

failures = []


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        failures.append(name)


def build(ctor, layers, **kw):
    return resnet_freqquant(ctor(weights="IMAGENET1K_V1"), layers=tuple(layers), **kw)


def mods(net):
    return [m for m in net.modules() if isinstance(m, FreqQuant)]


def set_state(ms, enabled, strength=1.0):
    for m in ms:
        m.enabled, m.strength = enabled, strength


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3"])
    ap.add_argument("--r18", action="store_true")
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(0)

    ctor = torchvision.models.resnet18 if args.r18 else torchvision.models.resnet50
    expect = sum(len(getattr(ctor(weights=None), n)) for n in args.layers)
    ref_w = ctor(weights="IMAGENET1K_V1").layer1[0].conv1.weight.detach().clone()
    x = torch.randn(args.batch, 3, 224, 224)

    # ---------- 0. 结构 ----------
    print("[0] 结构")
    net = build(ctor, args.layers)
    ms = mods(net)
    report("0a 模块数", len(ms) == expect, f"{len(ms)} (期望 {expect})")
    keys = [k for k in net.state_dict() if k.startswith(f"{args.layers[0]}.0.0.")]
    report("0b state_dict 多一层 .0.", bool(keys), keys[0] if keys else "没找到")
    same = torch.equal(net.layer1[0][0].conv1.weight.detach(), ref_w) \
        if args.layers[0] == "layer1" else True
    report("0c 预训练权重还在", same, "包装在加载之后" if same else "权重丢了！")
    try:
        resnet_freqquant(net, tuple(args.layers))
        report("0d 不幂等保护", False, "重复包装没报错")
    except RuntimeError:
        report("0d 不幂等保护", True)

    for mode in ("quant", "clip", "cb_freq", "cb_feat"):
        print(f"\n[{mode}]")
        net = build(ctor, args.layers, mode=mode, codebook=64, groups=1)
        net.eval()
        ms = mods(net)

        # ---------- 1. 关闭时恒等 ----------
        set_state(ms, False)
        with torch.no_grad():
            y_off = net(x)
        set_state(ms, True, 0.0)
        with torch.no_grad():
            y_s0 = net(x)
        report(f"1 {mode} 关闭/strength=0 恒等", torch.equal(y_off, y_s0),
               f"max|d|={(y_off - y_s0).abs().max():.2e}")

        # ---------- 2. FFT 往返（把量化器短路） ----------
        if mode != "cb_feat":
            orig = {"quant": FreqQuant._scalar, "clip": FreqQuant._clip,
                    "cb_freq": FreqQuant._vq_amp}[mode]
            setattr(FreqQuant, {"quant": "_scalar", "clip": "_clip",
                                "cb_freq": "_vq_amp"}[mode],
                    lambda self, amp, m: (amp, None))
            try:
                set_state(ms, True, 1.0)
                with torch.no_grad():
                    err = (net(x) - y_off).abs().max().item()
                report(f"2 {mode} FFT 往返恒等", err < 1e-4, f"max|d|={err:.2e}")
                report(f"2b {mode} 非精确零（早退没写 training）",
                       err > 0, f"{err:.2e}")
            finally:
                setattr(FreqQuant, {"quant": "_scalar", "clip": "_clip",
                                    "cb_freq": "_vq_amp"}[mode], orig)

        # ---------- 3. 频带朝向：只有被选中的那一带被改动 ----------
        if mode != "cb_feat":
            for band in ("high", "low"):
                m0 = FreqQuant(64, mode=mode, band=band, codebook=64, quantile=0.01)
                m0.enabled, m0.strength = True, 1.0
                m0.train()
                f = torch.randn(2, 64, 28, 28)
                m0(f)                       # 先估一次区间
                m0.eval()
                with torch.no_grad():
                    g = m0(f)
                sp0 = torch.fft.fftshift(torch.fft.fft2(f.float(), norm="ortho"), dim=(2, 3))
                sp1 = torch.fft.fftshift(torch.fft.fft2(g.float(), norm="ortho"), dim=(2, 3))
                sel = m0._mask(28, 28, f.device)
                d = (sp1 - sp0).abs()
                inn, out = d.masked_select(sel).max().item(), d.masked_select(~sel).max().item()
                report(f"3 {mode}/{band} 只动选中带", out < 1e-4 and inn > 10 * max(out, 1e-6),
                       f"带内 {inn:.2e} / 带外 {out:.2e}")

        # ---------- 4. 码本健康度 ----------
        if mode.startswith("cb_"):
            net.train()
            set_state(ms, True, 1.0)
            for _ in range(12):
                with torch.no_grad():
                    net(torch.randn(args.batch, 3, 224, 224))
            used, tot = codebook_usage(net)
            report(f"4 {mode} 码本未坍缩", used > tot / 8,
                   f"在用 {used}/{tot} ({used/tot:.1%})")

        # ---------- 5. 梯度 ----------
        net.train()
        set_state(ms, True, 1.0)
        net.zero_grad()
        out = net(torch.randn(args.batch, 3, 224, 224))
        aux = collect_aux_loss(net)
        (out.sum() + (aux if torch.is_tensor(aux) else 0.0)).backward()
        g = net.conv1.weight.grad
        ok = g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
        report(f"5 {mode} 梯度回到 conv1", bool(ok),
               f"|grad|={g.abs().sum():.3e}" if g is not None else "None")
        if mode.startswith("cb_"):
            report(f"5b {mode} commitment loss 有限",
                   torch.is_tensor(aux) and torch.isfinite(aux).all(),
                   f"aux={float(aux):.4f}" if torch.is_tensor(aux) else "不是张量")

    print()
    if failures:
        print("FAILED:", ", ".join(failures))
        sys.exit(1)
    print("全部通过，可以上服务器探开销了。")


if __name__ == "__main__":
    main()
