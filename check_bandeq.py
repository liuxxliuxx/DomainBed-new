"""频带均衡接线自检。

    python check_bandeq.py --dataset PACS --data_dir <包含 PACS/ 的上级目录> --test_env 3

六项检查，任何一项 FAIL 都不要开训。
"""
import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.getcwd())

from domainbed.datasets import transforms as DBT          # noqa: E402
from domainbed.datasets import get_dataset, get_band_target  # noqa: E402
from domainbed import hparams_registry                     # noqa: E402

BandEqualize = DBT.BandEqualize
OK, BAD = "  [PASS]", "  [FAIL]"
failures = []


def report(name, ok, detail=""):
    print(f"{OK if ok else BAD} {name}  {detail}")
    if not ok:
        failures.append(name)


def ref_equalize(x, target, edges, preserve_total=True, eps=1e-12):
    """独立的 numpy 参考实现，只用来和 torch 版对拍。"""
    C, H, W = x.shape
    nb = min(H, W) // 2
    yy, xx = np.mgrid[0:H, 0:W]
    rad = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
    rbin = np.clip(rad.astype(np.int64), 0, nb - 1)
    inside = rad < nb
    K = len(edges) - 1
    centers = np.array([(edges[k] + edges[k + 1] - 1) / 2.0 for k in range(K)])

    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    F = np.fft.fftshift(np.fft.fft2(lum - lum.mean()))
    P = np.abs(F) ** 2
    ring = np.bincount(rbin[inside], weights=P[inside], minlength=nb)
    ring[0] = 0.0
    e = np.array([ring[edges[k]:edges[k + 1]].sum() for k in range(K)])
    tot = max(e.sum(), eps)
    scale = tot if preserve_total else 1.0
    gb = np.sqrt(np.asarray(target) * scale / np.maximum(e, eps))

    i1 = np.clip(np.searchsorted(centers, np.arange(nb)), 0, K - 1)
    i0 = np.clip(i1 - 1, 0, K - 1)
    denom = np.where(i1 > i0, centers[i1] - centers[i0], 1.0)
    w = np.clip((np.arange(nb) - centers[i0]) / denom, 0.0, 1.0)
    gr = gb[i0] * (1 - w) + gb[i1] * w
    gr[0] = 1.0
    gain = gr[rbin]

    out = np.empty_like(x)
    for c in range(C):
        m = x[c].mean()
        Fc = np.fft.fftshift(np.fft.fft2(x[c] - m))
        out[c] = np.real(np.fft.ifft2(np.fft.ifftshift(Fc * gain))) + m
    return out


def band_share(x, edges):
    H = x.shape[1]
    nb = H // 2
    yy, xx = np.mgrid[0:H, 0:H]
    rad = np.sqrt((yy - H // 2) ** 2 + (xx - H // 2) ** 2)
    rbin = np.clip(rad.astype(np.int64), 0, nb - 1)
    inside = rad < nb
    lum = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    P = np.abs(np.fft.fftshift(np.fft.fft2(lum - lum.mean()))) ** 2
    ring = np.bincount(rbin[inside], weights=P[inside], minlength=nb)
    ring[0] = 0.0
    e = np.array([ring[edges[k]:edges[k + 1]].sum() for k in range(len(edges) - 1)])
    return e / max(e.sum(), 1e-30)


def chain(compose):
    return [type(o).__name__ for o in compose.transforms]


def has_bandeq(compose):
    ops = compose.transforms
    idx = [i for i, o in enumerate(ops) if isinstance(o, BandEqualize)]
    names = chain(compose)
    if not idx:
        return False, " -> ".join(names)
    i = idx[0]
    if not ("ToTensor" in names and "Normalize" in names
            and names.index("ToTensor") < i < names.index("Normalize")):
        return False, f"位置不对: {' -> '.join(names)}"
    return True, " -> ".join(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_env", type=int, default=0)
    a = ap.parse_args()

    args = types.SimpleNamespace(
        dataset=a.dataset, data_dir=a.data_dir, cache="none", cache_size=None,
        cache_root="cache", resize_mode="stretch", holdout_fraction=0.2, trial_seed=0,
    )
    hp = hparams_registry.default_hparams("ERM", a.dataset)
    hp["band_eq"] = 1
    hp["band_eq_preserve_total"] = True
    hp["band_eq_target_mode"] = "source_mean"
    hp["band_eq_mode"] = "both"

    te = [a.test_env]
    dataset, in_splits, out_splits = get_dataset(te, args, hp)
    target, edges = get_band_target(dataset, te, args, hp)
    target, edges = np.asarray(target), np.asarray(edges)
    print(f"K={len(edges) - 1}  target={np.array2string(target, precision=4)}\n")

    train_i = next(i for i in range(len(dataset)) if i not in te)
    for tag, dset in [("train", in_splits[train_i][0]),
                      ("valid", out_splits[train_i][0]),
                      ("test ", in_splits[te[0]][0])]:
        ok, detail = has_bandeq(dset.transforms["x"])
        report(f"1 {tag} 分支挂上均衡", ok, detail)

    hp0 = dict(hp)
    hp0["band_eq"] = 0
    _, in0, _ = get_dataset(te, args, hp0)
    ok0, _ = has_bandeq(in0[te[0]][0].transforms["x"])
    report("2 band_eq=0 时不插入模块", not ok0, "关闭时链里不应出现 BandEqualize")

    # ---------- 2b 三种 mode 挂在正确的分支上 ----------
    want = {"both": (True, True, True),
            "test_only": (False, False, True),
            "train_only": (True, True, False)}
    for mode, (w_tr, w_va, w_te) in want.items():
        hpm = dict(hp)
        hpm["band_eq_mode"] = mode
        _, i_m, o_m = get_dataset(te, args, hpm)
        got = (has_bandeq(i_m[train_i][0].transforms["x"])[0],
               has_bandeq(o_m[train_i][0].transforms["x"])[0],
               has_bandeq(i_m[te[0]][0].transforms["x"])[0])
        report(f"2b mode={mode:<10} (train,valid,test)", got == (w_tr, w_va, w_te),
               f"实际 {got} 期望 {(w_tr, w_va, w_te)}")

    beq = next(o for o in in_splits[train_i][0].transforms["x"].transforms
               if isinstance(o, BandEqualize))
    size = int(hp["image_size"])
    x = np.random.RandomState(0).rand(3, size, size).astype(np.float32)
    got = beq(torch.from_numpy(x.copy())).numpy().astype(np.float64)
    want = ref_equalize(x.astype(np.float64), target, edges, preserve_total=True)
    err = np.abs(got - want).max()
    report("3 torch 实现 vs numpy 参考", err < 1e-5, f"max|diff|={err:.2e} (阈值 1e-5)")

    from torchvision import transforms as T
    pre = T.Compose([T.Resize((size, size)), T.ToTensor()])
    src = in_splits[train_i][0]
    beq_hard = BandEqualize(target, edges, preserve_total=True, smooth=False)
    e_before, e_after, e_hard = [], [], []
    for k in range(16):
        raw = src.underlying_dataset[src.keys[k]][0]
        x0 = pre(raw)
        b0 = x0.numpy().astype(np.float64)
        e_before.append(np.abs(band_share(b0, edges) - target).max())
        e_after.append(np.abs(band_share(beq(x0).numpy().astype(np.float64), edges)
                              - target).max())
        e_hard.append(np.abs(band_share(beq_hard(x0).numpy().astype(np.float64), edges)
                             - target).max())
    b, af, hd = float(np.mean(e_before)), float(np.mean(e_after)), float(np.mean(e_hard))
    # 平滑增益按设计留有残差，所以看的是"往目标靠拢了多少倍"，不是绝对值
    report("4 均衡把剖面拉向 target", af < b / 5,
           f"n=16 平均偏差 {b:.4f} -> {af:.4f}（要求缩小 5 倍以上）")
    report("4b smooth=False 时精确命中 target", hd < 1e-6,
           f"硬分档平均偏差={hd:.2e}；不为 0 说明增益或分档写错了")

    other = next(i for i in range(len(dataset)) if i != a.test_env)
    t2, _ = get_band_target(dataset, [other], args, hp)
    d = float(np.abs(target - np.asarray(t2)).max())
    report("5 目标随 test_env 改变（统计时排除了目标域）", d > 1e-6,
           f"两个划分的目标最大差={d:.2e}")

    lo, hi = float(got.min()), float(got.max())
    report("6 输出数值范围", -2.0 < lo and hi < 3.0,
           f"min={lo:.3f} max={hi:.3f}（超出 [0,1] 正常，到 ±5 就是目标太激进）")

    # ---------- 7 真正要的指标：域间高频占比的离散度塌下来 ----------
    CUT = size // 4          # r=56 @224
    hi_idx = [k for k in range(len(edges) - 1) if (edges[k] + edges[k + 1] - 1) / 2 >= CUT]
    hb_before, hb_after = {}, {}
    for env_i in range(len(dataset)):
        env = dataset[env_i]
        idx = np.random.RandomState(1).choice(len(env), min(40, len(env)), replace=False)
        sb, sa = [], []
        for i in idx:
            x0 = pre(env[int(i)][0])
            b0 = x0.numpy().astype(np.float64)
            sb.append(band_share(b0, edges)[hi_idx].sum())
            sa.append(band_share(beq(x0).numpy().astype(np.float64), edges)[hi_idx].sum())
        hb_before[env_i], hb_after[env_i] = float(np.mean(sb)), float(np.mean(sa))
    sp = lambda d: max(d.values()) / max(min(d.values()), 1e-12)
    print("      各域高频(r>=%d)占比: " % CUT
          + "  ".join(f"env{k} {hb_before[k]:.4f}->{hb_after[k]:.4f}" for k in hb_before))
    report("7 域间高频占比离散度塌缩", sp(hb_after) < 1.5 and sp(hb_after) < sp(hb_before),
           f"最大/最小 {sp(hb_before):.2f}x -> {sp(hb_after):.2f}x")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("全部通过。")


if __name__ == "__main__":
    main()
