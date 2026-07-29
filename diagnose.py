"""HTP 诊断：换标签、换表征，看模型能不能学出来。

  python diagnose.py --label domain --env all
  python diagnose.py --label class  --env all
  python diagnose.py --label class  --env 01 --region gray     --cache_size 640 --input_size 448
  python diagnose.py --label class  --env 01 --region binary   --cache_size 640 --input_size 448
  python diagnose.py --label class  --env 01 --region skeleton --cache_size 1024 --input_size 640 --batch_size 8
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def collect(data_dir, env_filter, label_kind):
    """文件名格式: {域}_{类别}_{批次}_{序号}.ext"""
    paths, raw = [], []
    envs = ["00", "01", "02"] if env_filter == "all" else [env_filter]
    for env in envs:
        for cls in sorted(os.listdir(os.path.join(data_dir, env))):
            for p in sorted(glob.glob(os.path.join(data_dir, env, cls, "*"))):
                if not os.path.isfile(p):
                    continue
                fields = os.path.splitext(os.path.basename(p))[0].split("_")
                paths.append(p)
                if label_kind == "class":
                    raw.append(cls)
                elif label_kind == "domain":
                    raw.append(env)
                elif label_kind == "batch":
                    raw.append(fields[2] if len(fields) > 2 else "NA")
                else:
                    raise ValueError(label_kind)
    names = sorted(set(raw))
    idx = {n: i for i, n in enumerate(names)}
    return paths, np.array([idx[r] for r in raw]), names


def _otsu(x):
    """x: [0,1] 的 float 数组，返回类间方差最大的阈值。"""
    hist, edges = np.histogram(x, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    p /= max(p.sum(), 1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = 1e-12
    sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    return float(edges[int(np.argmax(sigma_b)) + 1])


def _ink_map(im, hires):
    """在高分辨率上估纸张、出笔迹强度图。0=纸, 1=最重笔迹。不裁剪。"""
    w, h = im.size
    s = hires / max(w, h)
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
    g = np.asarray(im.convert("L"), dtype=np.float32)
    paper = max(float(np.percentile(g, 95)), 1.0)
    return np.clip((paper - g) / paper, 0.0, 1.0)


def _square(a2d, size):
    """单通道 float [0,1] -> uint8 三通道 (size,size,3)，整页压成正方形，不裁剪。"""
    im = Image.fromarray((a2d * 255.0).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.stack([np.asarray(im)] * 3, axis=-1)


def _corner_tile(im, size):
    """只保留四角空白纸。"""
    a = np.asarray(im.resize((size, size), Image.BICUBIC))
    k = max(8, size // 16)
    top = np.concatenate([a[:k, :k], a[:k, -k:]], axis=1)
    bot = np.concatenate([a[-k:, :k], a[-k:, -k:]], axis=1)
    tile = Image.fromarray(np.concatenate([top, bot], axis=0))
    return np.asarray(tile.resize((size, size), Image.BICUBIC))


def decode_all(paths, size, region, hires):
    arr = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB")
        if region == "full":
            arr[i] = np.asarray(im.resize((size, size), Image.BICUBIC))
        elif region == "corner":
            arr[i] = _corner_tile(im, size)
        elif region == "geonorm":
            w, h = im.size
            s = hires / max(w, h)
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32)
            lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            a[lum >= 230] = 255.0
            m = lum < 230
            ys, xs = np.where(m)
            if len(ys) > 50:
                a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            bh, bw = a.shape[:2]                      # 按笔迹外接框自身比例填成正方形
            side = max(bh, bw)
            canvas = np.full((side, side, 3), 255.0, dtype=np.float32)
            y0, x0 = (side - bh) // 2, (side - bw) // 2
            canvas[y0:y0 + bh, x0:x0 + bw] = a
            arr[i] = np.asarray(
                Image.fromarray(canvas.astype(np.uint8)).resize((size, size), Image.BICUBIC))
        elif region == "corner_clean":
            w, h = im.size
            s = hires / max(w, h)
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
            a = np.asarray(im, dtype=np.float32)
            lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            a[lum >= 230] = 255.0
            arr[i] = _corner_tile(Image.fromarray(a.astype(np.uint8)), size)
        elif region == "paper":
            w, h = im.size
            s = hires / max(w, h)
            if s < 1.0:
                im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
            g = np.asarray(im, dtype=np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            b = max(8, min(g.shape) // 30)
            H, Wd = g.shape
            frame = np.concatenate([
                g[:b, :].ravel(), g[-b:, :].ravel(),
                g[:, :b].ravel(), g[:, -b:].ravel()])
            frame = frame[: (len(frame) // b) * b].reshape(-1, b)   # 排成条带
            frame = np.where(frame < 230, 255.0, frame)             # 把笔迹也抹白，只留纸
            frame = np.clip((frame - 230.0) / 25.0 * 255.0, 0, 255) # 把 230-255 拉到全量程
            side = int(np.sqrt(frame.size))
            tile = frame.ravel()[: side * side].reshape(side, side)
            arr[i] = _square(tile / 255.0, size)
        else:
            ink = _ink_map(im, hires)
            if region == "gray":
                arr[i] = _square(ink, size)
            elif region == "binary":
                arr[i] = _square((ink > _otsu(ink)).astype(np.float32), size)
            elif region == "skeleton":
                from skimage.morphology import skeletonize
                sk = skeletonize(ink > _otsu(ink))
                arr[i] = _square(sk.astype(np.float32), size)
            elif region == "clean":
                w, h = im.size
                s = hires / max(w, h)
                if s < 1.0:
                    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
                a = np.asarray(im.convert("RGB"), dtype=np.float32)
                lum = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
                a[lum >= 230] = 255.0        # 亮度够白的整个像素置纯白，其余保留原色
                arr[i] = np.asarray(
                    Image.fromarray(a.astype(np.uint8)).resize((size, size), Image.BICUBIC))
            else:
                raise ValueError(region)
        if (i + 1) % 100 == 0:
            print(f"  处理 {i + 1}/{len(paths)}", flush=True)
    return arr


class MemSet(Dataset):
    def __init__(self, arr, labels, idx, train, input_size):
        self.arr, self.labels, self.idx = arr, labels, idx
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.3, 0.3),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(Image.fromarray(self.arr[j])), int(self.labels[j])


@torch.no_grad()
def evaluate(net, loader, n_cls):
    net.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(torch.softmax(net(x.cuda()), 1).cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    acc = float((P.argmax(1) == Y).mean())
    try:
        if n_cls == 2:
            auc = float(roc_auc_score(Y, P[:, 1]))
        else:
            auc = float(roc_auc_score(Y, P, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    net.train()
    return acc, auc

@torch.no_grad()
def extract_feats(net, arr, labels, idx, input_size, batch_size):
    """冻结网络，返回 (特征 [n,2048], 源域分类头给出的正类概率 [n], 标签 [n])"""
    ld = DataLoader(MemSet(arr, labels, idx, False, input_size),
                    batch_size=batch_size, shuffle=False, num_workers=4)
    body = nn.Sequential(*list(net.children())[:-1])
    net.eval()
    F, S, Y = [], [], []
    for x, y in ld:
        f = body(x.cuda()).flatten(1)
        S.append(torch.softmax(net.fc(f), 1)[:, 1].cpu().numpy())
        F.append(f.cpu().numpy())
        Y.append(y.numpy())
    net.train()
    return np.concatenate(F), np.concatenate(S), np.concatenate(Y)


def few_shot_report(F, S, Y, ns, repeats, C):
    """固定评估集，从适应池里抽 N 张，比较三种适应方式。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    rng0 = np.random.RandomState(12345)
    pool, ev = [], []
    for c in (0, 1):
        ids = np.where(Y == c)[0]
        rng0.shuffle(ids)
        k = int(round(len(ids) * 0.4))
        pool += list(ids[:k])
        ev += list(ids[k:])
    pool, ev = np.array(pool), np.array(ev)
    print(f"\n  适应池 {len(pool)} 张 / 固定评估集 {len(ev)} 张")

    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)   # 原型法用归一化特征

    base_auc = roc_auc_score(Y[ev], S[ev])
    base_acc = float(((S[ev] > 0.5).astype(int) == Y[ev]).mean())
    maj = max(np.mean(Y[ev]), 1 - np.mean(Y[ev]))
    print(f"  多数类基线                          acc {maj:.3f}")
    print(f"  N=0    源域分类头直接用              AUC {base_auc:.3f}   acc {base_acc:.3f}")
    print(f"  N=0    （若强行翻转符号）            AUC {1 - base_auc:.3f}")

    for n in ns:
        per = max(1, n // 2)
        res = {k: [] for k in ("sign", "thr_acc", "proto", "linear")}
        flips = []
        for r in range(repeats):
            rng = np.random.RandomState(1000 + r)
            ad = []
            for c in (0, 1):
                ids = pool[Y[pool] == c].copy()
                rng.shuffle(ids)
                ad += list(ids[:per])
            ad = np.array(ad)

            # (1) 只用这 n 张决定要不要翻符号
            a = roc_auc_score(Y[ad], S[ad]) if len(set(Y[ad])) == 2 else 0.5
            flip = a < 0.5
            flips.append(flip)
            s_ev = 1 - S[ev] if flip else S[ev]
            res["sign"].append(roc_auc_score(Y[ev], s_ev))

            # (2) 再用这 n 张挑最优阈值（只影响准确率，不影响 AUC）
            s_ad = 1 - S[ad] if flip else S[ad]
            cand = np.unique(s_ad)
            best = max(cand, key=lambda t: ((s_ad >= t).astype(int) == Y[ad]).mean())
            res["thr_acc"].append(float(((s_ev >= best).astype(int) == Y[ev]).mean()))

            # (3) 原型法：两个类均值之差当方向
            p0 = Fn[ad][Y[ad] == 0].mean(0)
            p1 = Fn[ad][Y[ad] == 1].mean(0)
            res["proto"].append(roc_auc_score(Y[ev], Fn[ev] @ (p1 - p0)))

            # (4) 在冻结特征上重拟合逻辑回归
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, C=C))
            clf.fit(F[ad], Y[ad])
            res["linear"].append(roc_auc_score(Y[ev], clf.predict_proba(F[ev])[:, 1]))

        m = {k: (np.mean(v), np.std(v)) for k, v in res.items()}
        print(f"  N={n:<5} 只定符号   AUC {m['sign'][0]:.3f} ± {m['sign'][1]:.3f} (翻转率 {np.mean(flips):.0%})"
              f" | 定符号+调阈值 acc {m['thr_acc'][0]:.3f} ± {m['thr_acc'][1]:.3f}")
        print(f"  {'':<7} 原型法     AUC {m['proto'][0]:.3f} ± {m['proto'][1]:.3f}"
              f" | 重拟合线性头 AUC {m['linear'][0]:.3f} ± {m['linear'][1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset/HTP")
    ap.add_argument("--label", choices=["class", "domain", "batch"], required=True)
    ap.add_argument("--env", default="all", choices=["all", "00", "01", "02"])
    ap.add_argument("--region", default="full",
                    choices=["full", "corner", "gray", "binary", "skeleton","clean","geonorm","corner_clean","paper"])
    ap.add_argument("--cache_size", type=int, default=256, help="缓存边长")
    ap.add_argument("--input_size", type=int, default=224, help="送进网络的边长")
    ap.add_argument("--hires", type=int, default=1280, help="提取笔迹时的工作分辨率")
    ap.add_argument("--arch", default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_env", default=None, choices=["00", "01", "02"],
                    help="留一域评估：用其余域训练、在这个域上测。设了它就忽略随机划分")
    ap.add_argument("--adapt_n", type=int, nargs="*", default=None,
                    help="few-shot 适应：从测试域抽 N 张带标注样本。可给多个 N，例如 --adapt_n 10 20 50 100")
    ap.add_argument("--adapt_repeats", type=int, default=20, help="每个 N 重复多少次随机抽样")
    ap.add_argument("--adapt_C", type=float, default=0.01, help="重拟合线性头的正则强度，越小正则越强")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    paths, labels, names = collect(a.data_dir, a.env, a.label)
    n_cls = len(names)
    print(f"任务: label={a.label} env={a.env} region={a.region} "
          f"cache={a.cache_size} input={a.input_size}  共 {len(paths)} 张, {n_cls} 类")
    for i, nm in enumerate(names):
        print(f"  类 {i} ({nm}): {(labels == i).sum()} 张")
    if n_cls < 2:
        print("只有一个类别，没什么可训的。")
        return
    print(f"多数类基线: {np.bincount(labels).max() / len(labels):.3f}")

    print("处理图像中...")
    arr = decode_all(paths, a.cache_size, a.region, a.hires)
    print(f"缓存占用 {arr.nbytes / 1e9:.2f} GB")

    if a.test_env is not None:
        assert a.env == "all", "--test_env 必须配 --env all"
        env_of = np.array([os.path.normpath(p).split(os.sep)[-3] for p in paths])
        tr_idx = list(np.where(env_of != a.test_env)[0])
        te_idx = list(np.where(env_of == a.test_env)[0])
        print(f"留一域: 训练域 {sorted(set(env_of[tr_idx].tolist()))} -> 测试域 {a.test_env}")
    else:
        rng = np.random.RandomState(a.seed)
        tr_idx, te_idx = [], []
        for c in range(n_cls):
            ids = np.where(labels == c)[0]
            rng.shuffle(ids)
            k = int(round(len(ids) * 0.8))
            tr_idx += list(ids[:k])
            te_idx += list(ids[k:])
    print(f"训练 {len(tr_idx)} 张 / 测试 {len(te_idx)} 张")

    tr = DataLoader(MemSet(arr, labels, tr_idx, True, a.input_size),
                    batch_size=a.batch_size, shuffle=True, num_workers=4, drop_last=True)
    te = DataLoader(MemSet(arr, labels, te_idx, False, a.input_size),
                    batch_size=max(4, a.batch_size // 2), shuffle=False, num_workers=4)

    net = getattr(models, a.arch)(weights="IMAGENET1K_V1")
    net.fc = nn.Linear(net.fc.in_features, n_cls)
    net = net.cuda()
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    crit = nn.CrossEntropyLoss()

    print(f"{'epoch':>6} {'train_loss':>11} {'test_acc':>9} {'test_auc':>9} {'秒':>6}")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        net.train()
        tot = 0.0
        for x, y in tr:
            loss = crit(net(x.cuda()), y.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        acc, auc = evaluate(net, te, n_cls)
        print(f"{ep:6d} {tot / max(1, len(tr)):11.4f} {acc:9.3f} {auc:9.3f} {time.time() - t0:6.1f}")
    
    if a.adapt_n:
        assert a.test_env is not None, "--adapt_n 必须配 --test_env"
        print("\n=== few-shot 域适应 ===")
        F, S, Y = extract_feats(net, arr, labels, te_idx, a.input_size,
                               max(4, a.batch_size // 2))
        few_shot_report(F, S, Y, a.adapt_n, a.adapt_repeats, a.adapt_C)


if __name__ == "__main__":
    main()