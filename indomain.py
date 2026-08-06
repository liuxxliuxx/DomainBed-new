"""最简域内（in-domain）分类基线：不做任何域泛化算法，就是普通图像分类。

跟 train_all.py 的区别：train_all.py 是"留一个域出来当测试域"的域泛化流程（还带 SWAD/GGA
那一套）；这个脚本是**域内**评测——训练集和测试集来自同一个分布，按比例随机划分，
用来给域泛化结果提供一个"同分布能做到多好"的上限参考。

三种域内口径（--scope）：
  pooled        所有域的图混成一个池子。每个域内部各自按比例划分，再把各域的训练部分拼起来
                训练、测试部分拼起来评测。只报一个总 acc。
  pooled_split  训练方式和 pooled 完全一样，只是评测时额外按域分开各报一个 acc。
  per_env       每个域完全独立：独立划分、独立新建模型训练、独立评测。报各域 acc 和宏平均。

重复实验用 --seeds 显式列出种子（比如 --seeds 0 1 2 就是跑三轮），结果报 mean ± std。
想补跑第四个种子直接 `--seeds 3` 单独跑一次，配上同一个 --out 文件即可，不用重跑前三个。

--image_size 可以一次给多个值，脚本会依次扫一遍，用来看输入分辨率对 acc 的影响。
同一个种子下，不同分辨率共用**完全相同的划分**，所以分辨率之间的对比是配对的。

用法举例：
  # 冒烟：1 个种子 1 个 epoch，顺便把磁盘缓存建起来
  python indomain.py --dataset HTP --data_dir ./dataset --arch resnet18 \
      --scope pooled --seeds 0 --epochs 1 --image_size 112 \
      --cache disk --cache_size 640 --batch_size 8

  # 正式跑分辨率扫描
  python indomain.py --dataset HTP --data_dir ./dataset --arch resnet50 \
      --scope pooled_split --seeds 0 1 2 --epochs 20 --image_size 112 224 448 \
      --cache disk --cache_size 640 --batch_size 32 --amp --out indomain_res.csv

  # ViT
  python indomain.py --dataset HTP --data_dir ./dataset --arch vit_b_16 \
      --scope pooled --seeds 0 1 2 --epochs 20 --image_size 224 \
      --cache disk --cache_size 640 --batch_size 16 --amp
"""
import argparse
import contextlib
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import models
from torchvision.models.vision_transformer import VisionTransformer, interpolate_embeddings

# 复用仓库现成的数据管线：
#   set_transfroms  给切分好的数据集挂 transform（仓库里这个函数名就是拼错的，别改）
#   split_dataset   仓库原生的随机划分（不分层），--stratify 0 时才走它
#   _SplitDataset   按下标取子集的包装类，__getitem__ 返回 dict 而不是 tuple
from domainbed.datasets import _SplitDataset, set_transfroms, split_dataset
from domainbed.datasets.datasets import get_dataset_class
from domainbed.lib import misc

ARCHS = ["resnet18", "resnet50", "vit_b_16", "vit_b_32"]


def _cache_size(v):
    """--cache_size 的 argparse 类型，跟 train_all.py:23 保持一致。full 表示存原图不缩放。"""
    if v.lower() in ("full", "orig", "original"):
        return None
    n = int(v)
    if n < 32:
        raise argparse.ArgumentTypeError("cache_size 至少 32")
    return n


def set_all_seeds(seed):
    """把 python / numpy / torch / cuda 四套随机源一起设定。

    每个 (种子, 分辨率) 格子开跑前都调一次。这样单独跑 --image_size 224，和跑
    --image_size 112 224 时里面 224 那一格，结果是完全一样的——模型初始化、DataLoader 的
    shuffle 顺序都只取决于种子，不受前面已经跑过什么影响。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------------------
def split_env(env, test_ratio, seed, stratify=True):
    """把一个域切成 (训练子集, 测试子集)。

    env        : 一个域对应的 ImageFolder 或 CachedFolder
    test_ratio : 测试集占比
    seed       : 划分种子。划分结果只由它决定，跟全局随机状态无关，所以同一个种子在任何
                 机器上、任何时候跑出来的划分都一样（跟仓库 get_dataset() 的思路一致）
    stratify   : True 表示按类别分层抽样。HTP 每个域每个类只有一两百张，不分层的话类别
                 比例会随机漂，acc 的方差明显变大，所以默认开着
    """
    if not stratify:
        # 仓库原生实现的第一个返回值是"打乱后的前 n 个"。这里把 n 设成测试集大小，
        # 所以拿到的顺序是 (测试, 训练)——跟 get_dataset() 里 `out, in_ = split_dataset(...)` 一样。
        test_split, train_split = split_dataset(env, int(len(env) * test_ratio), seed)
        return train_split, test_split

    y = np.asarray(env.targets)  # ImageFolder 和 CachedFolder 都提供 .targets
    rng = np.random.RandomState(seed)
    train_keys, test_keys = [], []
    for c in np.unique(y):
        ids = np.where(y == c)[0]
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test_ratio))
        n_test = min(max(n_test, 1), max(len(ids) - 1, 0))  # 每类训练和测试都尽量各留至少 1 张
        test_keys += ids[:n_test].tolist()
        train_keys += ids[n_test:].tolist()
    rng.shuffle(train_keys)
    rng.shuffle(test_keys)
    return _SplitDataset(env, train_keys), _SplitDataset(env, test_keys)


def attach_transform(split, kind, size):
    """给切分好的子集挂上 transform。kind 取 "train"（用 aug）或 "test"（用 basic）。

    分辨率扫描之所以能只建一次 dataset，靠的就是这里：底层 ImageFolder / CachedFolder 本身
    不带 transform，transform 挂在 _SplitDataset.transforms 上，换分辨率重挂一次就行。

    两个坑：
    1) ConcatDataset 只是个壳，往它身上设 transforms 属性不会报错但完全无效，底层子集
       还是没 transform，取样本时会因为缺 "x" 键直接崩。所以要递归到它包的每个子集。
    2) set_transfroms 内部有 `assert hparams["data_augmentation"]`，想关掉增强不能把它设成
       False，得直接按 "test" 挂——--no_aug 就是这么做的。
    """
    hparams = {"data_augmentation": True, "val_augment": False, "image_size": size}
    children = split.datasets if isinstance(split, ConcatDataset) else [split]
    for child in children:
        set_transfroms(child, kind, hparams)


def print_dataset_stats(env_names, envs, classes):
    """开头打印每个域每个类多少张 + 多数类基线。没有基线，acc 是多少根本没法解读。"""
    print(f"数据集：{len(envs)} 个域，{len(classes)} 个类 {classes}")
    total = np.zeros(len(classes), dtype=int)
    for name, env in zip(env_names, envs):
        y = np.asarray(env.targets)
        counts = np.array([(y == c).sum() for c in range(len(classes))])
        total += counts
        detail = "  ".join(f"{classes[c]}:{counts[c]}" for c in range(len(classes)))
        print(f"  域 {name}: 共 {len(y):5d} 张   {detail}   多数类基线 {counts.max() / len(y):.3f}")
    print(f"  合计   : 共 {total.sum():5d} 张   多数类基线 {total.max() / total.sum():.3f}")


# --------------------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------------------
def build_model(arch, num_classes, size, pretrained=True):
    """按 arch 建一个分类网络，分类头换成 num_classes 类。

    ResNet 是全卷积 + 全局平均池化，任意输入边长都能跑，不用管 size。
    ViT 的位置编码个数跟 (size/patch)^2 绑死，而 torchvision 的 builder 在传了 weights 时
    会强制把 image_size 覆盖回 224（预训练权重的 meta 里写的就是 224）。所以非 224 分辨率
    必须自己用 interpolate_embeddings 把预训练位置编码重采样一遍，再灌进一个按目标尺寸
    新建的 VisionTransformer。
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    if arch.startswith("resnet"):
        net = getattr(models, arch)(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net

    if arch.startswith("vit_b_"):
        patch = int(arch.rsplit("_", 1)[1])  # vit_b_16 -> 16
        if size % patch != 0:
            raise ValueError(f"{arch} 要求 image_size 能被 patch({patch}) 整除，当前是 {size}")

        if not pretrained:
            # 没有预训练权重时 builder 不会覆盖 image_size，直接按目标尺寸建就行
            net = getattr(models, arch)(weights=None, image_size=size)
        else:
            net = getattr(models, arch)(weights=weights)
            if size != 224:
                state = interpolate_embeddings(
                    image_size=size, patch_size=patch, model_state=net.state_dict()
                )
                # ViT-B 的结构参数，vit_b_16 和 vit_b_32 只差一个 patch_size
                net = VisionTransformer(
                    image_size=size, patch_size=patch, num_layers=12, num_heads=12,
                    hidden_dim=768, mlp_dim=3072, num_classes=1000,
                )
                net.load_state_dict(state)

        net.heads.head = nn.Linear(net.heads.head.in_features, num_classes)
        return net

    raise ValueError(f"不认识的 arch: {arch}")


# --------------------------------------------------------------------------------------
# 评测
# --------------------------------------------------------------------------------------
def amp_context(use_amp):
    """混合精度上下文。关掉时返回空上下文，免得在纯 CPU 环境下碰 torch.cuda.amp。

    新旧两套 API 都兼容：torch>=1.10 是 torch.amp.autocast("cuda")，
    再老的只有 torch.cuda.amp.autocast()（后者在 torch>=2.4 会刷 FutureWarning）。
    """
    if not use_amp:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def make_scaler(use_amp):
    """建 GradScaler，同样兼容新旧 API：torch>=2.3 要求带 device_type 参数。"""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def _metrics(prob, y, num_classes):
    """由概率矩阵和真值算 (acc, auc)。二分类算正类 AUC，多分类算 ovr 宏平均 AUC。"""
    acc = float((prob.argmax(1) == y).mean())
    try:
        if num_classes == 2:
            auc = float(roc_auc_score(y, prob[:, 1]))
        else:
            auc = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")  # 测试集里某个类一张都没有时 sklearn 会抛这个
    return acc, auc


@torch.no_grad()
def predict(net, loader, device, use_amp):
    """跑完一个 loader，返回 (概率矩阵 [n, C], 真值 [n])。"""
    net.eval()
    probs, ys = [], []
    for batch in loader:
        # _SplitDataset 返回的是 dict {"x":…, "y":…}，不是 (x, y) 元组，这是这个仓库的约定
        x = batch["x"].to(device, non_blocking=True)
        with amp_context(use_amp):
            logit = net(x)
        probs.append(torch.softmax(logit.float(), 1).cpu().numpy())
        ys.append(batch["y"].numpy())
    net.train()
    return np.concatenate(probs), np.concatenate(ys)


def evaluate(net, eval_loaders, device, use_amp, num_classes):
    """对每个评测子集单独算指标，再把所有预测拼起来算一个总指标。

    eval_loaders : [(域名, DataLoader), ...]
    返回 ({域名: (acc, auc)}, (总 acc, 总 auc))
    总指标是把全部测试图拼起来算的（微平均），所以图多的域权重大。
    """
    per_env, all_prob, all_y = {}, [], []
    for name, loader in eval_loaders:
        prob, y = predict(net, loader, device, use_amp)
        per_env[name] = _metrics(prob, y, num_classes)
        all_prob.append(prob)
        all_y.append(y)
    overall = _metrics(np.concatenate(all_prob), np.concatenate(all_y), num_classes)
    return per_env, overall


# --------------------------------------------------------------------------------------
# 训练一格：一个种子 × 一个分辨率 × 一个 scope 单元
# --------------------------------------------------------------------------------------
def run_one(train_split, eval_splits, size, seed, num_classes, args, device, tag):
    """从预训练权重开始训一个模型，每个 epoch 评一次，返回逐 epoch 的记录。

    train_split : 训练集。pooled 时是各域训练部分的 ConcatDataset
    eval_splits : [(域名, 测试子集), ...]
    seed        : 本格用的种子，只用来播种模型初始化和 shuffle（划分已经在外面做好了）
    tag         : 打印用的前缀，比如 "seed=0 size=224 pooled"
    返回 [{"epoch":e, "loss":l, "envs":{域名:(acc,auc)}, "overall":(acc,auc)}, ...]
    """
    # 挂 transform 必须在建 DataLoader 之前：worker 是在开始迭代时才把 dataset 复制过去的，
    # 之后再改 transform 就不生效了
    attach_transform(train_split, "test" if args.no_aug else "train", size)
    for _, split in eval_splits:
        attach_transform(split, "test", size)

    eval_bs = args.eval_batch_size or args.batch_size
    # drop_last 能避免最后一个不满的 batch 干扰 BN 统计，但训练集太小时会把 batch 全丢光，加个保护
    drop_last = len(train_split) >= 2 * args.batch_size
    pin = device == "cuda"
    train_loader = DataLoader(
        train_split, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=drop_last, pin_memory=pin,
    )
    eval_loaders = [
        (name, DataLoader(split, batch_size=eval_bs, shuffle=False,
                          num_workers=args.workers, pin_memory=pin))
        for name, split in eval_splits
    ]

    set_all_seeds(seed)  # 见 set_all_seeds 的说明：保证这一格的结果只取决于种子
    net = build_model(args.arch, num_classes, size, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    use_amp = args.amp and device == "cuda"
    scaler = make_scaler(use_amp)

    n_test = sum(len(s) for _, s in eval_splits)
    print(f"\n[{tag}] 训练 {len(train_split)} 张 / 测试 {n_test} 张 "
          f"/ 每 epoch {len(train_loader)} 个 batch")
    print(f"{'epoch':>6} {'loss':>10} {'acc':>8} {'auc':>8} {'秒':>7}")

    records = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        total = 0.0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            with amp_context(use_amp):
                loss = crit(net(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()

        per_env, overall = evaluate(net, eval_loaders, device, use_amp, num_classes)
        loss_avg = total / max(1, len(train_loader))
        records.append({"epoch": ep, "loss": loss_avg, "envs": per_env, "overall": overall})
        print(f"{ep:6d} {loss_avg:10.4f} {overall[0]:8.3f} {overall[1]:8.3f} "
              f"{time.time() - t0:7.1f}")

    best = max(r["overall"][0] for r in records)
    print(f"[{tag}] 末轮 acc {records[-1]['overall'][0]:.3f}（记进汇总的是这个）"
          f"   最好 epoch acc {best:.3f}（挑了测试集上最好的一轮，只能参考，不能当结果报）")
    return records


# --------------------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------------------
def fmt_cell(values):
    """把一组数格式化成 `均值±标准差`。只有一个值时不显示标准差。"""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    return f"{v.mean():.3f}±{v.std():.3f}"


def print_summary(results, sizes, seeds, col_names, args):
    """打印汇总表：行是分辨率，列是各域 + overall，均值标准差在种子上取。

    results[size][seed] = {"envs": {域名: (acc, auc)}, "overall": (acc, auc)}
    """
    print(f"\n=== 汇总  dataset={args.dataset}  scope={args.scope}  arch={args.arch}  "
          f"seeds={seeds}  epochs={args.epochs}  test_ratio={args.test_ratio} ===")
    if args.scope == "per_env":
        print("（per_env 的 overall 是各域 acc 的宏平均。各域是各自独立的模型，"
              "把预测混起来算微平均没有意义）")

    for metric_i, metric in enumerate(["acc", "auc"]):
        print(f"\n{metric}")
        print(f"{'image_size':>11}" + "".join(f"{c:>16}" for c in col_names))
        for size in sizes:
            cells = []
            for col in col_names:
                vals = []
                for seed in seeds:
                    r = results[size][seed]
                    vals.append(r["overall"][metric_i] if col == "overall"
                                else r["envs"][col][metric_i])
                cells.append(fmt_cell(vals))
            print(f"{size:>11}" + "".join(f"{c:>16}" for c in cells))


def make_row(args, size, seed, env, rec, metric):
    """拼一行 csv 记录。metric 是 (acc, auc)。"""
    return {
        "dataset": args.dataset, "scope": args.scope, "arch": args.arch,
        "image_size": size, "seed": seed, "env": env, "epoch": rec["epoch"],
        "acc": round(metric[0], 5), "auc": round(metric[1], 5),
        "loss": round(rec["loss"], 5),
    }


def dump_csv(path, rows):
    """逐行写 csv，追加模式：文件不存在才写表头。
    这样分几次补种子（比如后来单独跑 --seeds 3 4）的结果能落进同一个文件，
    直接拿去画分辨率-acc 曲线。
    """
    fields = ["dataset", "scope", "arch", "image_size", "seed", "env", "epoch",
              "acc", "auc", "loss"]
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"\n结果已{'追加到' if exists else '写入'} {path}（本次 {len(rows)} 行）")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="最简域内分类评测：按比例划分 train/test，跑 ResNet 或 ViT，统计 acc")
    ap.add_argument("--dataset", default="HTP", help="domainbed/datasets/datasets.py 里的数据集名")
    ap.add_argument("--data_dir", default="./dataset",
                    help="数据根目录。数据集类会自己往下拼子目录，比如 HTP 拼的是 HTP/")
    ap.add_argument("--scope", default="pooled", choices=["pooled", "pooled_split", "per_env"],
                    help="域内口径：pooled 混成一个池子报一个 acc；pooled_split 训练同 pooled "
                         "但按域分开报；per_env 每个域独立训练独立评测")
    ap.add_argument("--arch", default="resnet50", choices=ARCHS)
    ap.add_argument("--image_size", type=int, nargs="+", default=[224],
                    help="送进网络的边长，可给多个（如 --image_size 112 224 448）依次扫。"
                         "注意缓存边长由 --cache_size 决定，要让扫描只反映网络输入分辨率这"
                         "一个变量，应保证 cache_size >= max(image_size)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="显式列出要跑哪几个种子，跑几轮由列表长度决定。每个种子对应一套"
                         "独立的随机划分和一次从预训练权重开始的重新训练")
    ap.add_argument("--test_ratio", type=float, default=0.2, help="测试集占比")
    ap.add_argument("--stratify", type=int, default=1, choices=[0, 1],
                    help="1 按类别分层划分（默认，小数据集上方差小很多）；0 用仓库原生的随机划分")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5, help="与 hparams_registry 的默认值一致")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=None,
                    help="不给就跟 --batch_size 一样。评测不反传，通常可以开大点")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 进程数")
    ap.add_argument("--amp", action="store_true", help="开混合精度。高分辨率的 ViT 基本必开")
    ap.add_argument("--no_aug", action="store_true", help="训练集也不做数据增强，只 resize")
    ap.add_argument("--no_pretrained", action="store_true", help="从随机初始化开始训")

    ap.add_argument("--cache", default="none", choices=["none", "disk", "ram"],
                    help="复用 domainbed 的图像缓存。HTP 是大扫描图，强烈建议用 disk")
    ap.add_argument("--cache_size", type=_cache_size, default=None,
                    help="缓存边长；full 表示存原图不缩放")
    ap.add_argument("--resize_mode", default="stretch", choices=["stretch", "pad"])
    ap.add_argument("--cache_root", default="cache")
    ap.add_argument("--out", default=None, help="把逐 epoch 结果写成 csv（追加模式）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 每一格里输入尺寸是固定的，开 benchmark 让 cudnn 挑最快的卷积算法
    torch.backends.cudnn.benchmark = True
    print(f"设备 {device} | arch {args.arch} | scope {args.scope} | "
          f"image_size {args.image_size} | seeds {args.seeds}")
    if args.cache_size is not None and args.cache_size < max(args.image_size):
        print(f"警告：cache_size={args.cache_size} 小于最大 image_size={max(args.image_size)}，"
              f"缓存图会先被缩小再放大回去，高分辨率那几档等于白跑")

    # ---------------- 数据：整个 run 只建一次 ----------------
    # 只支持 MultipleEnvironmentImageFolder 那一系（真实图片数据集）。MNIST 那几个是
    # 张量数据集，构造签名和 .classes/.environments 都不一样，早点报错比让它在里面崩强。
    if "MNIST" in args.dataset or args.dataset.startswith("Debug"):
        raise SystemExit(f"{args.dataset} 不是图片文件夹型数据集，这个脚本不支持。"
                         f"可用的有 VLCS / PACS / OfficeHome / TerraIncognita / DomainNet / HTP")

    # 这里传 max(image_size) 只是给数据集类内部做合法性校验，实际送进网络的尺寸完全由挂在
    # _SplitDataset 上的 transform 决定，所以换分辨率不需要重建数据集，缓存也只建一份。
    ds = get_dataset_class(args.dataset)(
        args.data_dir,
        cache=args.cache,
        cache_size=args.cache_size,
        resize_mode=args.resize_mode,
        cache_root=args.cache_root,
        image_size=max(args.image_size),
    )
    envs = [env for env in ds]   # MultipleDomainDataset 支持按下标取第 i 个域
    env_names = ds.environments  # HTP 是 ["00","01","02"]，不是 ENVIRONMENTS 里写的那几个名字
    classes = envs[0].classes
    for name, env in zip(env_names, envs):
        # 各域的类别列表必须一致，否则 pooled 会把不同域的不同类别当成同一类
        if env.classes != classes:
            raise ValueError(f"域 {name} 的类别 {env.classes} 与域 {env_names[0]} 的 {classes} 不一致")
    num_classes = len(classes)
    print_dataset_stats(env_names, envs, classes)

    # ---------------- 主循环：种子在外，分辨率在内 ----------------
    # 这个嵌套顺序保证同一个种子下所有分辨率用的是完全相同的划分，分辨率之间的对比是配对的
    results = {size: {} for size in args.image_size}
    csv_rows = []

    for seed in args.seeds:
        splits = [
            split_env(env, args.test_ratio, misc.seed_hash(seed, i), stratify=bool(args.stratify))
            for i, env in enumerate(envs)
        ]

        for size in args.image_size:
            if args.scope == "per_env":
                # 每个域一个独立的模型，各训各的
                per_env = {}
                for i, name in enumerate(env_names):
                    train_split, test_split = splits[i]
                    recs = run_one(train_split, [(name, test_split)], size, seed, num_classes,
                                   args, device, f"seed={seed} size={size} env={name}")
                    per_env[name] = recs[-1]["envs"][name]
                    csv_rows += [make_row(args, size, seed, name, r, r["envs"][name])
                                 for r in recs]
                # 各域是不同的模型，总指标只能取各域 acc 的宏平均
                overall = tuple(float(np.mean([per_env[n][k] for n in env_names])) for k in (0, 1))
                results[size][seed] = {"envs": per_env, "overall": overall}
            else:
                # pooled / pooled_split：一个模型，训练集是各域训练部分拼起来
                train_split = ConcatDataset([tr for tr, _ in splits])
                eval_splits = [(name, te) for name, (_, te) in zip(env_names, splits)]
                recs = run_one(train_split, eval_splits, size, seed, num_classes, args, device,
                               f"seed={seed} size={size} {args.scope}")
                results[size][seed] = {"envs": recs[-1]["envs"], "overall": recs[-1]["overall"]}
                for r in recs:
                    csv_rows.append(make_row(args, size, seed, "overall", r, r["overall"]))
                    if args.scope == "pooled_split":
                        csv_rows += [make_row(args, size, seed, n, r, r["envs"][n])
                                     for n in env_names]

    # ---------------- 汇总 ----------------
    col_names = ["overall"] if args.scope == "pooled" else list(env_names) + ["overall"]
    print_summary(results, args.image_size, args.seeds, col_names, args)
    if args.out:
        dump_csv(args.out, csv_rows)


if __name__ == "__main__":
    main()
