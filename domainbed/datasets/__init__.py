from pathlib import Path

import torch
import numpy as np
from torchvision import transforms as T

from domainbed.datasets import datasets
from domainbed.lib import misc
from domainbed.datasets import transforms as DBT


def set_transfroms(dset, data_type, hparams, algorithm_class=None, band_eq=None):

    # Originally, DomainBed use same training augmentation policy to validation.
    # We turn off the augmentation for validation as default,
    # but left the option to reproducibility.
    assert hparams["data_augmentation"]
    size = int(hparams["image_size"])

    additional_data = False
    if data_type == "train":
        dset.transforms = {"x": DBT.aug(size, band_eq)}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            dset.transforms = {"x": DBT.basic(size, band_eq)}
        else:
            dset.transforms = {"x": DBT.aug(size, band_eq)}
    elif data_type == "test":
        # 均衡是归一化不是增强，测试域也必须过，否则训练和推理的分布对不上
        dset.transforms = {"x": DBT.basic(size, band_eq)}
    elif data_type == "mnist":
        dset.transforms = {"x": lambda x: x}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform


def _as_bool(v):
    """命令行传进 hparams 的值可能是字符串，bool("0") 和 bool("False") 都是 True，
    直接 bool() 会让 --band_eq 0 静默变成开启。这里统一按语义解析。"""
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "none", "off")
    return bool(v)


def _band_share(x, edges):
    """(C,H,W) 张量 -> K 维频带能量占比，按亮度算、去直流。"""
    C, H, W = x.shape
    nb = min(H, W) // 2
    yy, xx = np.mgrid[0:H, 0:W]
    rad = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2)
    rbin = np.clip(rad.astype(np.int64), 0, nb - 1)
    inside = rad < nb
    a = x.numpy().astype(np.float64)
    lum = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2] if C >= 3 else a.mean(0)
    P = np.abs(np.fft.fftshift(np.fft.fft2(lum - lum.mean()))) ** 2
    ring = np.bincount(rbin[inside], weights=P[inside], minlength=nb)
    ring[0] = 0.0
    e = np.array([ring[edges[k]:edges[k + 1]].sum() for k in range(len(edges) - 1)])
    s = e.sum()
    return e / s if s > 0 else None


def get_band_target(dataset, test_envs, args, hparams, n_per_env=400):
    """频带均衡的目标剖面。

    只统计源域——目标域的频谱统计进了目标剖面就是泄漏，所以每个留一域划分各有
    一份目标，按 test_envs 缓存到磁盘。target_mode="fixed" 时用解析剖面，
    不统计、也就没有泄漏和簿记。
    """
    size = int(hparams["image_size"])
    mode = str(hparams["band_eq_target_mode"])
    if mode == "fixed":
        return DBT.fixed_target(size)

    edges = DBT.band_edges(size)
    tag = f"{args.dataset}_te{'-'.join(map(str, sorted(test_envs)))}_{size}.npz"
    path = Path(args.cache_root) / "band_eq" / tag
    if path.exists():
        z = np.load(path)
        return z["target"], z["edges"]

    pre = T.Compose([T.Resize((size, size)), T.ToTensor()])
    rng = np.random.RandomState(0)
    profs = []
    for env_i, env in enumerate(dataset):
        if env_i in test_envs:          # 泄漏就出在这一行，别省
            continue
        idx = rng.choice(len(env), min(n_per_env, len(env)), replace=False)
        for i in idx:
            x = env[int(i)][0]
            if not torch.is_tensor(x):
                x = pre(x)
            elif x.shape[-1] != size:
                x = T.Resize((size, size))(x)
            p = _band_share(x, edges)
            if p is not None:
                profs.append(p)
    if not profs:
        raise RuntimeError("统计目标剖面时没取到任何样本")
    # 几何平均：能量跨数量级，算术平均会被少数大值主导
    target = np.exp(np.log(np.clip(np.array(profs), 1e-30, None)).mean(axis=0))
    target = target / target.sum()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, target=target, edges=edges)
    return target, edges


def get_dataset(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    if is_mnist:
        dataset = vars(datasets)[args.dataset](args.data_dir)
    else:
        dataset = vars(datasets)[args.dataset](
            args.data_dir,
            cache=args.cache,
            cache_size=args.cache_size,
            resize_mode=args.resize_mode,
            cache_root=args.cache_root,
            image_size=int(hparams["image_size"]),
        )
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")

    # 目标剖面要在这里建：只有这个函数知道 test_envs，变换类内部不该读文件
    band_eq = None
    if not is_mnist and _as_bool(hparams["band_eq"]):
        target, edges = get_band_target(dataset, test_envs, args, hparams)
        band_eq = DBT.BandEqualize(
            target, edges,
            preserve_total=_as_bool(hparams["band_eq_preserve_total"]),
        )

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else:
            in_type = "train"
            out_type = "valid"

        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"

        set_transfroms(in_, in_type, hparams, algorithm_class, band_eq)
        set_transfroms(out, out_type, hparams, algorithm_class, band_eq)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}

        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {"y": y}

        for key, transform in self.transforms.items():
            ret[key] = transform(x)

        return ret

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)
