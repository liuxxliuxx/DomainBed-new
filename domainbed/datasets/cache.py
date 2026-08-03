import hashlib
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resize(im, size, mode):
    if size is None:
        return im
    if mode == "stretch":
        return im.resize((size, size), Image.BICUBIC)
    # 长边缩到 size，保持比例，其余补纯白
    w, h = im.size
    r = size / max(w, h)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    im = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _dump_one(job):
    src, dst, size, resize_mode = job
    if dst.exists():
        return
    im = Image.open(src).convert("RGB")
    a = np.asarray(_resize(im, size, resize_mode), dtype=np.uint8)
    tmp = dst.parent / f"{dst.stem}.{os.getpid()}.tmp.npy"
    np.save(tmp, a, allow_pickle=False)
    os.replace(tmp, dst)


class CachedFolder:
    """全分辨率缓存，不做任何缩放。
    __getitem__ 返回的 PIL 图像与 ImageFolder 逐位相同。

    mode = "disk"  每张图一个 .npy，用 mmap 读，页缓存自动共享
    mode = "ram"   启动时全部读进内存
    """

    def __init__(self, folder, mode="disk", size=None, resize_mode="stretch",
                 cache_root="cache", workers=None):
        self.samples = folder.samples
        self.classes = folder.classes
        self.class_to_idx = folder.class_to_idx
        self.targets = [y for _, y in folder.samples]
        self.mode = mode

        tag = "full" if size is None else f"{size}{resize_mode[0]}"   # full / 640s / 640p
        key = hashlib.sha1(
            ("\n".join(p for p, _ in self.samples) + f"|{tag}").encode("utf8")
        ).hexdigest()[:16]
        self.dir = Path(cache_root) / f"{tag}_{key}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.npy = [self.dir / f"{i:06d}.npy" for i in range(len(self.samples))]

        todo = [(p, f, size, resize_mode)
                for (p, _), f in zip(self.samples, self.npy) if not f.exists()]
        if todo:
            print(f"[cache] 建缓存 {self.dir.name}: {len(todo)} 张", flush=True)
            with Pool(workers or os.cpu_count()) as pool:
                for k, _ in enumerate(pool.imap_unordered(_dump_one, todo, chunksize=8), 1):
                    if k % 200 == 0:
                        print(f"[cache] {k}/{len(todo)}", flush=True)

        self._ram = None
        if mode == "ram":
            self._ram = [np.load(f) for f in self.npy]

    def __getstate__(self):
        # ram 模式下 fork 会共享；spawn 的话这里要重新加载，先置空避免 pickle 巨大对象
        d = self.__dict__.copy()
        if self.mode == "ram":
            d["_ram"] = None
        return d

    def __getitem__(self, index):
        if self.mode == "ram":
            if self._ram is None:               # spawn 后的 worker 兜底
                self._ram = [np.load(f) for f in self.npy]
            a = self._ram[index]
        else:
            a = np.load(self.npy[index], mmap_mode="r")
        return Image.fromarray(np.asarray(a)), self.samples[index][1]

    def __len__(self):
        return len(self.samples)