import hashlib
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _decode_one(job):
    idx, path, size = job
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return idx, np.asarray(im, dtype=np.uint8)


class CachedImageFolder:
    """包一层 ImageFolder。首次构造时把所有图解码并缩到 (size, size)，
    写成磁盘上的 uint8 memmap；之后所有进程直接 mmap 读。
    对外接口和 ImageFolder 一致，__getitem__ 返回 (PIL.Image, label)。
    """

    def __init__(self, folder, size=256, cache_root="cache", workers=None):
        self.samples = folder.samples          # [(path, class_idx), ...] 顺序照抄
        self.classes = folder.classes
        self.class_to_idx = folder.class_to_idx
        self.targets = [y for _, y in folder.samples]
        self.size = size
        self._arr = None

        cache_root = Path(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(
            ("\n".join(p for p, _ in self.samples) + f"|{size}").encode("utf8")
        ).hexdigest()[:16]
        self.arr_path = cache_root / f"{key}.npy"
        self.meta_path = cache_root / f"{key}.json"

        if not self.meta_path.exists():
            self._build(workers or os.cpu_count())
        self._wait_ready()

    def _build(self, workers):
        lock = self.meta_path.with_suffix(".lock")
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            return                              # 别的进程在建，去 _wait_ready 等

        try:
            n = len(self.samples)
            tmp = self.arr_path.with_suffix(".tmp.npy")
            arr = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.uint8, shape=(n, self.size, self.size, 3)
            )
            jobs = [(i, p, self.size) for i, (p, _) in enumerate(self.samples)]
            print(f"[cache] building {self.arr_path.name}: {n} images", flush=True)
            with Pool(workers) as pool:
                for done, (i, a) in enumerate(
                    pool.imap_unordered(_decode_one, jobs, chunksize=16), 1
                ):
                    arr[i] = a
                    if done % 2000 == 0:
                        print(f"[cache] {done}/{n}", flush=True)
            arr.flush()
            del arr
            os.replace(tmp, self.arr_path)      # 原子替换，中途崩了不会留半成品
            self.meta_path.write_text(
                json.dumps({"n": n, "size": self.size}), encoding="utf8"
            )
            print(f"[cache] done: {self.arr_path}", flush=True)
        finally:
            lock.unlink(missing_ok=True)

    def _wait_ready(self, timeout=7200):
        t0 = time.time()
        while not self.meta_path.exists():
            if time.time() - t0 > timeout:
                raise RuntimeError(f"等待缓存超时: {self.meta_path}")
            time.sleep(5)

    @property
    def arr(self):
        if self._arr is None:                   # 每个 worker 进程各自 mmap
            self._arr = np.load(self.arr_path, mmap_mode="r")
        return self._arr

    def __getstate__(self):                     # memmap 不能被 pickle，跨进程时丢掉
        d = self.__dict__.copy()
        d["_arr"] = None
        return d

    def __getitem__(self, index):
        return Image.fromarray(np.array(self.arr[index])), self.samples[index][1]

    def __len__(self):
        return len(self.samples)