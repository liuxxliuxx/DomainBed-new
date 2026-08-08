"""Zero-shot / few-shot evaluation of a Qwen-VL model on the HTP dataset.

Assumes a vLLM OpenAI-compatible server is already running, e.g.

    vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
        --served-model-name qwen-vl \
        --max-model-len 8192 \
        --limit-mm-per-prompt image=1 \
        --gpu-memory-utilization 0.9

Usage:
    python qwen_htp_eval.py --data-dir dataset/HTP --out output/qwen_htp.csv
"""

import argparse
import base64
import collections
import csv
import io
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from openai import OpenAI

# HTP 的目录名是 00 / 01，这里必须写清楚每个标签的真实含义，
# 否则模型只是在猜两个没有语义的符号。
CLASS_DESC = {
    "00": "存在心理异常倾向",
    "01": "心理状态正常",
}
CLASS_DIRS = ["00", "01"]
CHOICES = [CLASS_DESC[c] for c in CLASS_DIRS]

ENV_NAME = {"00": "child", "01": "college", "02": "social"}

PROMPT = (
    "这是一张房树人（HTP）绘画测验的作品，由被试手绘完成。\n"
    "请结合画面内容判断被试的心理状态，只能从下面两个选项中选一个，"
    "不要输出任何其他文字。\n"
    f"选项：{CHOICES[0]} / {CHOICES[1]}"
)


def encode_image(path, max_side=1024):
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def list_samples(data_dir):
    samples = []
    for env in sorted(os.listdir(data_dir)):
        env_dir = os.path.join(data_dir, env)
        if not os.path.isdir(env_dir):
            continue
        for cls in CLASS_DIRS:
            cls_dir = os.path.join(env_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                samples.append((os.path.join(cls_dir, fname), env, cls))
    return samples


def predict(client, model, path):
    b64 = encode_image(path)
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        temperature=0,
        max_tokens=16,
        # vLLM 的引导解码，保证输出一定落在两个选项里，不用再写正则去解析
        extra_body={"guided_choice": CHOICES},
    )
    text = resp.choices[0].message.content.strip()
    return CLASS_DIRS[CHOICES.index(text)] if text in CHOICES else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/HTP")
    # 多卡时每张卡起一个独立实例，这里用逗号分隔多个地址，请求轮流发
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="qwen-vl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="output/qwen_htp.csv")
    args = ap.parse_args()

    clients = [OpenAI(base_url=u.strip(), api_key=args.api_key)
               for u in args.base_url.split(",")]
    samples = list_samples(args.data_dir)
    print(f"{len(samples)} images, {len(clients)} endpoint(s)")

    def work(idx_item):
        idx, (path, env, gt) = idx_item
        client = clients[idx % len(clients)]
        try:
            pred = predict(client, args.model, path)
        except Exception as e:  # 单张失败不影响整体，记为 None
            print(f"[fail] {path}: {e}")
            pred = None
        return path, env, gt, pred

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(work, enumerate(samples)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "env", "label", "pred"])
        w.writerows(rows)

    hit = collections.Counter()
    tot = collections.Counter()
    cm = collections.Counter()
    for _, env, gt, pred in rows:
        tot[env] += 1
        tot[(env, gt)] += 1
        if pred == gt:
            hit[env] += 1
            hit[(env, gt)] += 1
        cm[(gt, pred)] += 1

    for env in sorted(tot):
        if not isinstance(env, str):
            continue
        per_cls = [hit[(env, c)] / max(tot[(env, c)], 1) for c in CLASS_DIRS]
        print(f"{ENV_NAME.get(env, env):8s} acc={hit[env] / tot[env]:.4f} "
              f"balanced={sum(per_cls) / len(per_cls):.4f} n={tot[env]}")
    total = sum(tot[e] for e in tot if isinstance(e, str))
    print(f"overall  acc={sum(hit[e] for e in tot if isinstance(e, str)) / total:.4f}")
    print("confusion (label -> pred):", dict(cm))


if __name__ == "__main__":
    main()
