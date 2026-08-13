"""Qwen3.5-9B 部署冒烟测试：确认能加载、能看图、显存够不够。

只依赖 transformers + torch，不需要 vLLM，因此不受 CUDA 13 的限制。

    pip install -U transformers accelerate
    CUDA_VISIBLE_DEVICES=2,3 python qwen35_smoke.py --model ./models/qwen3.5-9b

跑通了再考虑上 vLLM 提吞吐。
"""

import argparse
import time

import torch
from PIL import Image

PROMPT = (
    "这是一张手绘图画。用三句话描述你看到的内容："
    "画面里有哪些物体，各自大概占画面多大比例，线条是深还是浅。"
    "只描述看得见的，不要解读。"
)


def load_model(path, dtype):
    """Qwen3.5 是统一多模态，优先用 image-text-to-text 的通用 auto 类。"""
    from transformers import AutoProcessor

    errors = []
    for name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        try:
            import transformers
            cls = getattr(transformers, name)
        except AttributeError:
            errors.append(f"{name}: 当前 transformers 没有这个类")
            continue
        try:
            model = cls.from_pretrained(
                path, dtype=dtype, device_map="auto", trust_remote_code=True)
            print(f"加载成功，用的是 {name} -> {type(model).__name__}")
            return model, AutoProcessor.from_pretrained(path, trust_remote_code=True)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    raise SystemExit("两种 auto 类都失败了：\n  " + "\n  ".join(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/qwen3.5-9b")
    ap.add_argument("--image", default="dataset/HTP/00/00/00_00_00_0000.jpg")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    import transformers
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"transformers {transformers.__version__}")
    print(f"可见 GPU: {torch.cuda.device_count()} 张")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory / 1024**3:.1f} GiB")

    t0 = time.time()
    model, processor = load_model(args.model, torch.bfloat16)
    print(f"加载耗时 {time.time() - t0:.1f}s")

    img = Image.open(args.image).convert("RGB")
    if max(img.size) > args.max_side:
        s = args.max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    print(f"输入图片 {args.image} -> {img.size}")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)
    print(f"输入 token 数 {inputs['input_ids'].shape[-1]}")

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=False)
    new = out[0][inputs["input_ids"].shape[-1]:]
    dt = time.time() - t0

    print("\n" + "=" * 60)
    print(processor.decode(new, skip_special_tokens=True).strip())
    print("=" * 60)
    print(f"\n生成 {len(new)} token，耗时 {dt:.1f}s，{len(new) / dt:.1f} tok/s")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU{i} 峰值显存 {torch.cuda.max_memory_allocated(i) / 1024**3:.1f} GiB")
    print(f"\n按这个速度，1164 张大约需要 {1164 * dt / 60:.0f} 分钟")


if __name__ == "__main__":
    main()
