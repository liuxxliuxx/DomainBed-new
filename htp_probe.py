"""在 VLM 抽出的客观视觉特征上训逻辑回归，按留一域协议评估。

输入是 `qwen_htp_eval.py --mode extract` 产出的 CSV，其中 features 列是 JSON。

    python htp_probe.py output/qwen_htp_feat.csv

协议和 DomainBed 一致：三个域轮流做测试域，另外两个域做训练。
正则强度只在训练域内部选（一个域训另一个域验，两个方向取平均），
测试域从头到尾不参与任何选择。
"""

import argparse
import collections
import csv
import json
import sys

import numpy as np

ENV_NAME = {"00": "child", "01": "college", "02": "social"}
POS = "00"          # 正类 = 存在心理异常倾向
# 101 个字段展开后维度可能接近样本量，正则要能选到更强的档位
LAMBDAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            raw = r.get("features") or r.get("analysis") or ""
            if not raw:
                continue
            try:
                feat = json.loads(raw)
            except json.JSONDecodeError:
                continue
            rows.append((r["path"], r["env"], r["label"], feat))
    return rows


def build_matrix(rows, min_count=15):
    """所有字段都是枚举字符串，直接 one-hot。取值集合从全量数据统计，
    这一步不看标签，不构成信息泄漏。

    出现次数少于 min_count 的取值直接丢掉：它们既学不出可泛化的权重，
    标准化之后还会被放大成巨大的数值把优化搞崩。"""
    values = collections.defaultdict(collections.Counter)
    for _, _, _, f in rows:
        for k, v in f.items():
            values[k][str(v)] += 1
    cols, dropped = [], 0
    for k in sorted(values):
        vs = sorted(values[k])
        keep = [v for v in vs if values[k][v] >= min_count]
        dropped += len(vs) - len(keep)
        if len(keep) < 2:                      # 整个字段没有变化，跳过
            continue
        # 二值字段只留一列，避免冗余
        for v in (keep[1:] if len(keep) == 2 else keep):
            cols.append((k, v))
    X = np.zeros((len(rows), len(cols)), dtype=np.float64)
    for i, (_, _, _, f) in enumerate(rows):
        for j, (k, v) in enumerate(cols):
            if str(f.get(k)) == v:
                X[i, j] = 1.0
    y = np.array([1.0 if r[2] == POS else 0.0 for r in rows])
    env = np.array([r[1] for r in rows])
    if dropped:
        print(f"  丢弃出现少于 {min_count} 次的取值 {dropped} 个")
    return X, y, env, cols


def fit(X, y, lam, iters=4000):
    """带类别权重的 L2 逻辑回归。

    步长按梯度的 Lipschitz 常数取 1/L，这是凸光滑问题上梯度下降的收敛保证。
    写死 lr 在维度变多、列尺度不齐时会直接发散（331 维那次就是这么炸的）。
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    sw = np.where(y == 1, 0.5 / max(n_pos, 1), 0.5 / max(n_neg, 1)) * n

    H = (X * sw[:, None]).T @ X / n          # 加权二阶矩
    L = 0.25 * float(np.linalg.eigvalsh(H).max()) + lam + 0.25 * sw.mean()
    lr = 1.0 / max(L, 1e-8)

    for _ in range(iters):
        z = np.clip(X @ w + b, -30.0, 30.0)   # 防 exp 溢出
        p = 1.0 / (1.0 + np.exp(-z))
        r = sw * (p - y)
        w -= lr * ((X.T @ r) / n + lam * w)
        b -= lr * r.mean()
    if not (np.all(np.isfinite(w)) and np.isfinite(b)):
        raise FloatingPointError("逻辑回归未收敛，出现 inf/nan")
    return w, b


def predict_proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


def auc(scores, y):
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    yy = y[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = yy.sum(), len(yy) - yy.sum()
    if not n_pos or not n_neg:
        return float("nan")
    return (ranks[yy == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def balanced(pred, y):
    tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
    tnr = 1 - pred[y == 0].mean() if (y == 0).any() else 0.0
    return (tpr + tnr) / 2


def standardize(Xtr, Xte):
    """列是 0/1 指示变量，标准差下限设 0.1，避免稀有取值被放大成几十倍。"""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.maximum(sd, 0.1)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--top", type=int, default=15, help="打印多少个权重最大的特征")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        sys.exit(f"{args.csv} 里没有可用的 features 列")
    X, y, env, cols = build_matrix(rows)
    print(f"{len(rows)} 样本，{X.shape[1]} 维 one-hot 特征（原始字段 "
          f"{len({k for k, _ in cols})} 个）")
    for e in sorted(set(env)):
        m = env == e
        print(f"  {ENV_NAME.get(e, e):8s} n={m.sum():4d} 正类占比={y[m].mean():.3f}")

    envs = sorted(set(env))
    accs, bals, aucs, coefs = [], [], [], []
    print(f"\n{'held-out':10s} {'lam':>7s} {'acc':>8s} {'balanced':>9s} "
          f"{'AUC':>7s} {'base':>7s}")
    for te in envs:
        te_m = env == te
        tr_envs = [e for e in envs if e != te]

        # 内层：训练域之间互为验证集，选 lambda
        best_lam, best_val = LAMBDAS[0], -1.0
        for lam in LAMBDAS:
            vals = []
            for va in tr_envs:
                a_m, b_m = env == [e for e in tr_envs if e != va][0], env == va
                Xa, Xb = standardize(X[a_m], X[b_m])
                w, b = fit(Xa, y[a_m], lam)
                vals.append(balanced((predict_proba(Xb, w, b) >= 0.5).astype(float),
                                     y[b_m]))
            if np.mean(vals) > best_val:
                best_val, best_lam = np.mean(vals), lam

        tr_m = ~te_m
        Xtr, Xte = standardize(X[tr_m], X[te_m])
        w, b = fit(Xtr, y[tr_m], best_lam)
        p = predict_proba(Xte, w, b)
        pred = (p >= 0.5).astype(float)
        acc = (pred == y[te_m]).mean()
        bal = balanced(pred, y[te_m])
        a = auc(p, y[te_m])
        base = max(y[te_m].mean(), 1 - y[te_m].mean())
        accs.append(acc)
        bals.append(bal)
        aucs.append(a)
        coefs.append(w)
        print(f"{ENV_NAME.get(te, te):10s} {best_lam:7.4f} {acc:8.4f} {bal:9.4f} "
              f"{a:7.4f} {base:7.4f}")

    print(f"{'mean':10s} {'':7s} {np.mean(accs):8.4f} {np.mean(bals):9.4f} "
          f"{np.mean(aucs):7.4f}")

    W = np.mean(coefs, axis=0)
    print(f"\n三折平均权重最大的 {args.top} 个特征（正号推向"
          f"「存在心理异常倾向」）：")
    for j in np.argsort(-np.abs(W))[:args.top]:
        print(f"  {W[j]:+.3f}  {cols[j][0]} = {cols[j][1]}")


if __name__ == "__main__":
    main()
