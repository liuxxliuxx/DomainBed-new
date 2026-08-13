"""HTP Qwen structured-feature domain-shift and abstraction analysis.

This script uses the objective categorical features produced by
``qwen_htp_eval.py --mode extract``.  It does not call a remote model and does
not modify the source CSV or images.

Primary outputs:
  * a deduplicated sample-level abstraction proxy table;
  * nested-CV domain-classifier results and within-label permutation tests;
  * Kruskal-Wallis, Dunn-Holm and Cliff's delta results;
  * PCA/UMAP, confusion-matrix and violin plots.

Example (from the repository root):
    conda run -n yolo python htp_shift_abstraction.py \
        --csv logs/q36_feat_1.csv --output-dir output/htp_shift

The abstraction score is an explicitly defined proxy for representational
simplification.  It is not a validated psychological scale.  Domain and class
labels are never used to construct the score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from scipy.stats import kruskal, mannwhitneyu, norm, rankdata
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DOMAIN_NAMES = {"00": "Child", "01": "College", "02": "Social"}
DOMAIN_ORDER = ["Child", "College", "Social"]
DOMAIN_COLORS = {"Child": "#3978B8", "College": "#E68A3F", "Social": "#718C44"}
LABEL_NAMES = {"00": "类别 00", "01": "类别 01"}
LABEL_COLORS = {"类别 00": "#4C78A8", "类别 01": "#E07A5F"}


# High values always mean more representational simplification / abstraction.
# The maps were fixed without using domain or psychological labels.
SCORE_MAPS = {
    "overall_detail_level": {
        "minimal": 1.0, "low": 2 / 3, "moderate": 1 / 3, "high": 0.0,
    },
    "house_detail_amount": {
        "none": 1.0, "few": 2 / 3, "moderate": 1 / 3, "many": 0.0,
    },
    "house_window_count": {
        "0": 1.0, "1": 0.75, "2": 0.50, "3": 0.25, "4+": 0.0,
    },
    "house_door_present": {"no": 1.0, "yes": 0.0},
    "tree_leaves_amount": {
        "none": 1.0, "sparse": 2 / 3, "moderate": 1 / 3, "dense": 0.0,
    },
    "tree_roots": {"none": 1.0, "simple": 0.5, "elaborate": 0.0},
    "tree_crown_present": {"no": 1.0, "yes": 0.0},
    "tree_branch_tips": {
        "none": 1.0, "open_ended": 0.0, "rounded": 0.0, "pointed": 0.0,
    },
    "person_style": {"stick_figure": 1.0, "outline": 0.5, "detailed": 0.0},
    "person_facial_feature_count": {
        "0": 1.0, "1": 0.75, "2": 0.50, "3": 0.25, "4+": 0.0,
    },
    "person_hands_detail": {
        "none": 1.0, "stub": 0.75, "simple": 0.5, "fingers": 0.0, "fist": 0.25,
    },
    "person_feet_detail": {
        "none": 1.0, "stub": 0.75, "simple": 0.5, "shoes": 0.0,
    },
    "person_clothing_detail": {
        "none": 1.0, "few": 2 / 3, "moderate": 1 / 3, "many": 0.0,
    },
    "person_hair_amount": {"none": 1.0, "sparse": 0.5, "full": 0.0},
    "person_body_shape": {
        "single_line": 1.0, "rectangle": 0.5, "rounded": 0.5, "detailed": 0.0,
    },
}

COMPONENT_FIELDS = {
    "global_simplification": ["overall_detail_level"],
    "house_simplification": [
        "house_detail_amount", "house_window_count", "house_door_present",
    ],
    "tree_simplification": [
        "tree_leaves_amount", "tree_roots", "tree_crown_present", "tree_branch_tips",
    ],
    "person_simplification": [
        "person_style", "person_facial_feature_count", "person_hands_detail",
        "person_feet_detail", "person_clothing_detail", "person_hair_amount",
        "person_body_shape",
    ],
}

PRESENCE_FIELD = {
    "house_simplification": "house_present",
    "tree_simplification": "tree_present",
    "person_simplification": "person_present",
}

ABSTRACTION_SOURCE_FIELDS = sorted(
    set(SCORE_MAPS) | set(PRESENCE_FIELD.values())
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="logs/q36_feat_1.csv")
    parser.add_argument("--output-dir", default="output/htp_shift")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.set_defaults(skip_umap=True)
    parser.add_argument(
        "--skip-umap", action="store_true",
        help="只画 PCA（yolo 环境的安全默认值）",
    )
    parser.add_argument(
        "--with-umap", dest="skip_umap", action="store_false",
        help="额外运行 UMAP；当前 yolo 环境导入 pynndescent 可能长时间停滞",
    )
    return parser.parse_args()


def normalize_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_audit(csv_path: Path, project_root: Path, output_dir: Path):
    raw = pd.read_csv(csv_path, dtype={"path": str, "env": str, "label": str})
    required = {"path", "env", "label", "features"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

    raw["env"] = raw["env"].map(normalize_code)
    raw["label"] = raw["label"].map(normalize_code)
    if not set(raw["env"]).issubset(DOMAIN_NAMES):
        raise ValueError(f"发现未知域编码: {sorted(set(raw['env']) - set(DOMAIN_NAMES))}")
    raw["domain"] = raw["env"].map(DOMAIN_NAMES)
    raw["psych_label"] = raw["label"].map(LABEL_NAMES).fillna(raw["label"])

    parsed = []
    bad_json = []
    for i, value in enumerate(raw["features"]):
        try:
            obj = json.loads(value)
            if not isinstance(obj, dict):
                raise TypeError("features is not a JSON object")
            parsed.append(obj)
        except Exception as exc:  # keep row indices auditable
            parsed.append({})
            bad_json.append({"row": i, "path": raw.loc[i, "path"], "error": str(exc)})
    if bad_json:
        pd.DataFrame(bad_json).to_csv(output_dir / "bad_json_rows.csv", index=False)
        raise ValueError(f"有 {len(bad_json)} 行 features 无法解析")

    feature_df = pd.DataFrame(parsed).astype("string")
    key_counts = feature_df.notna().sum()
    if key_counts.min() != len(raw):
        absent = key_counts[key_counts != len(raw)].to_dict()
        raise ValueError(f"特征字段并非每行齐全: {absent}")

    resolved_paths = []
    exists = []
    hashes = []
    for text in raw["path"]:
        p = Path(text)
        if not p.is_absolute():
            p = project_root / p
        p = p.resolve()
        resolved_paths.append(str(p))
        ok = p.exists()
        exists.append(ok)
        hashes.append(sha256_file(p) if ok else "")
    raw["resolved_path"] = resolved_paths
    raw["file_exists"] = exists
    raw["sha256"] = hashes

    duplicate_mask = raw["sha256"].ne("") & raw.duplicated("sha256", keep=False)
    duplicate_rows = raw.loc[
        duplicate_mask, ["path", "env", "label", "domain", "sha256"]
    ].sort_values(["sha256", "path"])
    duplicate_rows.to_csv(output_dir / "duplicate_rows.csv", index=False)

    keep_mask = ~(raw["sha256"].ne("") & raw.duplicated("sha256", keep="first"))
    clean = raw.loc[keep_mask].reset_index(drop=True)
    clean_features = feature_df.loc[keep_mask.to_numpy()].reset_index(drop=True)

    audit_rows = []
    for (domain, label), group in clean.groupby(["domain", "label"], sort=False):
        audit_rows.append({
            "domain": domain,
            "psych_label": label,
            "n_after_dedup": len(group),
        })
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "data_audit.csv", index=False)

    audit_meta = {
        "source_csv": str(csv_path.resolve()),
        "rows_source": int(len(raw)),
        "rows_after_dedup": int(len(clean)),
        "feature_fields": int(feature_df.shape[1]),
        "bad_json_rows": len(bad_json),
        "missing_image_paths": int((~raw["file_exists"]).sum()),
        "duplicate_files_removed": int(len(raw) - len(clean)),
        "domain_counts_after_dedup": clean["domain"].value_counts().to_dict(),
        "label_counts_after_dedup": clean["label"].value_counts().to_dict(),
    }
    (output_dir / "audit.json").write_text(
        json.dumps(audit_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return clean, clean_features, audit_meta


def mapped_values(series: pd.Series, field: str) -> pd.Series:
    result = series.map(SCORE_MAPS[field]).astype(float)
    invalid = sorted(set(series.dropna().astype(str)) - set(SCORE_MAPS[field]) - {"none", "unclear"})
    if invalid:
        raise ValueError(f"{field} 出现评分表未覆盖的取值: {invalid}")
    return result


def build_abstraction_scores(meta: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    item_scores = pd.DataFrame(index=features.index)
    for field in SCORE_MAPS:
        item_scores[field] = mapped_values(features[field], field)

    components = pd.DataFrame(index=features.index)
    components["global_simplification"] = item_scores["overall_detail_level"]
    for component in ["house_simplification", "tree_simplification", "person_simplification"]:
        values = item_scores[COMPONENT_FIELDS[component]].mean(axis=1, skipna=True)
        present = features[PRESENCE_FIELD[component]].eq("yes")
        components[component] = values.where(present, np.nan)

    # Primary score: each depicted H/T/P object has equal weight; absent objects
    # are not automatically treated as abstract. Global detail is one component.
    primary = components.mean(axis=1, skipna=True)

    # Sensitivity score: every mapped item has equal weight. This intentionally
    # gives the more richly described person fields more influence.
    gated_items = item_scores.copy()
    for component, presence_field in PRESENCE_FIELD.items():
        absent = ~features[presence_field].eq("yes")
        gated_items.loc[absent, COMPONENT_FIELDS[component]] = np.nan
    field_weighted = gated_items.mean(axis=1, skipna=True)

    present_count = sum(features[f].eq("yes").astype(int) for f in PRESENCE_FIELD.values())
    completion = present_count / len(PRESENCE_FIELD)

    out = meta[[
        "path", "resolved_path", "env", "domain", "label", "psych_label", "sha256"
    ]].copy()
    out = pd.concat([out, components], axis=1)
    out["abstraction_score"] = primary.clip(0, 1)
    out["abstraction_field_weighted"] = field_weighted.clip(0, 1)
    out["abstraction_global_only"] = components["global_simplification"]
    out["task_completion"] = completion
    return out, item_scores


def holm_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def dunn_pairwise(values: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    mask = np.isfinite(values)
    values = values[mask]
    groups = groups[mask]
    labels = [x for x in DOMAIN_ORDER if x in set(groups)]
    ranks = rankdata(values, method="average")
    n = len(values)
    _, counts = np.unique(values, return_counts=True)
    tie_term = np.sum(counts ** 3 - counts)
    rank_variance = n * (n + 1) / 12 - tie_term / (12 * (n - 1))
    mean_ranks = {label: ranks[groups == label].mean() for label in labels}
    sizes = {label: int(np.sum(groups == label)) for label in labels}

    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            denom = math.sqrt(rank_variance * (1 / sizes[a] + 1 / sizes[b]))
            z = (mean_ranks[a] - mean_ranks[b]) / denom
            rows.append({
                "group_1": a,
                "group_2": b,
                "z": z,
                "p_raw": 2 * norm.sf(abs(z)),
            })
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = p_holm
    return pd.DataFrame(rows)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    u = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic
    return float(2 * u / (len(x) * len(y)) - 1)


def bootstrap_delta_ci(x: np.ndarray, y: np.ndarray, n_boot: int, rng: np.random.Generator):
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        xb = x[rng.integers(0, len(x), len(x))]
        yb = y[rng.integers(0, len(y), len(y))]
        values[i] = cliffs_delta(xb, yb)
    return np.quantile(values, [0.025, 0.975])


def run_abstraction_statistics(
    scores: pd.DataFrame, output_dir: Path, n_boot: int, seed: int
):
    rng = np.random.default_rng(seed)
    summary_rows = []
    kw_rows = []
    dunn_tables = []
    score_columns = [
        "abstraction_score", "abstraction_field_weighted", "abstraction_global_only"
    ]

    analyses = [("all", scores)]
    analyses.extend((f"label_{label}", g) for label, g in scores.groupby("label"))
    for analysis, frame in analyses:
        for score_col in score_columns:
            arrays = []
            for domain in DOMAIN_ORDER:
                x = frame.loc[frame["domain"].eq(domain), score_col].dropna().to_numpy()
                arrays.append(x)
                summary_rows.append({
                    "analysis": analysis,
                    "score": score_col,
                    "domain": domain,
                    "n": len(x),
                    "mean": np.mean(x),
                    "std": np.std(x, ddof=1),
                    "median": np.median(x),
                    "q1": np.quantile(x, 0.25),
                    "q3": np.quantile(x, 0.75),
                })
            h, p = kruskal(*arrays)
            n = sum(map(len, arrays))
            k = len(arrays)
            epsilon_sq = max(0.0, (h - k + 1) / (n - k))
            kw_rows.append({
                "analysis": analysis,
                "score": score_col,
                "H": h,
                "df": k - 1,
                "p": p,
                "epsilon_squared": epsilon_sq,
                "n": n,
            })

            dunn = dunn_pairwise(
                frame[score_col].to_numpy(float), frame["domain"].to_numpy(str)
            )
            dunn.insert(0, "score", score_col)
            dunn.insert(0, "analysis", analysis)
            deltas = []
            lows = []
            highs = []
            for row in dunn.itertuples(index=False):
                x = frame.loc[frame["domain"].eq(row.group_1), score_col].dropna().to_numpy()
                y = frame.loc[frame["domain"].eq(row.group_2), score_col].dropna().to_numpy()
                delta = cliffs_delta(x, y)
                lo, hi = bootstrap_delta_ci(x, y, n_boot, rng)
                deltas.append(delta)
                lows.append(lo)
                highs.append(hi)
            dunn["cliffs_delta_group1_minus_group2"] = deltas
            dunn["cliffs_delta_ci_low"] = lows
            dunn["cliffs_delta_ci_high"] = highs
            dunn_tables.append(dunn)

    summary = pd.DataFrame(summary_rows)
    kw = pd.DataFrame(kw_rows)
    dunn = pd.concat(dunn_tables, ignore_index=True)
    summary.to_csv(output_dir / "abstraction_group_summary.csv", index=False)
    kw.to_csv(output_dir / "kruskal_wallis.csv", index=False)
    dunn.to_csv(output_dir / "dunn_holm_cliffs_delta.csv", index=False)
    return summary, kw, dunn


def make_onehot_encoder(dense: bool = False):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:  # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


def categorical_pipeline(c: float = 1.0) -> Pipeline:
    return Pipeline([
        ("onehot", make_onehot_encoder(dense=False)),
        ("classifier", LogisticRegression(
            C=c, max_iter=4000, solver="lbfgs", class_weight="balanced",
        )),
    ])


def numeric_pipeline(c: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c, max_iter=4000, solver="lbfgs", class_weight="balanced",
        )),
    ])


def repeated_nested_cv(
    x,
    y: np.ndarray,
    psych_label: np.ndarray,
    kind: str,
    repeats: int,
    seed: int,
):
    classes = np.array(sorted(np.unique(y)))
    c_grid = [0.01, 0.1, 1.0, 10.0]
    rows = []
    best_cs = []
    cms = []
    for repeat in range(repeats):
        repeat_seed = seed + 1009 * repeat
        joint = np.char.add(np.char.add(y.astype(str), "|"), psych_label.astype(str))
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
        pred = np.empty(len(y), dtype=object)
        prob = np.zeros((len(y), len(classes)), dtype=float)
        for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), joint)):
            y_train = y[train_idx]
            label_train = psych_label[train_idx]
            inner_joint = np.char.add(np.char.add(y_train.astype(str), "|"), label_train.astype(str))
            inner = StratifiedKFold(
                n_splits=4, shuffle=True, random_state=repeat_seed + fold + 1
            )
            inner_splits = list(inner.split(np.zeros(len(train_idx)), inner_joint))
            pipeline = categorical_pipeline() if kind == "categorical" else numeric_pipeline()
            grid = GridSearchCV(
                pipeline,
                {"classifier__C": c_grid},
                scoring="balanced_accuracy",
                cv=inner_splits,
                n_jobs=1,
                refit=True,
            )
            if isinstance(x, pd.DataFrame):
                x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            else:
                x_train, x_test = x[train_idx], x[test_idx]
            grid.fit(x_train, y_train)
            pred[test_idx] = grid.predict(x_test)
            fold_prob = grid.predict_proba(x_test)
            fold_classes = grid.best_estimator_.named_steps["classifier"].classes_
            for j, cls in enumerate(fold_classes):
                prob[test_idx, np.where(classes == cls)[0][0]] = fold_prob[:, j]
            best_cs.append(float(grid.best_params_["classifier__C"]))

        ba = balanced_accuracy_score(y, pred)
        macro_f1 = f1_score(y, pred, average="macro")
        auc = roc_auc_score(y, prob, labels=classes, multi_class="ovr", average="macro")
        rows.append({
            "repeat": repeat,
            "balanced_accuracy": ba,
            "macro_f1": macro_f1,
            "macro_auc_ovr": auc,
        })
        cms.append(confusion_matrix(y, pred, labels=classes, normalize="true"))
    return pd.DataFrame(rows), best_cs, np.mean(cms, axis=0), classes


def mode_c(values: list[float]) -> float:
    counts = Counter(values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def fixed_cv_score(x, y, psych_label, pipeline, seed: int) -> float:
    joint = np.char.add(np.char.add(y.astype(str), "|"), psych_label.astype(str))
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), joint):
        model = clone(pipeline)
        if isinstance(x, pd.DataFrame):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        else:
            x_train, x_test = x[train_idx], x[test_idx]
        model.fit(x_train, y[train_idx])
        pred[test_idx] = model.predict(x_test)
    return float(balanced_accuracy_score(y, pred))


def within_label_permutation_test(
    x,
    y: np.ndarray,
    psych_label: np.ndarray,
    pipeline: Pipeline,
    n_permutations: int,
    seed: int,
):
    observed = fixed_cv_score(x, y, psych_label, pipeline, seed)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for b in range(n_permutations):
        y_perm = y.copy()
        for label in np.unique(psych_label):
            idx = np.flatnonzero(psych_label == label)
            y_perm[idx] = rng.permutation(y_perm[idx])
        null[b] = fixed_cv_score(
            x, y_perm, psych_label, pipeline, seed + 17 + b
        )
    p = (1 + np.sum(null >= observed)) / (n_permutations + 1)
    return {
        "observed_fixed_cv_balanced_accuracy": observed,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null, ddof=1)),
        "null_q95": float(np.quantile(null, 0.95)),
        "permutations": n_permutations,
        "p": float(p),
    }, null


def run_domain_classifiers(
    meta: pd.DataFrame,
    features: pd.DataFrame,
    scores: pd.DataFrame,
    output_dir: Path,
    repeats: int,
    permutations: int,
    seed: int,
):
    y = meta["env"].to_numpy(str)
    labels = meta["label"].to_numpy(str)
    non_abstraction = features.drop(columns=ABSTRACTION_SOURCE_FIELDS)
    setups = [
        ("all_qwen_features", features, "categorical"),
        ("non_abstraction_qwen_features", non_abstraction, "categorical"),
        ("abstraction_score_only", scores[["abstraction_score"]].to_numpy(), "numeric"),
    ]
    result_tables = []
    details = {}
    for name, x, kind in setups:
        table, best_cs, cm, classes = repeated_nested_cv(
            x, y, labels, kind, repeats, seed
        )
        table.insert(0, "analysis", "all")
        table.insert(0, "feature_set", name)
        result_tables.append(table)
        details[name] = {
            "x": x, "kind": kind, "best_cs": best_cs,
            "cm": cm, "classes": classes,
        }

    # Conditional checks: full Qwen features within each psychological label.
    for label in sorted(np.unique(labels)):
        mask = labels == label
        table, best_cs, cm, classes = repeated_nested_cv(
            features.loc[mask].reset_index(drop=True),
            y[mask],
            labels[mask],
            "categorical",
            repeats,
            seed + int(label) + 100,
        )
        table.insert(0, "analysis", f"label_{label}")
        table.insert(0, "feature_set", "all_qwen_features")
        result_tables.append(table)
        details[f"all_qwen_features_label_{label}"] = {
            "best_cs": best_cs, "cm": cm, "classes": classes,
        }

    results = pd.concat(result_tables, ignore_index=True)
    results.to_csv(output_dir / "domain_classifier_repeated_cv.csv", index=False)
    summary = results.groupby(["feature_set", "analysis"], sort=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        macro_auc_mean=("macro_auc_ovr", "mean"),
        macro_auc_sd=("macro_auc_ovr", "std"),
        repeats=("repeat", "count"),
    ).reset_index()
    summary.to_csv(output_dir / "domain_classifier_summary.csv", index=False)

    permutation_rows = []
    nulls = {}
    for name in ["all_qwen_features", "abstraction_score_only"]:
        detail = details[name]
        c = mode_c(detail["best_cs"])
        pipe = categorical_pipeline(c) if detail["kind"] == "categorical" else numeric_pipeline(c)
        stat, null = within_label_permutation_test(
            detail["x"], y, labels, pipe, permutations, seed + 700
        )
        stat.update({"feature_set": name, "fixed_C": c})
        permutation_rows.append(stat)
        nulls[name] = null
    permutation_table = pd.DataFrame(permutation_rows)
    permutation_table.to_csv(output_dir / "domain_classifier_permutation.csv", index=False)

    return results, summary, permutation_table, details, nulls


def set_plot_style():
    matplotlib.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.titleweight": "semibold",
    })
    sns.set_theme(style="whitegrid", rc={
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.7,
    })


def format_p(p: float) -> str:
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def add_sig_bracket(ax, x1, x2, y, text, height=0.012):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y],
            color="#444444", linewidth=0.8, clip_on=False)
    ax.text((x1 + x2) / 2, y + height + 0.002, text,
            ha="center", va="bottom", fontsize=8, color="#333333")


def plot_violin(scores, kw, dunn, output_dir: Path):
    frame = scores.copy()
    counts = frame["domain"].value_counts()
    xlabels = [f"{d}\n(n={counts[d]})" for d in DOMAIN_ORDER]
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    sns.violinplot(
        data=frame, x="domain", y="abstraction_score", hue="domain",
        order=DOMAIN_ORDER, hue_order=DOMAIN_ORDER, palette=DOMAIN_COLORS,
        legend=False, inner=None, cut=0, density_norm="width",
        linewidth=1.0, saturation=0.82, ax=ax,
    )
    sns.boxplot(
        data=frame, x="domain", y="abstraction_score", order=DOMAIN_ORDER,
        width=0.16, showfliers=False, color="white",
        boxprops={"facecolor": "white", "edgecolor": "#333333", "alpha": 0.88},
        medianprops={"color": "#111111", "linewidth": 1.6},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0}, ax=ax,
    )
    sns.stripplot(
        data=frame, x="domain", y="abstraction_score", order=DOMAIN_ORDER,
        color="#222222", size=1.7, jitter=0.22, alpha=0.18, ax=ax,
    )
    row = kw[(kw["analysis"] == "all") & (kw["score"] == "abstraction_score")].iloc[0]
    primary_dunn = dunn[
        (dunn["analysis"] == "all") & (dunn["score"] == "abstraction_score")
    ]
    pair_index = {d: i for i, d in enumerate(DOMAIN_ORDER)}
    for level, pair in enumerate(primary_dunn.itertuples(index=False)):
        p_text = f"Holm p={format_p(pair.p_holm)}"
        add_sig_bracket(
            ax, pair_index[pair.group_1], pair_index[pair.group_2],
            1.015 + level * 0.043, p_text,
        )

    ax.set_xticks(range(len(DOMAIN_ORDER)), labels=xlabels)
    ax.set_xlabel("Population domain")
    ax.set_ylabel("Qwen-derived abstraction proxy (0–1; higher = more simplified)")
    ax.set_ylim(0, 1.18)
    ax.set_title("HTP abstraction proxy distribution by population domain", pad=42, fontsize=14)
    fig.text(
        0.5, 0.935,
        "Equal-object composite; absent H/T/P objects excluded from object subscales",
        ha="center", va="center", fontsize=9.5, color="#555555",
    )
    fig.text(
        0.5, 0.907,
        f"Kruskal–Wallis H(2)={row.H:.2f}, p={format_p(row.p)}, ε²={row.epsilon_squared:.3f}",
        ha="center", va="center", fontsize=9.2, color="#444444",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(output_dir / f"abstraction_violin.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 6.3))
    sns.violinplot(
        data=frame, x="domain", y="abstraction_score", hue="psych_label",
        order=DOMAIN_ORDER, hue_order=["类别 00", "类别 01"],
        palette=LABEL_COLORS, split=True, inner="quart", cut=0,
        density_norm="width", linewidth=0.9, ax=ax,
    )
    ax.set_xlabel("Population domain")
    ax.set_ylabel("Qwen-derived abstraction proxy")
    ax.set_ylim(0, 1.03)
    ax.set_title("HTP abstraction proxy by domain and psychological class", fontsize=14)
    ax.legend(
        title="Psychological class", frameon=False,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "abstraction_violin_by_label.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_embedding(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    output_dir: Path,
    seed: int,
    neighbors: int,
    skip_umap: bool,
):
    encoder = make_onehot_encoder(dense=True)
    x = encoder.fit_transform(features)
    x = StandardScaler().fit_transform(x)
    n_pca = min(50, x.shape[1], x.shape[0] - 1)
    pca50 = PCA(n_components=n_pca, random_state=seed).fit_transform(x)
    pca2 = pca50[:, :2]
    embedding = None
    if not skip_umap:
        # ``umap.__init__`` eagerly imports ParametricUMAP when TensorFlow is
        # present.  This analysis only needs ordinary UMAP, so block that
        # optional TensorFlow import before lazily loading umap-learn.
        sys.modules.setdefault("tensorflow", None)
        import umap
        embedding = umap.UMAP(
            n_neighbors=neighbors, min_dist=0.1, n_components=2,
            metric="cosine", random_state=seed,
        ).fit_transform(pca50)

    panels = [(pca2, "PCA of Qwen structured features")]
    if embedding is not None:
        panels.append((embedding, "UMAP after PCA-50"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 5.8), squeeze=False)
    axes = axes.ravel()
    for ax, (coords, title) in zip(axes, panels):
        for domain in DOMAIN_ORDER:
            mask = meta["domain"].eq(domain).to_numpy()
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=15, alpha=0.62,
                c=DOMAIN_COLORS[domain], label=domain, edgecolors="none",
            )
        ax.set_title(title)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.grid(True, color="#ECECEC", linewidth=0.6)
    axes[-1].legend(
        frameon=False, title="Domain", loc="upper left",
        bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
    )
    fig.suptitle("HTP domain structure in the latest Qwen feature set", fontsize=15, y=1.02)
    fig.tight_layout()
    filename = "qwen_feature_pca.png" if skip_umap else "qwen_feature_pca_umap.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_and_permutation(details, permutation_table, nulls, output_dir: Path):
    detail = details["all_qwen_features"]
    cm = detail["cm"]
    class_names = [DOMAIN_NAMES[c] for c in detail["classes"]]
    perm_row = permutation_table[
        permutation_table["feature_set"] == "all_qwen_features"
    ].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    sns.heatmap(
        cm, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1,
        xticklabels=class_names, yticklabels=class_names, cbar_kws={"label": "Row proportion"},
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted domain")
    axes[0].set_ylabel("True domain")
    axes[0].set_title("Repeated nested-CV confusion matrix")

    null = nulls["all_qwen_features"]
    axes[1].hist(null, bins=24, color="#A7B8C8", edgecolor="white", linewidth=0.7)
    axes[1].axvline(
        perm_row.observed_fixed_cv_balanced_accuracy,
        color="#C94C4C", linewidth=2.0,
        label=f"Observed = {perm_row.observed_fixed_cv_balanced_accuracy:.3f}",
    )
    axes[1].axvline(1 / 3, color="#333333", linestyle="--", linewidth=1.2, label="Chance = 0.333")
    axes[1].set_xlabel("Balanced accuracy under within-label permutation")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Permutation null ({int(perm_row.permutations)} permutations; p={format_p(perm_row.p)})")
    axes[1].legend(frameon=False)
    sns.despine(ax=axes[1])
    fig.tight_layout()
    fig.savefig(output_dir / "domain_classifier_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_component_correlations(scores: pd.DataFrame, output_dir: Path):
    cols = ["global_simplification", "house_simplification", "tree_simplification", "person_simplification"]
    corr = scores[cols].corr(method="spearman")
    corr.to_csv(output_dir / "abstraction_component_spearman.csv")


def write_scoring_rules(output_dir: Path):
    rules = {
        "construct": "Qwen-derived representational abstraction/simplification proxy",
        "direction": "0=more detailed/concrete, 1=more simplified/abstract",
        "primary_aggregation": (
            "Mean of global, house, tree and person components. Object components are included "
            "only when that object is present, so absence is not scored as abstraction."
        ),
        "presence_fields": PRESENCE_FIELD,
        "component_fields": COMPONENT_FIELDS,
        "score_maps": SCORE_MAPS,
        "validation_status": "Not validated against blinded human abstraction ratings",
    }
    (output_dir / "abstraction_scoring_rules.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_summary(
    output_dir: Path,
    audit: dict,
    group_summary: pd.DataFrame,
    kw: pd.DataFrame,
    dunn: pd.DataFrame,
    classifier_summary: pd.DataFrame,
    permutation: pd.DataFrame,
    elapsed: float,
):
    primary_summary = group_summary[
        (group_summary["analysis"] == "all") &
        (group_summary["score"] == "abstraction_score")
    ].set_index("domain")
    primary_kw = kw[
        (kw["analysis"] == "all") & (kw["score"] == "abstraction_score")
    ].iloc[0]
    primary_dunn = dunn[
        (dunn["analysis"] == "all") & (dunn["score"] == "abstraction_score")
    ]

    lines = [
        "# HTP population-domain shift and abstraction analysis",
        "",
        f"Source: `{audit['source_csv']}`",
        f"Rows: {audit['rows_source']} source, {audit['rows_after_dedup']} after SHA256 deduplication.",
        f"Qwen feature fields: {audit['feature_fields']}; missing image paths: {audit['missing_image_paths']}.",
        "",
        "## Domain classifier",
        "",
        "| Feature set | Analysis | Balanced accuracy | Macro-F1 | Macro-AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for row in classifier_summary.itertuples(index=False):
        lines.append(
            f"| {row.feature_set} | {row.analysis} | "
            f"{row.balanced_accuracy_mean:.3f} ± {row.balanced_accuracy_sd:.3f} | "
            f"{row.macro_f1_mean:.3f} ± {row.macro_f1_sd:.3f} | "
            f"{row.macro_auc_mean:.3f} ± {row.macro_auc_sd:.3f} |"
        )
    lines.extend(["", "Within-psychological-label permutation tests:", ""])
    for row in permutation.itertuples(index=False):
        lines.append(
            f"- `{row.feature_set}`: BA={row.observed_fixed_cv_balanced_accuracy:.3f}, "
            f"null mean={row.null_mean:.3f}, p={row.p:.6g} ({int(row.permutations)} permutations)."
        )

    lines.extend([
        "",
        "## Qwen-derived abstraction proxy",
        "",
        "The primary score is an equal-object composite in [0,1]. Higher values indicate more "
        "representational simplification. Missing H/T/P objects are excluded from their object "
        "subscale and are recorded separately as task completion.",
        "",
        "| Domain | n | Median [IQR] | Mean ± SD |",
        "|---|---:|---:|---:|",
    ])
    for domain in DOMAIN_ORDER:
        row = primary_summary.loc[domain]
        lines.append(
            f"| {domain} | {int(row['n'])} | {row['median']:.3f} "
            f"[{row['q1']:.3f}, {row['q3']:.3f}] | {row['mean']:.3f} ± {row['std']:.3f} |"
        )
    lines.extend([
        "",
        f"Kruskal–Wallis: H(2)={primary_kw.H:.3f}, p={primary_kw.p:.6g}, "
        f"epsilon-squared={primary_kw.epsilon_squared:.3f}.",
        "",
        "Pairwise Dunn tests use Holm correction; Cliff's delta is group 1 minus group 2:",
        "",
        "| Group 1 | Group 2 | Holm p | Cliff's delta [95% CI] |",
        "|---|---|---:|---:|",
    ])
    for row in primary_dunn.itertuples(index=False):
        lines.append(
            f"| {row.group_1} | {row.group_2} | {row.p_holm:.6g} | "
            f"{row.cliffs_delta_group1_minus_group2:.3f} "
            f"[{row.cliffs_delta_ci_low:.3f}, {row.cliffs_delta_ci_high:.3f}] |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "- The classifier tests domain information in the latest Qwen structured representation, "
        "not in a frozen ResNet representation.",
        "- A significant abstraction result establishes association with domain for this proxy; it "
        "does not prove that abstraction causes the full domain shift.",
        "- The proxy has not yet been validated against blinded human ratings, so it should be "
        "reported as a Qwen-derived abstraction proxy rather than a psychological ground-truth score.",
        "- Scanner/template/text differences can influence Qwen features. Pixel-level cleaned-image "
        "analysis is still required to separate acquisition shift from drawing-content shift.",
        "",
        f"Runtime: {elapsed:.1f} seconds.",
    ])
    (output_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    start = time.time()
    project_root = Path(args.project_root).resolve()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading and auditing {csv_path}", flush=True)
    meta, features, audit = load_and_audit(csv_path, project_root, output_dir)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)

    print("[2/6] Building the preregistered Qwen-derived abstraction proxy", flush=True)
    scores, _ = build_abstraction_scores(meta, features)
    scores.to_csv(output_dir / "abstraction_scores.csv", index=False)
    write_scoring_rules(output_dir)
    write_component_correlations(scores, output_dir)

    print("[3/6] Running Kruskal-Wallis, Dunn-Holm and bootstrap effect sizes", flush=True)
    group_summary, kw, dunn = run_abstraction_statistics(
        scores, output_dir, args.bootstrap, args.seed
    )

    print("[4/6] Running repeated nested-CV domain classifiers", flush=True)
    results, classifier_summary, permutation, details, nulls = run_domain_classifiers(
        meta, features, scores, output_dir,
        args.cv_repeats, args.permutations, args.seed,
    )

    print("[5/6] Rendering violin, PCA/UMAP and classifier diagnostic figures", flush=True)
    set_plot_style()
    plot_violin(scores, kw, dunn, output_dir)
    plot_feature_embedding(
        features, meta, output_dir, args.seed, args.umap_neighbors, args.skip_umap
    )
    plot_confusion_and_permutation(details, permutation, nulls, output_dir)

    runtime = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "umap_learn": version("umap-learn") if not args.skip_umap else "not imported",
        "arguments": vars(args),
    }
    (output_dir / "runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    elapsed = time.time() - start
    write_summary(
        output_dir, audit, group_summary, kw, dunn,
        classifier_summary, permutation, elapsed,
    )
    print(f"[6/6] Done in {elapsed:.1f}s. Outputs: {output_dir}", flush=True)
    print(classifier_summary.to_string(index=False), flush=True)
    print(kw[(kw.analysis == "all") & (kw.score == "abstraction_score")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
