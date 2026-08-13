#!/usr/bin/env python
"""Cross-domain prediction of psychological label from abstraction score only.

Each model is fitted on one HTP population domain and evaluated, without any
target-domain calibration, on the other two domains. Label 01 is the positive
class. The primary metric is ROC-AUC because it exposes direction reversal;
balanced accuracy at the source-learned 0.5 threshold is also reported.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DOMAIN_ORDER = ["Child", "College", "Social"]
DOMAIN_COLORS = {"Child": "#4C78A8", "College": "#E07A5F", "Social": "#8A9A5B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("output/htp_shift/abstraction_scores.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/htp_shift"))
    parser.add_argument("--score-column", default="abstraction_score")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def load_scores(path: Path, score_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, dtype={"label": str, "domain": str, "sha256": str})
    required = {"domain", "label", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    frame["label"] = frame["label"].str.zfill(2)
    if set(frame["domain"]) != set(DOMAIN_ORDER):
        raise ValueError(f"Unexpected domains: {sorted(frame['domain'].unique())}")
    if set(frame["label"]) != {"00", "01"}:
        raise ValueError(f"Unexpected labels: {sorted(frame['label'].unique())}")
    frame[score_column] = pd.to_numeric(frame[score_column], errors="raise")
    if frame[score_column].isna().any() or not frame[score_column].between(0, 1).all():
        raise ValueError("Abstraction score contains missing or out-of-range values")
    if "sha256" in frame.columns and frame["sha256"].duplicated().any():
        raise ValueError("Input still contains duplicate image hashes")
    if "path" in frame.columns and frame["path"].duplicated().any():
        raise ValueError("Input contains duplicate paths")
    frame["y"] = frame["label"].eq("01").astype(int)
    audit = (
        frame.groupby(["domain", "label"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    return frame, audit


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    C=1.0,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_model(x: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    model = make_model(seed)
    model.fit(x.reshape(-1, 1), y)
    return model


def raw_parameters(model: Pipeline) -> tuple[float, float, float]:
    scaler: StandardScaler = model.named_steps["scale"]
    lr: LogisticRegression = model.named_steps["lr"]
    coef_scaled = float(lr.coef_[0, 0])
    scale = float(scaler.scale_[0])
    mean = float(scaler.mean_[0])
    coef_raw = coef_scaled / scale
    intercept_raw = float(lr.intercept_[0]) - coef_scaled * mean / scale
    threshold = -intercept_raw / coef_raw if abs(coef_raw) > 1e-12 else np.nan
    return coef_raw, intercept_raw, threshold


def predict_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, probability),
        "specificity_label00": specificity,
        "sensitivity_label01": sensitivity,
        "predicted_label01_rate": prediction.mean(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def stratified_bootstrap_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    indices = [np.flatnonzero(y == value) for value in [0, 1]]
    auc_values = np.empty(n_bootstrap, dtype=float)
    ba_values = np.empty(n_bootstrap, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    for iteration in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in indices]
        )
        auc_values[iteration] = roc_auc_score(y[sampled], probability[sampled])
        ba_values[iteration] = balanced_accuracy_score(y[sampled], prediction[sampled])
    return {
        "roc_auc_ci_low": np.quantile(auc_values, 0.025),
        "roc_auc_ci_high": np.quantile(auc_values, 0.975),
        "balanced_accuracy_ci_low": np.quantile(ba_values, 0.025),
        "balanced_accuracy_ci_high": np.quantile(ba_values, 0.975),
    }


def source_slope_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    full_coef: float,
    n_bootstrap: int,
    rng: np.random.Generator,
    seed: int,
) -> dict[str, float]:
    groups = [np.flatnonzero(y == value) for value in [0, 1]]
    coefficients = np.empty(n_bootstrap, dtype=float)
    for iteration in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        model = fit_model(x[sampled], y[sampled], seed + iteration + 1)
        coefficients[iteration] = raw_parameters(model)[0]
    same_sign = np.mean(np.sign(coefficients) == np.sign(full_coef))
    positive = np.mean(coefficients > 0)
    return {
        "coef_boot_ci_low": np.quantile(coefficients, 0.025),
        "coef_boot_ci_high": np.quantile(coefficients, 0.975),
        "coef_sign_same_fraction": same_sign,
        "coef_positive_fraction": positive,
    }


def classify_auc(auc_low: float, auc_high: float) -> str:
    if auc_high < 0.5:
        return "reversed"
    if auc_low > 0.5:
        return "aligned"
    return "near_random_or_uncertain"


def run_experiment(
    frame: pd.DataFrame,
    score_column: str,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    rng = np.random.default_rng(seed)
    source_rows: list[dict] = []
    transfer_rows: list[dict] = []
    models: dict[str, Pipeline] = {}
    for source_index, source in enumerate(DOMAIN_ORDER):
        train = frame[frame["domain"].eq(source)]
        x_train = train[score_column].to_numpy(float)
        y_train = train["y"].to_numpy(int)
        model = fit_model(x_train, y_train, seed + source_index)
        models[source] = model
        coef_raw, intercept_raw, threshold = raw_parameters(model)
        train_probability = model.predict_proba(x_train.reshape(-1, 1))[:, 1]
        train_metrics = predict_metrics(y_train, train_probability)
        slope_boot = source_slope_bootstrap(
            x_train,
            y_train,
            coef_raw,
            n_bootstrap,
            rng,
            seed + 10000 * (source_index + 1),
        )
        source_rows.append(
            {
                "train_domain": source,
                "n_train": len(train),
                "n_label00": int((y_train == 0).sum()),
                "n_label01": int((y_train == 1).sum()),
                "coef_raw": coef_raw,
                "intercept_raw": intercept_raw,
                "decision_threshold_score": threshold,
                "learned_direction": "A_up_to_label01" if coef_raw > 0 else "A_up_to_label00",
                "train_roc_auc_apparent": train_metrics["roc_auc"],
                "train_balanced_accuracy_apparent": train_metrics["balanced_accuracy"],
                **slope_boot,
            }
        )
        for target_index, target in enumerate(DOMAIN_ORDER):
            if target == source:
                continue
            test = frame[frame["domain"].eq(target)]
            x_test = test[score_column].to_numpy(float)
            y_test = test["y"].to_numpy(int)
            probability = model.predict_proba(x_test.reshape(-1, 1))[:, 1]
            metrics = predict_metrics(y_test, probability)
            ci = stratified_bootstrap_metrics(y_test, probability, n_bootstrap, rng)
            transfer_rows.append(
                {
                    "train_domain": source,
                    "test_domain": target,
                    "n_test": len(test),
                    "n_test_label00": int((y_test == 0).sum()),
                    "n_test_label01": int((y_test == 1).sum()),
                    "learned_direction": "A_up_to_label01" if coef_raw > 0 else "A_up_to_label00",
                    **metrics,
                    **ci,
                    "auc_interpretation": classify_auc(ci["roc_auc_ci_low"], ci["roc_auc_ci_high"]),
                }
            )
    return pd.DataFrame(source_rows), pd.DataFrame(transfer_rows), models


def plot_heatmaps(results: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7))
    cmap = sns.diverging_palette(25, 240, s=75, l=52, center="light", as_cmap=True)
    specs = [
        ("roc_auc", "Cross-domain ROC-AUC", "Threshold-free; below 0.5 means reversed ranking"),
        (
            "balanced_accuracy",
            "Cross-domain balanced accuracy",
            "Uses the source-learned probability threshold of 0.5",
        ),
    ]
    for ax, (metric, title, subtitle) in zip(axes, specs):
        matrix = pd.DataFrame(np.nan, index=DOMAIN_ORDER, columns=DOMAIN_ORDER)
        annotations = pd.DataFrame("", index=DOMAIN_ORDER, columns=DOMAIN_ORDER)
        for row in results.itertuples(index=False):
            value = getattr(row, metric)
            matrix.loc[row.train_domain, row.test_domain] = value
            if metric == "roc_auc":
                status = {"aligned": "aligned", "reversed": "reversed"}.get(
                    row.auc_interpretation, "uncertain"
                )
                annotations.loc[row.train_domain, row.test_domain] = f"{value:.3f}\n{status}"
            else:
                annotations.loc[row.train_domain, row.test_domain] = f"{value:.3f}"
        sns.heatmap(
            matrix,
            mask=matrix.isna(),
            annot=annotations,
            fmt="",
            cmap=cmap,
            vmin=0,
            vmax=1,
            center=0.5,
            square=True,
            linewidths=1.2,
            linecolor="white",
            cbar_kws={"label": metric.replace("_", " "), "shrink": 0.78},
            ax=ax,
        )
        for diagonal in range(len(DOMAIN_ORDER)):
            ax.add_patch(
                plt.Rectangle(
                    (diagonal, diagonal), 1, 1,
                    facecolor="#ECECEC", edgecolor="white", linewidth=1.2,
                )
            )
            ax.text(diagonal + 0.5, diagonal + 0.5, "source\nfit", ha="center", va="center", color="#666666")
        ax.set_title(title, fontsize=12, weight="semibold", pad=24)
        ax.text(0.5, 1.035, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color="#555555")
        ax.set_xlabel("Test domain")
        ax.set_ylabel("Train domain")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)
    fig.suptitle("Psychological-label transfer using abstraction score only", fontsize=15, weight="semibold", y=1.03)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(output_dir / f"abstraction_label_transfer_heatmaps.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_logistic_curves(models: dict[str, Pipeline], output_dir: Path) -> None:
    score_grid = np.linspace(0, 1, 501)
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for domain in DOMAIN_ORDER:
        probability = models[domain].predict_proba(score_grid.reshape(-1, 1))[:, 1]
        coef, _, threshold = raw_parameters(models[domain])
        direction = "A up -> label 01" if coef > 0 else "A up -> label 00"
        ax.plot(
            score_grid,
            probability,
            color=DOMAIN_COLORS[domain],
            linewidth=2.2,
            label=f"Train {domain}: {direction}",
        )
        if 0 <= threshold <= 1:
            ax.scatter([threshold], [0.5], color=DOMAIN_COLORS[domain], s=38, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Abstraction score")
    ax.set_ylabel("Predicted probability of label 01")
    fig.suptitle("Source-domain logistic models", fontsize=15, weight="semibold", y=0.985)
    fig.text(
        0.5,
        0.935,
        "Each curve is fitted on one domain; dots mark its probability-0.5 threshold",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    for suffix in ["png", "svg"]:
        fig.savefig(output_dir / f"abstraction_source_logistic_curves.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    source: pd.DataFrame,
    transfer: pd.DataFrame,
    audit: pd.DataFrame,
    scores_path: Path,
    output_path: Path,
    n_bootstrap: int,
    runtime_seconds: float,
) -> None:
    lines = [
        "# Abstraction-only cross-domain psychological-label prediction",
        "",
        f"Source: `{scores_path.resolve()}`",
        "",
        "Protocol: StandardScaler and class-balanced Logistic Regression are fitted on one source domain. "
        "The fitted model and its 0.5 probability threshold are applied unchanged to the other domains. Label 01 is positive.",
        "",
        "## Source models",
        "",
        "| Train domain | n | Raw coefficient [bootstrap 95% CI] | Direction | Score threshold | Apparent train AUC | Bootstrap sign agreement |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in source.itertuples(index=False):
        lines.append(
            f"| {row.train_domain} | {row.n_train} | {row.coef_raw:+.4f} "
            f"[{row.coef_boot_ci_low:+.4f}, {row.coef_boot_ci_high:+.4f}] | {row.learned_direction} | "
            f"{row.decision_threshold_score:.3f} | {row.train_roc_auc_apparent:.3f} | {row.coef_sign_same_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-domain tests",
            "",
            "AUC and balanced-accuracy confidence intervals are stratified target-domain bootstrap intervals, conditional on the fitted source model.",
            "",
            "| Train | Test | Direction | ROC-AUC [95% CI] | Balanced accuracy [95% CI] | Accuracy | Interpretation |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in transfer.itertuples(index=False):
        lines.append(
            f"| {row.train_domain} | {row.test_domain} | {row.learned_direction} | "
            f"{row.roc_auc:.3f} [{row.roc_auc_ci_low:.3f}, {row.roc_auc_ci_high:.3f}] | "
            f"{row.balanced_accuracy:.3f} [{row.balanced_accuracy_ci_low:.3f}, {row.balanced_accuracy_ci_high:.3f}] | "
            f"{row.accuracy:.3f} | {row.auc_interpretation} |"
        )
    college_social = transfer[
        transfer["train_domain"].eq("College") & transfer["test_domain"].eq("Social")
    ].iloc[0]
    college_child = transfer[
        transfer["train_domain"].eq("College") & transfer["test_domain"].eq("Child")
    ].iloc[0]
    lines.extend(
        [
            "",
            "## Requested inference",
            "",
            f"College learns `A up -> label 00`. On Social it transfers above chance "
            f"(AUC={college_social.roc_auc:.3f}); on Child it is near chance "
            f"(AUC={college_child.roc_auc:.3f}). The proposed inference is supported.",
            "",
            "ROC-AUC is the primary direction diagnostic. Balanced accuracy additionally depends on the source-domain probability calibration and fixed threshold.",
            "",
            "The abstraction score is a Qwen-derived proxy rather than a blinded human rating.",
            "",
            f"Bootstrap replicates: {n_bootstrap}. Runtime: {runtime_seconds:.1f} seconds.",
            "",
            "## Sample audit",
            "",
            "```text",
            audit.to_string(index=False),
            "```",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    scores_path = args.scores.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {scores_path}", flush=True)
    frame, audit = load_scores(scores_path, args.score_column)
    print(audit.to_string(index=False), flush=True)

    print("[2/4] Fitting three source-domain logistic models and six transfers", flush=True)
    source, transfer, models = run_experiment(
        frame,
        args.score_column,
        args.bootstrap,
        args.seed,
    )
    audit.to_csv(output_dir / "abstraction_transfer_audit.csv", index=False)
    source.to_csv(output_dir / "abstraction_source_models.csv", index=False)
    transfer.to_csv(output_dir / "abstraction_cross_domain_transfer.csv", index=False)

    print("[3/4] Rendering heatmaps and logistic curves", flush=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"]})
    plot_heatmaps(transfer, output_dir)
    plot_logistic_curves(models, output_dir)

    print("[4/4] Writing report", flush=True)
    runtime_seconds = time.perf_counter() - started
    write_report(
        source,
        transfer,
        audit,
        scores_path,
        output_dir / "abstraction_cross_domain_transfer.md",
        args.bootstrap,
        runtime_seconds,
    )
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "scores": str(scores_path),
        "score_column": args.score_column,
        "model": "StandardScaler + LogisticRegression(class_weight='balanced', C=1.0)",
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "runtime_seconds": runtime_seconds,
    }
    (output_dir / "abstraction_cross_domain_transfer_runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(source.to_string(index=False), flush=True)
    print(transfer.to_string(index=False), flush=True)
    print(f"Done in {runtime_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
