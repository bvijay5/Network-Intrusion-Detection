"""
Feature-selection and PCA experiment.

Question: do we need all 42 features, or does a smaller set work as well?

For every classifier we train four variants and evaluate them on the same
official test set:

  1. all        - every allowed feature (after one-hot encoding)
  2. mutual_info - top-K columns ranked by mutual information with the label
  3. tree_importance - top-K columns ranked by a Random Forest's importance
  4. pca        - PCA components that keep 95% of the variance

Feature SELECTION keeps a subset of the ORIGINAL columns (still readable:
"sttl", "sbytes", ...). PCA builds NEW columns that are mixtures of all the
original ones, so it reduces dimensionality but loses direct interpretability.

All selectors live INSIDE the pipeline, so they are fitted on training rows
only. Run with:   python -m src.feature_selection
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src import config
from src.models import MODEL_KEYS, build_pipeline
from src.preprocessing import build_preprocessor
from src.train import load_and_split, stratified_sample

VARIANTS: tuple[str, ...] = ("all", "mutual_info", "tree_importance", "pca")


def ranking_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Scores for ROC/PR-AUC: probabilities if available, else SVM distances."""
    if hasattr(pipeline, "predict_proba"):
        try:
            return pipeline.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return pipeline.decision_function(X)


def load_best_params() -> dict[str, dict]:
    """Re-use the hyperparameters found by src.train, if it has been run."""
    path = config.REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return {row.model_key: json.loads(row.best_params) for row in table.itertuples()}


def build_variant(model_key: str, variant: str, y_train: np.ndarray,
                  best_params: dict, k: int) -> Pipeline:
    """Insert a selection / PCA step between preprocessing and the model."""
    base = build_pipeline(model_key, y_train)
    base.set_params(**{f"model__{p}": v for p, v in best_params.items()})
    preprocess, model = base.named_steps["preprocess"], base.named_steps["model"]
    rs = config.RANDOM_STATE

    if variant == "all":
        return base
    if variant == "mutual_info":
        selector = SelectKBest(partial(mutual_info_classif, random_state=rs), k=k)
    elif variant == "tree_importance":
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=rs,
                                   class_weight="balanced"),
            max_features=k, threshold=-np.inf,  # keep exactly the top-k
        )
    elif variant == "pca":
        # PCA needs scaled inputs, so every model gets the scaled preprocessor.
        preprocess = build_preprocessor(scale_numeric=True)
        selector = PCA(n_components=config.PCA_VARIANCE, random_state=rs)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return Pipeline([("preprocess", preprocess), ("select", selector), ("model", model)])


def n_model_inputs(pipeline: Pipeline) -> int:
    """How many columns actually reach the classifier."""
    if "select" in pipeline.named_steps:
        step = pipeline.named_steps["select"]
        return int(step.n_components_) if isinstance(step, PCA) else int(step.get_support().sum())
    return len(pipeline.named_steps["preprocess"].get_feature_names_out())


def selected_names(pipeline: Pipeline) -> list[str]:
    if "select" not in pipeline.named_steps or isinstance(pipeline.named_steps["select"], PCA):
        return []
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    return list(names[pipeline.named_steps["select"].get_support()])


def run_experiment(model_keys: list[str], sample_size: int | None, k: int) -> pd.DataFrame:
    data = load_and_split(verbose=False)
    best = load_best_params()
    X_test, y_test = data["X_test"], data["y_test"].to_numpy()
    rows, selected = [], {}

    for key in model_keys:
        cap = config.MAX_TRAIN_ROWS.get(key)
        n = min(x for x in (sample_size, cap, len(data["X_fit"])) if x is not None)
        X_tr, y_tr = stratified_sample(data["X_fit"], data["y_fit"], n)
        for variant in VARIANTS:
            pipe = build_variant(key, variant, y_tr.to_numpy(), best.get(key, {}), k)
            t0 = time.perf_counter()
            pipe.fit(X_tr, y_tr)
            train_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            scores = ranking_scores(pipe, X_test)
            y_pred = pipe.predict(X_test)
            pred_s = time.perf_counter() - t0
            row = {
                "model_key": key,
                "model_name": config.MODEL_DISPLAY_NAMES[key],
                "variant": variant,
                "n_inputs": n_model_inputs(pipe),
                "train_rows": len(X_tr),
                "f1_default_threshold": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, scores),
                "pr_auc": average_precision_score(y_test, scores),
                "train_seconds": train_s,
                "predict_seconds_test": pred_s,
            }
            rows.append(row)
            if variant in ("mutual_info", "tree_importance") and variant not in selected:
                selected[variant] = selected_names(pipe)
            print(f"{row['model_name']:<24} {variant:<16} inputs={row['n_inputs']:<3} "
                  f"F1={row['f1_default_threshold']:.4f} ROC-AUC={row['roc_auc']:.4f} "
                  f"train={train_s:.1f}s")

    results = pd.DataFrame(rows)
    results.to_csv(config.REPORTS_DIR / "feature_selection_results.csv", index=False)
    (config.REPORTS_DIR / "selected_features.json").write_text(json.dumps({
        "top_k": k, "train_sample_size": sample_size, **selected}, indent=2))
    _plot(results)
    return results


def _plot(results: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = results.pivot(index="model_name", columns="variant", values="roc_auc")[list(VARIANTS)]
    ax = pivot.plot(kind="bar", figsize=(9, 4.5), rot=0)
    ax.set_ylim(max(0.5, pivot.min().min() - 0.03), 1.0)
    ax.set_ylabel("ROC-AUC (test set)")
    ax.set_title("Feature-set comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.FIGURES_DIR / "feature_selection.png", dpi=130)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature selection / PCA experiment.")
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--sample-size", type=int, default=config.FS_TRAIN_SAMPLE_SIZE)
    parser.add_argument("--top-k", type=int, default=config.FS_TOP_K)
    args = parser.parse_args()
    run_experiment(args.models, args.sample_size, args.top_k)
    print("\nSaved reports/feature_selection_results.csv")


if __name__ == "__main__":
    main()
