"""
Optional experiment: Isolation Forest anomaly detection.

Supervised classifiers learn "what attacks look like" from labeled examples.
An anomaly detector instead learns "what is usual" and flags records that are
unusual. It can, in principle, flag attack types never seen in training, but
"unusual" is not the same as "malicious", so it usually raises more false
alarms. Isolation Forest is a tree ensemble, not a neural network.

How it works: random trees repeatedly split the data on random features at
random values. Unusual points get isolated after only a few splits, so a short
average path length = high anomaly score.

Two variants, kept completely separate from the supervised pipeline:

  A. unsupervised  - fit on a random sample of training rows, labels ignored;
                     threshold from contamination="auto" (no labels used at all)
  B. benign_only   - fit only on BENIGN training rows (novelty detection);
                     threshold = 95th percentile of scores on benign VALIDATION
                     rows, i.e. we accept ~5% false alarms on normal traffic.
                     Labels are used only to choose which rows to fit on.

Both are evaluated on the official test file against its ground-truth labels.
Run with:   python -m src.anomaly_detection
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from src import config
from src.evaluate import compute_metrics
from src.preprocessing import build_preprocessor
from src.train import load_and_split

FIT_SAMPLE_SIZE = 50_000


def build_iforest(contamination="auto") -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=False)),
        ("model", IsolationForest(n_estimators=config.IFOREST_N_ESTIMATORS,
                                  contamination=contamination, n_jobs=-1,
                                  random_state=config.RANDOM_STATE)),
    ])


def anomaly_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous. (sklearn's score_samples is higher = more normal.)"""
    return -pipeline.score_samples(X)


def per_category_detection(y_pred: np.ndarray, attack_cat: pd.Series | None) -> dict:
    """Share of each attack category that was flagged (recall per category)."""
    if attack_cat is None:
        return {}
    frame = pd.DataFrame({"cat": attack_cat.to_numpy(), "pred": y_pred})
    return frame.groupby("cat")["pred"].mean().round(4).to_dict()


def run_experiment() -> pd.DataFrame:
    data = load_and_split(verbose=False)
    X_fit, y_fit = data["X_fit"], data["y_fit"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"].to_numpy()
    rng = np.random.RandomState(config.RANDOM_STATE)
    rows, categories = [], {}

    # ---- Variant A: fully unsupervised -----------------------------------
    idx = rng.choice(len(X_fit), size=min(FIT_SAMPLE_SIZE, len(X_fit)), replace=False)
    pipe_a = build_iforest("auto")
    t0 = time.perf_counter()
    pipe_a.fit(X_fit.iloc[idx])  # NOTE: no y passed
    train_a = time.perf_counter() - t0
    scores_a = anomaly_scores(pipe_a, X_test)
    # sklearn's own decision boundary: predict() == -1 means "anomaly"
    threshold_a = float(-pipe_a.named_steps["model"].offset_)
    rows.append(("unsupervised (all traffic, contamination=auto)", len(idx), train_a,
                 compute_metrics(y_test, scores_a, threshold_a)))

    # ---- Variant B: novelty detection on benign traffic ------------------
    benign = X_fit[y_fit.to_numpy() == 0]
    benign = benign.iloc[rng.choice(len(benign), size=min(FIT_SAMPLE_SIZE, len(benign)),
                                    replace=False)]
    pipe_b = build_iforest("auto")
    t0 = time.perf_counter()
    pipe_b.fit(benign)
    train_b = time.perf_counter() - t0
    val_benign_scores = anomaly_scores(pipe_b, X_val[y_val.to_numpy() == 0])
    threshold_b = float(np.percentile(val_benign_scores, config.IFOREST_BENIGN_PERCENTILE))
    scores_b = anomaly_scores(pipe_b, X_test)
    rows.append((f"benign-only (threshold = {config.IFOREST_BENIGN_PERCENTILE:.0f}th pct "
                 "of benign validation scores)", len(benign), train_b,
                 compute_metrics(y_test, scores_b, threshold_b)))

    for name, scores, thr in (("unsupervised", scores_a, threshold_a),
                              ("benign_only", scores_b, threshold_b)):
        categories[name] = per_category_detection((scores >= thr).astype(int),
                                                  data["test_attack_cat"])

    results = pd.DataFrame([
        {"variant": name, "fit_rows": n, "train_seconds": t, **metrics}
        for name, n, t, metrics in rows
    ])
    results.to_csv(config.REPORTS_DIR / "anomaly_detection_results.csv", index=False)
    pd.DataFrame(categories).rename_axis("attack_cat").to_csv(
        config.REPORTS_DIR / "anomaly_detection_by_category.csv")
    return results


def main() -> None:
    results = run_experiment()
    cols = ["variant", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]
    print(results[cols].round(4).to_string(index=False))
    print("\nSaved reports/anomaly_detection_results.csv")


if __name__ == "__main__":
    main()
