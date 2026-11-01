"""
The five supervised classifiers, each wrapped in a full Pipeline:

    raw CSV columns  ->  preprocessing  ->  classifier

Because preprocessing is inside the pipeline, a saved model file contains
everything needed to go from a raw CSV row to a prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src import config
from src.preprocessing import build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    """Static facts about one algorithm (used by training and the docs)."""

    key: str
    display_name: str
    needs_scaling: bool
    handles_imbalance_with: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "logistic_regression": ModelSpec(
        "logistic_regression", "Logistic Regression", True, "class_weight='balanced'"
    ),
    "knn": ModelSpec(
        "knn", "K-Nearest Neighbors", True,
        "not supported by the algorithm (reported as a limitation)",
    ),
    "svm": ModelSpec("svm", "Support Vector Machine", True, "class_weight='balanced'"),
    "random_forest": ModelSpec(
        "random_forest", "Random Forest", False, "class_weight='balanced'"
    ),
    "xgboost": ModelSpec(
        "xgboost", "XGBoost", False, "scale_pos_weight = #benign / #attack"
    ),
}

MODEL_KEYS: tuple[str, ...] = tuple(MODEL_SPECS)


def _make_classifier(key: str, y_train: np.ndarray):
    """Create an un-tuned classifier with sensible, reproducible defaults."""
    rs = config.RANDOM_STATE
    if key == "logistic_regression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rs)
    if key == "knn":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    if key == "svm":
        # Tuned without probabilities (much faster); the final fit is wrapped
        # in a calibrator, see add_probability_calibration().
        return SVC(kernel="rbf", class_weight="balanced", cache_size=1000, random_state=rs)
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=rs
        )
    if key == "xgboost":
        n_pos = max(int(np.sum(y_train == 1)), 1)
        n_neg = int(np.sum(y_train == 0))
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=rs,
        )
    raise KeyError(f"Unknown model '{key}'. Choose from: {', '.join(MODEL_KEYS)}")


def build_pipeline(key: str, y_train: np.ndarray) -> Pipeline:
    """Return an unfitted Pipeline(preprocess -> model) for the given algorithm."""
    spec = MODEL_SPECS[key]
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=spec.needs_scaling)),
            ("model", _make_classifier(key, y_train)),
        ]
    )


def add_probability_calibration(pipeline: Pipeline, key: str) -> Pipeline:
    """Give the SVM a predict_proba method for the final fit.

    An SVM outputs a signed distance from its decision boundary, not a
    probability. Platt scaling (method="sigmoid") fits a small logistic curve
    that maps those distances to scores in [0, 1], using internal 3-fold
    cross-validation on the training rows only. Other models already have
    predict_proba, so they are returned unchanged.
    """
    if key != "svm":
        return pipeline
    svc = pipeline.named_steps["model"]
    pipeline.steps[-1] = (
        "model",
        CalibratedClassifierCV(svc, method="sigmoid", cv=3, ensemble=False),
    )
    return pipeline


def predict_scores(pipeline: Pipeline, X) -> np.ndarray:
    """Return a score in [0, 1] where higher means 'more likely an attack'.

    For every model here this is `predict_proba(X)[:, 1]`. These numbers are
    model confidence scores, NOT guaranteed real-world attack probabilities.
    """
    return pipeline.predict_proba(X)[:, 1]
