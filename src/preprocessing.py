"""
Reusable preprocessing built with scikit-learn's ColumnTransformer.

Key idea: preprocessing is a *step inside the model pipeline*, not a separate
script. When we call `pipeline.fit(X_train, y_train)`, the imputer medians, the
scaler means and the list of known categories are learned from the training
rows ONLY. When we later call `pipeline.predict(new_data)`, exactly the same
transformation is applied. This prevents data leakage and train/serve skew.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

# A category that appears fewer than this many times in training is grouped
# into a single "infrequent" column. `proto` has 133 values, most of them rare.
MIN_CATEGORY_FREQUENCY: int = 20


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """log(1 + |x|) with the original sign kept.

    Network features such as byte counts span from 0 to ~10^9. Distance-based
    and linear models are dominated by such huge values, so we compress them
    with a log first. (Must be a top-level function so joblib can save it.)
    """
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Create the (unfitted) preprocessing step.

    Args:
        scale_numeric: True for Logistic Regression, KNN and SVM, which are
            sensitive to feature scale (distances / dot products). Tree models (Random Forest, XGBoost) split on thresholds, so they
            do not need scaling and we skip it to keep them simple.

    Numeric columns:   median imputation -> [log transform -> standard scaling]
    Categorical:       constant imputation -> one-hot encoding
    """
    numeric_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps += [
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]

    categorical_steps = [
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        (
            "onehot",
            OneHotEncoder(
                # Categories never seen in training are mapped to the
                # "infrequent" column instead of crashing at prediction time.
                handle_unknown="infrequent_if_exist",
                min_frequency=MIN_CATEGORY_FREQUENCY,
                sparse_output=False,
            ),
        ),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(config.NUMERIC_FEATURES)),
            ("cat", Pipeline(categorical_steps), list(config.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",  # anything not in the allow-list is ignored
        verbose_feature_names_out=False,
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns produced by a *fitted* pipeline's preprocessor."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def original_feature_of(transformed_name: str) -> str:
    """Map a one-hot column such as 'proto_tcp' back to its source 'proto'."""
    for cat in config.CATEGORICAL_FEATURES:
        if transformed_name.startswith(f"{cat}_"):
            return cat
    return transformed_name
