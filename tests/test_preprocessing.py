"""Preprocessing consistency and target-leakage prevention."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_loader import split_features_target
from src.models import MODEL_KEYS, build_pipeline
from src.preprocessing import build_preprocessor, original_feature_of, signed_log1p


def test_target_and_leaky_columns_are_excluded(synthetic_df):
    X, y = split_features_target(synthetic_df.assign(suspicious_copy=synthetic_df["label"]))
    for col in ("id", "label", "attack_cat", "suspicious_copy"):
        assert col not in X.columns
    assert list(X.columns) == list(config.FEATURE_COLUMNS)
    assert y.equals(synthetic_df["label"].astype(int))


def test_preprocessor_never_sees_target(synthetic_df):
    X, y = split_features_target(synthetic_df)
    pre = build_preprocessor(scale_numeric=True).fit(X, y)
    names = pre.get_feature_names_out()
    assert not any(n in ("label", "attack_cat", "id") or n.startswith("attack_cat")
                   for n in names)


def test_statistics_are_learned_from_training_rows_only(synthetic_df):
    X, _ = split_features_target(synthetic_df)
    train, other = X.iloc[:300], X.iloc[300:].copy()
    other["dur"] = other["dur"] * 1000  # very different "test" data
    pre = build_preprocessor(scale_numeric=False).fit(train)
    imputer = pre.named_transformers_["num"].named_steps["impute"]
    dur_pos = list(config.NUMERIC_FEATURES).index("dur")
    assert imputer.statistics_[dur_pos] == train["dur"].median()


def test_missing_values_are_imputed(synthetic_df):
    X, _ = split_features_target(synthetic_df)
    X = X.copy()
    X.loc[:10, "sbytes"] = np.nan
    X.loc[:10, "service"] = np.nan
    out = build_preprocessor(scale_numeric=True).fit_transform(X)
    assert not np.isnan(out).any()


def test_unseen_category_does_not_crash(synthetic_df):
    X, _ = split_features_target(synthetic_df)
    pre = build_preprocessor(scale_numeric=True).fit(X)
    new = X.iloc[:5].copy()
    new["proto"] = "never-seen-protocol"
    out = pre.transform(new)
    assert out.shape == (5, len(pre.get_feature_names_out()))


def test_same_output_width_for_training_and_new_data(synthetic_df):
    X, y = split_features_target(synthetic_df)
    for scale in (True, False):
        pre = build_preprocessor(scale).fit(X.iloc[:400], y.iloc[:400])
        assert pre.transform(X.iloc[:400]).shape[1] == pre.transform(X.iloc[400:]).shape[1]


def test_scaled_features_are_standardised_on_training_data(synthetic_df):
    X, _ = split_features_target(synthetic_df)
    out = build_preprocessor(scale_numeric=True).fit_transform(X)
    numeric = out[:, : len(config.NUMERIC_FEATURES)]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-6)


def test_signed_log1p():
    x = np.array([-9.0, 0.0, 9.0])
    assert np.allclose(signed_log1p(x), [-np.log(10), 0, np.log(10)])


def test_one_hot_names_map_back_to_original_feature():
    assert original_feature_of("proto_tcp") == "proto"
    assert original_feature_of("service_-") == "service"
    assert original_feature_of("sbytes") == "sbytes"


def test_every_model_pipeline_starts_with_preprocessing(synthetic_df):
    _, y = split_features_target(synthetic_df)
    for key in MODEL_KEYS:
        pipe = build_pipeline(key, y.to_numpy())
        assert list(pipe.named_steps) == ["preprocess", "model"]


def test_pipeline_ignores_column_order(synthetic_df):
    X, y = split_features_target(synthetic_df)
    pipe = build_pipeline("logistic_regression", y.to_numpy()).fit(X, y)
    shuffled = X[X.columns[::-1]]
    assert np.allclose(pipe.predict_proba(X), pipe.predict_proba(shuffled))
    assert isinstance(shuffled, pd.DataFrame)
