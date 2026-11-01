"""Training/inference compatibility, model saving/loading, prediction output."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import DatasetError, split_features_target
from src.evaluate import choose_threshold, compute_metrics
from src.predict import (
    ModelNotAvailableError,
    available_models,
    load_bundle,
    predict_dataframe,
    predictions_to_csv,
    save_bundle,
)
from src.train import tune_and_fit

# Tiny grids so every model trains in about a second.
TINY_GRIDS = {
    "logistic_regression": {"model__C": [1.0]},
    "knn": {"model__n_neighbors": [5]},
    "svm": {"model__kernel": ["rbf"], "model__C": [1.0]},
    "random_forest": {"model__n_estimators": [20], "model__max_depth": [5]},
    "xgboost": {"model__n_estimators": [20], "model__max_depth": [3]},
}


def _fit(key: str, df: pd.DataFrame):
    X, y = split_features_target(df)
    pipe, *_ = tune_and_fit(key, X, y, tuning_size=None, max_rows=None,
                            param_grid=TINY_GRIDS[key])
    return pipe


@pytest.fixture(scope="module")
def trained_rf():
    from tests.conftest import make_synthetic
    return _fit("random_forest", make_synthetic())


@pytest.mark.parametrize("key", list(TINY_GRIDS))
def test_every_model_trains_and_predicts_scores(key, synthetic_df):
    pipe = _fit(key, synthetic_df)
    X, y = split_features_target(synthetic_df)
    scores = pipe.predict_proba(X)[:, 1]
    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))
    # The synthetic signal is easy; every model should beat a coin flip.
    assert compute_metrics(y, scores)["roc_auc"] > 0.7


def test_save_and_load_roundtrip(tmp_path, trained_rf, synthetic_df):
    save_bundle(trained_rf, "random_forest", 0.42, {"note": "test"}, models_dir=tmp_path)
    assert available_models(tmp_path) == ["random_forest"]
    bundle = load_bundle("random_forest", models_dir=tmp_path)
    X, _ = split_features_target(synthetic_df)
    assert bundle["threshold"] == pytest.approx(0.42)
    assert np.allclose(bundle["pipeline"].predict_proba(X), trained_rf.predict_proba(X))
    assert "library_versions" in bundle["metadata"]


def test_missing_model_gives_helpful_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="python -m src.train"):
        load_bundle("xgboost", models_dir=tmp_path)


def test_tampered_model_file_is_refused(tmp_path, trained_rf):
    path = save_bundle(trained_rf, "random_forest", 0.5, {}, models_dir=tmp_path)
    with open(path, "ab") as fh:
        fh.write(b"tampered")
    with pytest.raises(PermissionError, match="checksum"):
        load_bundle("random_forest", models_dir=tmp_path)


def test_model_outside_models_folder_is_refused(tmp_path):
    with pytest.raises(PermissionError):
        load_bundle("../outside", models_dir=tmp_path / "models")


def test_prediction_on_unlabeled_data(tmp_path, trained_rf, synthetic_df):
    save_bundle(trained_rf, "random_forest", 0.5, {}, models_dir=tmp_path)
    bundle = load_bundle("random_forest", models_dir=tmp_path)
    unlabeled = synthetic_df.drop(columns=["label", "attack_cat", "id"])
    out = predict_dataframe(bundle, unlabeled)
    assert len(out) == len(unlabeled)
    assert {"attack_score", "prediction", "predicted_label"} <= set(out.columns)
    assert set(out["prediction"].unique()) <= {0, 1}
    assert set(out["predicted_label"].unique()) <= {"Benign", "Malicious"}


def test_labels_in_input_do_not_change_predictions(tmp_path, trained_rf, synthetic_df):
    save_bundle(trained_rf, "random_forest", 0.5, {}, models_dir=tmp_path)
    bundle = load_bundle("random_forest", models_dir=tmp_path)
    with_labels = predict_dataframe(bundle, synthetic_df)
    flipped = synthetic_df.assign(label=1 - synthetic_df["label"], attack_cat="Normal")
    assert with_labels["attack_score"].equals(predict_dataframe(bundle, flipped)["attack_score"])


def test_threshold_changes_flag_count(tmp_path, trained_rf, synthetic_df):
    save_bundle(trained_rf, "random_forest", 0.5, {}, models_dir=tmp_path)
    bundle = load_bundle("random_forest", models_dir=tmp_path)
    low = predict_dataframe(bundle, synthetic_df, threshold=0.05)["prediction"].sum()
    high = predict_dataframe(bundle, synthetic_df, threshold=0.95)["prediction"].sum()
    assert low >= high


def test_prediction_rejects_incompatible_file(tmp_path, trained_rf, synthetic_df):
    save_bundle(trained_rf, "random_forest", 0.5, {}, models_dir=tmp_path)
    bundle = load_bundle("random_forest", models_dir=tmp_path)
    with pytest.raises(DatasetError, match="missing"):
        predict_dataframe(bundle, synthetic_df.drop(columns=["sbytes"]))


def test_prediction_export_roundtrip(tmp_path, trained_rf, synthetic_df):
    save_bundle(trained_rf, "random_forest", 0.5, {}, models_dir=tmp_path)
    out = predict_dataframe(load_bundle("random_forest", models_dir=tmp_path), synthetic_df)
    exported = pd.read_csv(io.BytesIO(predictions_to_csv(out)))
    assert len(exported) == len(out)
    assert {"attack_score", "prediction", "predicted_label"} <= set(exported.columns)


def test_metrics_on_known_example():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.8, 0.4, 0.9])
    m = compute_metrics(y, s, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == 0.5
    assert m["false_positive_rate"] == 0.5


def test_threshold_is_chosen_from_validation_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.35, 0.3, 0.8, 0.9])
    t = choose_threshold(y, s)
    assert t in s
    assert compute_metrics(y, s, t)["f1"] >= compute_metrics(y, s, 0.5)["f1"]


def test_feature_columns_constant_is_consistent():
    assert len(config.FEATURE_COLUMNS) == len(set(config.FEATURE_COLUMNS)) == 42
    assert not set(config.EXCLUDED_COLUMNS) & set(config.FEATURE_COLUMNS)
