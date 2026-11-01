"""Dataset loading, schema validation and CSV upload checks."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import (
    DatasetError,
    coerce_feature_types,
    data_quality_summary,
    load_labeled_dataset,
    read_csv_safely,
    train_test_overlap,
    validate_schema,
)


def test_valid_file_loads(synthetic_csv):
    df = load_labeled_dataset(synthetic_csv)
    assert len(df) == 600
    assert set(config.FEATURE_COLUMNS) <= set(df.columns)
    assert set(df["label"].unique()) <= {0, 1}


def test_byte_order_mark_is_removed(tmp_path, synthetic_df):
    # The official training CSV starts with a UTF-8 BOM before "id".
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbf" + synthetic_df.to_csv(index=False).encode())
    df = read_csv_safely(path)
    assert "id" in df.columns and not any("﻿" in c for c in df.columns)


def test_column_names_are_normalised(synthetic_df):
    upper = synthetic_df.rename(columns={"sbytes": " SBYTES "})
    df = read_csv_safely(io.StringIO(upper.to_csv(index=False)))
    assert "sbytes" in df.columns


def test_missing_columns_are_reported(synthetic_df):
    report = validate_schema(synthetic_df.drop(columns=["sttl", "proto"]))
    assert not report.is_valid
    assert report.missing_columns == ["proto", "sttl"]


def test_unexpected_columns_are_ignored_not_fatal(synthetic_df):
    report = validate_schema(synthetic_df.assign(my_extra_column=1))
    assert report.is_valid
    assert report.extra_columns == ["my_extra_column"]


def test_target_required_for_training_data(synthetic_df):
    unlabeled = synthetic_df.drop(columns=["label"])
    assert validate_schema(unlabeled, require_target=False).is_valid
    assert not validate_schema(unlabeled, require_target=True).is_valid


def test_wrong_schema_raises_clear_error(tmp_path):
    path = tmp_path / "other.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
    with pytest.raises(DatasetError, match="missing"):
        load_labeled_dataset(path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DatasetError):
        read_csv_safely(path)


def test_header_only_file_raises(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text(",".join(config.FEATURE_COLUMNS) + "\n")
    with pytest.raises(DatasetError, match="no data rows"):
        read_csv_safely(path)


def test_empty_upload_raises():
    with pytest.raises(DatasetError, match="empty"):
        read_csv_safely(io.BytesIO(b"   "))


def test_missing_file_raises(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        read_csv_safely(tmp_path / "nope.csv")


def test_row_limit_is_enforced(synthetic_df):
    buffer = io.BytesIO(synthetic_df.to_csv(index=False).encode())
    with pytest.raises(DatasetError, match="limit"):
        read_csv_safely(buffer, max_rows=100)


def test_invalid_numbers_become_nan(synthetic_df):
    bad = synthetic_df.astype({"sbytes": object, "dur": object})
    bad.loc[0, "sbytes"] = "not-a-number"
    bad.loc[1, "dur"] = np.inf
    cleaned, n_invalid = coerce_feature_types(bad)
    assert n_invalid == 2
    assert np.isnan(cleaned.loc[0, "sbytes"]) and np.isnan(cleaned.loc[1, "dur"])


def test_rows_with_invalid_labels_are_dropped(tmp_path, synthetic_df):
    bad = synthetic_df.astype({"label": object})
    bad.loc[:4, "label"] = "unknown"
    path = tmp_path / "bad_labels.csv"
    bad.to_csv(path, index=False)
    df = load_labeled_dataset(path)
    assert len(df) == len(synthetic_df) - 5


def test_quality_summary_counts(synthetic_df):
    dup = pd.concat([synthetic_df, synthetic_df.iloc[:3]], ignore_index=True)
    summary = data_quality_summary(dup)
    assert summary["duplicate_rows_features_and_label"] == 3
    assert summary["missing_values_total"] == 0
    assert sum(summary["class_counts"].values()) == len(dup)


def test_train_test_overlap(synthetic_df):
    train, test = synthetic_df.iloc[:400], synthetic_df.iloc[350:]
    result = train_test_overlap(train, test)
    assert result["test_rows_seen_in_train"] >= 50
