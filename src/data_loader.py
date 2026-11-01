"""
Loading and validating UNSW-NB15 CSV files.

Why a separate module? A model is only as trustworthy as the data it reads.
This file makes sure a CSV really looks like UNSW-NB15 *before* any model sees
it, and explains clearly what is wrong when it does not.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union

import numpy as np
import pandas as pd

from src import config

CsvSource = Union[str, Path, IO[bytes], IO[str]]


class DatasetError(ValueError):
    """Raised when a CSV file cannot be used safely (wrong schema, empty, ...)."""


@dataclass
class SchemaReport:
    """Result of comparing a DataFrame against the expected UNSW-NB15 schema."""

    n_rows: int
    n_columns: int
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    has_target: bool = False
    invalid_target_values: int = 0
    messages: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """A file is usable if it has rows and every feature column is present."""
        return self.n_rows > 0 and not self.missing_columns


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, invisible byte-order marks and upper-case letters.

    The official training CSV starts with a UTF-8 byte-order mark, which would
    otherwise turn the first column name into '\\ufeffid' instead of 'id'.
    """
    df = df.copy()
    df.columns = [str(c).replace("﻿", "").strip().lower() for c in df.columns]
    return df


def read_csv_safely(source: CsvSource, max_rows: int | None = None) -> pd.DataFrame:
    """Read a CSV into a DataFrame without executing anything inside it.

    Args:
        source: file path or an uploaded file-like object.
        max_rows: refuse files with more rows than this (protects memory).

    Raises:
        DatasetError: if the file is empty, unreadable or too large.
    """
    try:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise DatasetError(f"File not found: {path}")
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        else:
            raw = source.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8-sig", errors="replace")
            if not raw.strip():
                raise DatasetError("The uploaded file is empty.")
            df = pd.read_csv(io.StringIO(raw), low_memory=False)
    except DatasetError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The file is empty or has no header row.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"The file could not be parsed as CSV: {exc}") from exc

    if df.empty:
        raise DatasetError("The file has a header but no data rows.")
    if max_rows is not None and len(df) > max_rows:
        raise DatasetError(
            f"The file has {len(df):,} rows; the limit is {max_rows:,}. "
            "Split it into smaller files."
        )
    return normalise_column_names(df)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, require_target: bool = False) -> SchemaReport:
    """Compare the columns of `df` with the columns the models expect.

    Args:
        df: data to check (column names should already be normalised).
        require_target: True for training/evaluation data, False for new,
            unlabeled traffic that we only want to predict on.
    """
    expected = set(config.FEATURE_COLUMNS)
    present = set(df.columns)
    report = SchemaReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        missing_columns=sorted(expected - present),
        extra_columns=sorted(present - expected - set(config.EXCLUDED_COLUMNS)),
        has_target=config.TARGET_COLUMN in present,
    )

    if report.n_rows == 0:
        report.messages.append("The file contains no rows.")
    if report.missing_columns:
        report.messages.append(
            f"{len(report.missing_columns)} required feature column(s) are missing: "
            + ", ".join(report.missing_columns)
        )
    if report.extra_columns:
        report.messages.append(
            "These columns are not used by the models and will be ignored: "
            + ", ".join(report.extra_columns)
        )

    if report.has_target:
        target = pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce")
        report.invalid_target_values = int((~target.isin([0, 1])).sum())
        if report.invalid_target_values:
            report.messages.append(
                f"{report.invalid_target_values} row(s) have a '{config.TARGET_COLUMN}' "
                "value that is not 0 or 1."
            )
    elif require_target:
        report.messages.append(
            f"The target column '{config.TARGET_COLUMN}' is required here but missing."
        )
        report.missing_columns.append(config.TARGET_COLUMN)

    return report


def coerce_feature_types(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Convert numeric columns to numbers and categorical columns to text.

    Values that cannot be converted (e.g. 'abc' in a byte-count column) and
    +/- infinity become NaN, which the preprocessing imputer later fills.

    Returns:
        (cleaned copy of df, number of invalid numeric values that became NaN)
    """
    df = df.copy()
    invalid = 0
    for col in config.NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        original_na = df[col].isna().sum()
        converted = pd.to_numeric(df[col], errors="coerce").astype("float64")
        converted = converted.replace([np.inf, -np.inf], np.nan)
        invalid += int(converted.isna().sum() - original_na)
        df[col] = converted
    for col in config.CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        # Keep real missing values as NaN so the imputer can handle them.
        df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
        df[col] = df[col].map(lambda v: v if pd.isna(v) else str(v).strip().lower())
    return df, invalid


def load_labeled_dataset(source: CsvSource) -> pd.DataFrame:
    """Load a labeled UNSW-NB15 file (training or testing set) for modelling.

    Rows with an invalid target value are dropped and reported, because we
    cannot train or evaluate on a record whose true class is unknown.

    Raises:
        DatasetError: if the schema does not match.
    """
    df = read_csv_safely(source)
    report = validate_schema(df, require_target=True)
    if not report.is_valid:
        raise DatasetError(
            "Dataset does not match the expected UNSW-NB15 schema:\n- "
            + "\n- ".join(report.messages)
        )
    df, n_invalid = coerce_feature_types(df)
    if n_invalid:
        print(f"[data_loader] {n_invalid} invalid numeric values were set to NaN.")

    target = pd.to_numeric(df[config.TARGET_COLUMN], errors="coerce")
    valid_rows = target.isin([0, 1])
    if (~valid_rows).any():
        print(f"[data_loader] Dropping {(~valid_rows).sum()} rows with an invalid label.")
    df = df.loc[valid_rows].copy()
    df[config.TARGET_COLUMN] = target[valid_rows].astype(int)
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) where X holds ONLY the allowed feature columns.

    Selecting columns from an explicit allow-list (instead of dropping a few
    columns) guarantees that id, label and attack_cat can never sneak in.
    """
    X = df.loc[:, list(config.FEATURE_COLUMNS)].copy()
    y = df[config.TARGET_COLUMN].astype(int)
    return X, y


# --------------------------------------------------------------------------
# Data-quality inspection
# --------------------------------------------------------------------------
def data_quality_summary(df: pd.DataFrame) -> dict:
    """Collect the facts you should know about a dataset before training."""
    features = [c for c in config.FEATURE_COLUMNS if c in df.columns]
    numeric = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    summary: dict = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "feature_columns_used": len(features),
        "missing_values_total": int(df[features].isna().sum().sum()),
        "missing_values_by_column": {
            c: int(n) for c, n in df[features].isna().sum().items() if n > 0
        },
        "negative_values_in_numeric": {
            c: int((df[c] < 0).sum()) for c in numeric if (df[c] < 0).any()
        },
        "duplicate_rows_features_and_label": int(
            df.duplicated(subset=features + [config.TARGET_COLUMN]).sum()
        )
        if config.TARGET_COLUMN in df.columns
        else None,
        "dtypes": {c: str(t) for c, t in df[features].dtypes.items()},
    }
    if config.TARGET_COLUMN in df.columns:
        counts = df[config.TARGET_COLUMN].value_counts().sort_index()
        summary["class_counts"] = {config.CLASS_NAMES[int(k)]: int(v) for k, v in counts.items()}
        summary["attack_fraction"] = float(df[config.TARGET_COLUMN].mean())
        # Same features but different labels: no model can get these all right.
        conflicts = df.groupby(features, dropna=False)[config.TARGET_COLUMN].nunique()
        summary["conflicting_duplicate_groups"] = int((conflicts > 1).sum())
    if "attack_cat" in df.columns:
        summary["attack_categories"] = {
            str(k): int(v) for k, v in df["attack_cat"].value_counts().items()
        }
    for col in config.CATEGORICAL_FEATURES:
        if col in df.columns:
            summary[f"unique_{col}"] = int(df[col].nunique())
    return summary


def train_test_overlap(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Count test rows whose feature values also appear exactly in training.

    We REPORT this instead of deleting rows, because the official split is the
    benchmark everyone compares against; removing rows would change it.
    """
    features = list(config.FEATURE_COLUMNS)
    train_keys = pd.util.hash_pandas_object(train[features], index=False)
    test_keys = pd.util.hash_pandas_object(test[features], index=False)
    overlap = test_keys.isin(set(train_keys))
    categories_unseen = {
        col: sorted(set(test[col].dropna()) - set(train[col].dropna()))
        for col in config.CATEGORICAL_FEATURES
    }
    return {
        "test_rows_seen_in_train": int(overlap.sum()),
        "test_rows_total": int(len(test)),
        "overlap_fraction": float(overlap.mean()),
        "test_categories_unseen_in_train": categories_unseen,
        "train_categories_unseen_in_test": {
            col: sorted(set(train[col].dropna()) - set(test[col].dropna()))
            for col in config.CATEGORICAL_FEATURES
        },
    }
