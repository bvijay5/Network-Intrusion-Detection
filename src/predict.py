"""
Saving, safely loading and using trained model bundles.

A "bundle" is one .joblib file holding a dict:
    pipeline        fitted Pipeline(preprocess -> model)
    model_key       e.g. "random_forest"
    threshold       decision threshold chosen on the validation set
    feature_columns columns the pipeline expects
    metadata        training rows, best hyperparameters, library versions, ...

SECURITY NOTE: joblib/pickle files can execute code when loaded. So this module
only loads files that (1) live inside the project's models/ folder and (2) match
the SHA-256 checksum written to models/manifest.json when we trained them.
Never load a model file you received from someone else.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from src import config
from src.data_loader import DatasetError, coerce_feature_types, validate_schema
from src.models import predict_scores

MANIFEST_NAME = "manifest.json"


class ModelNotAvailableError(FileNotFoundError):
    """Raised when a requested model has not been trained yet."""


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------
def library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(models_dir: Path) -> dict:
    path = models_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_bundle(pipeline, model_key: str, threshold: float, metadata: dict,
                models_dir: Path = config.MODELS_DIR) -> Path:
    """Save a fitted pipeline plus everything needed to use it later."""
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_key": model_key,
        "threshold": float(threshold),
        "feature_columns": list(config.FEATURE_COLUMNS),
        "metadata": {
            **metadata,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "library_versions": library_versions(),
        },
    }
    path = models_dir / f"{model_key}.joblib"
    joblib.dump(bundle, path, compress=3)

    manifest = _read_manifest(models_dir)
    manifest[path.name] = {"sha256": _sha256(path), "model_key": model_key}
    (models_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def available_models(models_dir: Path = config.MODELS_DIR) -> list[str]:
    """Model keys that have a trained file listed in the manifest."""
    manifest = _read_manifest(models_dir)
    return [
        info["model_key"] for name, info in manifest.items()
        if (models_dir / name).exists()
    ]


def load_bundle(model_key: str, models_dir: Path = config.MODELS_DIR) -> dict:
    """Load a bundle produced by `save_bundle`, after safety checks.

    Raises:
        ModelNotAvailableError: if the model has not been trained.
        PermissionError: if the file is outside models/ or its checksum changed.
    """
    models_dir = models_dir.resolve()
    path = (models_dir / f"{model_key}.joblib").resolve()
    if models_dir not in path.parents:
        raise PermissionError("Refusing to load a model from outside the models folder.")
    if not path.exists():
        raise ModelNotAvailableError(
            f"No trained model '{model_key}' found in {models_dir}. "
            "Train the models first with:  python -m src.train"
        )
    expected = _read_manifest(models_dir).get(path.name, {}).get("sha256")
    if expected is None or expected != _sha256(path):
        raise PermissionError(
            f"{path.name} is not listed in models/{MANIFEST_NAME} or its checksum "
            "does not match. It was not produced by this project's training run, "
            "so it will not be loaded. Re-train with:  python -m src.train"
        )

    bundle = joblib.load(path)
    saved = bundle["metadata"].get("library_versions", {})
    current = library_versions()
    mismatched = {
        lib: (saved.get(lib), current[lib])
        for lib in ("scikit-learn", "xgboost")
        if saved.get(lib) and saved.get(lib) != current[lib]
    }
    bundle["version_warnings"] = [
        f"{lib} was {old} at training time but is {new} now; "
        "if predictions look wrong, re-train with: python -m src.train"
        for lib, (old, new) in mismatched.items()
    ]
    return bundle


# --------------------------------------------------------------------------
# Predicting on new (possibly unlabeled) traffic
# --------------------------------------------------------------------------
def prepare_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an input table and return only the model's feature columns.

    A target label is NOT required: real traffic to be analysed is unlabeled.
    If label / attack_cat happen to be present they are simply not used.
    """
    report = validate_schema(df, require_target=False)
    if not report.is_valid:
        raise DatasetError(
            "This file cannot be analysed by the model:\n- " + "\n- ".join(report.messages)
        )
    cleaned, _ = coerce_feature_types(df)
    return cleaned.loc[:, list(config.FEATURE_COLUMNS)]


def predict_dataframe(bundle: dict, df: pd.DataFrame,
                      threshold: float | None = None) -> pd.DataFrame:
    """Run a model on a DataFrame and return the input plus prediction columns.

    Added columns:
        attack_score     model score in [0, 1] (a confidence, not a guarantee)
        prediction       0 = benign, 1 = potentially malicious
        predicted_label  human-readable version of `prediction`
    """
    X = prepare_for_prediction(df)
    threshold = bundle["threshold"] if threshold is None else float(threshold)
    scores = predict_scores(bundle["pipeline"], X)
    out = df.copy()
    out["attack_score"] = np.round(scores, 6)
    out["prediction"] = (scores >= threshold).astype(int)
    out["predicted_label"] = out["prediction"].map(config.CLASS_NAMES)
    return out


def predictions_to_csv(predictions: pd.DataFrame) -> bytes:
    """Serialise prediction results for download."""
    return predictions.to_csv(index=False).encode("utf-8")
