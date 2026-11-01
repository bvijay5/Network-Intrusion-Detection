"""
Builds the downloadable Markdown summary report from files in reports/.

The report only contains numbers that were written by an actual training run.
If a results file is missing, the section says so instead of inventing values.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config

LIMITATIONS = """\
- Results come from the UNSW-NB15 benchmark (synthetic traffic generated in a
  testbed in 2015). Performance on other networks will differ.
- The official training and testing files have different distributions:
  cross-validation scores on training data are clearly higher than test scores.
- Some features (e.g. `sttl`, `ct_state_ttl`) are strong testbed artefacts and
  may not generalise to real networks.
- The system analyses offline CSV records. It does NOT capture or monitor live
  network traffic.
- Model scores are confidence values, not verified probabilities of attack.
  Alerts are indicators for investigation, not proof of malicious activity.
- Feature importance and SHAP values show association with the model's output,
  not causation.
"""


def _read_csv(name: str) -> pd.DataFrame | None:
    path = config.REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def _read_json(name: str) -> dict | None:
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    cols = [c for c in columns if c in df.columns]
    shown = df[cols].copy()
    for c in shown.select_dtypes("number").columns:
        shown[c] = shown[c].round(4)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "---|" * len(cols)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in shown.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def build_markdown_report(prediction_summary: dict | None = None) -> str:
    """Return the full report as a Markdown string."""
    lines = [
        "# Explainable ML-Based Network Intrusion Detection - Analysis Report",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. "
        "This is an offline analysis summary, not a verified security incident report._",
        "",
        "## Dataset",
        "UNSW-NB15 (official training and testing CSV files), "
        "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
        "",
    ]
    quality = _read_json("data_quality.json")
    if quality:
        tr, te, ov = quality["train"], quality["test"], quality["train_test_overlap"]
        lines += [
            f"- Training rows: {tr['rows']:,} ({tr['class_counts']})",
            f"- Testing rows: {te['rows']:,} ({te['class_counts']})",
            f"- Feature columns used: {tr['feature_columns_used']} "
            "(id, label and attack_cat excluded to prevent leakage)",
            f"- Duplicate rows in training: {tr['duplicate_rows_features_and_label']:,}",
            f"- Test rows whose features also appear in training: "
            f"{ov['test_rows_seen_in_train']:,} ({ov['overlap_fraction']:.1%})",
            "",
        ]
    else:
        lines += ["_Data-quality report not found. Run `python -m src.train`._", ""]

    run = _read_json("run_config.json")
    lines += ["## Experimental configuration"]
    if run:
        lines += [
            f"- Fit / validation / test rows: {run['rows']['fit']:,} / "
            f"{run['rows']['validation']:,} / {run['rows']['test']:,}",
            f"- Hyperparameter tuning sample: {run['tuning_sample_size']} rows, "
            f"{run['cv_folds']}-fold stratified CV, scored by attack-class F1",
            f"- Training row caps: {run['max_train_rows']}",
            f"- Decision threshold: chosen on validation set ({run['threshold_strategy']})",
            f"- Random seed: {run['random_state']}",
            f"- Quick (smoke-test) mode: {run['quick_mode']}",
            "",
        ]
    else:
        lines += ["_No training run found._", ""]

    results = _read_csv("model_comparison.csv")
    lines += ["## Model comparison (official test set)"]
    if results is not None:
        lines += [_table(results, ["model_name", "threshold", "accuracy", "precision",
                                   "recall", "f1", "roc_auc", "pr_auc",
                                   "false_positive_rate", "false_negative_rate",
                                   "train_rows", "train_seconds"]), ""]
    else:
        lines += ["_No results yet. Run `python -m src.train`._", ""]

    fs = _read_csv("feature_selection_results.csv")
    if fs is not None:
        lines += ["## Feature-selection experiment",
                  _table(fs, ["model_name", "variant", "n_inputs", "f1_default_threshold",
                              "roc_auc", "train_seconds"]), ""]
    anomaly = _read_csv("anomaly_detection_results.csv")
    if anomaly is not None:
        lines += ["## Isolation Forest (anomaly detection)",
                  _table(anomaly, ["variant", "precision", "recall", "f1", "roc_auc",
                                   "false_positive_rate"]), ""]

    if prediction_summary:
        lines += ["## Predictions on the analysed file"]
        lines += [f"- {k}: {v}" for k, v in prediction_summary.items()]
        lines += [""]

    lines += ["## Limitations", LIMITATIONS]
    return "\n".join(lines)


def save_markdown_report(path: Path | None = None) -> Path:
    path = path or config.REPORTS_DIR / "analysis_report.md"
    path.write_text(build_markdown_report())
    return path
