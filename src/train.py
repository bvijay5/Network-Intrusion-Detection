"""
Train, tune, evaluate and save all five classifiers.

Run from the project folder:
    python -m src.train                    # all five models
    python -m src.train --models xgboost   # just one (or several) models
    python -m src.train --quick            # small samples, for a fast smoke test

What happens, step by step:
  1. Load + validate the official training and testing CSV files.
  2. Write a data-quality report (missing values, duplicates, overlap ...).
  3. Split the TRAINING file into  fit (85%)  and  validation (15%).
  4. For each model:
       a. tune hyperparameters with GridSearchCV on a sample of the fit part
       b. re-fit the best configuration on the fit part (row cap if configured)
       c. choose the decision threshold on the validation part
       d. evaluate ONCE on the untouched official test file
       e. save the fitted pipeline to models/<model>.joblib
  5. Save the comparison table, test-set scores and figures to reports/.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src import config
from src.data_loader import (
    data_quality_summary,
    load_labeled_dataset,
    split_features_target,
    train_test_overlap,
)
from src.evaluate import (
    choose_threshold,
    compute_metrics,
    plot_confusion_matrices,
    plot_metric_bars,
    plot_pr_curves,
    plot_roc_curves,
)
from src.models import (
    MODEL_KEYS,
    MODEL_SPECS,
    add_probability_calibration,
    build_pipeline,
    predict_scores,
)
from src.predict import library_versions, save_bundle

# Models that are single-threaded internally benefit from parallel CV folds.
_PARALLEL_SEARCH = {"logistic_regression", "svm"}


def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int | None,
                      seed: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.Series]:
    """Return at most n rows, keeping the benign/attack ratio unchanged."""
    if n is None or n >= len(X):
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return X_s, y_s


def load_and_split(verbose: bool = True) -> dict:
    """Load both official files and create the fit / validation / test sets."""
    train_df = load_labeled_dataset(config.TRAIN_FILE)
    test_df = load_labeled_dataset(config.TEST_FILE)

    quality = {
        "train": data_quality_summary(train_df),
        "test": data_quality_summary(test_df),
        "train_test_overlap": train_test_overlap(train_df, test_df),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))
    if verbose:
        print(f"Training file: {len(train_df):,} rows | Testing file: {len(test_df):,} rows")
        ov = quality["train_test_overlap"]
        print(f"Test rows whose features also occur in training: "
              f"{ov['test_rows_seen_in_train']:,} ({ov['overlap_fraction']:.1%}) - reported, not removed")

    X_train_all, y_train_all = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_all, y_train_all,
        test_size=config.VALIDATION_FRACTION,
        stratify=y_train_all,
        random_state=config.RANDOM_STATE,
    )
    return {
        "X_fit": X_fit, "y_fit": y_fit,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "test_attack_cat": test_df.get("attack_cat"),
        "quality": quality,
    }


def tune_and_fit(key: str, X_fit: pd.DataFrame, y_fit: pd.Series,
                 tuning_size: int | None, max_rows: int | None,
                 param_grid: dict | None = None) -> tuple:
    """Grid-search one model on a sample, then re-fit the best settings.

    Returns:
        (fitted pipeline, best params, cv f1, rows used for final fit,
         tuning seconds, training seconds)
    """
    grid = config.PARAM_GRIDS[key] if param_grid is None else param_grid
    X_tune, y_tune = stratified_sample(X_fit, y_fit, tuning_size)

    t0 = time.perf_counter()
    search = GridSearchCV(
        build_pipeline(key, y_tune.to_numpy()),
        param_grid=grid,
        scoring="f1",  # F1 of the attack class
        cv=StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        n_jobs=-1 if key in _PARALLEL_SEARCH else 1,
        refit=False,
    )
    search.fit(X_tune, y_tune)
    tuning_seconds = time.perf_counter() - t0

    X_train, y_train = stratified_sample(X_fit, y_fit, max_rows)
    final = build_pipeline(key, y_train.to_numpy())
    final.set_params(**search.best_params_)
    final = add_probability_calibration(final, key)  # SVM only

    t0 = time.perf_counter()
    final.fit(X_train, y_train)
    training_seconds = time.perf_counter() - t0
    return (final, search.best_params_, float(search.best_score_), len(X_train),
            tuning_seconds, training_seconds)


def run_training(model_keys: list[str], quick: bool = False) -> pd.DataFrame:
    """Train every requested model and write all results to reports/."""
    data = load_and_split()
    tuning_size = 3_000 if quick else config.TUNING_SAMPLE_SIZE
    results, test_scores = [], {}

    for key in model_keys:
        name = config.MODEL_DISPLAY_NAMES[key]
        cap = config.MAX_TRAIN_ROWS.get(key)
        if quick:
            cap = min(cap or 10_000, 10_000)
        print(f"\n=== {name} ===")
        pipe, params, cv_f1, n_rows, tune_s, train_s = tune_and_fit(
            key, data["X_fit"], data["y_fit"], tuning_size, cap)
        print(f"best params {params} | CV F1 {cv_f1:.4f} | fit on {n_rows:,} rows "
              f"in {train_s:.1f}s (tuning {tune_s:.1f}s)")

        # Threshold chosen on validation data only.
        val_scores = predict_scores(pipe, data["X_val"])
        threshold = choose_threshold(data["y_val"].to_numpy(), val_scores)

        # The ONE evaluation on the official test file.
        t0 = time.perf_counter()
        scores = predict_scores(pipe, data["X_test"])
        predict_s = time.perf_counter() - t0
        test_scores[key] = scores

        tuned = compute_metrics(data["y_test"], scores, threshold)
        default = compute_metrics(data["y_test"], scores, 0.5)
        print(f"test @ threshold {threshold:.3f}: F1 {tuned['f1']:.4f}  "
              f"precision {tuned['precision']:.4f}  recall {tuned['recall']:.4f}  "
              f"ROC-AUC {tuned['roc_auc']:.4f}")

        results.append({
            "model_key": key,
            "model_name": name,
            **tuned,
            "f1_at_0.5": default["f1"],
            "precision_at_0.5": default["precision"],
            "recall_at_0.5": default["recall"],
            "cv_f1_tuning": cv_f1,
            "train_rows": n_rows,
            "train_seconds": train_s,
            "tuning_seconds": tune_s,
            "predict_seconds_test": predict_s,
            "predict_ms_per_1k_rows": 1000 * predict_s / len(data["X_test"]) * 1000,
            "imbalance_handling": MODEL_SPECS[key].handles_imbalance_with,
            "best_params": json.dumps({k.replace("model__", ""): v for k, v in params.items()}),
        })

        save_bundle(pipe, key, threshold, metadata={
            "train_rows": n_rows,
            "best_params": {k.replace("model__", ""): v for k, v in params.items()},
            "cv_f1_tuning": cv_f1,
            "validation_threshold": threshold,
            "test_metrics": {k: float(v) for k, v in tuned.items()},
            "quick_mode": quick,
        })

    results_df = pd.DataFrame(results)
    _write_reports(results_df, test_scores, data, model_keys, quick, tuning_size)
    return results_df


def _write_reports(results_df: pd.DataFrame, test_scores: dict, data: dict,
                   model_keys: list[str], quick: bool, tuning_size) -> None:
    """Persist the comparison table, raw scores, figures and run configuration."""
    reports = config.REPORTS_DIR
    figures = config.FIGURES_DIR
    figures.mkdir(parents=True, exist_ok=True)

    # Merge with an existing table so training one model doesn't erase others.
    table_path = reports / "model_comparison.csv"
    if table_path.exists():
        old = pd.read_csv(table_path)
        old = old[~old["model_key"].isin(results_df["model_key"])]
        results_df = pd.concat([old, results_df], ignore_index=True)
    order = {k: i for i, k in enumerate(MODEL_KEYS)}
    results_df = results_df.sort_values("model_key", key=lambda s: s.map(order))
    results_df.to_csv(table_path, index=False)

    scores_path = reports / "test_scores.csv"
    scores_df = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame(
        {"y_true": data["y_test"].to_numpy()})
    if len(scores_df) != len(data["y_test"]):  # stale file from another dataset
        scores_df = pd.DataFrame({"y_true": data["y_test"].to_numpy()})
    if data["test_attack_cat"] is not None:
        scores_df["attack_cat"] = data["test_attack_cat"].to_numpy()
    for key, s in test_scores.items():
        scores_df[f"score_{key}"] = s
    scores_df.to_csv(scores_path, index=False)

    all_scores = {c.replace("score_", ""): scores_df[c].to_numpy()
                  for c in scores_df.columns if c.startswith("score_")}
    y = scores_df["y_true"].to_numpy()
    plot_roc_curves(y, all_scores, figures / "roc_curves.png")
    plot_pr_curves(y, all_scores, figures / "pr_curves.png")
    plot_confusion_matrices(results_df, figures / "confusion_matrices.png")
    plot_metric_bars(results_df, figures / "metric_comparison.png")

    run_info = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "quick_mode": quick,
        "models_trained_this_run": model_keys,
        "random_state": config.RANDOM_STATE,
        "rows": {"fit": len(data["X_fit"]), "validation": len(data["X_val"]),
                 "test": len(data["X_test"])},
        "validation_fraction": config.VALIDATION_FRACTION,
        "tuning_sample_size": tuning_size,
        "cv_folds": config.CV_FOLDS,
        "max_train_rows": config.MAX_TRAIN_ROWS,
        "threshold_strategy": config.THRESHOLD_STRATEGY,
        "param_grids": config.PARAM_GRIDS,
        "library_versions": library_versions(),
    }
    (reports / "run_config.json").write_text(json.dumps(run_info, indent=2, default=str))
    print(f"\nSaved results to {table_path.relative_to(config.PROJECT_ROOT)} "
          f"and figures to {figures.relative_to(config.PROJECT_ROOT)}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NIDS classifiers.")
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--quick", action="store_true",
                        help="use small samples for a fast smoke test (results not reportable)")
    args = parser.parse_args()
    np.random.seed(config.RANDOM_STATE)
    results = run_training(args.models, quick=args.quick)
    cols = ["model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
            "false_positive_rate", "train_seconds"]
    print("\n" + results[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
