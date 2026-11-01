# Technical README (for developers)

This document explains **how the system works internally**: the data flow,
each module's responsibilities and public functions, the design decisions and
their reasons, the on-disk artefacts, and how to extend or debug the project.
For the project overview, results and setup steps, see [README.md](README.md).

---

## 1. Design principles

| Principle | How it is enforced |
|---|---|
| **No target leakage** | Features are selected from an explicit *allow-list* (`config.FEATURE_COLUMNS`), not by dropping a few columns. `id`, `label`, `attack_cat` can never reach a model. |
| **Fit on training data only** | Every preprocessing / selection step lives *inside* an sklearn `Pipeline`, so `fit` only ever sees training rows. |
| **Test set touched once** | Hyperparameters are tuned with CV on training data; the decision threshold is chosen on a validation split of the training file; the official test file is used only for the final report. |
| **One source of truth** | The dashboard calls `src/` functions; it has no ML logic of its own. All settings are in `src/config.py`. |
| **No invented numbers** | Every metric shown is read from files written by an actual run (`reports/`). If a file is missing, the UI says so. |
| **Safe model loading** | Pickle files can execute code, so `load_bundle` only loads files inside `models/` whose SHA-256 matches `models/manifest.json`. |
| **Keep it explainable** | Plain functions, no class hierarchies, no framework magic. |

---

## 2. Directory layout

```
network-intrusion-detection/
├── app.py                     Streamlit dashboard (UI only)
├── requirements.txt           Dependency lower bounds
├── requirements-lock.txt      Exact versions used for the reported results
├── pyproject.toml             pytest configuration (pythonpath, testpaths)
├── .streamlit/config.toml     Dark theme + 200 MB upload limit
├── .vscode/                   Interpreter, pytest and run/debug configurations
├── data/
│   ├── README.md              How to download and place the dataset
│   ├── raw/                   Official CSVs go here (git-ignored)
│   └── demo/demo_traffic.csv  500-row excerpt for the dashboard
├── models/                    Trained bundles *.joblib + manifest.json (git-ignored)
├── reports/                   Metrics CSV/JSON + figures/ (generated)
├── notebooks/exploration.ipynb  Optional EDA (nothing depends on it)
├── docs/PROJECT_REPORT_OUTLINE.md
├── src/
│   ├── config.py              Paths, schema, seeds, sample sizes, grids
│   ├── data_loader.py         Read CSV, validate schema, coerce types, quality report
│   ├── preprocessing.py       ColumnTransformer factory
│   ├── models.py              The five pipelines
│   ├── train.py               CLI: tune, fit, threshold, evaluate, save
│   ├── evaluate.py            Metrics, threshold selection, matplotlib figures
│   ├── predict.py             Save/load bundles safely, predict on new data
│   ├── feature_selection.py   CLI: all vs MI vs tree-importance vs PCA
│   ├── anomaly_detection.py   CLI: Isolation Forest experiment
│   ├── explainability.py      Native/permutation importance, SHAP; CLI
│   ├── report.py              Markdown summary report builder
│   └── make_demo.py           CLI: build the demo CSV
└── tests/                     pytest suite (uses synthetic data, no dataset needed)
```

---

## 3. End-to-end data flow

```
            data/raw/UNSW_NB15_training-set.csv          data/raw/UNSW_NB15_testing-set.csv
                          │                                            │
              data_loader.load_labeled_dataset()           data_loader.load_labeled_dataset()
       (strip BOM, validate schema, coerce types, drop bad labels)     │
                          │                                            │
              split_features_target()  ──► X (42 allow-listed cols), y │
                          │                                            │
          train_test_split(stratified, 15%)                            │
             ┌────────────┴─────────────┐                              │
         X_fit (85%)               X_val (15%)                     X_test (100%)
             │                          │                              │
   stratified sample (20k)              │                              │
   GridSearchCV, 3-fold, F1             │                              │
             │ best params              │                              │
   refit Pipeline on X_fit              │                              │
   (row cap for KNN/SVM)                │                              │
             └──────────► pipeline ─────┤                              │
                                        ▼                              │
                         choose_threshold (max F1) ──► threshold       │
                                        │                              ▼
                                        └──────────► compute_metrics(y_test, scores, threshold)
                                                                       │
                     models/<key>.joblib  ◄── save_bundle    reports/model_comparison.csv
                                                             reports/test_scores.csv
                                                             reports/figures/*.png
```

Inference (dashboard or your own script):

```
CSV ─► read_csv_safely (size/row limits) ─► validate_schema(require_target=False)
    ─► prepare_for_prediction (allow-listed columns only) ─► bundle["pipeline"].predict_proba
    ─► attack_score ≥ threshold ─► prediction / predicted_label
```

---

## 4. Module reference

### `src/config.py`
Constants only. Important ones:

| Name | Purpose |
|---|---|
| `FEATURE_COLUMNS` | The 42 allowed inputs (39 numeric + `proto`, `service`, `state`). |
| `EXCLUDED_COLUMNS` | `id`, `label`, `attack_cat` - documented reasons in the file. |
| `VALIDATION_FRACTION` | Share of the training file held out for threshold selection (0.15). |
| `TUNING_SAMPLE_SIZE`, `CV_FOLDS` | Grid-search cost control (20,000 rows, 3 folds). |
| `MAX_TRAIN_ROWS` | Per-model row caps (KNN 30k, SVM 20k, others all rows). |
| `PARAM_GRIDS` | Hyperparameter grids, keys prefixed with `model__`. |
| `FS_*`, `PCA_VARIANCE` | Feature-selection experiment settings. |
| `IFOREST_*` | Isolation Forest settings. |
| `PERMUTATION_*`, `SHAP_BACKGROUND_SIZE` | Explainability cost control. |
| `MAX_UPLOAD_MB`, `MAX_UPLOAD_ROWS` | Dashboard upload limits. |

### `src/data_loader.py`
| Function | Description |
|---|---|
| `read_csv_safely(source, max_rows)` | Reads a path or file-like object with `utf-8-sig`, normalises column names (strip BOM/whitespace, lower-case), raises `DatasetError` for missing/empty/unparseable/too-large files. Never executes content. |
| `validate_schema(df, require_target)` | Returns a `SchemaReport` (missing, extra columns, invalid labels, human-readable messages, `is_valid`). Extra columns are allowed and ignored. |
| `coerce_feature_types(df)` | Numeric columns → `float64` (invalid strings and ±inf → NaN, counted); categoricals → stripped lower-case strings. |
| `load_labeled_dataset(source)` | Read + validate (target required) + coerce + drop rows whose label is not 0/1. |
| `split_features_target(df)` | Allow-list selection → `(X, y)`. |
| `data_quality_summary(df)` | Missing values, negatives, duplicates, conflicting duplicates, class counts, categories. |
| `train_test_overlap(train, test)` | Row-hash overlap between files and unseen categories. |

### `src/preprocessing.py`
`build_preprocessor(scale_numeric)` returns a `ColumnTransformer`:

| Branch | Steps | Why |
|---|---|---|
| numeric (scaled models) | `SimpleImputer(median)` → `signed_log1p` → `StandardScaler` | Byte/packet counts span ~9 orders of magnitude; log compresses them so distance/linear models are not dominated by a few huge columns. Median is robust to outliers. |
| numeric (tree models) | `SimpleImputer(median)` | Trees split on thresholds; monotone transforms and scaling change nothing. |
| categorical | `SimpleImputer(constant="missing")` → `OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)` | One column per category with no fake ordering. Categories seen < 20 times in training (most of `proto`'s 133 values) and categories never seen at all share an "infrequent" column, so new data never crashes the model. |

Output width depends on how many rows the encoder saw (rare categories cross
the 20-occurrence threshold more often in bigger samples). In the reported run:
188 columns for LR/RF/XGBoost (all 149,039 fit rows: 39 numeric + 149 one-hot),
98 for KNN (30k rows) and 67 for SVM (20k rows).

Helpers: `transformed_feature_names(pipeline)`, `original_feature_of("proto_tcp") -> "proto"`.

### `src/models.py`
`build_pipeline(key, y_train)` → `Pipeline([("preprocess", ...), ("model", ...)])`.

| Key | Estimator | Scaled | Imbalance handling |
|---|---|---|---|
| `logistic_regression` | `LogisticRegression(max_iter=2000)` | yes | `class_weight="balanced"` |
| `knn` | `KNeighborsClassifier` | yes | none available (documented limitation) |
| `svm` | `SVC` (+ `CalibratedClassifierCV(method="sigmoid", ensemble=False)` for the final fit) | yes | `class_weight="balanced"` |
| `random_forest` | `RandomForestClassifier` | no | `class_weight="balanced"` |
| `xgboost` | `XGBClassifier(tree_method="hist")` | no | `scale_pos_weight = n_benign / n_attack` |

`add_probability_calibration` wraps the SVM so it has `predict_proba` (Platt
scaling). `SVC(probability=True)` is deprecated in scikit-learn 1.9, so the
calibrator is used instead. `predict_scores(pipeline, X)` returns
`predict_proba(X)[:, 1]`.

> In this dataset attacks are the **majority** (68% of training rows), so
> "balanced" weighting *up-weights benign traffic*. That reduces the model's
> bias toward predicting "attack" and therefore targets false alarms.

### `src/train.py` (CLI)
`python -m src.train [--models KEY ...] [--quick]`

- `load_and_split()` - loads both files, writes `reports/data_quality.json`, returns fit/val/test sets.
- `stratified_sample(X, y, n)` - class-ratio-preserving subsample (no-op if `n` is `None`).
- `tune_and_fit(key, X_fit, y_fit, tuning_size, max_rows, param_grid=None)` - `GridSearchCV(scoring="f1", refit=False)` on a sample, then a fresh pipeline is refit with the best params on (capped) `X_fit`. Returns `(pipeline, best_params, cv_f1, n_rows, tuning_s, training_s)`.
- `run_training(keys, quick)` - per model: tune/fit → threshold on validation → one evaluation on test → `save_bundle`. Then `_write_reports` merges into existing CSVs, so retraining one model keeps the others' rows.
- `--quick` uses 3k tuning rows and ≤10k training rows; the run is flagged `quick_mode: true` in `run_config.json` and the dashboard shows a red banner.

### `src/evaluate.py`
- `compute_metrics(y, scores, threshold)` → accuracy, attack-class precision/recall/F1, ROC-AUC, PR-AUC (average precision), FPR, FNR, tn/fp/fn/tp.
- `choose_threshold(y_val, val_scores, "max_f1")` → threshold maximising F1 on the precision-recall curve.
- `threshold_table(y, scores)` → metrics at thresholds 0.05 … 0.95.
- `plot_*` → PNGs in `reports/figures/`.

### `src/predict.py`
- `save_bundle(pipeline, key, threshold, metadata, models_dir)` - `joblib.dump(compress=3)` + updates `manifest.json` with SHA-256.
- `load_bundle(key, models_dir)` - path must resolve inside `models_dir`; checksum must match manifest; adds `version_warnings` if scikit-learn/xgboost versions differ from training.
- `prepare_for_prediction(df)` - validates schema (no target needed) and returns allow-listed columns.
- `predict_dataframe(bundle, df, threshold=None)` - returns `df` + `attack_score`, `prediction`, `predicted_label`.
- `predictions_to_csv(df)` - bytes for download.

Bundle structure:
```python
{
  "pipeline": Pipeline,           # preprocess -> model, fitted
  "model_key": "xgboost",
  "threshold": 0.316,             # chosen on validation
  "feature_columns": [...42 names...],
  "metadata": {"train_rows", "best_params", "cv_f1_tuning", "validation_threshold",
               "test_metrics", "quick_mode", "saved_at", "library_versions"},
}
```

### `src/feature_selection.py` (CLI)
`python -m src.feature_selection [--models ...] [--sample-size N] [--top-k K]`

Inserts a `("select", ...)` step between preprocessing and the model:
`SelectKBest(mutual_info_classif)`, `SelectFromModel(RandomForest, max_features=K, threshold=-inf)`
or `PCA(n_components=0.95)` (PCA variant always uses the scaled preprocessor).
Each variant reuses the tuned hyperparameters from `model_comparison.csv`, is
fit on a 30k stratified sample (capped by `MAX_TRAIN_ROWS`), and is scored on
the full test set with threshold-free ROC-AUC/PR-AUC plus F1 at the model's
default decision rule. SVM is not calibrated here (decision-function scores
are enough for AUC). Outputs `feature_selection_results.csv`,
`selected_features.json`, `figures/feature_selection.png`.

### `src/anomaly_detection.py` (CLI)
Two Isolation Forest variants inside `Pipeline(preprocess(unscaled) -> IsolationForest)`:
1. **unsupervised** - fit on 50k random training rows, no labels; threshold = sklearn's `offset_` from `contamination="auto"`.
2. **benign-only** - fit on 50k benign training rows; threshold = 95th percentile of anomaly scores on benign *validation* rows (≈5% target false-alarm rate).

Anomaly score = `-score_samples(X)` (higher = more unusual). Outputs
`anomaly_detection_results.csv` and per-category detection rates in
`anomaly_detection_by_category.csv`. Completely independent of `train.py`.

### `src/explainability.py`
- `native_importance(bundle)` - tree importances or |LR coefficients|, one-hot columns summed back to the original feature, normalised to sum to 1. `None` for KNN/SVM.
- `permutation_importance_table(bundle, X, y)` - shuffles **original** columns through the whole pipeline; scoring = ROC-AUC drop. Works for all five models.
- `explain_record(bundle, record, background)` - SHAP for one row. `TreeExplainer` for RF (units: probability) and XGBoost (units: log-odds), `LinearExplainer` with an `Independent` masker for LR (log-odds). One-hot contributions are summed back to the original column. Raises `NotImplementedError` for KNN/SVM (KernelExplainer is impractically slow).
- CLI `python -m src.explainability` - permutation importance on a 5k validation sample for every trained model (~8 min on 2 cores; KNN/SVM dominate). Outputs `permutation_importance.csv`, `native_importance.csv`, `figures/permutation_importance.png`.

### `src/report.py`
`build_markdown_report(prediction_summary=None)` assembles dataset facts,
configuration, all result tables and fixed limitations text from `reports/`.

### `app.py`
Sidebar: page, data source (demo / upload), model, threshold slider (default =
bundle threshold). Caching: `st.cache_resource` for bundles (keyed by file
mtime so retraining invalidates it), `st.cache_data` for predictions and report
CSVs. Pages are plain functions (`page_overview`, `page_traffic`,
`page_detection`, `page_comparison`, `page_explainability`, `page_reports`).

---

## 5. Generated artefacts

| File | Written by | Content |
|---|---|---|
| `models/<key>.joblib` | train | Fitted bundle (see above) |
| `models/manifest.json` | train | SHA-256 per bundle |
| `reports/data_quality.json` | train | Quality summary for both files + overlap |
| `reports/run_config.json` | train | Row counts, sampling, grids, versions, quick flag |
| `reports/model_comparison.csv` | train | One row per model (metrics at validation threshold, metrics at 0.5, timings, params) |
| `reports/test_scores.csv` | train | `y_true`, `attack_cat`, `score_<key>` for every test row (for curves) |
| `reports/figures/*.png` | train / experiments | ROC, PR, confusion matrices, metric bars, feature selection, permutation importance |
| `reports/feature_selection_results.csv`, `selected_features.json` | feature_selection | Variant comparison |
| `reports/anomaly_detection_results.csv`, `anomaly_detection_by_category.csv` | anomaly_detection | Isolation Forest results |
| `reports/permutation_importance.csv`, `native_importance.csv` | explainability | Global importance |

---

## 6. Reproducibility

- Seed `config.RANDOM_STATE = 42` is passed to every splitter, sampler and estimator.
- Exact library versions are in `requirements-lock.txt` and inside every bundle.
- Runtimes measured on a 2-core / 8 GB Linux machine: training ≈ 5 min,
  permutation importance ≈ 8 min, feature-selection experiment ≈ 5 min,
  Isolation Forest ≈ 10 s.
- Multi-threaded estimators (RF, XGBoost, KNN) can produce tiny run-to-run
  differences on different hardware.

To regenerate everything after changing `config.py`:
```bash
python -m src.train
python -m src.feature_selection
python -m src.anomaly_detection
python -m src.explainability
```

---

## 7. Testing

`python -m pytest` (configured in `pyproject.toml`) runs 44 tests in ~5 s using a
synthetic 600-row table with the UNSW-NB15 schema (`tests/conftest.py`), so
no dataset is needed.

| File | Covers |
|---|---|
| `test_data_validation.py` | BOM handling, name normalisation, missing/extra columns, target requirement, wrong schema, empty/header-only/missing files, row limit, invalid numbers, invalid labels, quality summary, overlap |
| `test_preprocessing.py` | Leakage (allow-list, injected label copies), train-only statistics, imputation, unseen categories, consistent width, scaling, log transform, name mapping, column-order independence |
| `test_predictions.py` | All five models train and score, save/load round-trip, missing model, tampered file, path traversal, unlabeled prediction, labels ignored, threshold behaviour, incompatible file, CSV export, metric arithmetic, threshold selection |

---

## 8. Extending the project

**Add a model**
1. Add a `ModelSpec` and a branch in `_make_classifier` (`src/models.py`).
2. Add a grid to `config.PARAM_GRIDS`, a display name to `config.MODEL_DISPLAY_NAMES`, and optionally a row cap.
3. Add a tiny grid to `TINY_GRIDS` in `tests/test_predictions.py`.
4. `python -m src.train --models <new_key>` - other models' results are kept.

**Change preprocessing** - edit `build_preprocessor`; retrain all models
(bundles contain the old preprocessor).

**Use a different dataset version** - update `NUMERIC_FEATURES` /
`CATEGORICAL_FEATURES`; the validator will then report any mismatch.

**Live traffic** - out of scope. You would need a flow exporter that produces
the same 42 features (UNSW-NB15 used Argus and Bro/Zeek); the models cannot
consume raw packets.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `File not found: data/raw/...` | Download the dataset, see `data/README.md`. |
| `No trained model ... found` | `python -m src.train` |
| `... checksum does not match` | The model file was changed or copied from elsewhere. Retrain. |
| Version warning in the sidebar | Your scikit-learn/xgboost differ from training. Retrain, or install `requirements-lock.txt`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root (the folder containing `app.py`). |
| Training too slow | Lower `TUNING_SAMPLE_SIZE` / `MAX_TRAIN_ROWS` in `config.py`, or `--quick` for a smoke test. |
| Upload rejected | Check the message: missing columns are listed by name; size limit 200 MB / 300k rows. |
