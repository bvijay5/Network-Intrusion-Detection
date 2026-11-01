# Explainable ML-Based Network Intrusion Detection System Using UNSW-NB15

A machine-learning Network Intrusion Detection System (NIDS) that classifies
network-flow records as **benign** or **potentially malicious**, compares five
traditional ML algorithms, explains its predictions, and provides an
interactive Streamlit dashboard for offline CSV analysis.

> **Scope:** offline analysis of CSV flow records from public / authorised
> datasets. It does **not** capture or monitor live network traffic.
> **No deep learning** is used (course constraint).

Developer documentation: [TECHNICAL_README.md](TECHNICAL_README.md) ·
Report outline: [docs/PROJECT_REPORT_OUTLINE.md](docs/PROJECT_REPORT_OUTLINE.md) ·
Dataset setup: [data/README.md](data/README.md)

---

## Contents
1. [Abstract](#1-abstract)
2. [Problem statement](#2-problem-statement)
3. [Motivation](#3-motivation)
4. [Objectives](#4-objectives)
5. [Dataset](#5-dataset)
6. [System architecture](#6-system-architecture)
7. [Algorithms used](#7-algorithms-used)
8. [Preprocessing methodology](#8-preprocessing-methodology)
9. [Evaluation methodology](#9-evaluation-methodology)
10. [Results](#10-results)
11. [Feature-selection experiments](#11-feature-selection-experiments)
12. [Anomaly detection (Isolation Forest)](#12-anomaly-detection-isolation-forest)
13. [Explainability](#13-explainability)
14. [Installation and execution](#14-installation-and-execution)
15. [Dashboard functionality](#15-dashboard-functionality)
16. [Testing](#16-testing)
17. [Known limitations](#17-known-limitations)
18. [Future improvements](#18-future-improvements)
19. [References](#19-references)

---

## 1. Abstract

Signature-based intrusion detection cannot recognise attacks it has no rule
for. This project trains supervised classifiers on the UNSW-NB15 benchmark to
learn the statistical difference between normal and attack traffic from 42
flow features (durations, byte and packet counts, TTLs, connection counts,
protocol, service and state). Five algorithms - Logistic Regression, K-Nearest
Neighbors, Support Vector Machine, Random Forest and XGBoost - are tuned with
cross-validation and evaluated on the official held-out test file using
attack-class precision, recall, F1, ROC-AUC and PR-AUC. Feature selection
(mutual information, tree importance), PCA and an Isolation Forest anomaly
detector are studied as additional experiments. Permutation importance and
SHAP explain which features drive predictions. In our run, Random Forest and
XGBoost performed best (ROC-AUC ≈ 0.98, attack F1 ≈ 0.89, attack recall
≈ 98-99%), but all models raised false alarms on roughly 29-34% of benign test
records, which we trace to a distribution shift between the official training
and testing files.

## 2. Problem statement

Given one network-flow record described by 42 features, predict whether it is
**benign (0)** or **potentially malicious (1)**, using only traditional
machine-learning methods, in a way that is reproducible, fair to compare
across algorithms, and explainable to a security analyst.

## 3. Motivation

- New and modified attacks evade fixed signatures; ML models generalise from examples.
- Raw accuracy hides the errors that matter: a **missed attack** (false negative) is dangerous, while too many **false alarms** (false positives) overwhelm analysts.
- An alert is only useful if an analyst can see *why* it fired, hence explainability.
- UNSW-NB15 is a modern, widely used, labelled benchmark that replaces the outdated KDD'99.

## 4. Objectives

1. Build a leakage-free, reusable preprocessing and training pipeline.
2. Train, tune and fairly compare five classifiers on the same data.
3. Evaluate with attack-focused metrics and explicit error analysis.
4. Compare feature-selection methods and PCA.
5. Test Isolation Forest as a label-free alternative.
6. Explain predictions globally and per record.
7. Deliver a tested dashboard for CSV analysis, prediction and export.

## 5. Dataset

**UNSW-NB15**, created by the Cyber Range Lab of UNSW Canberra (2015) with
the IXIA PerfectStorm tool, mixing real normal traffic with synthetic attacks.
Official source: <https://research.unsw.edu.au/projects/unsw-nb15-dataset> [1, 2].

We use the official partitioned files:

| File | Records | Benign | Attack | Attack share |
|---|---:|---:|---:|---:|
| `UNSW_NB15_training-set.csv` | 175,341 | 56,000 | 119,341 | 68.1% |
| `UNSW_NB15_testing-set.csv` | 82,332 | 37,000 | 45,332 | 55.1% |

Each file has 45 columns: `id`, 42 features, `attack_cat`, `label`.

**Important features** (meanings from the dataset documentation [1]):

| Feature | Meaning |
|---|---|
| `dur` | Flow duration (seconds) |
| `proto`, `service`, `state` | Transport protocol (tcp, udp, ...), application service (http, dns, `-` = none), connection state (FIN, INT, CON, ...) |
| `spkts`, `dpkts` / `sbytes`, `dbytes` | Packets / bytes sent from source and from destination |
| `rate`, `sload`, `dload` | Packets per second; source and destination bits per second |
| `sttl`, `dttl` | Source-to-destination and destination-to-source IP time-to-live |
| `sloss`, `dloss` | Retransmitted or dropped packets |
| `sinpkt`, `dinpkt`, `sjit`, `djit` | Inter-packet arrival times and jitter |
| `swin`, `dwin`, `stcpb`, `dtcpb`, `tcprtt`, `synack`, `ackdat` | TCP window sizes, base sequence numbers and handshake timings |
| `smean`, `dmean` | Mean packet size in each direction |
| `trans_depth`, `response_body_len`, `ct_flw_http_mthd` | HTTP request depth, response size, number of HTTP methods |
| `is_ftp_login`, `ct_ftp_cmd` | FTP login flag and FTP command count |
| `ct_srv_src`, `ct_dst_ltm`, `ct_src_ltm`, `ct_dst_src_ltm`, ... | Counts of connections with the same service / address / port within the last 100 connections |
| `ct_state_ttl` | Count based on state and TTL ranges |
| `is_sm_ips_ports` | 1 if source and destination IP and port are equal |
| `attack_cat` | Attack family: Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms |
| `label` | **Target**: 0 = normal, 1 = attack |

**Data-quality findings** (from `reports/data_quality.json`, our run):

- No missing or infinite values; no negative numeric values.
- The training file starts with an invisible UTF-8 byte-order mark (`﻿`) before `id`; the loader strips it.
- 74,072 duplicate rows in training and 28,380 in testing (same features and label); 229 groups of identical features with *conflicting* labels in training.
- 8,541 test rows (10.4%) have exactly the same feature values as some training row.
- The test file contains two `state` values (`ACC`, `CLO`) never seen in training.

Duplicates and overlap are **reported, not removed**: the official split is
the benchmark that published work uses, and deleting rows would change it.

**Columns never used as inputs:** `id` (row counter; rows are grouped by
class, so it would leak the label), `label` (the target), `attack_cat`
(derived from the target: `Normal` ⇔ `label = 0`). Features are chosen from an
explicit allow-list, so these cannot leak in by accident.

## 6. System architecture

```
                ┌──────────────────────────── app.py (Streamlit UI) ───────────────────────────┐
                │ Overview · Traffic Analysis · Intrusion Detection · Model Comparison ·       │
                │ Explainability · Reports                                                     │
                └───────────────┬──────────────────────┬──────────────────────┬────────────────┘
                                │ calls                │ reads                │ calls
           ┌────────────────────▼───┐        ┌─────────▼─────────┐   ┌────────▼─────────┐
           │ src/predict.py         │        │ reports/*.csv,    │   │ src/explain-     │
           │ safe load + predict    │        │ *.json, figures/  │   │ ability.py       │
           └────────────┬───────────┘        └─────────▲─────────┘   └──────────────────┘
                        │ loads                        │ writes
           ┌────────────▼───────────┐        ┌─────────┴──────────────────────────────┐
           │ models/*.joblib        │◄───────│ src/train.py  feature_selection.py      │
           │ (+ SHA-256 manifest)   │ saves  │ anomaly_detection.py  explainability.py │
           └────────────────────────┘        └─────────▲──────────────────────────────┘
                                                       │ uses
           ┌───────────────────────────────────────────┴────────────────────────────────┐
           │ src/data_loader.py → src/preprocessing.py → src/models.py → src/evaluate.py │
           │ (validate)           (ColumnTransformer)    (5 Pipelines)   (metrics/plots) │
           └────────────────────────────────────────────────────────────────────────────┘
                                   all settings in src/config.py
```

Each saved model is a complete `Pipeline(preprocess → classifier)`, so the
dashboard never re-implements preprocessing.

## 7. Algorithms used

| Algorithm | Idea in one line | Why it is here | Main limitation | Grid searched |
|---|---|---|---|---|
| **Logistic Regression** [4] | Weighted sum of features → sigmoid → probability | Fast, interpretable baseline (coefficients) | Only linear decision boundaries | `C` ∈ {0.01, 0.1, 1, 10} |
| **K-Nearest Neighbors** [5] | Label = vote of the *k* most similar training records | No training phase; captures local patterns | Prediction is slow (compares to every stored row); needs scaling; no class weights | `k` ∈ {3, 5, 11, 21}, weights ∈ {uniform, distance} |
| **Support Vector Machine** [6] | Maximum-margin boundary; RBF kernel allows curved boundaries | Strong on medium-sized, scaled data | Training is roughly O(n²): we cap it at 20k rows; scores need calibration [9] | kernel ∈ {linear, rbf}, `C` ∈ {0.1, 1, 10} |
| **Random Forest** [7] | Many decision trees on bootstrap samples, majority vote | Handles mixed, skewed features without scaling; robust | Large model files; less interpretable than one tree | trees ∈ {100, 200}, depth ∈ {12, 20, None}, min split ∈ {2, 10} |
| **XGBoost** [8] | Trees added one at a time, each correcting the previous errors (gradient boosting) | Usually the strongest tabular model | More hyperparameters; can overfit without tuning | trees ∈ {200, 400}, depth ∈ {4, 6, 8}, learning rate ∈ {0.05, 0.1} |

**Class imbalance.** Attacks are the *majority* (68% of training rows). LR,
SVM and RF use `class_weight="balanced"`; XGBoost uses
`scale_pos_weight = #benign / #attack`. Here this *up-weights benign traffic*,
which pushes the models away from flagging everything and targets false
alarms. KNN has no class-weight option; this is noted as a limitation.

**Hyperparameter tuning.** `GridSearchCV`, 3-fold stratified CV on a 20,000-row
stratified sample of the training data, scored by **attack-class F1**. The best
settings are then refit on the full training portion (KNN capped at 30,000
rows and SVM at 20,000 rows, for laptop-friendly run times; caps are in
`src/config.py` and reported in the results table).

## 8. Preprocessing methodology

All steps are inside a scikit-learn `ColumnTransformer` within each model's
`Pipeline`, so they are **fitted on training rows only** and applied
identically at prediction time.

| Column type | Steps | Reason |
|---|---|---|
| Numeric (LR, KNN, SVM) | median imputation → `sign(x)·log(1+|x|)` → standardisation | Byte/packet counts range from 0 to ~10⁹; the log stops a few huge columns dominating distances and dot products |
| Numeric (RF, XGBoost) | median imputation | Trees split on thresholds, so scaling is unnecessary |
| Categorical (`proto`, `service`, `state`) | constant imputation → one-hot encoding; categories seen < 20 times, or never seen, go to an "infrequent" column | No artificial ordering; new categories at prediction time do not crash the model |

The loader also: strips the byte-order mark; normalises column names;
converts invalid numbers and ±∞ to missing (then imputed); drops rows whose
label is not 0/1 (and reports how many).

## 9. Evaluation methodology

```
Official training file (175,341)
 ├─ 85% fit set ──────► tune (3-fold CV on 20k sample) ─► refit best model
 └─ 15% validation ───► choose decision threshold (max attack-class F1)
Official testing file (82,332) ──► evaluated ONCE per model, same rows for all
```

- **Positive class = attack.** Precision, recall and F1 are for attacks.
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC (average precision), confusion matrix, false-positive rate (benign flagged), false-negative rate (attacks missed), training time, prediction time.
- The **threshold** is chosen on validation data, never on the test set. Metrics at the default 0.5 threshold are also reported for comparison.
- Every number in this README was produced by the commands in [§14](#14-installation-and-execution) (configuration in `reports/run_config.json`; environment in `requirements-lock.txt`).

## 10. Results

> Actual results from our run on 24 Oct 2026 (Python 3.13, scikit-learn 1.9.1,
> XGBoost 3.4.1, 2-core CPU). Your numbers may differ slightly on other
> hardware or library versions. Source: `reports/model_comparison.csv`.

### 10.1 Main comparison (official test set, validation-chosen threshold)

| Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | False-alarm rate | Missed attacks | Train rows | Train time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.308 | 0.839 | 0.783 | 0.979 | 0.870 | 0.967 | 0.976 | 33.3% | 952 | 149,039 | 21.8 s |
| K-Nearest Neighbors | 0.472 | 0.851 | 0.797 | 0.978 | 0.879 | 0.969 | 0.972 | 30.5% | 982 | 30,000 | 0.2 s |
| SVM (RBF) | 0.321 | 0.834 | 0.777 | 0.978 | 0.866 | 0.945 | 0.937 | 34.4% | 981 | 20,000 | 19.1 s |
| **Random Forest** | 0.360 | **0.863** | 0.806 | **0.991** | **0.889** | **0.984** | **0.988** | 29.3% | **420** | 149,039 | 14.4 s |
| **XGBoost** | 0.316 | 0.862 | **0.808** | 0.983 | 0.887 | 0.982 | 0.987 | **28.7%** | 760 | 149,039 | 13.4 s |

Best hyperparameters found: LR `C=10`; KNN `k=21, weights=distance`;
SVM `rbf, C=0.1`; RF `100 trees, max_depth=None, min_samples_split=10`;
XGBoost `400 trees, max_depth=8, learning_rate=0.1`.

Prediction time per 1,000 records: LR 4 ms, XGBoost 7 ms, RF 8 ms, KNN 102 ms,
SVM 348 ms.

Figures: `reports/figures/roc_curves.png`, `pr_curves.png`,
`confusion_matrices.png`, `metric_comparison.png`.

### 10.2 Interpretation

- **Tree ensembles win.** Random Forest and XGBoost lead on every ranking metric and are also among the fastest at prediction time. Random Forest misses the fewest attacks (420 of 45,332).
- **Errors are mostly false alarms.** Every model catches ~98-99% of attacks but flags 29-34% of benign records (10.6k-12.7k false positives vs 420-982 false negatives).
- **Distribution shift between the official files.** Cross-validation F1 on training data was ~0.95-0.96 for all models, but test F1 is ~0.87-0.89. The two official files were sampled differently, so a model tuned on one sees noticeably different benign traffic in the other.
- **The validation threshold does not transfer.** Because validation data comes from the training file, the F1-optimal threshold there (0.31-0.47) is lower than 0.5, and on the test file 0.5 would actually have been better (e.g. Random Forest F1 0.914, XGBoost 0.910 at 0.5). We still report the validation-chosen threshold: picking the threshold *because* it looks better on the test set would be tuning on the test set. Both are in `model_comparison.csv` (`*_at_0.5` columns), and the dashboard's threshold slider lets you explore the trade-off.
- **Per attack type** (share flagged, XGBoost): Generic, DoS, Backdoor, Reconnaissance, Worms 100%; Exploits 99.5%; Shellcode 99.2%; Analysis 95.9%; **Fuzzers 88.9%** (hardest for all models, 85-94%). Fuzzers send semi-random data that can resemble ordinary traffic.

## 11. Feature-selection experiments

Does the model need all 42 features? `src/feature_selection.py` trains
four variants of every model on the **same 30,000-row stratified training
sample** (SVM: 20,000) with the tuned hyperparameters, and scores them on the
full official test set. Selectors sit *inside* the pipeline, so they are fitted
on training rows only.

| Variant | What it does |
|---|---|
| `all` | Every encoded feature |
| `mutual_info` | Top 20 encoded columns by mutual information with the label [17] (`SelectKBest`) |
| `tree_importance` | Top 20 encoded columns by a Random Forest's importance (`SelectFromModel`) |
| `pca` | Principal components keeping 95% of the variance (on scaled inputs) |

**Feature selection vs PCA.** Selection keeps a *subset of the original
columns*, so the model stays readable ("it uses `sttl` and `sbytes`"). PCA
creates *new columns that are mixtures of all original ones*: fewer
dimensions, but each component has no direct meaning, which hurts
explainability.

ROC-AUC on the test set (our run; F1 at each model's default decision rule in
brackets). Source: `reports/feature_selection_results.csv`.

| Model | all | mutual_info (20) | tree_importance (20) | PCA (12 comps) |
|---|---:|---:|---:|---:|
| Logistic Regression | **0.964** (0.884) | 0.872 (0.842) | 0.854 (0.832) | 0.892 (0.839) |
| K-Nearest Neighbors | **0.969** (0.882) | 0.962 (0.867) | 0.941 (0.871) | 0.961 (0.882) |
| SVM (RBF) | **0.945** (0.873) | 0.902 (0.852) | 0.919 (0.877) | 0.927 (0.861) |
| Random Forest | **0.982** (0.910) | 0.972 (0.900) | 0.977 (0.909) | 0.967 (0.891) |
| XGBoost | **0.981** (0.904) | 0.974 (0.898) | 0.976 (0.899) | 0.968 (0.888) |

(Input widths: 98 encoded columns from a 30k sample, 67 for SVM's 20k sample.)

**Observations:**
- Using all features was best for every model; fewer features did **not** improve results here.
- Tree ensembles barely notice the reduction (ROC-AUC −0.005 to −0.015), because they can already ignore unhelpful columns.
- Logistic Regression suffers most (−0.07 to −0.11). Both selectors kept only numeric columns (no `proto`/`service`/`state` one-hot columns made the top 20), yet LR relied heavily on `state` and `proto`.
- PCA kept 95% of variance with just 12 components and trains fastest, but costs accuracy and interpretability.
- Selected columns (`reports/selected_features.json`): both methods agree on byte, packet, load, timing and TTL features (`sbytes`, `dbytes`, `sttl`, `dttl`, `sload`, `smean`, `ct_state_ttl`, ...).
- Runtime of the whole experiment: ~5 minutes on 2 cores. Mutual-information scoring is the slowest step (~10 s per model).

## 12. Anomaly detection (Isolation Forest)

Isolation Forest [10] isolates records with random splits; records isolated
in few splits are "unusual". It learns *what is normal*, not *what attacks
look like*. Two variants (`src/anomaly_detection.py`), kept separate from the
supervised pipeline, evaluated on the official test file:

| Variant | Fit on | Threshold | Precision | Recall | F1 | ROC-AUC | False-alarm rate |
|---|---|---|---:|---:|---:|---:|---:|
| Unsupervised | 50k random training rows, labels ignored | `contamination="auto"` | 0.387 | 0.005 | 0.010 | 0.411 | 1.0% |
| Benign-only (novelty) | 50k benign training rows | 95th percentile of benign *validation* scores | 0.845 | 0.294 | 0.436 | 0.793 | 6.6% |

**What this shows:**
- Fully unsupervised detection fails here (ROC-AUC 0.41 is *worse than random*). Attacks are the majority and many are highly repetitive (e.g. Generic floods), so to a density-based detector the attacks look normal and the varied benign traffic looks unusual. "Unusual" ≠ "malicious".
- Learning only from benign traffic works better: at a ~6.6% false-alarm rate it catches 29% of attacks, far below the supervised models (98%+) but without needing any attack labels. Per-category detection rates are in `reports/anomaly_detection_by_category.csv`.

## 13. Explainability

| Method | Scope | Models | Where |
|---|---|---|---|
| Built-in importance (impurity/gain, \|coefficients\|) | Global | LR, RF, XGBoost | Dashboard, `reports/native_importance.csv` |
| Permutation importance (drop in ROC-AUC when one column is shuffled, 5k validation rows, 5 repeats) [13] | Global | All five | `reports/permutation_importance.csv`, `figures/permutation_importance.png` |
| SHAP [11, 12] | One record | RF, XGBoost (TreeExplainer), LR (LinearExplainer) | Dashboard → Explainability |

**Observed (our run):** TTL-related features dominate. Shuffling `sttl` drops
XGBoost's ROC-AUC by 0.24; `sttl` is also the top feature for Random Forest,
and `ct_state_ttl`/`sttl` top the SVM list. Logistic Regression relies most
on `state`, `dttl` and `swin`. TTL values are known to be largely an artefact
of the UNSW testbed configuration, so this dependence is a warning sign for
generalisation, not a discovery about attackers [16].

**Limitations of explanations:** they describe the *model*, not the attack;
they show association, not causation; correlated features share credit
unpredictably; SHAP on one-hot features is summed back to the original column;
KNN and SVM get permutation importance only (KernelExplainer is too slow).

## 14. Installation and execution

Requirements: Python 3.10 or newer (tested with 3.13), ~2 GB free RAM, and
the dataset (~47 MB). Run every command **from the project folder** (the one
containing `app.py`).

### 14.1 Open in VS Code
`File → Open Folder…` → select `network-intrusion-detection`. Open a terminal
with `` Ctrl+` `` (macOS: `` ⌃` ``).

### 14.2 Create and activate a virtual environment

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```
(If PowerShell blocks the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.)

Then in VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`.

### 14.3 Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(`requirements-lock.txt` pins the exact versions used for the reported results.)

### 14.4 Download and place the dataset
Follow [data/README.md](data/README.md): put `UNSW_NB15_training-set.csv` and
`UNSW_NB15_testing-set.csv` into `data/raw/`.

> The project zip already contains trained models and results, so the
> dashboard works immediately. You need the dataset only to retrain or re-run
> experiments. A fresh clone from GitHub has no models: train first.
>
> If you received the project as a zip, `data/raw/` already contains the two
> CSVs. They were taken from a public GitHub mirror whose row counts (175,341 /
> 82,332), class counts and columns match the official files. Replace them with
> the official download if your course requires it; nothing else changes.

### 14.5 Train and evaluate the models (~5 min on a 2-core laptop)
```bash
python -m src.train                  # all five models
python -m src.train --quick          # fast smoke test (results NOT reportable)
python -m src.train --models xgboost random_forest   # a subset
```
This tunes, trains, evaluates and saves: `models/*.joblib`,
`reports/model_comparison.csv`, `reports/test_scores.csv`,
`reports/data_quality.json`, `reports/run_config.json`, `reports/figures/*.png`.

### 14.6 Run the experiments (optional)
```bash
python -m src.anomaly_detection      # ~10 s
python -m src.explainability         # ~8 min (permutation importance)
python -m src.feature_selection      # ~5 min
python -m src.make_demo              # rebuild the 500-row demo file
```

### 14.7 Start the dashboard
```bash
streamlit run app.py
```
It opens at <http://localhost:8501>. In VS Code you can also use
**Run and Debug → "Dashboard (streamlit run app.py)"**.

### 14.8 Run the tests
```bash
python -m pytest
```

### 14.9 Changing the configuration
Edit `src/config.py` (sample sizes, row caps, grids, validation fraction,
seeds), then regenerate everything:
```bash
python -m src.train
python -m src.feature_selection
python -m src.anomaly_detection
python -m src.explainability
```
The dashboard picks up new models and reports automatically (reload the page).

## 15. Dashboard functionality

| Page | What you can do |
|---|---|
| **Overview** | Project summary, records analysed, predicted benign/malicious counts, score distribution, the selected model's test metrics, dataset information, investigation notice |
| **Traffic Analysis** | Record/feature counts, schema check with clear messages, dtypes, missing values, statistics, numeric distributions (log scale, split by label if present), protocol/service/state charts, attack-category and predicted-class charts |
| **Intrusion Detection** | Pick a model and threshold, run predictions on the demo file or an uploaded CSV (no label needed), filter by class and score, inspect any record, download predictions as CSV; if the file has labels, see how the model did on it |
| **Model Comparison** | Results table, metric bars, confusion matrices, ROC and PR curves, threshold trade-off, per-attack-category detection, timings, feature-selection and Isolation Forest results |
| **Explainability** | Built-in and permutation importance, SHAP explanation for any record (RF, XGBoost, LR), limitations |
| **Reports** | Download predictions, comparison and experiment CSVs, and a Markdown summary report |

Upload safety: CSV only, 200 MB / 300,000-row limits, schema validation, no
code is ever executed from an uploaded file, and models are loaded only from
`models/` after a SHA-256 check.

## 16. Testing

```bash
python -m pytest
```
44 tests run in ~5 seconds on synthetic data with the UNSW-NB15 schema (no
dataset required). They cover schema validation, missing/unexpected columns,
empty and invalid files, preprocessing consistency, target-column exclusion,
training/inference compatibility for all five models, model saving/loading
(including tampered files and path traversal), prediction output format,
upload validation and prediction export. See
[TECHNICAL_README.md §7](TECHNICAL_README.md#7-testing).

Result in our environment: **44 passed**.

## 17. Known limitations

- **Benchmark data.** UNSW-NB15 was generated in a 2015 testbed. Results do not prove performance on any real network [15, 16].
- **Dataset artefacts.** TTL features separate classes unusually well in this testbed and dominate the models.
- **Distribution shift** between the official files lowers test performance and makes validation-tuned thresholds too aggressive (≈29-34% false alarms).
- **Sampling.** Tuning uses 20k rows; KNN and SVM are trained on 30k and 20k rows respectively.
- **KNN** cannot use class weights.
- **Scores are not probabilities of attack.** They are model confidence values.
- **Explanations** show association, not causation.
- **Offline only.** No packet capture or live monitoring; inputs must already contain the 42 UNSW-NB15 flow features.
- **Duplicates/overlap** between train and test (10.4% of test rows) may make results slightly optimistic.

## 18. Future improvements

- Multi-class detection of the attack category.
- Cross-dataset testing (e.g. CIC-IDS2017, ToN-IoT).
- Retraining without artefact features (`sttl`, `dttl`, `ct_state_ttl`) to measure their impact.
- Probability calibration and cost-based thresholds (cost of a missed attack vs a false alarm).
- A flow exporter (Zeek/Argus) producing the same features for authorised live analysis.
- Drift monitoring and scheduled retraining.

## 19. References

[1] N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems," *MilCIS*, 2015.
[2] N. Moustafa and J. Slay, "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set," *Information Security Journal: A Global Perspective*, 25(1-3), 2016.
[3] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 12, 2011.
[4] D. R. Cox, "The regression analysis of binary sequences," *J. Royal Statistical Society B*, 20(2), 1958.
[5] T. Cover and P. Hart, "Nearest neighbor pattern classification," *IEEE Trans. Information Theory*, 13(1), 1967.
[6] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, 20(3), 1995.
[7] L. Breiman, "Random forests," *Machine Learning*, 45(1), 2001.
[8] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *KDD*, 2016.
[9] J. Platt, "Probabilistic outputs for support vector machines...," *Advances in Large Margin Classifiers*, 1999.
[10] F. T. Liu, K. M. Ting and Z.-H. Zhou, "Isolation forest," *ICDM*, 2008.
[11] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," *NeurIPS*, 2017.
[12] S. M. Lundberg et al., "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, 2, 2020.
[13] A. Fisher, C. Rudin and F. Dominici, "All models are wrong, but many are useful...," *JMLR*, 20(177), 2019.
[14] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLoS ONE*, 10(3), 2015.
[15] R. Sommer and V. Paxson, "Outside the closed world: On using machine learning for network intrusion detection," *IEEE S&P*, 2010.
[16] D. Arp et al., "Dos and don'ts of machine learning in computer security," *USENIX Security*, 2022.
[17] A. Kraskov, H. Stögbauer and P. Grassberger, "Estimating mutual information," *Physical Review E*, 69, 2004.

---

*For defensive, educational analysis of public or authorised data only.*
