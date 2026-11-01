"""
Central configuration for the whole project.

Every "knob" you might want to change (file paths, sample sizes, hyperparameter
grids, random seed) lives here, so experiments are reproducible and you never
have to hunt through the code to change a setting.

After editing this file, regenerate models and reports with:
    python -m src.train
    python -m src.feature_selection
    python -m src.anomaly_detection
    python -m src.explainability
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project folder, never to your personal machine)
# --------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
DEMO_DATA_DIR: Path = DATA_DIR / "demo"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

TRAIN_FILE: Path = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE: Path = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"
DEMO_FILE: Path = DEMO_DATA_DIR / "demo_traffic.csv"

# --------------------------------------------------------------------------
# Dataset schema (UNSW-NB15 official training/testing CSV files)
# --------------------------------------------------------------------------
TARGET_COLUMN: str = "label"  # 0 = normal, 1 = attack

# Columns that must NEVER be used as model inputs:
#   id         -> a row counter; rows are grouped by class, so it leaks the label
#   label      -> the target itself
#   attack_cat -> attack category; "Normal" <=> label 0, so it reveals the target
EXCLUDED_COLUMNS: tuple[str, ...] = ("id", "label", "attack_cat")

CATEGORICAL_FEATURES: tuple[str, ...] = ("proto", "service", "state")

NUMERIC_FEATURES: tuple[str, ...] = (
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
)

FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CLASS_NAMES: dict[int, str] = {0: "Benign", 1: "Malicious"}

# --------------------------------------------------------------------------
# Reproducibility and data splitting
# --------------------------------------------------------------------------
RANDOM_STATE: int = 42

# Part of the OFFICIAL TRAINING file held back as a validation set. It is used
# to pick the decision threshold, so the test set is only touched once.
VALIDATION_FRACTION: float = 0.15

# Hyperparameter search runs on a stratified sample of the training data so it
# finishes in minutes on a laptop. Set to None to tune on everything (slow).
TUNING_SAMPLE_SIZE: int | None = 20_000
CV_FOLDS: int = 3

# Some algorithms scale badly with the number of rows (SVM is roughly O(n^2),
# KNN must compare every test row to every stored training row). We cap their
# training rows. The cap is written into the results table so it is reported.
# None = use all available training rows.
MAX_TRAIN_ROWS: dict[str, int | None] = {
    "logistic_regression": None,
    "knn": 30_000,
    "svm": 20_000,
    "random_forest": None,
    "xgboost": None,
}

# Upload safety limits for the dashboard.
MAX_UPLOAD_MB: int = 200
MAX_UPLOAD_ROWS: int = 300_000

# --------------------------------------------------------------------------
# Hyperparameter grids (searched with GridSearchCV, scored by F1 on attacks)
# The "model__" prefix targets the classifier step inside the Pipeline.
# --------------------------------------------------------------------------
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logistic_regression": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
    },
    "knn": {
        "model__n_neighbors": [3, 5, 11, 21],
        "model__weights": ["uniform", "distance"],
    },
    "svm": {
        "model__kernel": ["linear", "rbf"],
        "model__C": [0.1, 1.0, 10.0],
    },
    "random_forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [12, 20, None],
        "model__min_samples_split": [2, 10],
    },
    "xgboost": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
    },
}

# Threshold selection on the validation set: choose the threshold that
# maximises F1 for the attack class.
THRESHOLD_STRATEGY: str = "max_f1"

# --------------------------------------------------------------------------
# Feature selection / PCA experiment
# --------------------------------------------------------------------------
FS_TRAIN_SAMPLE_SIZE: int | None = 30_000  # rows used to fit each variant
FS_TOP_K: int = 20                          # features kept by MI / tree importance
PCA_VARIANCE: float = 0.95                  # keep components explaining 95% variance

# --------------------------------------------------------------------------
# Isolation Forest (optional anomaly detection experiment)
# --------------------------------------------------------------------------
IFOREST_N_ESTIMATORS: int = 200
# Threshold = this percentile of anomaly scores on BENIGN validation records,
# i.e. we accept roughly (100 - percentile)% false alarms on normal traffic.
IFOREST_BENIGN_PERCENTILE: float = 95.0

# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
PERMUTATION_SAMPLE_SIZE: int = 5_000
PERMUTATION_REPEATS: int = 5
SHAP_BACKGROUND_SIZE: int = 200

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
