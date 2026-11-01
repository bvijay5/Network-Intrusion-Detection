"""
Evaluation metrics, threshold selection and comparison plots.

Every model is scored with the SAME functions on the SAME held-out test set,
so the comparison is fair. "Positive" always means the attack class (label 1).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    """All comparison metrics for one model at one decision threshold.

    Args:
        y_true: true labels (0 benign, 1 attack).
        scores: model scores, higher = more likely attack.
        threshold: score at or above which a record is flagged as malicious.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        # The next three are for the ATTACK class (pos_label=1)
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # Ranking quality, independent of the threshold
        "roc_auc": roc_auc_score(y_true, scores) if both_classes else np.nan,
        "pr_auc": average_precision_score(y_true, scores) if both_classes else np.nan,
        # Error analysis
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,  # false alarms
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,  # missed attacks
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_val: np.ndarray, val_scores: np.ndarray,
                     strategy: str = config.THRESHOLD_STRATEGY) -> float:
    """Pick a decision threshold on the VALIDATION set (never the test set).

    strategy="max_f1": the threshold with the best attack-class F1, i.e. the
    best balance between catching attacks (recall) and avoiding false alarms
    (precision).
    """
    if strategy != "max_f1":
        raise ValueError(f"Unknown threshold strategy: {strategy}")
    precision, recall, thresholds = precision_recall_curve(y_val, val_scores)
    # precision/recall have one more element than thresholds; drop the last.
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(thresholds[int(np.argmax(f1))])


def threshold_table(y_true: np.ndarray, scores: np.ndarray,
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Precision / recall / false-alarm rate at a range of thresholds."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0, 0.05), 2)
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, scores, t)
        rows.append({k: m[k] for k in ("threshold", "precision", "recall", "f1",
                                       "false_positive_rate", "fp", "fn")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Plots (saved as PNG files into reports/figures)
# --------------------------------------------------------------------------
def _name(key: str) -> str:
    return config.MODEL_DISPLAY_NAMES.get(key, key)


def plot_roc_curves(y_true: np.ndarray, scores_by_model: dict[str, np.ndarray],
                    path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key, scores in scores_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, label=f"{_name(key)} (AUC={roc_auc_score(y_true, scores):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Random guess")
    ax.set(xlabel="False positive rate (benign flagged as attack)",
           ylabel="True positive rate (attacks caught)", title="ROC curves - test set")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr_curves(y_true: np.ndarray, scores_by_model: dict[str, np.ndarray],
                   path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for key, scores in scores_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, label=f"{_name(key)} (AP={ap:.3f})")
    ax.axhline(np.mean(y_true), linestyle="--", color="grey", linewidth=1,
               label="Attack base rate")
    ax.set(xlabel="Recall (attacks caught)", ylabel="Precision (alerts that are real)",
           title="Precision-recall curves - test set")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_matrices(results: pd.DataFrame, path: Path) -> None:
    """One small confusion matrix per model, from the tn/fp/fn/tp columns."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.3))
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, results.iterrows()):
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["Benign", "Malicious"], fontsize=8)
        ax.set_yticks([0, 1], ["Benign", "Malicious"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(row.model_name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metric_bars(results: pd.DataFrame, path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(results))
    width = 0.16
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * width, results[m], width, label=m)
    ax.set_xticks(x, results["model_name"], fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Score (test set)")
    ax.set_title("Model comparison (attack = positive class)")
    ax.legend(fontsize=8, ncol=5, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
