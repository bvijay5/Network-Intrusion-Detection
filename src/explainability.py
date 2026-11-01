"""
Explainability: which features influence the models' predictions?

Three complementary tools:

1. Native importance (global, cheap)
   - Random Forest / XGBoost: how much each feature reduced impurity / loss.
   - Logistic Regression: absolute coefficient size (features are scaled, so
     coefficients are comparable).
   - KNN and SVM (RBF) have no built-in importance.

2. Permutation importance (global, works for ANY model)
   Shuffle one original column and measure how much ROC-AUC drops. A large drop
   means the model relies on that column. Computed on validation rows.

3. SHAP (local, one record at a time)
   Splits one prediction into additive contributions per feature, relative to
   the model's average output. Supported here for Random Forest, XGBoost
   (TreeExplainer, exact and fast) and Logistic Regression (LinearExplainer).
   KNN/SVM would need KernelExplainer, which is too slow to be practical.

IMPORTANT: all of these show ASSOCIATION, not causation. "sttl was important"
means the model used it, not that the TTL value caused an attack.

Run with:   python -m src.explainability
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.preprocessing import original_feature_of, transformed_feature_names

SHAP_SUPPORTED: tuple[str, ...] = ("random_forest", "xgboost", "logistic_regression")


# --------------------------------------------------------------------------
# Global importance
# --------------------------------------------------------------------------
def native_importance(bundle: dict) -> pd.DataFrame | None:
    """Built-in importance aggregated back to original columns, or None."""
    pipe, key = bundle["pipeline"], bundle["model_key"]
    model = pipe.named_steps["model"]
    names = transformed_feature_names(pipe)
    if key in ("random_forest", "xgboost"):
        values = np.asarray(model.feature_importances_, dtype=float)
        kind = "impurity / gain importance"
    elif key == "logistic_regression":
        values = np.abs(model.coef_.ravel())
        kind = "|coefficient| (scaled inputs)"
    else:
        return None
    frame = pd.DataFrame({"column": names, "importance": values})
    frame["feature"] = frame["column"].map(original_feature_of)
    out = frame.groupby("feature", as_index=False)["importance"].sum()
    out["importance"] = out["importance"] / out["importance"].sum()  # share of total
    out["method"] = kind
    return out.sort_values("importance", ascending=False, ignore_index=True)


def permutation_importance_table(bundle: dict, X: pd.DataFrame, y: pd.Series,
                                 n_repeats: int = config.PERMUTATION_REPEATS) -> pd.DataFrame:
    """Drop in ROC-AUC when each ORIGINAL column is shuffled (mean +/- std)."""
    result = permutation_importance(
        bundle["pipeline"], X, y, scoring="roc_auc", n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------
# Local (single-record) SHAP explanations
# --------------------------------------------------------------------------
def _positive_class(values: np.ndarray) -> np.ndarray:
    """SHAP returns (n, features) or (n, features, classes); keep class 1."""
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values


def explain_record(bundle: dict, record: pd.DataFrame,
                   background: pd.DataFrame | None = None) -> dict:
    """SHAP contributions for ONE record, aggregated to original features.

    Args:
        bundle: loaded model bundle.
        record: a one-row DataFrame with the model's feature columns.
        background: reference rows (needed for Logistic Regression only).

    Returns:
        dict with 'table' (feature, value, contribution), 'base_value' and
        'units' (what the contribution numbers are measured in).

    Raises:
        NotImplementedError: for models without a practical SHAP explainer.
    """
    import shap  # imported lazily: it is slow to import

    key, pipe = bundle["model_key"], bundle["pipeline"]
    if key not in SHAP_SUPPORTED:
        raise NotImplementedError(
            f"SHAP record-level explanations are not provided for "
            f"{config.MODEL_DISPLAY_NAMES[key]} (KernelExplainer would be too slow). "
            "Use the permutation importance view for this model instead."
        )
    record = record.loc[:, list(config.FEATURE_COLUMNS)]
    pre, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    x = pre.transform(record)
    names = transformed_feature_names(pipe)

    if key == "logistic_regression":
        if background is None:
            raise ValueError("Logistic Regression explanations need background rows.")
        bg = pre.transform(background)
        masker = shap.maskers.Independent(bg, max_samples=len(bg))
        explainer = shap.LinearExplainer(model, masker)
        units = "log-odds of 'attack'"
    else:
        explainer = shap.TreeExplainer(model)
        units = "log-odds of 'attack'" if key == "xgboost" else "probability of 'attack'"

    values = _positive_class(explainer.shap_values(x))[0]
    base = np.ravel(explainer.expected_value)
    base_value = float(base[1] if base.size == 2 else base[0])

    frame = pd.DataFrame({"column": names, "contribution": values})
    frame["feature"] = frame["column"].map(original_feature_of)
    table = frame.groupby("feature", as_index=False)["contribution"].sum()
    table["value"] = table["feature"].map(lambda f: record.iloc[0][f])
    table["abs"] = table["contribution"].abs()
    table = table.sort_values("abs", ascending=False, ignore_index=True).drop(columns="abs")
    return {"table": table[["feature", "value", "contribution"]],
            "base_value": base_value, "units": units}


# --------------------------------------------------------------------------
# Command line: precompute global importance for the dashboard
# --------------------------------------------------------------------------
def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.predict import available_models, load_bundle
    from src.train import load_and_split, stratified_sample

    keys = available_models()
    if not keys:
        raise SystemExit("No trained models found. Run:  python -m src.train")
    data = load_and_split(verbose=False)
    X_val, y_val = stratified_sample(data["X_val"], data["y_val"],
                                     config.PERMUTATION_SAMPLE_SIZE)
    perm_frames, native_frames = [], []
    for key in keys:
        bundle = load_bundle(key)
        print(f"Permutation importance: {config.MODEL_DISPLAY_NAMES[key]} ...")
        perm = permutation_importance_table(bundle, X_val, y_val)
        perm.insert(0, "model_key", key)
        perm_frames.append(perm)
        native = native_importance(bundle)
        if native is not None:
            native.insert(0, "model_key", key)
            native_frames.append(native)

    perm_all = pd.concat(perm_frames, ignore_index=True)
    perm_all.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)
    if native_frames:
        pd.concat(native_frames, ignore_index=True).to_csv(
            config.REPORTS_DIR / "native_importance.csv", index=False)

    # One figure: top-10 permutation importance per model
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 4), sharex=False)
    for ax, key in zip(np.atleast_1d(axes), keys):
        top = perm_all[perm_all.model_key == key].head(10)[::-1]
        ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
        ax.set_title(config.MODEL_DISPLAY_NAMES[key], fontsize=9)
        ax.set_xlabel("ROC-AUC drop", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("Permutation importance (validation sample)", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "permutation_importance.png", dpi=130)
    plt.close(fig)
    print("Saved reports/permutation_importance.csv and native_importance.csv")


if __name__ == "__main__":
    main()
