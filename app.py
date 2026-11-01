"""
Streamlit dashboard for the Explainable ML-Based Network Intrusion Detection System.

Start it from the project folder with:
    streamlit run app.py

The dashboard contains NO machine-learning logic of its own. It calls the
functions in src/ (loading, validation, prediction, explanation), so what you
see here is exactly what the training and test code produce.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config
from src.data_loader import DatasetError, read_csv_safely, validate_schema
from src.evaluate import compute_metrics, threshold_table
from src.explainability import SHAP_SUPPORTED, explain_record, native_importance
from src.predict import (
    ModelNotAvailableError,
    available_models,
    load_bundle,
    predict_dataframe,
    predictions_to_csv,
)
from src.report import build_markdown_report

st.set_page_config(page_title="ML Intrusion Detection", page_icon="🛡️", layout="wide")

TEMPLATE = "plotly_dark"
COLORS = {"Benign": "#2ec4b6", "Malicious": "#ff6b6b"}
DISCLAIMER = (
    "Model alerts are **indicators for investigation, not proof of an attack**. "
    "Scores are model confidence values, not guaranteed probabilities."
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      .small-note {color: #9aa4b2; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# Cached helpers (each calls into src/)
# ==========================================================================
def _model_mtime(key: str) -> float:
    path = config.MODELS_DIR / f"{key}.joblib"
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner="Loading model ...")
def get_bundle(key: str, mtime: float) -> dict:  # mtime busts the cache after retraining
    return load_bundle(key)


@st.cache_data(show_spinner="Running predictions ...")
def get_predictions(key: str, mtime: float, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return predict_dataframe(get_bundle(key, mtime), data, threshold)


@st.cache_data
def read_report_csv(name: str, mtime: float) -> pd.DataFrame | None:
    path = config.REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def report_csv(name: str) -> pd.DataFrame | None:
    path = config.REPORTS_DIR / name
    return read_report_csv(name, path.stat().st_mtime if path.exists() else 0.0)


def report_json(name: str) -> dict | None:
    path = config.REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner="Reading file ...")
def load_demo() -> pd.DataFrame:
    return read_csv_safely(config.DEMO_FILE)


def load_upload(uploaded) -> pd.DataFrame:
    if uploaded.size > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise DatasetError(f"File is larger than the {config.MAX_UPLOAD_MB} MB limit.")
    return read_csv_safely(uploaded, max_rows=config.MAX_UPLOAD_ROWS)


def train_instructions() -> None:
    st.info(
        "No trained models found yet. In a terminal, from the project folder, run:\n\n"
        "```\npython -m src.train\n```\n"
        "See README.md for downloading the dataset first."
    )


# ==========================================================================
# Sidebar: data source, model and threshold
# ==========================================================================
st.sidebar.title("🛡️ ML-NIDS")
st.sidebar.caption("Explainable intrusion detection on UNSW-NB15 records (offline CSV analysis)")

page = st.sidebar.radio(
    "Page",
    ["Overview", "Traffic Analysis", "Intrusion Detection", "Model Comparison",
     "Explainability", "Reports"],
)

st.sidebar.divider()
source = st.sidebar.radio("Data source", ["Demo sample (included)", "Upload CSV"])
data: pd.DataFrame | None = None
data_name = ""
if source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("UNSW-NB15-format CSV", type=["csv"])
    if uploaded is not None:
        try:
            data, data_name = load_upload(uploaded), uploaded.name
        except DatasetError as exc:
            st.sidebar.error(str(exc))
else:
    if config.DEMO_FILE.exists():
        data, data_name = load_demo(), config.DEMO_FILE.name
    else:
        st.sidebar.warning("Demo file missing. Create it with: python -m src.make_demo")

schema = validate_schema(data) if data is not None else None
if schema is not None and not schema.is_valid:
    st.sidebar.error("This file does not match the expected schema (see Traffic Analysis).")

st.sidebar.divider()
models = available_models()
model_key: str | None = None
bundle: dict | None = None
threshold = 0.5
if models:
    model_key = st.sidebar.selectbox(
        "Model", models, format_func=lambda k: config.MODEL_DISPLAY_NAMES.get(k, k),
        index=models.index("xgboost") if "xgboost" in models else 0,
    )
    try:
        bundle = get_bundle(model_key, _model_mtime(model_key))
        for warning in bundle.get("version_warnings", []):
            st.sidebar.warning(warning)
        threshold = st.sidebar.slider(
            "Decision threshold", 0.01, 0.99, float(round(bundle["threshold"], 2)), 0.01,
            help="Records with an attack score at or above this value are flagged. "
                 "The default was chosen on the validation set (best F1). Lower = catch "
                 "more attacks but more false alarms; higher = fewer alarms, more misses.",
        )
    except (ModelNotAvailableError, PermissionError) as exc:
        st.sidebar.error(str(exc))
        bundle = None
else:
    st.sidebar.warning("No trained models yet.")


def current_predictions() -> pd.DataFrame | None:
    """Predictions for the active file, or None (with a message) if impossible."""
    if bundle is None:
        train_instructions()
        return None
    if data is None:
        st.info("Choose the demo sample or upload a CSV in the sidebar.")
        return None
    if not schema.is_valid:
        st.error("Cannot run the model on this file:\n\n- " + "\n- ".join(schema.messages))
        return None
    try:
        return get_predictions(model_key, _model_mtime(model_key), data, threshold)
    except DatasetError as exc:
        st.error(str(exc))
        return None


def has_labels(df: pd.DataFrame) -> bool:
    return config.TARGET_COLUMN in df.columns and df[config.TARGET_COLUMN].isin([0, 1]).all()


# ==========================================================================
# Page A: Overview
# ==========================================================================
def page_overview() -> None:
    st.title("Explainable ML-Based Network Intrusion Detection")
    st.markdown(
        "This tool classifies network-flow records from the **UNSW-NB15** dataset as "
        "**benign** or **potentially malicious** using five traditional machine-learning "
        "models (Logistic Regression, KNN, SVM, Random Forest, XGBoost), and explains "
        "which features drove each prediction. It analyses **offline CSV files only**; "
        "it does not capture live traffic."
    )
    st.warning(DISCLAIMER, icon="⚠️")

    preds = current_predictions()
    if preds is not None:
        n_mal = int(preds["prediction"].sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records analysed", f"{len(preds):,}")
        c2.metric("Predicted benign", f"{len(preds) - n_mal:,}")
        c3.metric("Potentially malicious", f"{n_mal:,}")
        c4.metric("Flag rate", f"{n_mal / len(preds):.1%}")
        st.caption(f"File: `{data_name}` | Model: {config.MODEL_DISPLAY_NAMES[model_key]} "
                   f"| Threshold: {threshold:.2f}")

        left, right = st.columns(2)
        counts = preds["predicted_label"].value_counts().reset_index()
        counts.columns = ["class", "records"]
        left.plotly_chart(px.pie(counts, names="class", values="records", hole=0.5,
                                 color="class", color_discrete_map=COLORS,
                                 template=TEMPLATE, title="Predicted classes"),
                          width="stretch")
        right.plotly_chart(px.histogram(preds, x="attack_score", nbins=40, template=TEMPLATE,
                                        title="Distribution of attack scores",
                                        color_discrete_sequence=["#7aa2f7"])
                           .add_vline(x=threshold, line_dash="dash", line_color="#ff6b6b",
                                      annotation_text="threshold"),
                           width="stretch")

    st.subheader("Selected model - evaluation on the official test set")
    results = report_csv("model_comparison.csv")
    if results is None or model_key is None or model_key not in set(results["model_key"]):
        st.info("No evaluation results yet. They appear after `python -m src.train` finishes.")
    else:
        row = results.set_index("model_key").loc[model_key]
        cols = st.columns(6)
        for col, (label, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                            ("Recall", "recall"), ("F1", "f1"),
                                            ("ROC-AUC", "roc_auc"), ("False alarm rate",
                                                                     "false_positive_rate")]):
            col.metric(label, f"{row[key]:.3f}")
        st.caption(f"Metrics for the attack class at the validation-chosen threshold "
                   f"{row['threshold']:.3f}, on {int(row.tn + row.fp + row.fn + row.tp):,} "
                   "test records.")

    quality = report_json("data_quality.json")
    if quality:
        with st.expander("Dataset information"):
            tr, te = quality["train"], quality["test"]
            st.markdown(
                f"- **Training file:** {tr['rows']:,} records, {tr['class_counts']}\n"
                f"- **Testing file:** {te['rows']:,} records, {te['class_counts']}\n"
                f"- **Features used:** {tr['feature_columns_used']} "
                f"(39 numeric + 3 categorical: proto, service, state)\n"
                f"- **Excluded:** `id`, `label`, `attack_cat` (target / leakage)\n"
                f"- Source: [UNSW-NB15 official page]"
                f"(https://research.unsw.edu.au/projects/unsw-nb15-dataset)"
            )


# ==========================================================================
# Page B: Traffic Analysis
# ==========================================================================
def page_traffic() -> None:
    st.title("Traffic Analysis")
    if data is None:
        st.info("Choose the demo sample or upload a CSV in the sidebar.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(data):,}")
    c2.metric("Columns", data.shape[1])
    c3.metric("Model features present",
              f"{len(config.FEATURE_COLUMNS) - len(schema.missing_columns)} / "
              f"{len(config.FEATURE_COLUMNS)}")

    if schema.messages:
        (st.error if not schema.is_valid else st.info)("\n\n".join(schema.messages))

    tab1, tab2, tab3, tab4 = st.tabs(["Columns & statistics", "Distributions",
                                      "Protocols & services", "Labels & predictions"])
    with tab1:
        info = pd.DataFrame({
            "dtype": data.dtypes.astype(str),
            "missing": data.isna().sum(),
            "unique": data.nunique(),
        })
        st.dataframe(info, width="stretch", height=300)
        st.markdown("**Numeric summary**")
        st.dataframe(data.describe().T, width="stretch", height=300)
        if data.isna().sum().sum() == 0:
            st.success("No missing values.")

    with tab2:
        numeric = [c for c in config.NUMERIC_FEATURES if c in data.columns]
        if numeric:
            feat = st.selectbox("Numeric feature", numeric,
                                index=numeric.index("sbytes") if "sbytes" in numeric else 0)
            log_x = st.checkbox("Log scale (most traffic features are very skewed)", value=True)
            values = pd.to_numeric(data[feat], errors="coerce")
            plot_df = pd.DataFrame({feat: np.log10(values + 1) if log_x else values})
            color = None
            if has_labels(data):
                plot_df["class"] = data[config.TARGET_COLUMN].map(config.CLASS_NAMES)
                color = "class"
            fig = px.histogram(plot_df, x=feat, color=color, nbins=50, barmode="overlay",
                               template=TEMPLATE, color_discrete_map=COLORS, opacity=0.75,
                               labels={feat: f"log10({feat}+1)" if log_x else feat})
            st.plotly_chart(fig, width="stretch")

    with tab3:
        cols = st.columns(3)
        for col, cat in zip(cols, config.CATEGORICAL_FEATURES):
            if cat in data.columns:
                vc = data[cat].astype(str).value_counts().head(10).reset_index()
                vc.columns = [cat, "records"]
                col.plotly_chart(px.bar(vc, x="records", y=cat, orientation="h",
                                        template=TEMPLATE, title=f"Top {cat} values")
                                 .update_yaxes(autorange="reversed"),
                                 width="stretch")

    with tab4:
        if "attack_cat" in data.columns and has_labels(data):
            vc = data["attack_cat"].astype(str).value_counts().reset_index()
            vc.columns = ["attack_cat", "records"]
            st.plotly_chart(px.bar(vc, x="attack_cat", y="records", template=TEMPLATE,
                                   title="Ground-truth attack categories (from the file)"),
                            width="stretch")
        else:
            st.caption("No valid ground-truth labels in this file.")
        preds = current_predictions()
        if preds is not None:
            vc = preds["predicted_label"].value_counts().reset_index()
            vc.columns = ["class", "records"]
            st.plotly_chart(px.bar(vc, x="class", y="records", color="class",
                                   color_discrete_map=COLORS, template=TEMPLATE,
                                   title="Benign vs potentially malicious (model prediction)"),
                            width="stretch")


# ==========================================================================
# Page C: Intrusion Detection
# ==========================================================================
def page_detection() -> None:
    st.title("Intrusion Detection")
    st.warning(DISCLAIMER, icon="⚠️")
    preds = current_predictions()
    if preds is None:
        return

    n_mal = int(preds["prediction"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{len(preds):,}")
    c2.metric("Potentially malicious", f"{n_mal:,}")
    c3.metric("Threshold", f"{threshold:.2f}")

    if has_labels(preds):
        m = compute_metrics(preds[config.TARGET_COLUMN], preds["attack_score"], threshold)
        st.caption(
            f"This file contains ground-truth labels, so we can check the model on it: "
            f"precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}, "
            f"{m['fp']} false alarms, {m['fn']} missed attacks. "
            "(Labels were NOT given to the model.)"
        )

    f1, f2 = st.columns([1, 2])
    only_mal = f1.checkbox("Show only potentially malicious records", value=True)
    min_score = f2.slider("Minimum attack score", 0.0, 1.0, 0.0, 0.01)
    view = preds[preds["attack_score"] >= min_score]
    if only_mal:
        view = view[view["prediction"] == 1]
    view = view.sort_values("attack_score", ascending=False)

    front = ["predicted_label", "attack_score"]
    if has_labels(preds):
        front += [c for c in ("label", "attack_cat") if c in preds.columns]
    shown_cols = front + [c for c in ("proto", "service", "state", "dur", "sbytes", "dbytes",
                                      "sttl", "dttl", "spkts", "dpkts") if c in view.columns]
    st.dataframe(view[shown_cols], width="stretch", height=360)
    st.caption(f"{len(view):,} records shown. Row numbers match the uploaded file "
               "(0 = first data row).")

    st.download_button("⬇️ Download all predictions (CSV)", predictions_to_csv(preds),
                       file_name=f"predictions_{model_key}.csv", mime="text/csv")

    st.subheader("Inspect a record")
    if len(view):
        idx = st.selectbox("Row number", view.index.tolist()[:2000])
        record = preds.loc[idx]
        a, b = st.columns([1, 2])
        a.metric("Prediction", record["predicted_label"])
        a.metric("Attack score", f"{record['attack_score']:.3f}")
        b.dataframe(record[list(config.FEATURE_COLUMNS)].astype(str).rename("value"),
                    width="stretch", height=300)
        st.caption("See the Explainability page for which features drove this prediction.")


# ==========================================================================
# Page D: Model Comparison
# ==========================================================================
def _curve_points(y: np.ndarray, s: np.ndarray, kind: str, n: int = 300) -> tuple:
    if kind == "roc":
        x, yv, _ = roc_curve(y, s)
    else:
        p, r, _ = precision_recall_curve(y, s)
        x, yv = r, p
    step = max(1, len(x) // n)
    return x[::step], yv[::step]


def page_comparison() -> None:
    st.title("Model Comparison")
    results = report_csv("model_comparison.csv")
    if results is None:
        train_instructions()
        return
    run = report_json("run_config.json") or {}
    if run.get("quick_mode"):
        st.error("These results come from a QUICK smoke-test run on small samples. "
                 "Re-run `python -m src.train` without --quick for reportable results.")
    st.caption("All models evaluated on the same official UNSW-NB15 test file. "
               "Precision, recall and F1 refer to the attack class. Thresholds were "
               "chosen on a validation split of the training file, never on the test set.")

    show = results[["model_name", "threshold", "accuracy", "precision", "recall", "f1",
                    "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate",
                    "fp", "fn", "train_rows", "train_seconds", "predict_ms_per_1k_rows"]]
    st.dataframe(show.style.format(precision=4).highlight_max(
        subset=["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"], color="#1f5f5b"),
        width="stretch", hide_index=True)

    tabs = st.tabs(["Metrics", "Confusion matrices", "ROC & PR curves", "Threshold trade-off",
                    "Per attack type", "Time", "Feature sets", "Anomaly detection"])
    with tabs[0]:
        long = results.melt(id_vars="model_name",
                            value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
                            var_name="metric", value_name="score")
        st.plotly_chart(px.bar(long, x="model_name", y="score", color="metric", barmode="group",
                               template=TEMPLATE, range_y=[0.5, 1.0]),
                        width="stretch")
        st.markdown(
            "**Reading the errors:** a *false positive* is benign traffic flagged as an "
            "attack (wasted analyst time); a *false negative* is a missed attack "
            "(the dangerous error). High recall with a high false-alarm rate means the model "
            "catches almost everything but cries wolf often."
        )

    with tabs[1]:
        pick = st.selectbox("Model", results["model_key"],
                            format_func=lambda k: config.MODEL_DISPLAY_NAMES[k], key="cm_model")
        r = results.set_index("model_key").loc[pick]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        fig = px.imshow(cm, text_auto=",", template=TEMPLATE, color_continuous_scale="Teal",
                        x=["Pred. benign", "Pred. malicious"], y=["Actual benign",
                                                               "Actual malicious"])
        st.plotly_chart(fig, width="stretch")
        st.caption(f"False alarms: {int(r.fp):,} of {int(r.tn + r.fp):,} benign records "
                   f"({r.false_positive_rate:.1%}). Missed attacks: {int(r.fn):,} of "
                   f"{int(r.fn + r.tp):,} ({r.false_negative_rate:.1%}).")

    scores = report_csv("test_scores.csv")
    with tabs[2]:
        if scores is None:
            st.info("test_scores.csv not found.")
        else:
            y = scores["y_true"].to_numpy()
            roc, pr = go.Figure(), go.Figure()
            for k in results["model_key"]:
                if f"score_{k}" not in scores:
                    continue
                s = scores[f"score_{k}"].to_numpy()
                x1, y1 = _curve_points(y, s, "roc")
                roc.add_scatter(x=x1, y=y1, name=config.MODEL_DISPLAY_NAMES[k], mode="lines")
                x2, y2 = _curve_points(y, s, "pr")
                pr.add_scatter(x=x2, y=y2, name=config.MODEL_DISPLAY_NAMES[k], mode="lines")
            roc.add_scatter(x=[0, 1], y=[0, 1], name="random", line=dict(dash="dash",
                                                                          color="grey"))
            roc.update_layout(template=TEMPLATE, title="ROC curve", xaxis_title="False positive "
                              "rate", yaxis_title="True positive rate (recall)")
            pr.update_layout(template=TEMPLATE, title="Precision-recall curve",
                             xaxis_title="Recall", yaxis_title="Precision")
            a, b = st.columns(2)
            a.plotly_chart(roc, width="stretch")
            b.plotly_chart(pr, width="stretch")

    with tabs[3]:
        if scores is not None:
            pick = st.selectbox("Model", results["model_key"],
                                format_func=lambda k: config.MODEL_DISPLAY_NAMES[k],
                                key="thr_model")
            table = threshold_table(scores["y_true"], scores[f"score_{pick}"])
            fig = px.line(table, x="threshold", y=["precision", "recall", "f1",
                                                   "false_positive_rate"],
                          template=TEMPLATE, markers=True)
            chosen = results.set_index("model_key").loc[pick, "threshold"]
            fig.add_vline(x=chosen, line_dash="dash", annotation_text="validation-chosen")
            st.plotly_chart(fig, width="stretch")
            st.caption("Illustration only: this shows how precision and recall move as the "
                       "threshold changes on the test set. The model's threshold was picked "
                       "on validation data, so the test set was not used for tuning.")

    with tabs[4]:
        if scores is not None and "attack_cat" in scores:
            rows = []
            for k in results["model_key"]:
                thr = results.set_index("model_key").loc[k, "threshold"]
                flagged = scores[f"score_{k}"] >= thr
                rate = flagged.groupby(scores["attack_cat"]).mean()
                rows += [{"model": config.MODEL_DISPLAY_NAMES[k], "attack_cat": c,
                          "flagged_share": v} for c, v in rate.items()]
            frame = pd.DataFrame(rows)
            st.plotly_chart(px.bar(frame, x="attack_cat", y="flagged_share", color="model",
                                   barmode="group", template=TEMPLATE,
                                   title="Share of each category flagged as malicious"),
                            width="stretch")
            st.caption("For 'Normal' this is the false-alarm rate; for attack categories it "
                       "is the detection rate (recall).")

    with tabs[5]:
        st.plotly_chart(px.bar(results, x="model_name",
                               y=["train_seconds", "tuning_seconds", "predict_seconds_test"],
                               barmode="group", template=TEMPLATE, log_y=True,
                               title="Time in seconds (log scale)"),
                        width="stretch")
        st.caption("Training rows differ per model (see table) because KNN and SVM are "
                   "capped for laptop-friendly run times.")

    with tabs[6]:
        fs = report_csv("feature_selection_results.csv")
        if fs is None:
            st.info("Run `python -m src.feature_selection` to produce this experiment.")
        else:
            st.plotly_chart(px.bar(fs, x="model_name", y="roc_auc", color="variant",
                                   barmode="group", template=TEMPLATE,
                                   range_y=[max(0.5, fs.roc_auc.min() - 0.03), 1]),
                            width="stretch")
            st.dataframe(fs.round(4), width="stretch", hide_index=True)
            sel = report_json("selected_features.json")
            if sel:
                st.markdown(f"**Mutual-information top {sel['top_k']}:** "
                            + ", ".join(sel.get("mutual_info", [])))
                st.markdown(f"**Tree-importance top {sel['top_k']}:** "
                            + ", ".join(sel.get("tree_importance", [])))

    with tabs[7]:
        ad = report_csv("anomaly_detection_results.csv")
        if ad is None:
            st.info("Run `python -m src.anomaly_detection` to produce this experiment.")
        else:
            st.dataframe(ad[["variant", "fit_rows", "threshold", "precision", "recall", "f1",
                             "roc_auc", "false_positive_rate", "fp", "fn"]].round(4),
                         width="stretch", hide_index=True)
            cat = report_csv("anomaly_detection_by_category.csv")
            if cat is not None:
                st.dataframe(cat.round(3), width="stretch", hide_index=True)
            st.caption("Isolation Forest flags *unusual* records without learning attack "
                       "labels. Unusual is not the same as malicious, so expect more false "
                       "alarms than the supervised models.")


# ==========================================================================
# Page E: Explainability
# ==========================================================================
def page_explainability() -> None:
    st.title("Explainability")
    st.info("Feature importance and SHAP values show **which inputs the model relied on**. "
            "They show association, **not causation**: an important feature did not "
            "necessarily *cause* malicious activity. " + DISCLAIMER)
    if bundle is None:
        train_instructions()
        return

    st.subheader(f"Global importance - {config.MODEL_DISPLAY_NAMES[model_key]}")
    a, b = st.columns(2)
    native = native_importance(bundle)
    if native is not None:
        top = native.head(15)[::-1]
        a.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h",
                              template=TEMPLATE, title=f"Built-in: {native.method[0]}"),
                       width="stretch")
    else:
        a.caption("This model has no built-in feature importance; see permutation importance.")
    perm = report_csv("permutation_importance.csv")
    if perm is not None and model_key in set(perm["model_key"]):
        top = perm[perm.model_key == model_key].head(15)[::-1]
        b.plotly_chart(px.bar(top, x="importance_mean", y="feature", orientation="h",
                              error_x="importance_std", template=TEMPLATE,
                              title="Permutation importance (drop in ROC-AUC)"),
                       width="stretch")
    else:
        b.info("Run `python -m src.explainability` to compute permutation importance.")

    st.subheader("Explain one record (SHAP)")
    if model_key not in SHAP_SUPPORTED:
        st.caption(f"Record-level SHAP is available for "
                   f"{', '.join(config.MODEL_DISPLAY_NAMES[k] for k in SHAP_SUPPORTED)}. "
                   "KNN and SVM would need the very slow KernelExplainer.")
        return
    preds = current_predictions()
    if preds is None:
        return
    flagged_first = preds.sort_values("attack_score", ascending=False).index.tolist()
    idx = st.selectbox("Row number (sorted by attack score)", flagged_first[:2000])
    record = preds.loc[[idx]]
    background = preds.sample(min(config.SHAP_BACKGROUND_SIZE, len(preds)),
                              random_state=config.RANDOM_STATE)
    with st.spinner("Computing SHAP values ..."):
        result = explain_record(bundle, record[list(config.FEATURE_COLUMNS)],
                                background[list(config.FEATURE_COLUMNS)])
    table = result["table"].head(12)
    c1, c2 = st.columns([1, 2])
    c1.metric("Prediction", record["predicted_label"].iloc[0])
    c1.metric("Attack score", f"{record['attack_score'].iloc[0]:.3f}")
    c1.caption(f"Base value (average model output): {result['base_value']:.3f} "
               f"in units of {result['units']}.")
    fig = px.bar(table[::-1], x="contribution", y="feature", orientation="h", template=TEMPLATE,
                 color=np.where(table[::-1]["contribution"] > 0, "towards attack",
                                "towards benign"),
                 color_discrete_map={"towards attack": "#ff6b6b", "towards benign": "#2ec4b6"},
                 hover_data=["value"], title="Top feature contributions for this record")
    c2.plotly_chart(fig, width="stretch")
    st.dataframe(table.assign(value=table["value"].astype(str)), width="stretch",
                 hide_index=True)

    with st.expander("Limitations of these explanations"):
        st.markdown(
            "- SHAP values are computed on the model's *internal* inputs (one-hot / scaled) "
            "and summed back to the original column.\n"
            "- Correlated features share credit in ways that can be hard to interpret.\n"
            "- Permutation importance can understate features that are correlated with others.\n"
            "- Explanations describe the model, which may have learned dataset artefacts "
            "(e.g. TTL values specific to the UNSW testbed)."
        )


# ==========================================================================
# Page F: Reports
# ==========================================================================
def page_reports() -> None:
    st.title("Downloadable Reports")
    st.caption("Offline analysis outputs. These are not verified security incident reports.")

    preds = current_predictions() if bundle is not None and data is not None else None
    summary = None
    if preds is not None:
        summary = {
            "file": data_name, "model": config.MODEL_DISPLAY_NAMES[model_key],
            "threshold": round(threshold, 3), "records": len(preds),
            "flagged_potentially_malicious": int(preds["prediction"].sum()),
        }
        st.download_button("⬇️ Predictions for the current file (CSV)",
                           predictions_to_csv(preds), f"predictions_{model_key}.csv",
                           "text/csv")
    for label, name in [("Model comparison (CSV)", "model_comparison.csv"),
                        ("Feature-selection experiment (CSV)", "feature_selection_results.csv"),
                        ("Anomaly-detection experiment (CSV)", "anomaly_detection_results.csv"),
                        ("Permutation importance (CSV)", "permutation_importance.csv")]:
        path = config.REPORTS_DIR / name
        if path.exists():
            st.download_button(f"⬇️ {label}", path.read_bytes(), name, "text/csv")
        else:
            st.caption(f"{label}: not generated yet.")

    report_md = build_markdown_report(summary)
    st.download_button("⬇️ Summary report (Markdown)", report_md.encode("utf-8"),
                       "nids_analysis_report.md", "text/markdown")
    with st.expander("Preview summary report"):
        st.markdown(report_md)


PAGES = {
    "Overview": page_overview,
    "Traffic Analysis": page_traffic,
    "Intrusion Detection": page_detection,
    "Model Comparison": page_comparison,
    "Explainability": page_explainability,
    "Reports": page_reports,
}
PAGES[page]()
st.sidebar.divider()
st.sidebar.caption("For defensive, educational analysis of public or authorised data only.")
