import streamlit as st;

import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)



ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST_CSV = ROOT / "test_data.csv"
TARGET = "Revenue"

metrics = {
  "Logistic Regression": "logistic_regression.joblib",
  "Decision Tree": "decision_tree.joblib",
  "kNN": "knn.joblib",
  "Naive Bayes": "naive_bayes.joblib",
  "Random Forest": "random_forest.joblib"
}

st.set_page_config(
    page_title="Online Shopper Conversion Classifier",
    page_icon="🛒",
    layout="wide",
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained models...")
def load_models() -> dict:

    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in metrics.items()
        if (MODEL_DIR / filename).exists()
    }

@st.cache_data(show_spinner=False)
def load_bundled_test_data() -> pd.DataFrame | None:
    if DEFAULT_TEST_CSV.exists():
        return pd.read_csv(DEFAULT_TEST_CSV)
    return None

def coerce_target(series: pd.Series) -> pd.Series:
    """Accept TRUE/FALSE, True/False or 1/0 in the uploaded file."""
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        return (
            series.astype(str)
            .str.strip()
            .str.upper()
            .map({"TRUE": 1, "FALSE": 0, "1": 1, "0": 0})
            .astype(int)
        )
    return series.astype(int)

def score(model, X, y_true) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
models = load_models()

st.sidebar.title("🛒 Controls")

if not models:
    st.sidebar.error("No trained models found.")
    st.title("Online Shopper Conversion Classifier")
    st.error("No models in `model/`")
    st.stop()

uploaded = st.sidebar.file_uploader(
    "Upload test data (CSV)",
    type="csv",
    help="Must contain the 17 feature columns plus the Revenue target column.",
)

chosen = st.sidebar.selectbox("Choose a model", list(models.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: UCI Online Shoppers Purchasing Intention ")

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
st.title("Online Shopper Conversion Classifier")

st.markdown(
    "Predicting whether an e-commerce browsing session ends in a purchase. "
    "Five classifiers trained on the UCI *Online Shoppers Purchasing Intention* dataset."
)

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.success(f"Loaded **{uploaded.name}** — {len(data):,} rows.")
else:
    data = load_bundled_test_data()
    if data is None:
        st.warning("Upload a CSV to begin — no bundled `test_data.csv` was found.")
        st.stop()
    st.info(f"Using the bundled `test_data.csv` — {len(data):,} rows. Upload your own from the sidebar.")

if TARGET not in data.columns:
    st.error(f"The uploaded file has no `{TARGET}` column, so it cannot be scored.")
    st.stop()

y_true = coerce_target(data[TARGET])
X = data.drop(columns=[TARGET])

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_single, tab_compare = st.tabs(
    ["Selected model", "Compare all models"]
)

with tab_single:
    try:
        result = score(models[chosen], X, y_true)
    except Exception as exc:
        st.error(f"Could not score this file with **{chosen}**: {exc}")
        st.stop()

    st.subheader(f"{chosen} — evaluation metrics")

    cols = st.columns(6)
    for col, metric in zip(
        cols, ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ):
        col.metric(metric, f"{result[metric]:.4f}")

    st.markdown("---")
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, result["y_pred"])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["No purchase", "Purchase"],
            yticklabels=["No purchase", "Purchase"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y_true, result["y_proba"])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        ax.plot(fpr, tpr, label=f"AUC = {result['AUC']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Classification report")
    report = classification_report(
        y_true,
        result["y_pred"],
        target_names=["No purchase", "Purchase"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with tab_compare:
    st.subheader("All models comparison")

    with st.spinner("Loading..."):
        rows = []
        for name, model in models.items():
            try:
                s = score(model, X, y_true)
            except Exception:
                continue
            rows.append(
                {
                    "ML Model Name": name,
                    **{m: round(s[m], 4) for m in
                    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]},
                }
            )

        if not rows:
            st.error("None of the models could score this file.")
        else:
            table = pd.DataFrame(rows).set_index("ML Model Name")
            st.dataframe(
                table.style.highlight_max(axis=0, color="#c6f0d4"),
                use_container_width=True,
            )
