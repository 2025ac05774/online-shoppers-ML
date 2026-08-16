import streamlit as st;

import joblib
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