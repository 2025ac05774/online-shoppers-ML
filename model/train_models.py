import sys

import pandas as pd
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"


RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Revenue"

CLASS_WEIGHT = "balanced"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

def load_dataset() -> pd.DataFrame:

    print("fetching from the UCI repository...")
    
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        sys.exit(
            "ucimlrepo is not installed and no local CSV was found"
        )

    repo = fetch_ucirepo(id=468)
    frame = pd.concat([repo.data.features, repo.data.targets], axis=1)

    return frame

def to_binary(series: pd.Series) -> pd.Series:
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

def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight=CLASS_WEIGHT,
            random_state=RANDOM_STATE,
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
        ),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight=CLASS_WEIGHT,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def main() -> None:
    frame = load_dataset()
    print(f"Loaded {len(frame):,} rows x {frame.shape[1]} columns")

    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = expected - set(frame.columns)
    if missing:
        sys.exit(f"Dataset is missing expected columns: {sorted(missing)}")

    X = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = to_binary(frame[TARGET])
    print(f"positive ({int(y.sum()):,} of {len(y):,})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,}   Test: {len(X_test):,}")

    MODEL_DIR.mkdir(exist_ok=True)
    results: dict = {}
    index: dict = {}

    for name, estimator in build_models().items():
        print(f"\nTraining {name}...")
        pipeline = Pipeline(
            [("preprocess", make_preprocessor()), ("classifier", estimator)]
        )
        pipeline.fit(X_train, y_train)

        results[name] = evaluate(pipeline, X_test, y_test)

        filename = f"{slugify(name)}.joblib"
        joblib.dump(pipeline, MODEL_DIR / filename)
        index[name] = filename
        print("   " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.values
    test_frame.to_csv(ROOT / "test_data.csv", index=False)


if __name__ == "__main__":
    main()