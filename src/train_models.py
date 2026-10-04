from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from .config import MODEL_DIR, RANDOM_STATE, RESULT_DIR, TARGET
from .data_preprocessing import clean_dataframe, load_raw_data, train_test_data
from .feature_engineering import engineer_features

warnings.filterwarnings("ignore")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def make_pipeline(model, X: pd.DataFrame) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("feature_selection", SelectPercentile(score_func=f_classif, percentile=80)),
        ("classifier", model),
    ])


def model_registry():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": BernoulliNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Support Vector Machine": SVC(
            probability=True, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }


def train_all():
    raw = load_raw_data()
    clean = clean_dataframe(raw)
    clean = engineer_features(clean)

    X_train, X_test, y_train, y_test = train_test_data(clean)

    results = []
    fitted = {}

    for name, classifier in model_registry().items():
        pipeline = make_pipeline(classifier, X_train)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        fitted[name] = pipeline

    comparison = pd.DataFrame(results).set_index("Model")
    # Churn-oriented selection: prioritize F1 and ROC-AUC rather than accuracy alone.
    comparison["SelectionScore"] = (
        0.45 * comparison["F1"] + 0.35 * comparison["Recall"]
        + 0.20 * comparison["ROC-AUC"]
    )
    best_name = comparison["SelectionScore"].idxmax()
    best_model = fitted[best_name]

    comparison.to_csv(RESULT_DIR / "model_comparison.csv")

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(best_model.named_steps["preprocessor"], MODEL_DIR / "scaler.pkl")
    joblib.dump(best_model.named_steps["preprocessor"], MODEL_DIR / "encoding_pipeline.pkl")
    joblib.dump(list(X_train.columns), MODEL_DIR / "feature_columns.pkl")

    metadata = {
        "best_model": best_name,
        "target": TARGET,
        "risk_thresholds": {"low_max": 0.30, "medium_max": 0.60},
        "selection_score": "0.45*F1 + 0.35*Recall + 0.20*ROC-AUC",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return comparison, best_name, best_model, (X_test, y_test)


if __name__ == "__main__":
    comparison, best_name, _, _ = train_all()
    print(comparison.round(4))
    print(f"\nSelected model: {best_name}")
