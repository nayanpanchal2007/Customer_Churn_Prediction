from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from .config import MODEL_DIR, RESULT_DIR, RANDOM_STATE
from .data_preprocessing import clean_dataframe, load_raw_data, train_test_data
from .feature_engineering import engineer_features
from .train_models import build_preprocessor


def tune_models():
    df = engineer_features(clean_dataframe(load_raw_data()))
    X_train, X_test, y_train, y_test = train_test_data(df)

    candidates = {
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {"classifier__n_estimators": [150, 300],
             "classifier__max_depth": [None, 8, 15],
             "classifier__min_samples_split": [2, 5],
             "classifier__min_samples_leaf": [1, 2]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"classifier__max_depth": [4, 8, None],
             "classifier__min_samples_split": [2, 5, 10],
             "classifier__min_samples_leaf": [1, 2, 4]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"classifier__n_neighbors": [5, 11, 15, 21],
             "classifier__weights": ["uniform", "distance"],
             "classifier__metric": ["euclidean", "manhattan"]},
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"classifier__C": [0.5, 1, 2],
             "classifier__kernel": ["rbf", "linear"],
             "classifier__gamma": ["scale", "auto"]},
        ),
    }

    rows = []
    tuned_models = {}

    from .train_models import make_pipeline, evaluate_model

    for name, (estimator, grid) in candidates.items():
        pipeline = make_pipeline(estimator, X_train)
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="f1",
            cv=5,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        metrics = evaluate_model(search.best_estimator_, X_test, y_test)
        metrics["Model"] = name
        metrics["Best Params"] = str(search.best_params_)
        rows.append(metrics)
        tuned_models[name] = search.best_estimator_

    result = pd.DataFrame(rows).set_index("Model")
    result.to_csv(RESULT_DIR / "tuned_model_comparison.csv")

    # Save the tuned model with highest F1.
    best_name = result["F1"].idxmax()
    joblib.dump(tuned_models[best_name], MODEL_DIR / "tuned_best_model.pkl")
    (MODEL_DIR / "tuning_metadata.json").write_text(
        json.dumps({"best_tuned_model": best_name}, indent=2)
    )
    return result


if __name__ == "__main__":
    print(tune_models().round(4))
