from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import MODEL_DIR, RESULT_DIR
from .data_preprocessing import clean_dataframe, load_raw_data, train_test_data
from .feature_engineering import engineer_features


def permutation_feature_importance(model, X_test, y_test, n_repeats=5):
    result = permutation_importance(
        model, X_test, y_test,
        scoring="f1",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean,
    }).sort_values("Importance", ascending=False)
    importance.to_csv(RESULT_DIR / "permutation_feature_importance.csv", index=False)
    return importance


def build_global_explanation():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    df = engineer_features(clean_dataframe(load_raw_data()))
    X_train, X_test, y_train, y_test = train_test_data(df)
    return permutation_feature_importance(model, X_test, y_test)


def local_explanation(model, customer_row: pd.DataFrame) -> pd.DataFrame:
    """Use SHAP when available; otherwise return a transparent fallback message."""
    try:
        import shap
        explainer = shap.Explainer(model.predict_proba, customer_row)
        values = explainer(customer_row)
        # Binary-class explanation: use class 1 when SHAP returns 3-D values.
        arr = values.values
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        scores = arr[0]
        return pd.DataFrame({
            "Feature": customer_row.columns,
            "Contribution": scores,
            "Direction": np.where(scores >= 0, "Increases churn", "Decreases churn"),
        }).sort_values("Contribution", key=np.abs, ascending=False)
    except Exception:
        return pd.DataFrame({
            "Feature": ["Explanation"],
            "Contribution": [0.0],
            "Direction": [
                "Local SHAP unavailable for this environment; use global permutation importance."
            ],
        })
