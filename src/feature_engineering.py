from __future__ import annotations

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create only features that can be computed from fields available at prediction time."""
    data = df.copy()

    if "tenure" in data.columns:
        data["tenure_group"] = pd.cut(
            pd.to_numeric(data["tenure"], errors="coerce"),
            bins=[-1, 6, 12, 24, 48, 72, float("inf")],
            labels=["0-6", "7-12", "13-24", "25-48", "49-72", "73+"],
        ).astype("object")

    if {"TotalCharges", "tenure"}.issubset(data.columns):
        tenure = pd.to_numeric(data["tenure"], errors="coerce").replace(0, pd.NA)
        total = pd.to_numeric(data["TotalCharges"], errors="coerce")
        data["avg_monthly_spend"] = (total / tenure).astype(float)

    if {"MonthlyCharges", "tenure"}.issubset(data.columns):
        monthly = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
        tenure = pd.to_numeric(data["tenure"], errors="coerce")
        data["estimated_account_value"] = monthly * tenure

    return data
